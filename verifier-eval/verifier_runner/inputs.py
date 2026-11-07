"""Resolve BVB generated tasks and immutable human-feedback artifacts."""

from __future__ import annotations

import json
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.bvb.config import COHORT
from adapters.bvb.release import BVBRelease


# ---------------------------------------------------------------------------
# Cohort registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CohortConfig:
    """Per-cohort layout knobs."""

    cohort_subdir: str
    task_id_prefix: str = "btb-"


COHORTS: dict[str, CohortConfig] = {
    COHORT: CohortConfig(cohort_subdir=COHORT),
}


KNOWN_COHORTS: tuple[str, ...] = tuple(COHORTS)


# ---------------------------------------------------------------------------
# Verifier kinds
# ---------------------------------------------------------------------------
#
VERIFIER_KINDS: tuple[str, ...] = ("gandalf", "rewardkit")
DEFAULT_VERIFIER_KIND = "gandalf"
_DATASET_ROOTS = {
    "gandalf": "gandalf",
    "rewardkit": "rewardkit",
}


def cohort_config(cohort: str) -> CohortConfig:
    if cohort not in COHORTS:
        raise ValueError(f"Unknown cohort: {cohort!r}. Known: {KNOWN_COHORTS}")
    return COHORTS[cohort]


def cohort_dataset_subdir(cohort: str, verifier_kind: str = DEFAULT_VERIFIER_KIND) -> str:
    """Return the BVB generated-task subdirectory for a cohort."""
    if verifier_kind not in VERIFIER_KINDS:
        raise ValueError(
            f"Unknown verifier_kind {verifier_kind!r}. Known: {VERIFIER_KINDS}"
        )
    return f"{_DATASET_ROOTS[verifier_kind]}/{cohort_config(cohort).cohort_subdir}"


def staging_jobs_subdir(cohort: str, verifier_kind: str = DEFAULT_VERIFIER_KIND) -> str:
    """Return the per-(cohort, kind) staging dir name.

    Used to derive ``jobs/<staging_jobs_subdir>/`` so the kinds can
    coexist on disk (re-staging one doesn't clobber the others).
    """
    if verifier_kind not in VERIFIER_KINDS:
        raise ValueError(
            f"Unknown verifier_kind {verifier_kind!r}. Known: {VERIFIER_KINDS}"
        )
    suffix = "" if verifier_kind == "gandalf" else f"__{verifier_kind}"
    return f"staging__{cohort}{suffix}"


# Marker fields on Harbor's placeholder trajectory (see test.sh in any task).
# Real ATIF trajectories have many steps and a real session_id.
_PLACEHOLDER_SESSION_IDS = {"verify-only"}


@dataclass(frozen=True)
class InputBundle:
    """Everything a runner needs to grade one (cohort, task_id) pair.

    Built once by :func:`resolve` and passed to each runner's ``invoke()``.
    Paths point at *staged* artifacts (under ``jobs/<staging_id>/<trial>/``)
    when the bundle is task-specific to a staging run, or at the dataset
    when only static fields (rubric, instructions, ground truth) are needed.
    """

    # Identity
    cohort: str
    task_id: str
    gt_task_id: str

    # Static (from dataset)
    rubric: list[dict]
    instructions: str
    judge_guidance: str | None
    task_dir: Path
    ground_truth_path: Path
    release_root: Path
    canonical_rubric_sha256: str

    # Dynamic (from a staging run, optional)
    workspace_dir: Path | None
    trajectory_path: Path | None
    trial_dir: Path | None
    trajectory_is_placeholder: bool
    trajectory_provenance: str | None
    input_identity: dict | None = None


def validate_bundle(bundle: InputBundle) -> None:
    """Recheck the values actually passed to a judge against the pinned release."""
    if bundle.input_identity is None:
        return  # Historical bundles retain their legacy contract.
    from adapters.bvb.contract import canonical_rows, input_identity, validate_rubric, tree_hash, file_hash
    release = BVBRelease.open(bundle.release_root)
    task = release.tasks[bundle.task_id]
    validate_rubric(bundle.rubric, canonical_rows(task))
    if bundle.input_identity != input_identity(release, task) or bundle.instructions != release.instruction_for(task):
        raise ValueError('Judge bundle differs from pinned task inputs')
    if bundle.workspace_dir is None or bundle.trajectory_path is None:
        raise ValueError('Judge bundle lacks authentic workspace or trajectory')
    if (tree_hash(bundle.workspace_dir) != bundle.input_identity['workspace_sha256']
            or file_hash(bundle.trajectory_path) != bundle.input_identity['trajectory_sha256']):
        raise ValueError('Judge bundle replay differs from human-reviewed rollout')


# ---------------------------------------------------------------------------
# Cohort / dataset / GT helpers
# ---------------------------------------------------------------------------

def cohort_dataset_dir(
    cohort: str,
    datasets_root: Path,
    verifier_kind: str = DEFAULT_VERIFIER_KIND,
) -> Path:
    """Return the dataset directory for ``(cohort, verifier_kind)``."""
    return datasets_root / cohort_dataset_subdir(cohort, verifier_kind)


def list_tasks(
    cohort: str,
    datasets_root: Path,
    verifier_kind: str = DEFAULT_VERIFIER_KIND,
    *,
    dataset_dir: Path | None = None,
) -> list[str]:
    """List task IDs in a cohort, in sorted order."""
    cfg = cohort_config(cohort)
    cohort_dir = dataset_dir if dataset_dir is not None else cohort_dataset_dir(
        cohort, datasets_root, verifier_kind
    )
    return sorted(
        p.name for p in cohort_dir.iterdir()
        if p.is_dir() and p.name.startswith(cfg.task_id_prefix)
    )


def gt_task_id_for(cohort: str, task_id: str, release_root: Path) -> str:
    """BVB ground truth always uses the exact release task ID."""
    if cohort not in KNOWN_COHORTS:
        raise ValueError(f"Unknown cohort: {cohort!r}")
    if task_id not in BVBRelease.open(release_root).tasks:
        raise FileNotFoundError(f"Task {task_id!r} is absent from the BVB release")
    return task_id


# ---------------------------------------------------------------------------
# grader.toml parsing — minimal, single-line string extraction
# ---------------------------------------------------------------------------

def _extract_toml_string(text: str, field: str) -> str | None:
    """Extract a single-line ``field = "..."`` value from TOML text."""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith(field):
            continue
        if "=" not in stripped:
            continue
        _, _, value = stripped.partition("=")
        value = value.strip()
        if value.startswith('"') and value.endswith('"'):
            return value[1:-1].replace('\\"', '"').replace("\\n", "\n")
    return None


def _read_grader_toml(task_dir: Path) -> dict[str, str]:
    """Pull ``instructions`` and ``model`` from ``tests/grader.toml``."""
    toml_path = task_dir / "tests" / "grader.toml"
    if not toml_path.is_file():
        raise FileNotFoundError(f"grader.toml not found: {toml_path}")
    data = tomllib.loads(toml_path.read_text())
    instructions = data.get("instructions")
    if not isinstance(instructions, str):
        raise ValueError(f"grader.toml at {toml_path} has no `instructions` field")
    model = str(data.get("model") or "")
    return {"instructions": instructions, "model": model}


# ---------------------------------------------------------------------------
# Trajectory placeholder detection
# ---------------------------------------------------------------------------

def is_placeholder_trajectory(trajectory_path: Path) -> bool:
    """Return True iff the trajectory file is Harbor's verify-only placeholder."""
    try:
        data = json.loads(trajectory_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if data.get("session_id") in _PLACEHOLDER_SESSION_IDS:
        return True
    if data.get("bvb_trajectory_provenance") == "synthetic_placeholder":
        return True
    steps = data.get("steps", [])
    if (
        len(steps) <= 1
        and data.get("agent", {}).get("name") == "nop"
    ):
        return True
    return False


# ---------------------------------------------------------------------------
# Bundle construction
# ---------------------------------------------------------------------------

def resolve(
    cohort: str,
    task_id: str,
    *,
    datasets_root: Path,
    release_root: Path,
    staging_jobs_dir: Path | None = None,
    trial_index: int | None = None,
    verifier_kind: str = DEFAULT_VERIFIER_KIND,
    dataset_dir: Path | None = None,
) -> InputBundle:
    """Build an :class:`InputBundle` for one (cohort, task_id, verifier_kind).

    The ``verifier_kind`` selects which dataset subdir is consulted (the
    rubric, instructions, and judge-guidance are kind-specific because the
    BTB rewardkit adapter writes them to a different ``tests/`` shape).
    Canonical labels and criterion identities always come from ``release_root``.
    """
    cohort_dir = dataset_dir if dataset_dir is not None else cohort_dataset_dir(
        cohort, datasets_root, verifier_kind
    )
    task_dir = cohort_dir / task_id
    if not task_dir.is_dir():
        raise FileNotFoundError(f"Task dir not found: {task_dir}")

    rubric_path = task_dir / "tests" / "rubric.json"
    if not rubric_path.is_file():
        raise FileNotFoundError(f"rubric.json not found: {rubric_path}")
    rubric = json.loads(rubric_path.read_text())

    # Gandalf-staged tasks carry instructions in tests/grader.toml.
    # Rewardkit-staged tasks don't ship grader.toml (rewardkit has no
    # equivalent config); their instructions live in instruction.md
    # (the agent-side spec, written by the base adapter).
    if (task_dir / "tests" / "grader.toml").is_file():
        instructions = _read_grader_toml(task_dir)["instructions"]
    elif (task_dir / "instruction.md").is_file():
        instructions = (task_dir / "instruction.md").read_text()
    else:
        raise FileNotFoundError(
            f"Neither tests/grader.toml nor instruction.md found in {task_dir}"
        )

    judge_guidance: str | None = None
    judge_path = task_dir / "tests" / "judge-guidance.md"
    if judge_path.is_file():
        judge_guidance = judge_path.read_text()

    release = BVBRelease.open(release_root)
    try:
        release_task = release.tasks[task_id]
    except KeyError as error:
        raise FileNotFoundError(f"Task {task_id} is absent from the BVB release") from error
    if task_id not in release.task_ids(cohort):
        raise ValueError(f"{task_id} does not belong to release cohort {cohort}")
    gt_task_id = task_id
    ground_truth_path = release.root / release_task.labels_path if release_task.labels_path else release.root / "human-adjudication"

    generated_manifest = task_dir / "bvb-task-manifest.json"
    if not generated_manifest.is_file():
        raise FileNotFoundError(f"BVB generated-task manifest not found: {generated_manifest}")
    generated = json.loads(generated_manifest.read_text())
    generated_kind = str(generated.get("verifier_kind") or "gandalf")
    if generated_kind != verifier_kind:
        raise ValueError(
            f"{task_id} was materialized for {generated_kind!r}, "
            f"not requested verifier kind {verifier_kind!r}"
        )
    canonical_digest = release_task.manifest.get("evaluation_rubric", {}).get("sha256")
    if generated.get("canonical_rubric_sha256") != canonical_digest:
        raise ValueError(f"{task_id} generated task uses a different canonical rubric")
    if release_task.criteria and len(rubric) != len(release_task.criteria):
        raise ValueError(f"{task_id} generated rubric count differs from release canonical rubric")

    from adapters.bvb.contract import canonical_rows, input_identity, validate_rubric, tree_hash, file_hash
    validate_rubric(rubric, canonical_rows(release_task))
    identity = input_identity(release, release_task)
    contract = json.loads((task_dir / "tests/input-contract.json").read_text())
    if contract["identity"] != identity or contract["instructions"] != instructions:
        raise ValueError(f"{task_id} input contract or task prompt differs from pinned release")
    validate_rubric(contract["rubric"], canonical_rows(release_task))
    if instructions != release.instruction_for(release_task):
        raise ValueError(f"{task_id} task prompt differs from frozen rollout prompt")
    workspace_dir: Path | None = None
    trajectory_path: Path | None = None
    trial_dir: Path | None = None
    trajectory_is_placeholder = False
    trajectory_provenance: str | None = generated.get("trajectory_provenance")

    if staging_jobs_dir is not None:
        candidates = list_staged_trials(staging_jobs_dir, task_id)
        if trial_index is None:
            trial_dir = _find_staged_trial(staging_jobs_dir, task_id)
        else:
            if trial_index < 0 or trial_index >= len(candidates):
                raise ValueError(
                    f"{task_id} has {len(candidates)} staged trial(s); "
                    f"cannot select trial index {trial_index}"
                )
            trial_dir = candidates[trial_index]
        if trial_dir is not None:
            from adapters.bvb.contract import staged_workspace
            ws = staged_workspace(trial_dir)
            if ws.is_dir():
                workspace_dir = ws
            traj = trial_dir / "agent" / "trajectory.json"
            if traj.is_file():
                trajectory_path = traj
                trajectory_is_placeholder = is_placeholder_trajectory(traj)
                if trajectory_is_placeholder:
                    trajectory_provenance = "synthetic_placeholder"

    if staging_jobs_dir is not None:
        if workspace_dir is None or trajectory_path is None:
            raise ValueError(f"{task_id} lacks authentic staged workspace or trajectory")
        if tree_hash(workspace_dir) != identity["workspace_sha256"] or file_hash(trajectory_path) != identity["trajectory_sha256"]:
            raise ValueError(f"{task_id} staged replay differs from human-reviewed rollout")

    return InputBundle(
        cohort=cohort,
        task_id=task_id,
        gt_task_id=gt_task_id,
        rubric=rubric,
        instructions=instructions,
        judge_guidance=judge_guidance,
        task_dir=task_dir,
        ground_truth_path=ground_truth_path,
        release_root=release.root,
        canonical_rubric_sha256=str(canonical_digest or ""),
        workspace_dir=workspace_dir,
        trajectory_path=trajectory_path,
        trial_dir=trial_dir,
        trajectory_is_placeholder=trajectory_is_placeholder,
        trajectory_provenance=trajectory_provenance,
        input_identity=identity,
    )


def _find_staged_trial(staging_jobs_dir: Path, task_id: str) -> Path | None:
    """Locate the trial subdirectory matching ``task_id`` under a staging dir.

    Harbor names trial dirs ``<task_id>__<rand>``. For BTB's short
    ``btb-XXXXXXXX`` IDs there's never any truncation, so we just match
    on the prefix.
    """
    candidates = list_staged_trials(staging_jobs_dir, task_id)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        f"Ambiguous staging for {task_id}: {', '.join(path.name for path in candidates)}. "
        "Pass a trial index when evaluating repeated verifier attempts."
    )


def list_staged_trials(staging_jobs_dir: Path, task_id: str) -> list[Path]:
    """Return every Harbor trial for a task in deterministic name order."""
    if not staging_jobs_dir.is_dir():
        return []
    candidates: list[Path] = []
    for entry in sorted(staging_jobs_dir.iterdir()):
        if not entry.is_dir():
            continue
        if "__" not in entry.name:
            continue
        prefix, _, _ = entry.name.rpartition("__")
        if prefix == task_id:
            candidates.append(entry)
    return candidates


__all__ = [
    "COHORTS",
    "CohortConfig",
    "InputBundle",
    "KNOWN_COHORTS",
    "cohort_config",
    "cohort_dataset_dir",
    "gt_task_id_for",
    "is_placeholder_trajectory",
    "list_tasks",
    "list_staged_trials",
    "resolve",
    "_read_grader_toml",
]
