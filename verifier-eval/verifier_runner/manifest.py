"""Run manifest — records what was run, with what deps, against what staging.

Persisted as ``results/<run_id>/manifest.json``. Captures enough context to
answer "what numbers did we publish?" reproducibly:

* Eval-repo git sha, dirty state.
* Staging dir + cohort.
* Verifiers + their resolved models + per-runner kwargs.
* Verifier-source pins (gandalf-the-grader version, archipelago repo sha,
  agent-as-a-judge repo sha, autorubric pip version).
* Env-var presence (NOT values — never write API keys to disk).
* Wall-clock window for the run.
* Counts of verdicts/skips/errors.

Manifests are append-update: re-running ``evaluate --resume`` adds entries
to existing fields rather than overwriting.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import subprocess
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path


_API_KEY_ENV_VARS = (
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "ANTHROPIC_API_KEY",
    "DEEPSEEK_API_KEY",
    "GROQ_API_KEY",
    "OPENROUTER_API_KEY",
    "REDUCTO_API_KEY",
    "XAI_API_KEY",
    "LLM_API_KEY",
    "CLAUDE_CODE_OAUTH_TOKEN",
)


@dataclass
class Manifest:
    run_id: str
    cohort: str
    staging_id: str
    staging_jobs_dir: str
    verifiers: list[str]
    models: dict[str, str]                 # resolved model per verifier
    runner_kwargs: dict[str, dict]         # explicit kwargs passed at evaluate time
    started_at: str                        # ISO 8601
    finished_at: str | None = None
    eval_git_sha: str | None = None
    eval_git_dirty: bool = False
    verifier_pins: dict[str, str] = field(default_factory=dict)
    api_keys_present: list[str] = field(default_factory=list)
    n_verdicts: int = 0
    n_skips: int = 0
    n_errors: int = 0
    # Verifier kind that produced the staging artifacts. "gandalf" |
    # "rewardkit". Defaulted to "gandalf" so old manifests parse cleanly.
    verifier_kind: str = "gandalf"
    trajectory_provenance: dict[str, str] = field(default_factory=dict)
    trial_index: int | None = None
    allow_synthetic_trajectories: bool = False


def begin(
    *,
    run_id: str,
    cohort: str,
    staging_id: str,
    staging_jobs_dir: Path,
    verifiers: list[str],
    models: dict[str, str],
    runner_kwargs: dict[str, dict] | None = None,
    eval_repo: Path | None = None,
    verifier_kind: str = "gandalf",
    trial_index: int | None = None,
    allow_synthetic_trajectories: bool = False,
) -> Manifest:
    """Build a fresh Manifest at the start of an evaluate run."""
    return Manifest(
        run_id=run_id,
        cohort=cohort,
        staging_id=staging_id,
        staging_jobs_dir=str(staging_jobs_dir),
        verifiers=list(verifiers),
        models=dict(models),
        runner_kwargs=runner_kwargs or {},
        started_at=_now_iso(),
        eval_git_sha=_git_sha(eval_repo),
        eval_git_dirty=_git_dirty(eval_repo),
        verifier_pins=_collect_verifier_pins(),
        api_keys_present=_present_api_keys(),
        verifier_kind=verifier_kind,
        trial_index=trial_index,
        allow_synthetic_trajectories=allow_synthetic_trajectories,
    )


def finalize(
    manifest: Manifest,
    *,
    n_verdicts: int,
    n_skips: int,
    n_errors: int,
) -> Manifest:
    manifest.finished_at = _now_iso()
    manifest.n_verdicts = n_verdicts
    manifest.n_skips = n_skips
    manifest.n_errors = n_errors
    return manifest


def write(manifest: Manifest, results_dir: Path) -> Path:
    p = results_dir / manifest.run_id / "manifest.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(manifest), indent=2) + "\n")
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _git_sha(repo: Path | None) -> str | None:
    if repo is None:
        repo = Path(__file__).resolve().parent.parent
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL,
        )
        return out.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _git_dirty(repo: Path | None) -> bool:
    if repo is None:
        repo = Path(__file__).resolve().parent.parent
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, text=True, stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _collect_verifier_pins() -> dict[str, str]:
    """Best-effort: record version/sha for each verifier source.

    Failures (missing CLI, repo not found) are recorded as the literal
    string ``"unknown"`` so the manifest still serializes.
    """
    pins: dict[str, str] = {}

    eval_root = Path(__file__).resolve().parents[1]
    repo_root = eval_root.parent

    # Gandalf executes inside Harbor, so the generated image pin is
    # authoritative. A host CLI on PATH may be an unrelated version.
    pins["gandalf-the-grader"] = (
        _dependency_pin(
            repo_root / "adapters" / "bvb" / "template" / "environment" / "Dockerfile",
            "gandalf-the-grader",
        )
        or "unknown"
    )

    # Vendored sources record their immutable upstream shas in UPSTREAM.md.
    for name, upstream in (
        ("archipelago", eval_root / "verifiers" / "archipelago" / "UPSTREAM.md"),
        ("agent-as-a-judge", eval_root / "verifiers" / "aaaj" / "UPSTREAM.md"),
    ):
        pins[name] = _upstream_sha(upstream) or "unknown"

    pins["autorubric"] = (
        _project_dependency_pin(
            eval_root / "verifiers" / "autorubric" / "pyproject.toml",
            "autorubric",
        )
        or "unknown"
    )
    pins["harbor-rewardkit"] = (
        _dependency_pin(
            repo_root / "adapters" / "bvb" / "template" / "_tests-rewardkit" / "test.sh",
            "harbor-rewardkit",
        )
        or "unknown"
    )

    return pins


def _upstream_sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    match = re.search(r"\b[0-9a-f]{40}\b", path.read_text())
    return match.group(0) if match else None


def _project_dependency_pin(path: Path, package: str) -> str | None:
    if not path.is_file():
        return None
    dependencies = tomllib.loads(path.read_text()).get("project", {}).get("dependencies", [])
    prefix = f"{package}=="
    for dependency in dependencies:
        if isinstance(dependency, str) and dependency.startswith(prefix):
            return dependency.removeprefix(prefix)
    return None


def _dependency_pin(path: Path, package: str) -> str | None:
    if not path.is_file():
        return None
    match = re.search(rf"{re.escape(package)}(?:\[[^\]]+\])?==([A-Za-z0-9_.+-]+)", path.read_text())
    return match.group(1) if match else None


def _present_api_keys() -> list[str]:
    return [k for k in _API_KEY_ENV_VARS if os.environ.get(k)]
