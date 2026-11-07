"""Stage rollouts via Harbor.

Wraps ``harbor run -a oracle`` so one command stages a cohort's rollouts
into a deterministic ``jobs/staging__<cohort>/`` directory. The same
invocation produces:

* ``jobs/staging__<cohort>/<trial>/agent/workspace/`` — the rehydrated
  rollout workspace (per the dataset's ``solution/solve.sh``)
* ``jobs/staging__<cohort>/<trial>/agent/trajectory.json`` — the rollout's
  ATIF trajectory
* ``jobs/staging__<cohort>/<trial>/verifier/info.json`` — Gandalf's verdict
  (Harbor's verifier phase runs gandalf-the-grader inside the container,
  with full MCP access)

Verification is left **enabled** so the in-image verifier (Gandalf by
default; RewardKit when ``--verifier-kind rewardkit`` is selected)
produces its real verdict during staging — that's what gives the
in-image verifier its production-grade fidelity in the multi-verifier
eval. The three host-side runners (aaaj, archipelago, autorubric) read
from the same staged ``agent/`` artifacts after the fact.

Naming convention
-----------------
``--staging-id`` defaults to ``staging__<cohort>``, deterministic per
cohort. Override only when running parallel comparison stages.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .inputs import (
    DEFAULT_VERIFIER_KIND,
    KNOWN_COHORTS,
    VERIFIER_KINDS,
    cohort_config,
    cohort_dataset_dir,
    staging_jobs_subdir,
)


# Repo root that owns the ``datasets/`` and ``jobs/`` directories.
# Resolved as the parent of this verifier-eval/ subdir by default.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class StageResult:
    staging_id: str
    jobs_dir: Path
    cohort: str
    n_tasks_requested: int
    returncode: int


def default_staging_id(cohort: str, verifier_kind: str = DEFAULT_VERIFIER_KIND) -> str:
    """Deterministic staging ID for ``(cohort, verifier_kind)``.

    Gandalf keeps the legacy unsuffixed name (``staging__<cohort>``);
    rewardkit appends ``__rewardkit`` so the two coexist in ``jobs/``.
    """
    return staging_jobs_subdir(cohort, verifier_kind)


def stage(
    *,
    cohort: str,
    tasks: list[str] | None = None,
    staging_id: str | None = None,
    force_build: bool = False,
    env_file: Path | None = None,
    repo_root: Path | None = None,
    extra_harbor_args: list[str] | None = None,
    dry_run: bool = False,
    verifier_kind: str = DEFAULT_VERIFIER_KIND,
    dataset_dir: Path | None = None,
    environment_type: str = "docker",
    n_concurrent: int | None = None,
    n_attempts: int = 1,
    max_retries: int = 0,
    retry_include: list[str] | None = None,
    environment_kwargs: list[str] | None = None,
) -> StageResult:
    """Run ``harbor run -a oracle`` for ``(cohort, verifier_kind)``.

    Args:
        cohort: One of :data:`KNOWN_COHORTS`.
        verifier_kind: ``"gandalf"`` (default) or ``"rewardkit"``.
            Selects both the generated dataset path and a distinct staging
            directory so in-environment verifier outputs coexist on disk.
        tasks: Subset of ``btb-XXXXXXXX`` task IDs. None = all tasks.
        staging_id: Override the jobs/<name> directory name. Defaults to
            ``default_staging_id(cohort, verifier_kind)``.
        force_build: Pass ``--force-build`` to Harbor.
        env_file: Path to ``.env``. Defaults to ``<repo_root>/.env`` when
            it exists.
        repo_root: Path to the repo root (default: derived from this
            file's location).
        extra_harbor_args: Extra args appended to the harbor command.
        dry_run: Print the command and skip execution. Returns a result
            with ``returncode=0``.
    """
    if cohort not in KNOWN_COHORTS:
        raise ValueError(f"Unknown cohort: {cohort!r}. Known: {KNOWN_COHORTS}")
    if verifier_kind not in VERIFIER_KINDS:
        raise ValueError(
            f"Unknown verifier_kind {verifier_kind!r}. Known: {VERIFIER_KINDS}"
        )
    repo_root = repo_root or REPO_ROOT
    if not repo_root.is_dir():
        raise FileNotFoundError(f"repo root not found: {repo_root}")

    if shutil.which("harbor") is None:
        raise FileNotFoundError("`harbor` CLI not on PATH; install it first.")

    # Validate the cohort's dataset path before invoking Harbor — saves a
    # confusing harbor error if the dataset isn't generated yet.
    if dataset_dir is not None:
        cohort_dir = dataset_dir if dataset_dir.is_absolute() else repo_root / dataset_dir
        if not cohort_dir.is_dir():
            raise FileNotFoundError(f"--dataset-dir not found: {cohort_dir}")
    else:
        cohort_dir = cohort_dataset_dir(cohort, repo_root / "datasets", verifier_kind)
        if not cohort_dir.is_dir():
            raise FileNotFoundError(
                f"Cohort dataset not found at {cohort_dir}. "
                f"Generate it via `python -m adapters.bvb.run_adapter "
                f"--release-dir <downloaded-release> --cohort {cohort}` first."
            )

    if tasks:
        missing = [t for t in tasks if not (cohort_dir / t).is_dir()]
        if missing:
            raise FileNotFoundError(
                f"Cohort {cohort!r} ({verifier_kind}) has no task dirs for: {missing}"
            )

    cfg = cohort_config(cohort)

    staging_id = staging_id or default_staging_id(cohort, verifier_kind)
    jobs_dir = repo_root / "jobs" / staging_id

    if env_file is None:
        candidate = repo_root / ".env"
        if candidate.is_file():
            env_file = candidate

    cmd: list[str] = [
        "harbor", "run",
        "-p", str(cohort_dir.relative_to(repo_root)),
        "-a", "oracle",
        "-e", environment_type,
        "--job-name", staging_id,
        # Auto-confirm Harbor's "tasks declare host env vars; proceed?" prompt.
        # Without this, the eval pipeline blocks waiting on stdin — defeats
        # the point of orchestrating it.
        "--yes",
        # Keep containers/images alive after each trial. Harbor's default
        # teardown is `docker compose down --rmi all`, which removes
        # locally-tagged base images — every subsequent trial then fails
        # with "No such image". --no-delete skips the --rmi/--volumes
        # flags so locally-tagged base images survive. Trade-off: each
        # trial leaves stopped containers behind in `docker ps -a` until
        # the next stage cycle reuses them.
        "--no-delete",
    ]
    if force_build:
        cmd.append("--force-build")
    if n_concurrent is not None:
        if n_concurrent < 1:
            raise ValueError("n_concurrent must be at least 1")
        cmd += ["-n", str(n_concurrent)]
    if n_attempts < 1:
        raise ValueError("n_attempts must be at least 1")
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    cmd += ["-k", str(n_attempts), "--max-retries", str(max_retries)]
    for exception_name in retry_include or []:
        cmd += ["--retry-include", exception_name]
    for value in environment_kwargs or []:
        cmd += ["--ek", value]
    if env_file is not None:
        cmd += ["--env-file", str(env_file)]
    if tasks:
        for t in tasks:
            cmd += ["-i", t]
    if extra_harbor_args:
        cmd += list(extra_harbor_args)

    print(f"[stage] {' '.join(cmd)}")
    print(f"[stage] cwd={repo_root}")

    if dry_run:
        return StageResult(
            staging_id=staging_id,
            jobs_dir=jobs_dir,
            cohort=cohort,
            n_tasks_requested=(
                len(tasks) if tasks else _count_cohort_tasks(cohort_dir, cfg.task_id_prefix)
            ),
            returncode=0,
        )

    if jobs_dir.exists():
        raise FileExistsError(f'Refusing to overwrite existing Harbor job metadata: {jobs_dir}')

    result = subprocess.run(cmd, cwd=repo_root, env={**os.environ}, check=False)
    returncode = result.returncode
    result_path = jobs_dir / "result.json"
    if returncode == 0 and result_path.is_file():
        try:
            harbor_result = json.loads(result_path.read_text())
            errored_trials = int(harbor_result.get("stats", {}).get("n_errored_trials", 0))
        except (OSError, ValueError, json.JSONDecodeError):
            errored_trials = 0
        if errored_trials:
            print(
                f"[stage] Harbor completed with {errored_trials} errored trial(s); "
                "inspect the job result before evaluating.",
                file=sys.stderr,
            )
            returncode = 1
    n_requested = (
        len(tasks) if tasks else _count_cohort_tasks(cohort_dir, cfg.task_id_prefix)
    )
    return StageResult(
        staging_id=staging_id,
        jobs_dir=jobs_dir,
        cohort=cohort,
        n_tasks_requested=n_requested,
        returncode=returncode,
    )


def _count_cohort_tasks(cohort_dir: Path, task_id_prefix: str) -> int:
    return sum(
        1 for p in cohort_dir.iterdir()
        if p.is_dir() and p.name.startswith(task_id_prefix)
    )
