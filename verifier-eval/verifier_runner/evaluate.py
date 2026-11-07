"""Orchestrate verifier runs over a staged cohort.

For each (verifier, task) pair:

1. Resolve an :class:`InputBundle` for the task, anchored at the staging
   jobs dir so workspace + trajectory paths come from there.
2. Check the runner's :class:`Capabilities` against the bundle. Missing
   workspace or trajectory (or placeholder trajectory when the runner
   doesn't accept it) → record a skip and continue.
3. Otherwise call ``runner.invoke(bundle, model=...)`` and persist the
   :class:`Verdict` to ``results/<run_id>/verdicts/<verifier>/<task>.json``.

Verdicts are written one at a time so a crash on task #7 of 15 keeps
verdicts 1–6 on disk. ``--resume`` (handled at the CLI layer) re-uses
verdicts already on disk and only invokes runners for missing ones.

Skips are recorded in ``results/<run_id>/skips.csv``. They show up in the
final scores table as ``N Skipped > 0`` for the relevant verifier.

Errors are caught per (verifier, task) so one failure doesn't abort the
whole run; they're reported in the run summary at the end and the
process exits non-zero so CI/scripts can detect partial failure.
"""

from __future__ import annotations

import csv
import importlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from . import manifest as _manifest
from .identity import stamp_criterion_identity
from .inputs import DEFAULT_VERIFIER_KIND, InputBundle, list_tasks, resolve
from .runners.base import (
    SKIP_GANDALF_NO_INFO_JSON,
    SKIP_NO_REWARDKIT_OUTPUT,
    SKIP_NO_TRAJECTORY,
    SKIP_NO_WORKSPACE,
    SKIP_PLACEHOLDER_TRAJECTORY,
    SKIP_SYNTHETIC_TRAJECTORY,
    Capabilities,
)
from .verdict import Verdict


# verifier name → runner module path
_RUNNER_MODULES = {
    "gandalf": "verifier_runner.runners.gandalf",
    "rewardkit": "verifier_runner.runners.rewardkit",
    "aaaj": "verifier_runner.runners.aaaj",
    "archipelago": "verifier_runner.runners.archipelago",
    "autorubric": "verifier_runner.runners.autorubric",
}

KNOWN_VERIFIERS = tuple(_RUNNER_MODULES)


# Single shared default model used when the user does not pass
# ``--model``. Applied uniformly across every verifier (gandalf, aaaj,
# archipelago, autorubric, rewardkit). Tuned for Gemini-flash judging —
# see `verifier-eval/verifiers/aaaj/UPSTREAM.md` for the Gemini-compat
# patches that this default depends on.
#
# Override per-invocation with ``--model <model>`` (bare; same model
# for every verifier) or ``--model verifier=model,...`` (per-verifier).
DEFAULT_MODEL = "gemini/gemini-3-flash-preview"


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SkipRecord:
    verifier: str
    task_id: str
    reason: str
    duration_sec: float = 0.0


@dataclass(frozen=True)
class ErrorRecord:
    verifier: str
    task_id: str
    error: str
    duration_sec: float


@dataclass
class EvaluateResult:
    run_id: str
    verdicts_written: list[Path]
    skips: list[SkipRecord]
    errors: list[ErrorRecord]
    n_runner_invocations: int

    @property
    def n_failed(self) -> int:
        return len(self.errors)


# ---------------------------------------------------------------------------
# Capability check
# ---------------------------------------------------------------------------

def _check_skip(
    caps: Capabilities,
    bundle: InputBundle,
    verifier: str,
    *,
    allow_synthetic_trajectories: bool = False,
) -> str | None:
    """Return a skip reason string, or None if the runner can proceed."""
    if (
        bundle.trajectory_provenance == "synthetic_placeholder"
        and verifier not in {"gandalf", "rewardkit"}
        and not allow_synthetic_trajectories
    ):
        return SKIP_SYNTHETIC_TRAJECTORY
    if verifier == "gandalf":
        # Gandalf's verdict is the staged verifier/info.json; check that
        # explicitly so the skip message points at the real cause (Harbor
        # verification was disabled, or the staging trial errored, or the
        # cohort was staged under --verifier-kind=rewardkit).
        if bundle.trial_dir is None:
            return SKIP_NO_WORKSPACE  # no staged trial at all
        info = bundle.trial_dir / "verifier" / "info.json"
        if not info.is_file():
            return SKIP_GANDALF_NO_INFO_JSON
        return None

    if verifier == "rewardkit":
        if bundle.trial_dir is None:
            return SKIP_NO_WORKSPACE
        details = bundle.trial_dir / "verifier" / "reward-details.json"
        reward = bundle.trial_dir / "verifier" / "reward.json"
        if not details.is_file() or not reward.is_file():
            return SKIP_NO_REWARDKIT_OUTPUT
        return None

    if caps.needs_workspace and bundle.workspace_dir is None:
        return SKIP_NO_WORKSPACE
    if caps.needs_trajectory:
        if bundle.trajectory_path is None:
            return SKIP_NO_TRAJECTORY
        if (
            bundle.trajectory_is_placeholder
            and not caps.accepts_placeholder_trajectory
            and not allow_synthetic_trajectories
        ):
            return SKIP_PLACEHOLDER_TRAJECTORY
    return None


# ---------------------------------------------------------------------------
# Per-(verifier, task) execution
# ---------------------------------------------------------------------------

def _load_runner(verifier: str):
    if verifier not in _RUNNER_MODULES:
        raise ValueError(
            f"Unknown verifier {verifier!r}. Known: {KNOWN_VERIFIERS}"
        )
    return importlib.import_module(_RUNNER_MODULES[verifier])


def _has_param(func, name: str) -> bool:
    """True if ``func`` declares a keyword parameter ``name``."""
    try:
        import inspect
        return name in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


def _runner_default_model(runner) -> str | None:
    """Read ``DEFAULT_MODEL`` constant from a runner module if defined."""
    return getattr(runner, "DEFAULT_MODEL", None)


def _model_for_verifier(verifier: str, models: dict[str, str]) -> str | None:
    explicit = models.get(verifier)
    if explicit is not None:
        return explicit
    if verifier in {"gandalf", "rewardkit"}:
        return None
    return DEFAULT_MODEL


def _verdict_path(results_dir: Path, run_id: str, verifier: str, task_id: str) -> Path:
    return results_dir / run_id / "verdicts" / verifier / f"{task_id}.json"


def _trace_dir(results_dir: Path, run_id: str, verifier: str, task_id: str) -> Path:
    """Per-(verifier, task) directory for native verifier trace artifacts.

    Created by the runner on demand; held even on partial failures so
    the trace from a crashed run is inspectable.
    """
    return results_dir / run_id / "traces" / verifier / task_id


def _write_verdict(verdict: Verdict, results_dir: Path, run_id: str) -> Path:
    p = _verdict_path(results_dir, run_id, verdict.verifier, verdict.task_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x") as stream:
        stream.write(verdict.model_dump_json(indent=2) + "\n")
    return p


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def evaluate(
    *,
    verifiers: list[str],
    cohort: str,
    staging_jobs_dir: Path,
    results_dir: Path,
    run_id: str,
    datasets_root: Path,
    release_root: Path,
    tasks: list[str] | None = None,
    models: dict[str, str] | None = None,
    runner_kwargs: dict[str, dict] | None = None,
    resume: bool = False,
    quiet: bool = False,
    verbose: bool = False,
    verifier_kind: str = DEFAULT_VERIFIER_KIND,
    dataset_dir: Path | None = None,
    trial_index: int | None = None,
    allow_synthetic_trajectories: bool = False,
) -> EvaluateResult:
    """Run all (verifier × task) combinations sequentially.

    Args:
        verifiers: Subset of :data:`KNOWN_VERIFIERS`.
        cohort: Which cohort to evaluate against.
        staging_jobs_dir: ``jobs/<staging_id>/`` from a prior `stage` run.
        results_dir: Root where ``<run_id>/verdicts/<verifier>/`` lands.
        run_id: User-chosen run handle.
        tasks: Subset of cohort task IDs. None = all tasks in cohort.
        models: Per-verifier model override, e.g.
            ``{"gandalf": "gemini/gemini-3-flash-preview", ...}``.
        runner_kwargs: Per-verifier extra kwargs passed to ``invoke()``.
        resume: If True, skip (verifier, task) pairs whose verdict file
            already exists.
        quiet: Suppress per-task "starting" heartbeat (only print
            ``ok:`` / ``skip:`` / ``ERROR`` lines). Default heartbeats
            on so operators can see which task is currently in flight.
        verbose: Stream subprocess (worker) stdout/stderr live to the
            user's terminal. Useful for debugging or watching progress
            inside a runner. Implies not-quiet. Pass via ``runner_kwargs``
            to forward to runners that support it (currently aaaj,
            archipelago, autorubric — gandalf is no-op since its work
            already happened during stage).

    Returns an :class:`EvaluateResult`. Skips and errors are recorded but
    don't abort the run.
    """
    models = models or {}
    runner_kwargs = runner_kwargs or {}
    if (results_dir / run_id).exists() and not resume:
        raise FileExistsError(f'Refusing to overwrite an existing evaluation run: {results_dir / run_id}')
    bad = [v for v in verifiers if v not in _RUNNER_MODULES]
    if bad:
        raise ValueError(f"Unknown verifier(s): {bad}. Known: {KNOWN_VERIFIERS}")
    staged_model_overrides = sorted(
        verifier
        for verifier in ("gandalf", "rewardkit")
        if verifier in verifiers and verifier in models
    )
    if staged_model_overrides:
        raise ValueError(
            "Cannot override the model for in-environment verifier(s) "
            f"{staged_model_overrides}; their model was fixed during materialization/staging."
        )

    if tasks is None:
        tasks = list_tasks(cohort, datasets_root, verifier_kind, dataset_dir=dataset_dir)
    if not tasks:
        raise ValueError(f"No tasks resolved for cohort {cohort!r}")

    skips: list[SkipRecord] = []
    errors: list[ErrorRecord] = []
    written: list[Path] = []
    n_invocations = 0

    manifest = _manifest.begin(
        run_id=run_id, cohort=cohort,
        staging_id=staging_jobs_dir.name,
        staging_jobs_dir=staging_jobs_dir,
        verifiers=verifiers, models=models,
        runner_kwargs=runner_kwargs,
        verifier_kind=verifier_kind,
        trial_index=trial_index,
        allow_synthetic_trajectories=allow_synthetic_trajectories,
    )

    print(f"[evaluate] run_id={run_id} cohort={cohort} kind={verifier_kind} "
          f"verifiers={verifiers} tasks={len(tasks)} trial_index={trial_index} "
          f"allow_synthetic={allow_synthetic_trajectories} resume={resume}")

    for verifier in verifiers:
        runner = _load_runner(verifier)
        caps = runner.capabilities()
        kwargs = runner_kwargs.get(verifier, {})
        # Host-side judges share the Gemini default. In-environment verifiers
        # already ran during staging, so their authoritative model comes from
        # the staged output/task metadata unless the user explicitly overrides
        # it for translation.
        model = _model_for_verifier(verifier, models)
        runner_accepts_verbose = _has_param(runner.invoke, "verbose")

        for task_id in tasks:
            verdict_path = _verdict_path(results_dir, run_id, verifier, task_id)
            if verdict_path.exists() and not resume:
                raise FileExistsError(f"Refusing to overwrite existing verdict: {verdict_path}")

            try:
                bundle = resolve(
                    cohort, task_id,
                    datasets_root=datasets_root,
                    release_root=release_root,
                    staging_jobs_dir=staging_jobs_dir,
                    trial_index=trial_index,
                    verifier_kind=verifier_kind,
                    dataset_dir=dataset_dir,
                )
            except Exception as e:  # noqa: BLE001
                errors.append(ErrorRecord(
                    verifier=verifier, task_id=task_id,
                    error=f"resolve failed: {e}", duration_sec=0.0,
                ))
                print(f"[evaluate]   ERROR resolve {verifier} / {task_id}: {e}",
                      file=sys.stderr)
                continue
            if resume and verdict_path.is_file():
                prior = Verdict.model_validate_json(verdict_path.read_text())
                if prior.input_identity != bundle.input_identity or (model and prior.model != model):
                    raise ValueError(f"Cannot resume {task_id}: existing verdict uses different inputs or model")
                print(f"[evaluate]   skip (validated existing verdict): {verifier} / {task_id}")
                continue
            if bundle.trajectory_provenance is not None:
                manifest.trajectory_provenance[task_id] = bundle.trajectory_provenance

            reason = _check_skip(
                caps,
                bundle,
                verifier,
                allow_synthetic_trajectories=allow_synthetic_trajectories,
            )
            if reason is not None:
                skips.append(SkipRecord(verifier=verifier, task_id=task_id, reason=reason))
                print(f"[evaluate]   skip: {verifier} / {task_id} — {reason}")
                continue

            trace_dir = _trace_dir(results_dir, run_id, verifier, task_id)

            if not quiet:
                # Heartbeat before the (possibly multi-minute) invoke.
                # Tells the user which task is in flight without waiting
                # for the ``ok:`` line.
                resolved_model = (
                    model
                    or _runner_default_model(runner)
                    or "<from staged output/task>"
                )
                n_criteria = len(bundle.rubric)
                print(
                    f"[evaluate]   starting: {verifier} / {task_id} "
                    f"(model={resolved_model}, {n_criteria} criteria)",
                    flush=True,
                )

            trace_dir.mkdir(parents=True, exist_ok=True)
            request_path = trace_dir / "judge-inputs.json"
            with request_path.open("x") as stream:
                json.dump({"identity": bundle.input_identity, "instructions": bundle.instructions,
                           "rubric": bundle.rubric, "workspace": str(bundle.workspace_dir),
                           "trajectory": str(bundle.trajectory_path)}, stream, indent=2)
            invoke_kwargs = dict(kwargs)
            if verbose and runner_accepts_verbose and "verbose" not in invoke_kwargs:
                invoke_kwargs["verbose"] = True
            t0 = time.time()
            try:
                verdict = runner.invoke(
                    bundle, model=model, trace_dir=trace_dir, **invoke_kwargs
                )
                verdict = stamp_criterion_identity(verdict, bundle)
            except Exception as e:  # noqa: BLE001
                duration = time.time() - t0
                errors.append(ErrorRecord(
                    verifier=verifier, task_id=task_id,
                    error=f"{type(e).__name__}: {e}", duration_sec=duration,
                ))
                print(f"[evaluate]   ERROR invoke {verifier} / {task_id}: {e}",
                      file=sys.stderr)
                continue
            duration = time.time() - t0

            # Stamp duration on the verdict before persisting.
            verdict = verdict.model_copy(update={"duration_sec": round(duration, 2)})
            p = _write_verdict(verdict, results_dir, run_id)
            written.append(p)
            # Record the *resolved* model in the manifest — what was actually
            # used for the call. The user's --model dict only captures
            # explicit overrides; the runner's default-model fallback (e.g.
            # AAAJ's openai/gpt-5.4-nano default) wouldn't otherwise show up.
            manifest.models[verifier] = verdict.model
            n_invocations += 1
            print(f"[evaluate]   ok: {verifier} / {task_id} "
                  f"({len(verdict.criteria)} criteria, {duration:.1f}s)")

    _write_skips_csv(results_dir, run_id, skips, errors)
    # Count verdict files on disk under the run dir (cumulative across
    # any prior --resume invocations) so the manifest's n_verdicts
    # reflects the run's total state, not just what *this* invocation
    # added. Skips/errors stay per-invocation — they describe what
    # happened during this call.
    #
    # Also reconstruct manifest.models from existing verdicts when this
    # invocation's runners were no-ops (--resume with everything already
    # on disk leaves manifest.models empty otherwise).
    run_verdicts_dir = results_dir / run_id / "verdicts"
    n_verdicts_total = 0
    if run_verdicts_dir.is_dir():
        for v_dir in run_verdicts_dir.iterdir():
            if not v_dir.is_dir():
                continue
            verifier = v_dir.name
            for vp in v_dir.glob("*.json"):
                n_verdicts_total += 1
                if verifier in manifest.models:
                    continue
                try:
                    v = Verdict.model_validate_json(vp.read_text())
                except Exception:  # noqa: BLE001
                    continue
                if v.model:
                    manifest.models[verifier] = v.model
    _manifest.finalize(
        manifest, n_verdicts=n_verdicts_total,
        n_skips=len(skips), n_errors=len(errors),
    )
    _manifest.write(manifest, results_dir)
    return EvaluateResult(
        run_id=run_id,
        verdicts_written=written,
        skips=skips,
        errors=errors,
        n_runner_invocations=n_invocations,
    )


def _write_skips_csv(
    results_dir: Path, run_id: str,
    skips: list[SkipRecord], errors: list[ErrorRecord],
) -> Path:
    """Persist skips + errors to ``<results>/<run_id>/skips.csv``.

    Both kinds end up in the same CSV with a ``kind`` column so the file
    can be inspected without joining two tables.
    """
    p = results_dir / run_id / "skips.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "verifier", "task_id", "reason_or_error", "duration_sec"])
        for s in skips:
            w.writerow(["skip", s.verifier, s.task_id, s.reason, s.duration_sec])
        for e in errors:
            w.writerow(["error", e.verifier, e.task_id, e.error, round(e.duration_sec, 2)])
    return p
