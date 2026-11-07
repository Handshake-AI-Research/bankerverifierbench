"""BVB verifier-evaluation command-line surface.

Subcommands, one per pipeline phase:

* ``stage``    — run Harbor (oracle agent + Gandalf verifier) for a cohort.
* ``evaluate`` — run the chosen verifiers against a staged cohort. Auto-
                 builds ``traces_atif/`` and runs ``score`` on success
                 (suppress with ``--no-normalize`` / ``--no-score``).
* ``score``    — score the verdicts on disk against ground truth.
* ``traces``   — trace post-processing (``normalize`` builds ``traces_atif/``).

Run via ``python -m verifier_runner <subcommand> ...``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from .atif_normalize import normalize_run
from .evaluate import KNOWN_VERIFIERS, evaluate
from .inputs import DEFAULT_VERIFIER_KIND, KNOWN_COHORTS, VERIFIER_KINDS
from .scoring import (
    ConfusionCounts,
    VerifierScore,
    baseline_metrics,
    load_ground_truth,
    score_verifier,
)
from .stage import REPO_ROOT, default_staging_id, stage
from .verdict import Verdict


# Repo-relative defaults: results under verifier-eval/results/, generated
# tasks under <repo>/datasets/.
EVAL_ROOT = Path(__file__).resolve().parent.parent
from adapters.bvb.config import DEFAULT_RELEASE_DIR
DEFAULT_RESULTS_DIR = EVAL_ROOT / "results"
DEFAULT_DATASETS_ROOT = REPO_ROOT / "datasets"


# ---------------------------------------------------------------------------
# Helpers (path resolution, table printing, CSV writing)
# ---------------------------------------------------------------------------

def _verdicts_dir(results_dir: Path, run_id: str, verifier: str) -> Path:
    return results_dir / run_id / "verdicts" / verifier


def _load_verdicts(results_dir: Path, run_id: str, verifier: str) -> list[Verdict]:
    d = _verdicts_dir(results_dir, run_id, verifier)
    if not d.is_dir():
        return []
    return [Verdict.model_validate_json(p.read_text()) for p in sorted(d.glob("*.json"))]


def _list_run_verifiers(results_dir: Path, run_id: str) -> list[str]:
    d = results_dir / run_id / "verdicts"
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_dir())


_CSV_COLUMNS = [
    "Verifier", "Model", "Cohort",
    "Precision", "Recall", "F1", "Accuracy", "MCC", "Unmet Rate",
    "Prompt Tokens", "Cached Prompt Tokens", "Completion Tokens",
    "Cost USD",
    "TP", "FP", "FN", "TN", "N Tasks", "N Skipped",
]


def _row_for_score(score: VerifierScore, verdicts: list[Verdict]) -> dict:
    m = score.metrics
    cohort = next((v.cohort for v in verdicts if v.cohort), "")
    return {
        "Verifier": score.verifier,
        "Model": score.model,
        "Cohort": cohort,
        "Precision": round(m.precision, 4),
        "Recall": round(m.recall, 4),
        "F1": round(m.f1, 4),
        "Accuracy": round(m.accuracy, 4),
        "MCC": round(m.mcc, 4),
        "Unmet Rate": round(m.unmet_rate, 4),
        "Prompt Tokens": sum(v.tokens.prompt_tokens for v in verdicts),
        "Cached Prompt Tokens": sum(v.tokens.cached_prompt_tokens for v in verdicts),
        "Completion Tokens": sum(v.tokens.completion_tokens for v in verdicts),
        "Cost USD": round(sum(v.cost_usd for v in verdicts), 4),
        "TP": m.counts.tp, "FP": m.counts.fp,
        "FN": m.counts.fn, "TN": m.counts.tn,
        "N Tasks": len(score.per_task),
        "N Skipped": len(score.skipped),
    }


def _print_table(rows: list[dict], columns: list[str]) -> None:
    if not rows:
        print("(no rows)")
        return
    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in columns}
    print("  ".join(c.ljust(widths[c]) for c in columns))
    print("  ".join("-" * widths[c] for c in columns))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in columns))


def _baseline_rows(release_dir: Path, cohort: str) -> list[dict]:
    """Always-Met / Always-Unmet baselines from immutable BVB labels."""
    from adapters.bvb.release import BVBRelease

    release = BVBRelease.open(release_dir)
    n_unmet = 0
    n_met = 0
    for task_id in release.task_ids(cohort):
        for label in load_ground_truth(release, task_id):
            if label["decision"] == "MET":
                n_met += 1
            else:
                n_unmet += 1
    bl = baseline_metrics(ConfusionCounts(tp=n_unmet, fp=0, fn=0, tn=n_met))
    rows = []
    for label, m in [("always-unmet (baseline)", bl["always-unmet"]),
                     ("always-met (baseline)", bl["always-met"])]:
        rows.append({
            "Verifier": label, "Model": "—", "Cohort": cohort,
            "Precision": round(m.precision, 4),
            "Recall": round(m.recall, 4),
            "F1": round(m.f1, 4),
            "Accuracy": round(m.accuracy, 4),
            "MCC": round(m.mcc, 4),
            "Unmet Rate": round(m.unmet_rate, 4),
            "Prompt Tokens": 0, "Cached Prompt Tokens": 0, "Completion Tokens": 0,
            "Cost USD": 0.0,
            "TP": m.counts.tp, "FP": m.counts.fp,
            "FN": m.counts.fn, "TN": m.counts.tn,
            "N Tasks": "", "N Skipped": "",
        })
    return rows


def _parse_models_arg(s: str | None) -> dict[str, str]:
    """Parse ``--model`` value into ``{verifier: model}``.

    Two forms are accepted, possibly mixed:

    * ``verifier=model`` — explicit override for one verifier.
    * a bare ``model`` (no ``=``) — applied to every selected host-side
      verifier as a single shared default. Explicit pairs always win when both are
      given (``--model gemini/x,aaaj=openai/y`` → aaaj uses openai/y,
      everything else uses gemini/x).
    """
    if not s:
        return {}
    out: dict[str, str] = {}
    bare: str | None = None
    for piece in s.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "=" in piece:
            k, v = piece.split("=", 1)
            out[k.strip()] = v.strip()
        else:
            if bare is not None and bare != piece:
                raise argparse.ArgumentTypeError(
                    f"--model has multiple bare model strings ({bare!r}, {piece!r}); "
                    "use at most one to set a shared default."
                )
            bare = piece
    bad = [k for k in out if k not in KNOWN_VERIFIERS]
    if bad:
        raise argparse.ArgumentTypeError(
            f"--model has unknown verifier(s): {bad}. Known: {KNOWN_VERIFIERS}"
        )
    if bare is not None:
        for v in KNOWN_VERIFIERS:
            out.setdefault(v, bare)
    return out


def _parse_csv_list(s: str | None, *, valid: tuple[str, ...] | None = None) -> list[str]:
    if not s:
        return []
    items = [x.strip() for x in s.split(",") if x.strip()]
    if valid is not None:
        bad = [x for x in items if x not in valid]
        if bad:
            raise argparse.ArgumentTypeError(
                f"Unknown values: {bad}. Valid: {valid}"
            )
    return items


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_stage(args: argparse.Namespace) -> int:
    result = stage(
        cohort=args.cohort,
        tasks=_parse_csv_list(args.tasks),
        staging_id=args.staging_id,
        force_build=args.force_build,
        dry_run=args.dry_run,
        verifier_kind=args.verifier_kind,
        dataset_dir=Path(args.dataset_dir) if args.dataset_dir else None,
        environment_type=args.environment,
        n_concurrent=args.n_concurrent,
        n_attempts=args.n_attempts,
        max_retries=args.max_retries,
        retry_include=args.retry_include,
        environment_kwargs=args.environment_kwargs,
    )
    print(f"\n[stage] cohort={result.cohort} kind={args.verifier_kind} "
          f"staging_id={result.staging_id} jobs_dir={result.jobs_dir}")
    print(f"[stage] tasks_requested={result.n_tasks_requested} returncode={result.returncode}")
    return result.returncode


def cmd_evaluate(args: argparse.Namespace) -> int:
    verifiers = _parse_csv_list(args.verifier, valid=KNOWN_VERIFIERS)
    if not verifiers:
        verifiers = list(KNOWN_VERIFIERS)
    tasks = _parse_csv_list(args.tasks) or None
    models = _parse_models_arg(args.model)

    staging_id = args.staging_id or default_staging_id(args.cohort, args.verifier_kind)
    staging_jobs_dir = REPO_ROOT / "jobs" / staging_id
    if not staging_jobs_dir.is_dir():
        print(
            f"ERROR: staging dir not found: {staging_jobs_dir}\n"
            f"       run `stage --cohort {args.cohort} "
            f"--verifier-kind {args.verifier_kind}` first.",
            file=sys.stderr,
        )
        return 2

    result = evaluate(
        verifiers=verifiers,
        cohort=args.cohort,
        staging_jobs_dir=staging_jobs_dir,
        results_dir=Path(args.results_dir),
        run_id=args.run_id,
        datasets_root=Path(args.datasets_root),
        release_root=Path(args.release_dir),
        tasks=tasks,
        models=models,
        resume=args.resume,
        quiet=args.quiet,
        verbose=args.verbose,
        verifier_kind=args.verifier_kind,
        dataset_dir=Path(args.dataset_dir) if args.dataset_dir else None,
        trial_index=args.trial_index,
    )

    print(
        f"\n[evaluate] verdicts={len(result.verdicts_written)} "
        f"skips={len(result.skips)} errors={len(result.errors)}"
    )
    if result.errors:
        for e in result.errors:
            print(f"  ERROR  {e.verifier} / {e.task_id}: {e.error}", file=sys.stderr)

    if not args.no_normalize:
        _do_traces_normalize(Path(args.results_dir) / args.run_id)

    if not args.no_score:
        return _do_score(args, run_id=args.run_id)

    return 1 if result.errors else 0


def _do_traces_normalize(run_dir: Path) -> None:
    """Run the ATIF normalizer; never fail the run on normalizer errors.

    Best-effort because the canonical Verdict + raw traces are already
    on disk — losing the ATIF view shouldn't abort a successful eval.
    """
    try:
        summary = normalize_run(run_dir)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"\n[traces normalize] ERROR: {exc}", file=sys.stderr)
        return
    print(f"\n[traces normalize] wrote {len(summary['written'])} ATIF docs "
          f"to {summary['out_dir']}")
    if summary["skipped"]:
        print(f"  ({len(summary['skipped'])} skipped — see traces/ for details)",
              file=sys.stderr)


def cmd_score(args: argparse.Namespace) -> int:
    return _do_score(args, run_id=args.run_id)


def cmd_traces_normalize(args: argparse.Namespace) -> int:
    """Convert raw per-verifier traces to ATIF JSON under traces_atif/."""
    results_dir = Path(args.results_dir)
    run_dir = results_dir / args.run_id
    if not run_dir.is_dir():
        print(f"ERROR: run dir not found: {run_dir}", file=sys.stderr)
        return 2
    summary = normalize_run(run_dir)
    print(f"\n[traces normalize] run={args.run_id}")
    print(f"  out: {summary['out_dir']}")
    print(f"  wrote {len(summary['written'])} ATIF docs, "
          f"skipped {len(summary['skipped'])}")
    by_verifier: dict[str, int] = {}
    for w in summary["written"]:
        by_verifier[w["verifier"]] = by_verifier.get(w["verifier"], 0) + 1
    for v in sorted(by_verifier):
        print(f"    {v}: {by_verifier[v]}")
    for s in summary["skipped"]:
        print(f"  skip: {s['verifier']}/{s['task_id']} — {s['reason']}",
              file=sys.stderr)
    return 0


def _do_score(args: argparse.Namespace, run_id: str) -> int:
    results_dir = Path(args.results_dir)
    run_dir = results_dir / run_id
    if not run_dir.is_dir():
        print(f"ERROR: run dir not found: {run_dir}", file=sys.stderr)
        return 2
    release_dir = Path(args.release_dir)

    verifiers = _list_run_verifiers(results_dir, run_id)
    if not verifiers:
        print(f"ERROR: no verifier subdirs under {run_dir / 'verdicts'}", file=sys.stderr)
        return 2

    rows: list[dict] = []
    per_task_rows: list[dict] = []
    cohort_for_baselines: str | None = None

    for verifier in verifiers:
        verdicts = _load_verdicts(results_dir, run_id, verifier)
        if not verdicts:
            print(f"[skip] {verifier}: no verdicts on disk", file=sys.stderr)
            continue
        cohorts = {v.cohort for v in verdicts}
        if len(cohorts) > 1:
            print(
                f"ERROR: {verifier} has verdicts spanning cohorts {cohorts}; "
                f"score one cohort at a time.",
                file=sys.stderr,
            )
            return 2
        cohort = cohorts.pop()
        if cohort is None:
            print(f"ERROR: {verifier} verdicts have no cohort set", file=sys.stderr)
            return 2
        cohort_for_baselines = cohort_for_baselines or cohort

        try:
            score = score_verifier(verdicts, release_dir, cohort=cohort)
        except ValueError as error:
            print(f"ERROR: strict scoring failed for {verifier}: {error}", file=sys.stderr)
            return 1
        rows.append(_row_for_score(score, verdicts))
        for ts in score.per_task:
            per_task_rows.append({
                "Verifier": verifier, "task_id": ts.task_id,
                "TP": ts.counts.tp, "FP": ts.counts.fp,
                "FN": ts.counts.fn, "TN": ts.counts.tn,
                "n_criteria": ts.n_criteria, "n_predictions": ts.n_predictions,
                "human_overall_score": ts.human_overall_score,
                "verifier_overall_score": ts.verifier_overall_score,
                "overall_score_difference": ts.overall_score_difference,
            })

    if not rows:
        print("ERROR: no verifier produced verdicts to score", file=sys.stderr)
        return 2

    baseline_rows = (
        _baseline_rows(release_dir, cohort_for_baselines)
        if cohort_for_baselines is not None else []
    )

    print(f"\n=== Run {run_id} ({cohort_for_baselines}) ===\n")
    _print_table(rows + baseline_rows, _CSV_COLUMNS)

    if args.write_csv:
        scores_csv = run_dir / "scores.csv"
        if scores_csv.exists() or (run_dir / "per_task.csv").exists():
            raise FileExistsError('Score outputs already exist; use --no-write-csv to inspect them safely')
        with scores_csv.open("x", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
            w.writeheader()
            w.writerows(rows + baseline_rows)
        print(f"\nwrote {scores_csv}")

        if per_task_rows:
            per_task_csv = run_dir / "per_task.csv"
            cols = ["Verifier", "task_id", "TP", "FP", "FN", "TN",
                    "n_criteria", "n_predictions", "human_overall_score",
                    "verifier_overall_score", "overall_score_difference"]
            with per_task_csv.open("x", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                w.writerows(per_task_rows)
            print(f"wrote {per_task_csv}")

    return 0


# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------

def _nonempty_run_id(value: str) -> str:
    """Argparse ``type=`` for ``--run-id`` — reject empty / whitespace-only.

    ``required=True`` only checks the flag is present; an unset shell var
    expands to ``--run-id ""``, which silently writes everything to
    ``results/`` without a subdir. Force the failure at parse time so the
    user sees an actionable error before any I/O happens.
    """
    if not value or not value.strip():
        raise argparse.ArgumentTypeError(
            "--run-id must be a non-empty string (got empty/whitespace; "
            "did a shell variable expand to nothing?)"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="verifier_runner",
        description="Materialize-aware BVB staging, evaluation, and strict scoring.",
    )
    p.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    p.add_argument(
        "--release-dir",
        default=str(DEFAULT_RELEASE_DIR),
        help="Downloaded immutable BVB release containing labels and manifests.",
    )
    p.add_argument("--datasets-root", default=str(DEFAULT_DATASETS_ROOT),
                   help="Root containing generated <gandalf|rewardkit>/<cohort>/ task directories.")

    sub = p.add_subparsers(dest="cmd", required=True)

    # stage --------------------------------------------------------------
    sp = sub.add_parser(
        "stage",
        help="Run `harbor run -a oracle` for a cohort. Verification stays "
             "enabled so Gandalf produces its real verdict.",
    )
    sp.add_argument("--cohort", default=KNOWN_COHORTS[0], choices=KNOWN_COHORTS)
    sp.add_argument(
        "--verifier-kind", choices=VERIFIER_KINDS, default=DEFAULT_VERIFIER_KIND,
        help="BVB in-image verifier shape (Gandalf or RewardKit).",
    )
    sp.add_argument("--tasks", default=None,
                    help="Comma-separated subset of task IDs in the cohort, "
                         "e.g. 'btb-XXXXXXXX'. Default: all tasks.")
    sp.add_argument("--staging-id", default=None,
                    help="Override jobs/<name>/. Default: staging__<cohort>.")
    sp.add_argument("--force-build", action="store_true",
                    help="Pass --force-build to harbor (rebuild docker image).")
    sp.add_argument(
        "--environment",
        choices=("docker", "daytona"),
        default="docker",
        help="Execution provider. Daytona requires DAYTONA_API_KEY and a BVB snapshot.",
    )
    sp.add_argument(
        "--n-concurrent",
        type=int,
        default=None,
        help="Maximum simultaneous Harbor trials (provider and model quotas still apply).",
    )
    sp.add_argument(
        "--n-attempts",
        type=int,
        default=1,
        help="Independent verifier attempts per task (default: 1).",
    )
    sp.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help="Retry count for transient Harbor trial failures (default: 0).",
    )
    sp.add_argument(
        "--retry-include",
        action="append",
        default=[],
        metavar="EXCEPTION",
        help="Exception type eligible for retry, e.g. DaytonaError.",
    )
    sp.add_argument(
        "--environment-kwarg",
        dest="environment_kwargs",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Provider setting passed to Harbor, e.g. snapshot_template_name=<snapshot>.",
    )
    sp.add_argument("--dry-run", action="store_true",
                    help="Print the harbor command and exit.")
    sp.add_argument(
        "--dataset-dir", default=None,
        help="Override the cohort dataset path (must live under this repo). "
             "Use when staging a parallel dataset that diverges from the "
             "kind→subdir mapping — e.g. one generated by an adapter's "
             "--output-dir for a different verifier model. Pair with "
             "--staging-id to keep the run separate.",
    )
    sp.set_defaults(func=cmd_stage)

    # evaluate -----------------------------------------------------------
    sp = sub.add_parser(
        "evaluate",
        help="Run verifiers against a staged cohort; persist verdicts; score.",
    )
    sp.add_argument("--cohort", default=KNOWN_COHORTS[0], choices=KNOWN_COHORTS)
    sp.add_argument(
        "--verifier-kind", choices=VERIFIER_KINDS, default=DEFAULT_VERIFIER_KIND,
        help="Must match the in-image verifier task shape used for staging.",
    )
    sp.add_argument("--run-id", required=True, type=_nonempty_run_id)
    sp.add_argument(
        "--verifier", default=None,
        help=f"Comma-separated subset of {','.join(KNOWN_VERIFIERS)} "
             f"(default: all)",
    )
    sp.add_argument(
        "--tasks", default=None,
        help="Comma-separated subset of task IDs in the cohort, "
             "e.g. 'btb-XXXXXXXX'. (default: all)",
    )
    sp.add_argument(
        "--staging-id", default=None,
        help=f"Override jobs/<name>/ to read from. Default: staging__<cohort>.",
    )
    sp.add_argument(
        "--model", default=None,
        help="Model override(s). A bare model string (e.g. "
             "'gemini/gemini-3-flash-preview') applies to host-side verifiers; "
             "verifier=model pairs override per verifier; the two forms can be "
             "mixed (e.g. 'gemini/gemini-3-flash-preview,aaaj=openai/gpt-5.4-nano'). "
             "When omitted, host-side verifiers use Gemini Flash while Gandalf "
             "and RewardKit retain their staged model. Gandalf/RewardKit model "
             "overrides are rejected because those judges already ran.",
    )
    sp.add_argument(
        "--resume", action="store_true",
        help="Skip (verifier, task) pairs whose verdict file already exists.",
    )
    sp.add_argument(
        "--trial-index",
        type=int,
        default=None,
        help=(
            "Zero-based deterministic trial selection for staging jobs with "
            "multiple Harbor attempts. Run once per index for repeated evaluation."
        ),
    )
    sp.add_argument(
        "--no-score", action="store_true",
        help="Don't auto-run `score` after evaluate (e.g. for partial runs).",
    )
    sp.add_argument(
        "--no-normalize", action="store_true",
        help="Don't auto-build traces_atif/ after evaluate. Normalization "
             "runs by default; rerun manually with `traces normalize`.",
    )
    sp.add_argument(
        "-q", "--quiet", action="store_true",
        help="Suppress per-task `starting:` heartbeats (only `ok:`/`skip:`/`ERROR`).",
    )
    sp.add_argument(
        "-v", "--verbose", action="store_true",
        help="Stream worker subprocess stdout/stderr live to terminal. "
             "Lets you see AAAJ's per-criterion log lines and autorubric's "
             "tqdm progress bar in real time.",
    )
    sp.add_argument("--write-csv", action="store_true", default=True,
                    help=argparse.SUPPRESS)  # always on; flag for symmetry
    sp.add_argument(
        "--dataset-dir", default=None,
        help="Override the cohort dataset path (mirrors `stage --dataset-dir`). "
             "Use when evaluating a parallel dataset that diverges from the "
             "kind→subdir mapping — e.g. one generated by an adapter's "
             "--output-dir for a different verifier model. Must point at the "
             "same dataset that was staged.",
    )
    sp.set_defaults(func=cmd_evaluate)

    # score --------------------------------------------------------------
    sp = sub.add_parser(
        "score",
        help="Re-score verdicts on disk without re-running any verifiers.",
    )
    sp.add_argument("--run-id", required=True, type=_nonempty_run_id)
    sp.add_argument("--no-csv", dest="write_csv", action="store_false", default=True,
                    help="Skip writing scores.csv / per_task.csv")
    sp.set_defaults(func=cmd_score)

    # traces -------------------------------------------------------------
    sp = sub.add_parser(
        "traces",
        help="Trace post-processing (normalize raw traces to ATIF, etc.)",
    )
    traces_sub = sp.add_subparsers(dest="traces_cmd", required=True)
    sp_norm = traces_sub.add_parser(
        "normalize",
        help="Convert raw per-verifier traces to a unified ATIF format under "
             "traces_atif/<verifier>/<task>.json.",
    )
    sp_norm.add_argument("--run-id", required=True, type=_nonempty_run_id)
    sp_norm.set_defaults(func=cmd_traces_normalize)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
