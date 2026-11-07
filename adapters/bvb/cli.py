"""Inspect and rerun the measured BVB conditions.

    bvb list
    bvb show M05
    bvb run --condition M05 --output outputs/m05 --tasks btb-68259d37 --repeats 1 --dry-run
    bvb score --predictions my-verifier.jsonl

Each condition in configs/paper/conditions.toml records the runner and
native settings it was measured with. `run` turns an entry into that runner's
own command line and executes it unchanged. Host-verifier conditions read
evidence staged by an M01 (`gandalf-daytona`) run; pass that run's output
directory with --evidence-run.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

from .config import COHORT, DEFAULT_RELEASE_DIR

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "configs" / "paper" / "conditions.toml"
RESULTS = ROOT / "results" / "paper" / "conditions.csv"

RUNNERS = {
    "gandalf-daytona": "scripts/run_bvb_gandalf_daytona.py",
    "gandalf-matrix": "scripts/run_bvb_gandalf_matrix.py",
    "rewardkit-daytona": "scripts/run_bvb_rewardkit_daytona.py",
    "host-matrix": "scripts/run_bvb_verifier_matrix.py",
    "gandalf-batching": "scripts/run_bvb_gandalf_batching.py",
}
REFERENCE_CONDITIONS = ("M01", "M13", "M16", "M19")
HOST_SETTINGS = ("autorubric_parallel_requests", "autorubric_request_timeout", "autorubric_worker_timeout")


def load_conditions(path: Path = REGISTRY) -> dict[str, dict]:
    data = tomllib.loads(path.read_text())
    if data.get("schema") != "bvb-conditions/v1":
        raise ValueError(f"Unsupported condition registry schema in {path}")
    conditions = {}
    for condition in data["condition"]:
        if condition["id"] in conditions:
            raise ValueError(f"Duplicate condition {condition['id']}")
        if condition["runner"] not in RUNNERS:
            raise ValueError(f"{condition['id']} uses unknown runner {condition['runner']!r}")
        conditions[condition["id"]] = condition
    return conditions


def load_results(path: Path = RESULTS) -> dict[str, dict]:
    with path.open(newline="") as stream:
        return {row["id"]: row for row in csv.DictReader(stream)}


def command(
    condition: dict,
    *,
    output: Path,
    tasks: list[str] | None = None,
    repeats: int | None = None,
    evidence_run: Path | None = None,
    release_dir: Path = DEFAULT_RELEASE_DIR,
) -> list[str]:
    runner = condition["runner"]
    if runner != "host-matrix" and not output.resolve().is_relative_to(ROOT):
        raise ValueError(f"--output must be inside the repository ({ROOT}); Harbor stages tasks by repository-relative path")
    repeats = condition["repeats"] if repeats is None else repeats
    if not 1 <= repeats <= 3:
        raise ValueError("Repeats must be between 1 and 3")
    cmd = [sys.executable, RUNNERS[runner]]
    if runner == "gandalf-daytona":
        from .daytona import snapshot_name
        from .release import BVBRelease

        release = BVBRelease.open(release_dir)
        cmd += [
            "--release-dir", str(release_dir),
            "--output-dir", str(output),
            "--run-id", output.name,
            "--snapshot", snapshot_name(release, release_revision=release.revision),
            "--verifier-model", condition["model"],
            "--reasoning-effort", condition["reasoning_effort"],
            "--attempts", str(repeats),
            "--concurrency", str(condition["concurrency"]),
        ]
        if tasks:
            cmd += ["--task-ids", ",".join(tasks)]
        return cmd
    if runner == "gandalf-batching":
        from .batching import MODEL, REASONING_EFFORT

        if condition["model"] != MODEL or condition["reasoning_effort"] != REASONING_EFFORT:
            raise ValueError(f"{condition['id']}: the batching runner measures {MODEL} at {REASONING_EFFORT} effort only")
        size = (["--batch-size", str(condition["batch_size"])] if "batch_size" in condition
                else ["--batch-splits", str(condition["batch_splits"])])
        cmd += ["--release-dir", str(release_dir), "--output", str(output), *size,
                "--attempts", str(repeats), "--concurrency", str(condition["concurrency"])]
        if tasks:
            cmd += ["--tasks", ",".join(tasks)]
        return cmd
    cmd += ["--output", str(output), "--model", condition["model"], "--attempts", str(repeats),
            "--concurrency", str(condition["concurrency"])]
    if tasks:
        cmd += ["--tasks", ",".join(tasks)]
    if runner in {"gandalf-matrix", "rewardkit-daytona"}:
        return cmd + ["--release-dir", str(release_dir), "--reasoning-effort", condition["reasoning_effort"]]
    if evidence_run is None:
        raise ValueError(f"{condition['id']} reads staged Gandalf evidence; pass --evidence-run <M01 output dir>")
    cmd += ["--release", str(release_dir), "--verifiers", condition["verifier"],
            "--job", str(ROOT / "jobs" / evidence_run.name), "--dataset", str(evidence_run / COHORT)]
    if "litellm_version" in condition:
        cmd += ["--litellm-version", condition["litellm_version"]]
    for key in HOST_SETTINGS:
        if key in condition:
            cmd += ["--" + key.replace("_", "-"), str(condition[key])]
    return cmd


def _show(condition: dict, result: dict | None) -> None:
    for key, value in condition.items():
        print(f"{key:30} {value}")
    if result:
        print()
        print(f"{'measured runs':30} {result['runs']} of {result['target']} ({result['status']})")
        for key in ("f1", "precision", "recall", "accuracy", "majority_f1"):
            print(f"{key:30} {float(result[key]):.3f}")
        print(f"{'total API cost (USD)':30} {float(result['total_cost']):.2f}")
    print()
    evidence = Path("outputs/m01") if condition["runner"] == "host-matrix" else None
    output = Path("outputs") / condition["id"].lower()
    print(shlex.join(["bvb", "run", "--condition", condition["id"], "--output", str(output)]
                     + (["--evidence-run", str(evidence)] if evidence else [])))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bvb", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    listing = sub.add_parser("list", help="List the measured conditions.")
    listing.add_argument("--runner", choices=sorted(RUNNERS), help="Only conditions of this runner.")
    show = sub.add_parser("show", help="Show one condition's settings, measured results and run command.")
    show.add_argument("condition")
    run = sub.add_parser("run", help="Run one condition with its recorded runner and settings.")
    run.add_argument("--condition", required=True)
    run.add_argument("--output", type=Path, required=True, help="Fresh output directory.")
    run.add_argument("--tasks", help="Comma-separated task IDs (default: all tasks).")
    run.add_argument("--repeats", type=int, help="Benchmark repeats, 1-3 (default: the condition's).")
    run.add_argument("--evidence-run", type=Path, help="Output directory of an M01 run (host verifiers only).")
    run.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE_DIR)
    run.add_argument("--dry-run", action="store_true", help="Print the command without running it.")
    scoring = sub.add_parser("score", help="Score your own verifier's predictions against the human labels.")
    source = scoring.add_mutually_exclusive_group(required=True)
    source.add_argument("--predictions", type=Path, help="Predictions JSONL (see adapters/bvb/score.py).")
    source.add_argument("--verdicts", type=Path, help="Output directory of a `bvb run` (one verifier and model).")
    scoring.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE_DIR)
    scoring.add_argument("--json", action="store_true", help="Print the metrics as JSON.")
    args = parser.parse_args(argv)

    if args.action == "score":
        from .release import BVBRelease
        from .score import ScoreError, load_predictions, load_verdicts, report, score

        try:
            release = BVBRelease.open(args.release_dir)
            predictions = load_predictions(args.predictions) if args.predictions else load_verdicts(args.verdicts, release)
            result = score(release, predictions)
        except ScoreError as error:
            parser.error(str(error))
        results = load_results()
        print(json.dumps(result, indent=2) if args.json
              else report(result, {cid: results[cid] for cid in REFERENCE_CONDITIONS}))
        return 0

    conditions = load_conditions()
    if args.action == "list":
        results = load_results()
        print(f"{'id':9} {'verifier':12} {'model':32} {'runs':>5} {'F1':>5}  {'status':18} runner")
        for condition in conditions.values():
            if args.runner and condition["runner"] != args.runner:
                continue
            result = results.get(condition["id"], {})
            f1 = result.get("f1")
            runs = f"{result['runs']}/{result['target']}" if result else "-"
            print(f"{condition['id']:9} {condition['verifier']:12} {condition['model']:32} {runs:>5} "
                  f"{float(f1) if f1 else float('nan'):5.3f}  {result.get('status', '-'):18} {condition['runner']}")
        print("\nruns = measured task runs / target (24 tasks x repeats). Only Complete rows are full-coverage results.")
        return 0
    name = args.condition
    if name not in conditions:
        parser.error(f"Unknown condition {name!r}; see `bvb list`")
    if args.action == "show":
        _show(conditions[name], load_results().get(name))
        return 0
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()] if args.tasks else None
    try:
        cmd = command(conditions[name], output=args.output.resolve(), tasks=tasks,
                      repeats=args.repeats, evidence_run=args.evidence_run and args.evidence_run.resolve(),
                      release_dir=args.release_dir.resolve())
    except ValueError as error:
        parser.error(str(error))
    print(shlex.join(cmd), flush=True)
    if args.dry_run:
        return 0
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    return subprocess.run(cmd, cwd=ROOT, env=os.environ.copy()).returncode


if __name__ == "__main__":
    raise SystemExit(main())
