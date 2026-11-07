#!/usr/bin/env python3
"""Run independent host-verifier repetitions and audit their native artifacts."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
import re
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "verifier-eval")]
from dotenv import load_dotenv
from verifier_runner.inputs import resolve, validate_bundle
from verifier_runner.identity import stamp_criterion_identity
from verifier_runner.scoring import _validate_verdict_identity
from adapters.bvb.config import COHORT, DEFAULT_RELEASE_DIR
from adapters.bvb.release import BVBRelease
from verifier_runner.runners._run_audit import criterion_errors, call_audit, reader_audit


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, default=str)


def run_one(args, verifier, task, repeat):
    if getattr(args, 'stop_file', None) and args.stop_file.exists():
        return dict(verifier=verifier, model=args.model, task_id=task,
                    repeat=repeat + 1, status="not_started_control_stop", checked_at=now())
    run = args.output / verifier / args.model.replace("/", "__") / f"repeat-{repeat + 1}"
    trace = run / "traces" / task
    trace.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = dict(verifier=verifier, model=args.model, task_id=task,
                  repeat=repeat + 1, started_at=now(), status="running")
    write(trace / "started.json", status)
    print("START", verifier, task, repeat + 1, flush=True)
    try:
        bundle = resolve(COHORT, task, datasets_root=ROOT / "datasets",
                         dataset_dir=args.dataset, release_root=args.release,
                         staging_jobs_dir=args.job, trial_index=repeat)
        validate_bundle(bundle)
        write(trace / "judge-inputs.json", dict(
            identity=bundle.input_identity, instructions=bundle.instructions,
            rubric=bundle.rubric, workspace=str(bundle.workspace_dir),
            trajectory=str(bundle.trajectory_path), source_trial=str(bundle.trial_dir),
            evidence_mode={"aaaj": "gray_box: file readers and trajectory retrieval",
                           "autorubric": "trajectory text, no filesystem or MCP tools",
                           "archipelago": "output_llm: initial/final snapshots and extracted artifacts"}[verifier],
            human_labels_passed=False,
        ))
        module = importlib.import_module("verifier_runner.runners." + verifier)
        kwargs = {"aaaj": dict(max_workers=2, timeout_sec=5400),
                  "autorubric": dict(max_parallel_requests=args.autorubric_parallel_requests,
                                     request_timeout_sec=args.autorubric_request_timeout,
                                     timeout_sec=args.autorubric_worker_timeout),
                           "archipelago": {}}[verifier]
        if args.litellm_version:
            kwargs["litellm_version"] = args.litellm_version
        verdict = module.invoke(bundle, model=args.model, trace_dir=trace, **kwargs)
        verdict = stamp_criterion_identity(verdict, bundle)
        verdict.duration_sec = time.monotonic() - started
        _validate_verdict_identity(BVBRelease.open(args.release), task, verdict)
        validate_bundle(bundle)  # Detect accidental mutation of reviewed evidence.
        if any(not c.reason for c in verdict.criteria):
            raise ValueError("Missing per-criterion explanation")
        infrastructure_errors = criterion_errors(verdict.criteria)
        if infrastructure_errors:
            write(trace / 'excluded-from-comparison.json', infrastructure_errors)
            raise ValueError(f'{len(infrastructure_errors)} native error-derived grades; refusing invalid evaluation')
        calls_path = trace / "llm_calls.jsonl"
        calls = [json.loads(s) for s in calls_path.read_text().splitlines() if s.strip()]
        if not calls:
            raise ValueError("No recorded model calls")
        truncated = sum(
            choice.get("finish_reason") == "length"
            for call in calls for choice in (call.get("response") or {}).get("choices", [])
        )
        if truncated:
            raise ValueError(f"{truncated} length-truncated model responses; refusing incomplete evaluation")
        actual_inputs=call_audit(trace)
        write(trace/'actual-input-audit.json',actual_inputs)
        if not actual_inputs['frozen_prompt_in_actual_messages'] or actual_inputs['criteria_missing_from_actual_messages']:
            raise ValueError('Frozen task prompt or canonical criterion text missing from actual model messages')
        flags = []
        readers = reader_audit(trace)
        write(trace/'reader-audit.json', readers)
        if readers['requires_evidence_review']:
            flags.append('Native document reader/extraction warnings; inspect reader-audit.json for evidence limitations')
        failed_calls = sum(bool(c.get("error")) for c in calls)
        if failed_calls:
            flags.append(f"{failed_calls} failed model calls (inspect retries)")
        abstentions = sum(c.raw_verdict in {"CANNOT_ASSESS", "UNDETERMINED", "NULL"}
                          for c in verdict.criteria)
        if abstentions:
            flags.append(f"{abstentions} native abstentions mapped to UNMET; report separately")
        if verifier == "autorubric":
            inp = json.loads((trace / "worker-inputs.json").read_text())
            expected = module._build_to_grade(bundle.trajectory_path)
            if inp["query"] != bundle.instructions or inp["to_grade"] != expected:
                raise ValueError("AutoRubric task/trajectory input mismatch")
            if "[to_grade truncated]" in expected:
                flags.append("Trajectory was truncated by adapter limit")
        raw_errors = []
        for p in trace.glob("worker-*.txt"):
            for line in p.read_text(errors="replace").splitlines():
                # Snapshot diffs can contain the evaluated agent's own errors.
                # Only flag verifier diagnostics, not quoted evidence lines.
                if not line.lstrip().startswith(("+", "-")) and (
                    re.search(r"\|\s*ERROR\s*\|| - ERROR - |^Traceback", line)
                    or line.startswith(("Error reading", "No reader found"))
                ):
                    raw_errors.append(line[:800])
        if raw_errors:
            flags.append("Reader/runtime error messages present; inspect raw_error_lines")
        cost_known = all(c.get("cost_usd") is not None for c in calls if c.get("response"))
        write(run / "verdicts" / (task + ".json"), verdict.model_dump())
        status.update(status="completed", finished_at=now(), duration_sec=verdict.duration_sec,
                      criteria=len(verdict.criteria), failed_model_calls=failed_calls,
                      native_abstentions=abstentions, flags=flags,
                      raw_error_lines=raw_errors, cost_complete_for_recorded_calls=cost_known,
                      recorded_cost_usd=sum(c.get("cost_usd") or 0 for c in calls),
                      recorded_calls=len(calls), inputs_validated_before_and_after=True)
        print("DONE", verifier, task, repeat + 1, "flags", len(flags), flush=True)
    except Exception as error:
        status.update(status="failed", finished_at=now(), duration_sec=time.monotonic()-started,
                      error_type=type(error).__name__, error=str(error))
        (trace / "exception.txt").write_text(traceback.format_exc())
        print("FAIL", verifier, task, repeat + 1, type(error).__name__, str(error), flush=True)
    write(trace / "audit.json", status)
    return status


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--release", type=Path, default=DEFAULT_RELEASE_DIR)
    p.add_argument("--job", type=Path, required=True,
                   help="Harbor job holding the staged Gandalf trials whose evidence the host verifiers read")
    p.add_argument("--dataset", type=Path, required=True,
                   help="Generated task directory used for that job (<output>/bvb)")
    p.add_argument("--verifiers", default="aaaj,autorubric")
    p.add_argument("--model", default="gemini/gemini-3-flash-preview")
    p.add_argument("--litellm-version", help="Explicit isolated transport upgrade for model ablations")
    p.add_argument("--tasks", default="")
    p.add_argument("--attempts", type=int, default=3)
    p.add_argument('--repeat-start', type=int, default=1,
                   help='First benchmark repetition slot; use with attempts to run only missing slots')
    p.add_argument('--stop-file', type=Path,
                   help='Finish active attempts but skip not-yet-started attempts when this file exists')
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument('--autorubric-parallel-requests', type=int, default=4)
    p.add_argument('--autorubric-request-timeout', type=float, default=60.0)
    p.add_argument('--autorubric-worker-timeout', type=int, default=1800)
    p.add_argument('--completed-runs-from', type=Path, nargs='*', default=[],
                   help='Skip only valid audited repetitions from earlier fresh output roots')
    args = p.parse_args()
    if args.repeat_start < 1 or args.attempts < 1 or args.repeat_start + args.attempts - 1 > 3:
        p.error('Requested repetitions must fit benchmark slots 1 through 3')
    if min(args.autorubric_parallel_requests, args.autorubric_request_timeout, args.autorubric_worker_timeout) <= 0:
        p.error('AutoRubric concurrency and timeouts must be positive')
    load_dotenv(ROOT / ".env")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    from verifier_runner.runners._runtime import pin_dependency_cache
    pin_dependency_cache(os.environ, args.litellm_version)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output / "matplotlib-cache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(args.output / "cache"))
    release = BVBRelease.open(args.release)
    tasks = args.tasks.split(",") if args.tasks else sorted(release.tasks)
    verifiers = args.verifiers.split(",")
    if "archipelago" in verifiers and not (os.getenv("REDUCTO_API_KEY") or os.getenv("MERCOR_DELIVERY_API_KEY")):
        raise ValueError("Archipelago requires a document extraction key; refusing degraded evaluation")
    write(args.output / "plan.json", dict(started_at=now(), tasks=tasks, attempts=args.attempts,
          repeat_start=args.repeat_start, stop_file=str(args.stop_file) if args.stop_file else None,
          concurrency=args.concurrency, verifiers=verifiers, model=args.model,
          litellm_version=args.litellm_version,
          uv_cache_dir=os.environ.get('UV_CACHE_DIR'),
          autorubric_parallel_requests=args.autorubric_parallel_requests,
          autorubric_request_timeout=args.autorubric_request_timeout,
          autorubric_worker_timeout=args.autorubric_worker_timeout,
          release_revision=release.revision, source_job=str(args.job), dataset=str(args.dataset)))
    jobs = [(v,t,r) for t in tasks
            for r in range(args.repeat_start - 1, args.repeat_start - 1 + args.attempts)
            for v in verifiers]
    completed = set()
    for prior in args.completed_runs_from:
        for path in prior.glob('*/**/verdicts/*.json'):
            trace = path.parent.parent / 'traces' / path.stem
            if (trace / 'excluded-from-comparison.json').exists(): continue
            audit = trace / 'posthoc-audit.json'
            if not audit.exists() or not json.loads(audit.read_text()).get('valid'):
                raise ValueError(f'Previous result requires a valid post-hoc audit: {path}')
            old = json.loads(path.read_text())
            if old['model'] == args.model:
                completed.add((old['verifier'], old['task_id'], int(path.parent.parent.name.removeprefix('repeat-'))-1))
    jobs = [job for job in jobs if job not in completed]
    write(args.output / 'pending-runs.json', dict(skipped_valid_runs=len(completed), jobs=jobs))
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(run_one,args,*j) for j in jobs]
        for future in as_completed(futures):
            results.append(future.result())
    write(args.output / "completion.json", dict(finished_at=now(), planned=len(jobs),
          completed=sum(r["status"]=="completed" for r in results),
          failed=sum(r["status"]=="failed" for r in results),
          not_started=sum(r["status"]=="not_started_control_stop" for r in results), runs=results))
    return int(any(r["status"]=="failed" for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
