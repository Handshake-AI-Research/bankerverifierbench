#!/usr/bin/env python3
"""Run one Gandalf criterion-batching condition (B-b4 ... B-b64, B-split2) on Daytona.

    python scripts/run_bvb_gandalf_batching.py --output outputs/b16 --batch-size 16
    python scripts/run_bvb_gandalf_batching.py --output outputs/split2 --batch-splits 2

Tasks run on the derived gandalf-llm snapshot with Gemini 3 Flash at high
reasoning effort. Each task's judge runs two batch sessions at a time. A run is
accepted only if every judge session received exactly its ordered share of the
canonical rubric and the reviewed evidence (see `collect`). The measured study
retried an invalid task/repeat at most twice; rerun failed tasks with --tasks
and --repeat-start into a new output directory.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "verifier-eval")]

from dotenv import load_dotenv  # noqa: E402

from adapters.bvb.adapter import BVBAdapter, effective_reasoning_config  # noqa: E402
from adapters.bvb.batching import (  # noqa: E402
    MODEL, REASONING_EFFORT, BatchingCondition, build_overlay, configure_task, native_source,
)
from adapters.bvb.config import COHORT, DEFAULT_RELEASE_DIR  # noqa: E402
from adapters.bvb.contract import (  # noqa: E402
    EXPECTED_MCP_TOOLS, canonical_rows, file_hash, input_identity, staged_workspace, tree_hash,
)
from adapters.bvb.release import BVBRelease  # noqa: E402
from adapters.bvb.runtime_snapshots import plan as runtime_plan  # noqa: E402
from scripts.run_bvb_gandalf_daytona import _validate_info, _validate_judge_trace  # noqa: E402
from scripts.run_bvb_gandalf_matrix import preflight  # noqa: E402
from verifier_runner.identity import stamp_criterion_identity  # noqa: E402
from verifier_runner.inputs import list_staged_trials, resolve, validate_bundle  # noqa: E402
from verifier_runner.runners.gandalf import invoke  # noqa: E402
from verifier_runner.scoring import _validate_verdict_identity  # noqa: E402
from verifier_runner.stage import stage  # noqa: E402

# USD per million tokens, standard synchronous tier, retrieved 2026-10-07:
# https://ai.google.dev/gemini-api/docs/pricing#gemini-3-flash-preview
PRICE = dict(input=0.5, cache_read=0.05, output_including_thinking=3.0)
READ_CALLS = ("load_workbook", "Presentation(", "PdfReader(", "pdfplumber.open", "read_text(")
ERROR_PATTERN = re.compile(
    r'[^\n]*(?:Traceback \(most recent call last\)|PermissionError|FileNotFoundError|BadZipFile|BadRequestError'
    r'|RateLimitError|RESOURCE_EXHAUSTED|Exit code: [1-9]|"isError": true)[^\n]*')


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _observations(text: str) -> list[str]:
    """Compact header of each tool observation in a judge trace."""
    return [_compact(part.split("Agent Action", 1)[0]) for part in re.split(r"Observation\s+─+", text)[1:]]


def collect(trial: Path, condition: BatchingCondition, release: BVBRelease, dataset: Path) -> dict:
    """Validate one native trial against the batching protocol; return its audit."""
    result = json.loads((trial / "result.json").read_text())
    if result.get("exception_info") is not None:
        raise ValueError("Harbor exception")
    task_id = result["task_name"]
    task = release.tasks[task_id]
    expected = canonical_rows(task)
    info = _validate_info(trial / "verifier/info.json")
    if len(info["criterion_results"]) != len(expected):
        raise ValueError("Criterion count")
    for row, canonical in zip(info["criterion_results"], expected):
        if row["criterion"] != canonical["criterion"] or float(row["weight"]) != float(canonical["weight"]):
            raise ValueError("Canonical criterion wording/weight")
        if not isinstance(row.get("reasoning"), str) or len(row["reasoning"].strip()) <= 3:
            raise ValueError("Missing explanation")
        if not isinstance(row.get("evidence"), list):
            raise ValueError("Missing native evidence array")

    audit = json.loads((trial / "verifier/input-audit.json").read_text())
    identity = input_identity(release, task)
    if audit["status"] != "validated" or audit["identity"] != identity or audit["judge_view_validated"] is not True:
        raise ValueError("Canonical replay audit")
    if set(audit["mcp_tools"]) != EXPECTED_MCP_TOOLS:
        raise ValueError("MCP tools")
    probe = audit["mcp_probe"]
    if probe["caller_uid"] != 10003 or 10001 in probe["caller_groups"]:
        raise ValueError("Sandbox isolation")
    if (tree_hash(staged_workspace(trial)) != identity["workspace_sha256"]
            or file_hash(trial / "agent/trajectory.json") != identity["trajectory_sha256"]):
        raise ValueError("Staged evidence hash")

    guidance = (dataset / task_id / "tests/judge-guidance.md").read_text()
    instruction = release.instruction_for(task)
    chunks = condition.partition([row["criterion"] for row in expected])
    reasoning = effective_reasoning_config(REASONING_EFFORT)
    deliverables = [entry["path"] for entry in audit["files"] if "/deliverables/" in "/" + entry["path"]]
    references = dict.fromkeys(deliverables, 0)
    initial, sessions, usage, flags = [], [], [], []
    successful_reads = mcp_observations = terminal_observations = terminal_successes = 0
    for trace in sorted((trial / "verifier").glob("judge_trace_batch*.txt")):
        text = trace.read_text(errors="replace")
        if "Error in sitecustomize;" in text:
            raise ValueError("Instrumentation startup error visible to judge")
        observed = json.loads(Path(str(trace) + ".input.json").read_text())
        actual = observed["input"]
        if (observed["canonical_identity"] != identity or observed["clone_hashes"]
                != {key: identity[key] for key in ("workspace_sha256", "trajectory_sha256")}):
            raise ValueError("Inner cloned evidence")
        if actual["instructions"] != instruction or actual["judge_guidance"] != guidance or actual["model"] != MODEL:
            raise ValueError("Actual judge inputs")
        if condition.batch_size is not None and len(actual["criteria"]) > condition.batch_size:
            raise ValueError("Retry or original batch exceeds batch size")
        retry = "_retry" in trace.name
        if not retry:
            index = int(re.search(r"_split(\d+)", trace.name)[1])
            if actual["criteria"] != chunks[index]:
                raise ValueError("Exact ordered partition")
            initial.append(index)
        captured = [json.loads(line.split("] ", 1)[1]) for line in text.splitlines()
                    if line.startswith("[BVB_ACTUAL_SESSION] ")]
        if len(captured) != 1:
            raise ValueError("Missing or duplicate actual session prompt")
        session = captured[0]
        if session["model"] != actual["model"] or session["workdir"] != actual["workdir"]:
            raise ValueError("Actual model/workdir")
        prompt = session["prompt"]
        if instruction not in prompt or guidance not in prompt or not all(c in prompt for c in actual["criteria"]):
            raise ValueError("Actual prompt text/rubric/guidance")
        if actual["final_output"] not in prompt:
            raise ValueError("Selected final response")
        sessions.append(dict(trace=trace.name, criteria=len(actual["criteria"]), workdir=actual["workdir"],
                             prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), initial=not retry))

        # A run-level screen for deliverable reads, not proof for every criterion.
        compact = _compact(text)
        headers = _observations(text)
        succeeded = sum("Tool:terminal" in h and "Exitcode:0" in h for h in headers)
        has_read = "/deliverables/" in compact and any(call in compact for call in READ_CALLS)
        successful_reads += int(has_read and succeeded > 0)
        terminal_successes += succeeded
        terminal_observations += sum("Tool:terminal" in h for h in headers)
        for path in references:
            if _compact(path) in compact:
                references[path] += 1
        mcp_observations += sum(any("Tool:" + tool in h for tool in EXPECTED_MCP_TOOLS) for h in headers)
        errors = ERROR_PATTERN.findall(text)
        if errors:
            flags.append(dict(trace=trace.name, type="inspection_or_provider_error", snippets=errors[:40],
                              count=len(errors), recovery="Not inferred from overall completion; retained for evidence review"))
        session_usage = [json.loads(line.split("] ", 1)[1]) for line in text.splitlines()
                         if line.startswith("[BVB_SESSION_USAGE] ")]
        usage.extend(session_usage)
        if text.startswith("exit_code: 0"):
            _validate_judge_trace(trace, verifier_model=MODEL, reasoning_config=reasoning)
        elif not retry or session_usage:
            flags.append(dict(trace=trace.name, type="native_failed_session", cost_known=bool(session_usage)))
    if sorted(initial) != list(range(len(chunks))):
        raise ValueError("Missing/duplicate initial chunk")
    if successful_reads == 0:
        raise ValueError("No evidence of successful deliverable inspection")

    total = info["llm_usage"]
    if total["prompt_tokens"] <= 0 or total["completion_tokens"] <= 0:
        raise ValueError("No token accounting")
    for field in ("prompt_tokens", "completion_tokens", "cache_read_tokens"):
        if sum(row[field] for row in usage) != total[field]:
            raise ValueError("Session token accounting differs: " + field)
    if abs(sum(row["cost_usd"] for row in usage) - total["cost_usd"]) >= 1e-6:
        raise ValueError("Session native cost sum")
    if not 0 <= total["cache_read_tokens"] <= total["prompt_tokens"]:
        raise ValueError("Cached token range")
    price = ((total["prompt_tokens"] - total["cache_read_tokens"]) * PRICE["input"]
             + total["cache_read_tokens"] * PRICE["cache_read"]
             + total["completion_tokens"] * PRICE["output_including_thinking"]) / 1e6
    rows = info["criterion_results"]
    score = sum(r["weight"] * r["met"] for r in rows) / sum(r["weight"] for r in rows)
    if json.loads((trial / "verifier/reward.json").read_text())["reward"] != round(score, 4):
        raise ValueError("Native four-decimal score aggregation")
    return dict(valid=True, condition=condition.name, task=task_id, native_trial=str(trial), identity=identity,
                criteria=len(expected), partition_sizes=[len(c) for c in chunks], sessions=sessions,
                actual_prompt_verified=True, private_evidence_hashes_verified=True, sandbox_probe=probe,
                successful_deliverable_session_screens=successful_reads, terminal_observations=terminal_observations,
                successful_terminal_observations=terminal_successes, deliverable_session_references=references,
                independent_mcp_observations=mcp_observations, mcp_catalog_verified=True, flags=flags, usage=total,
                normalized_api_cost_usd=price, native_api_cost_usd=total["cost_usd"],
                unreported_failed_request_cost="unknown", score=score)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="Fresh output directory.")
    size = parser.add_mutually_exclusive_group(required=True)
    size.add_argument("--batch-size", type=int, help="Criteria per judge session (B-b4 ... B-b64).")
    size.add_argument("--batch-splits", type=int, help="Native near-equal split count (B-split2 uses 2).")
    parser.add_argument("--tasks", default="", help="Comma-separated task IDs (default: all tasks).")
    parser.add_argument("--attempts", type=int, default=3, help="Benchmark repeats to run.")
    parser.add_argument("--repeat-start", type=int, default=1, help="First benchmark repeat slot (1-3).")
    parser.add_argument("--concurrency", type=int, default=6, help="Task sandboxes at a time.")
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE_DIR)
    parser.add_argument("--snapshot-plan", type=Path,
                        help="Derived runtime plan JSON (default: computed gandalf-llm plan for the pinned release)")
    args = parser.parse_args()
    if args.repeat_start < 1 or args.attempts < 1 or args.repeat_start + args.attempts - 1 > 3:
        parser.error("Requested repetitions must fit benchmark slots 1 through 3")
    name = f"b{args.batch_size}" if args.batch_size is not None else f"split{args.batch_splits}"
    try:
        condition = BatchingCondition(name, batch_size=args.batch_size, batch_splits=args.batch_splits)
    except ValueError as error:
        parser.error(str(error))

    load_dotenv(ROOT / ".env")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    release = BVBRelease.open(args.release_dir)
    plan = json.loads(args.snapshot_plan.read_text()) if args.snapshot_plan else runtime_plan("gandalf-llm", release)
    (output / "runtime-preflight.json").write_text(json.dumps(asyncio.run(preflight(plan, release)), indent=2))
    overlay = build_overlay(native_source(output / "gandalf-native"), output / "gandalf-overlay")

    tasks = args.tasks.split(",") if args.tasks else sorted(release.tasks)
    dataset = output / "dataset" / COHORT
    BVBAdapter(release, output_dir=dataset, tool_root=release.root, verifier_model=MODEL,
               reasoning_effort=REASONING_EFFORT, execution_environment="daytona",
               daytona_snapshot=plan["snapshot"]).generate([release.tasks[t] for t in tasks])
    for task_id in tasks:
        configure_task(dataset / task_id, condition, overlay)

    outcome = stage(cohort=COHORT, tasks=tasks, staging_id=f"bvb-gb-{output.name}-{condition.name}", repo_root=ROOT,
                    dataset_dir=dataset, environment_type="daytona", n_concurrent=args.concurrency,
                    n_attempts=args.attempts, max_retries=2, retry_include=["DaytonaError", "DaytonaConnectionError"],
                    environment_kwargs=["snapshot_template_name=" + plan["snapshot"]], extra_harbor_args=["--delete"])
    accepted, failed = [], []
    for task_id in tasks:
        trials = list_staged_trials(outcome.jobs_dir, task_id)
        if len(trials) != args.attempts:
            failed.append(dict(task=task_id, error=f"Expected {args.attempts} native trials, found {len(trials)}"))
        for index, trial in enumerate(trials[:args.attempts]):
            repeat = args.repeat_start + index
            try:
                report = collect(trial, condition, release, dataset)
                bundle = resolve(COHORT, task_id, datasets_root=ROOT / "datasets", dataset_dir=dataset,
                                 release_root=release.root, staging_jobs_dir=outcome.jobs_dir, trial_index=index)
                validate_bundle(bundle)
                verdict = stamp_criterion_identity(invoke(bundle), bundle)
                _validate_verdict_identity(release, task_id, verdict)
                timing = json.loads((trial / "result.json").read_text())["verifier"]
                verdict.duration_sec = (datetime.fromisoformat(timing["finished_at"])
                                        - datetime.fromisoformat(timing["started_at"])).total_seconds()
                verdict.cost_usd = report["normalized_api_cost_usd"]
            except Exception as error:
                failed.append(dict(task=task_id, repeat=repeat, native_trial=str(trial),
                                   error=str(error), type=type(error).__name__))
                continue
            target = output / "results" / condition.name / f"repeat-{repeat}" / task_id
            target.mkdir(parents=True)
            (target / "verdict.json").write_text(verdict.model_dump_json(indent=2))
            (target / "audit.json").write_text(json.dumps(dict(report, repeat=repeat,
                                                               duration_sec=verdict.duration_sec), indent=2))
            accepted.append(dict(task=task_id, repeat=repeat))
    summary = dict(condition=condition.__dict__, model=MODEL, reasoning_effort=REASONING_EFFORT,
                   snapshot=plan["snapshot"], tasks=tasks, attempts=args.attempts, repeat_start=args.repeat_start,
                   native_returncode=outcome.returncode, jobs_dir=str(outcome.jobs_dir), accepted=accepted, failed=failed)
    (output / "completion.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(dict(condition=condition.name, accepted=len(accepted), invalid=len(failed))), flush=True)
    return 0 if not failed and len(accepted) == len(tasks) * args.attempts else 1


if __name__ == "__main__":
    raise SystemExit(main())
