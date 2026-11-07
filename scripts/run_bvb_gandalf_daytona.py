#!/usr/bin/env python3
"""Materialize and run the complete BVB Gandalf evaluation on Daytona."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "verifier-eval")]

from adapters.bvb.adapter import (
    DEFAULT_EXTENDED_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_VERIFIER_MODEL,
    REASONING_EFFORTS,
    BVBAdapter,
    effective_reasoning_config,
)
from adapters.bvb.config import COHORT, DEFAULT_RELEASE_DIR
from adapters.bvb.daytona import validate_snapshot_runtime
from adapters.bvb.prerequisites import validate_release
from adapters.bvb.release import BVBRelease, ReleaseError
from verifier_runner.stage import stage


_JUDGE_RUNTIME_FAILURE_MARKERS = (
    "ModuleNotFoundError: No module named 'openpyxl'",
    "ModuleNotFoundError: No module named 'pandas'",
    "ModuleNotFoundError: No module named 'pptx'",
    "ModuleNotFoundError: No module named 'pdfplumber'",
    "ModuleNotFoundError: No module named 'pypdf'",
    "ModuleNotFoundError: No module named 'reportlab'",
    "Created 0 MCP tools",
    "Tools Available: 4",
)
_JUDGE_RUNTIME_SUCCESS_MARKERS = (
    "Tools Available: 19",
    "Created 15 MCP tools",
)


def _validate_json(path: Path) -> None:
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseError(f"Missing or invalid run artifact {path}: {error}") from error


def _validate_info(path: Path) -> dict:
    _validate_json(path)
    info = json.loads(path.read_text(encoding="utf-8"))
    criteria = info.get("criterion_results")
    if not isinstance(criteria, list) or not criteria:
        raise ReleaseError(f"{path} has no criterion results")
    for index, criterion in enumerate(criteria):
        if (
            not isinstance(criterion, dict)
            or not isinstance(criterion.get("criterion"), str)
            or not isinstance(criterion.get("weight"), (int, float))
            or not isinstance(criterion.get("met"), bool)
        ):
            raise ReleaseError(f"{path} has invalid criterion JSON at index {index}")
    if int(info.get("errored_criterion_count", 0) or 0) != 0:
        raise ReleaseError(f"{path} records errored criteria")
    if float(info.get("evaluated_criteria_pct", 100.0) or 0.0) != 100.0:
        raise ReleaseError(f"{path} has incomplete criterion evaluation")
    return info


def _validate_judge_trace(
    path: Path,
    *,
    verifier_model: str | None = None,
    reasoning_config: dict[str, object] | None = None,
    provider_routing: dict[str, object] | None = None,
) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    failures = [
        marker for marker in _JUDGE_RUNTIME_FAILURE_MARKERS if marker in text
    ]
    if failures:
        raise ReleaseError(
            f"{path} contains broken inner-judge runtime markers: {failures}"
        )
    missing = [
        marker for marker in _JUDGE_RUNTIME_SUCCESS_MARKERS if marker not in text
    ]
    if missing:
        raise ReleaseError(
            f"{path} is missing inner-judge runtime markers: {missing}"
        )
    compact_text = re.sub(r"\s+", "", text)
    if verifier_model and re.sub(r"\s+", "", verifier_model) not in compact_text:
        raise ReleaseError(
            f"{path} does not report the selected verifier model {verifier_model!r}"
        )
    if reasoning_config:
        reasoning_effort = reasoning_config.get("reasoning_effort")
        thinking_budget = reasoning_config.get("extended_thinking_budget")
        reasoning_max_tokens = reasoning_config.get("reasoning_max_tokens")
        expected_markers = [
            f"'reasoning_effort':{reasoning_effort!r}",
            f"'extended_thinking_budget':{thinking_budget}",
        ]
        if reasoning_max_tokens is not None:
            expected_markers.append(
                f"'reasoning':{{'enabled':True,'max_tokens':{reasoning_max_tokens}}}"
            )
        missing_reasoning = [
            marker for marker in expected_markers if marker not in compact_text
        ]
        if missing_reasoning:
            raise ReleaseError(
                f"{path} is missing effective reasoning markers: {missing_reasoning}"
            )
    if provider_routing:
        routing_markers = [
            *(f"'{provider}'" for provider in provider_routing.get("only", [])),
            f"'allow_fallbacks':{provider_routing.get('allow_fallbacks')}",
            f"'require_parameters':{provider_routing.get('require_parameters')}",
        ]
        missing_routing = [
            marker for marker in routing_markers if marker not in compact_text
        ]
        if missing_routing:
            raise ReleaseError(
                f"{path} is missing provider-routing markers: {missing_routing}"
            )


def validate_job(
    jobs_dir: Path,
    *,
    task_ids: list[str],
    attempts: int,
    verifier_model: str | None = None,
    reasoning_config: dict[str, object] | None = None,
    provider_routing: dict[str, object] | None = None,
    release_dir: Path | None = None,
) -> dict:
    release = BVBRelease.open(release_dir) if release_dir is not None else None
    result_path = jobs_dir / "result.json"
    _validate_json(result_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    expected_trials = len(task_ids) * attempts
    stats = result.get("stats", {})
    if int(result.get("n_total_trials", -1)) != expected_trials:
        raise ReleaseError(
            f"Expected {expected_trials} trials, found {result.get('n_total_trials')}"
        )
    if int(stats.get("n_errored_trials", -1)) != 0:
        raise ReleaseError(
            f"Harbor recorded {stats.get('n_errored_trials')} errored trial(s)"
        )
    if int(stats.get("n_completed_trials", expected_trials)) != expected_trials:
        raise ReleaseError(
            f"Harbor completed {stats.get('n_completed_trials')} of {expected_trials} trials"
        )

    task_counts: Counter[str] = Counter()
    trial_summaries = []
    for trial_dir in sorted(path for path in jobs_dir.iterdir() if path.is_dir()):
        trial_result_path = trial_dir / "result.json"
        if not trial_result_path.is_file():
            continue
        _validate_json(trial_result_path)
        trial_result = json.loads(trial_result_path.read_text(encoding="utf-8"))
        task_id = str(trial_result.get("task_name", ""))
        if task_id not in task_ids:
            raise ReleaseError(f"Unexpected task in run artifacts: {task_id}")
        if trial_result.get("exception_info") is not None:
            raise ReleaseError(f"{trial_dir.name} has exception_info")
        required = (
            trial_dir / "agent" / "trajectory.json",
            trial_dir / "verifier" / "info.json",
            trial_dir / "verifier" / "reward.json",
        )
        _validate_json(required[0])
        info = _validate_info(required[1])
        if release is not None:
            from adapters.bvb.contract import EXPECTED_MCP_TOOLS, REPLAY_VIEW, input_identity, canonical_rows, tree_hash, file_hash, staged_workspace
            identity = input_identity(release, release.tasks[task_id])
            audit = json.loads((trial_dir / 'verifier/input-audit.json').read_text())
            if audit.get('status') != 'validated' or audit.get('identity') != identity:
                raise ReleaseError(f'{trial_dir.name} canonical input audit mismatch')
            if (audit.get('judge_view_validated') is not True or audit.get('judge_workdir') != REPLAY_VIEW
                    or set(audit.get('mcp_tools', [])) != EXPECTED_MCP_TOOLS):
                raise ReleaseError(f'{trial_dir.name} full replay view or MCP access not verified')
            if (tree_hash(staged_workspace(trial_dir)) != identity['workspace_sha256']
                    or file_hash(required[0]) != identity['trajectory_sha256']):
                raise ReleaseError(f'{trial_dir.name} selected replay mismatch')
            expected = canonical_rows(release.tasks[task_id])
            rows = info['criterion_results']
            if len(rows) != len(expected) or any(r['criterion'] != e['criterion'] or float(r['weight']) != float(e['weight']) for r, e in zip(rows, expected)):
                raise ReleaseError(f'{trial_dir.name} native criterion text or weight mismatch')
        _validate_json(required[2])
        judge_traces = sorted(
            (trial_dir / "verifier").glob("judge_trace_batch_split*.txt")
        )
        if not judge_traces or any(path.stat().st_size == 0 for path in judge_traces):
            raise ReleaseError(f"{trial_dir.name} has incomplete judge traces")
        for path in judge_traces:
            _validate_judge_trace(
                path,
                verifier_model=verifier_model,
                reasoning_config=reasoning_config,
                provider_routing=provider_routing,
            )
        task_counts[task_id] += 1
        trial_summaries.append(
            {
                "task_id": task_id,
                "trial_name": trial_dir.name,
                "reward": trial_result.get("verifier_result", {})
                .get("rewards", {})
                .get("reward"),
                "trajectory": str(required[0]),
                "info": str(required[1]),
                "reward_file": str(required[2]),
                "judge_traces": [str(path) for path in judge_traces],
                "judge_retry_traces": [
                    str(path) for path in sorted((trial_dir / "verifier").glob("judge_trace_batch_retry*.txt"))
                ],
                "llm_usage": info.get("llm_usage", {}),
            }
        )
    wrong_counts = {
        task_id: task_counts[task_id]
        for task_id in task_ids
        if task_counts[task_id] != attempts
    }
    if wrong_counts:
        raise ReleaseError(
            f"Per-task attempt coverage mismatch (expected {attempts}): {wrong_counts}"
        )
    return {
        "schema_version": "bvb-daytona-gandalf-run/v2",
        "validated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "tasks": len(task_ids),
        "attempts_per_task": attempts,
        "trials": len(trial_summaries),
        "errored_trials": 0,
        "harbor_retries": int(stats.get("n_retries", 0) or 0),
        "artifacts_complete": True,
        "trial_results": trial_summaries,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "datasets" / "gandalf-daytona-full",
    )
    parser.add_argument("--snapshot", required=True)
    parser.add_argument(
        "--verifier-model",
        default=DEFAULT_VERIFIER_MODEL,
        help=(
            "LiteLLM verifier model ID written to every generated grader.toml "
            f"(default: {DEFAULT_VERIFIER_MODEL})."
        ),
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=REASONING_EFFORTS,
        default=DEFAULT_REASONING_EFFORT,
        help=(
            "Explicit OpenHands reasoning effort passed to every inner judge "
            f"(default: {DEFAULT_REASONING_EFFORT})."
        ),
    )
    parser.add_argument(
        "--extended-thinking-budget",
        type=int,
        default=DEFAULT_EXTENDED_THINKING_BUDGET,
        help=(
            "OpenHands extended-thinking token budget "
            f"(default: {DEFAULT_EXTENDED_THINKING_BUDGET})."
        ),
    )
    parser.add_argument(
        "--omit-reasoning-effort",
        action="store_true",
        help=(
            "Pass reasoning_effort=None for providers that support reasoning "
            "but do not expose effort levels."
        ),
    )
    parser.add_argument(
        "--reasoning-max-tokens",
        type=int,
        help="Optional provider reasoning.max_tokens request override.",
    )
    parser.add_argument(
        "--openrouter-provider",
        help=(
            "Optional OpenRouter provider slug to hard-pin with fallbacks "
            "disabled and parameter support required."
        ),
    )
    parser.add_argument(
        "--task-ids",
        help="Optional comma-separated adjudicated task IDs (default: full cohort).",
    )
    parser.add_argument(
        "--run-id",
        default=datetime.now(UTC).strftime("daytona-full-%Y%m%dT%H%M%SZ"),
    )
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=60,
        help="Global Harbor trial concurrency. 60 fully parallelizes 20×3.",
    )
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument(
        "--cache-max-bytes",
        type=int,
        default=2 * 1024 * 1024 * 1024,
        help="Expected bounded lazy-tool cache encoded in the snapshot identity.",
    )
    parser.add_argument(
        "--require-authentic",
        action="store_true",
        help="Refuse tasks whose trajectory provenance is synthetic.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        dataset_dir = args.output_dir / COHORT
        jobs_dir = REPO_ROOT / "jobs" / args.run_id
        existing = [path for path in (dataset_dir, jobs_dir) if path.exists()]
        if existing:
            raise ReleaseError(
                "Refusing to overwrite existing experiment output(s): "
                + ", ".join(str(path) for path in existing)
            )
        release, _ = validate_release(args.release_dir, require_tools=True)
        task_ids = (
            [task_id.strip() for task_id in args.task_ids.split(",") if task_id.strip()]
            if args.task_ids
            else None
        )
        if args.task_ids and not task_ids:
            raise ReleaseError("--task-ids must contain at least one task ID")
        tasks = release.select_tasks(COHORT, task_ids)
        runtime_preflight = validate_snapshot_runtime(
            release,
            args.snapshot,
            cache_max_bytes=args.cache_max_bytes,
        )
        selected_reasoning_effort = (
            None if args.omit_reasoning_effort else args.reasoning_effort
        )
        reasoning_config = effective_reasoning_config(
            selected_reasoning_effort,
            args.extended_thinking_budget,
            args.reasoning_max_tokens,
        )
        provider_routing = (
            {
                "only": [args.openrouter_provider],
                "allow_fallbacks": False,
                "require_parameters": True,
            }
            if args.openrouter_provider
            else None
        )
        generated = BVBAdapter(
            release,
            output_dir=dataset_dir,
            tool_root=release.root / "shared-tools",
            verifier_model=args.verifier_model,
            reasoning_effort=selected_reasoning_effort,
            extended_thinking_budget=args.extended_thinking_budget,
            reasoning_max_tokens=args.reasoning_max_tokens,
            openrouter_provider=args.openrouter_provider,
            execution_environment="daytona",
            daytona_snapshot=args.snapshot,
        ).generate(tasks)
        provenance = Counter(item.trajectory_provenance for item in generated)
        if args.require_authentic and provenance.get("synthetic_placeholder", 0):
            raise ReleaseError(
                f"{provenance['synthetic_placeholder']} task(s) lack authentic trajectories"
            )
        result = stage(
            cohort=COHORT,
            tasks=[task.task_id for task in tasks],
            staging_id=args.run_id,
            dataset_dir=dataset_dir,
            environment_type="daytona",
            n_concurrent=args.concurrency,
            n_attempts=args.attempts,
            max_retries=args.max_retries,
            retry_include=["DaytonaError", "DaytonaConnectionError"],
            environment_kwargs=[f"snapshot_template_name={args.snapshot}"],
        )
        if result.returncode != 0:
            raise ReleaseError(
                f"Harbor run failed; inspect {result.jobs_dir / 'result.json'}"
            )
        summary = validate_job(
            result.jobs_dir,
            release_dir=release.root,
            task_ids=[task.task_id for task in tasks],
            attempts=args.attempts,
            verifier_model=args.verifier_model,
            reasoning_config=reasoning_config,
            provider_routing=provider_routing,
        )
        summary.update(
            {
                "release_dir": str(release.root),
                "snapshot": args.snapshot,
                "run_id": args.run_id,
                "verifier_model": args.verifier_model,
                "effective_reasoning_configuration": reasoning_config,
                "provider_routing": provider_routing,
                "concurrency": args.concurrency,
                "max_retries": args.max_retries,
                "runtime_preflight": runtime_preflight,
                "trajectory_provenance": dict(sorted(provenance.items())),
            }
        )
        summary_path = result.jobs_dir / "bvb-run-summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, ReleaseError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
