"""Materialize BankerVerifierBench tasks into Harbor build output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapter import (
    DEFAULT_REWARDKIT_AGENT,
    DEFAULT_REWARDKIT_HARBOR_TIMEOUT_SEC,
    DEFAULT_REWARDKIT_JUDGE_TIMEOUT_SEC,
    DEFAULT_REWARDKIT_VERIFIER_MODEL,
    DEFAULT_VERIFIER_MODEL,
    BVBAdapter,
    RewardKitBVBAdapter,
)
from .config import COHORT, DEFAULT_OUTPUT_DIR, DEFAULT_RELEASE_DIR, DEFAULT_REWARDKIT_OUTPUT_DIR
from .prerequisites import validate_release
from .release import ReleaseError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-dir",
        type=Path,
        default=DEFAULT_RELEASE_DIR,
        help=f"Downloaded release with shared tools (default: {DEFAULT_RELEASE_DIR}).",
    )
    parser.add_argument("--task-ids", nargs="*", help="Optional BVB task IDs (default: all 24).")
    parser.add_argument(
        "--verifier-kind",
        choices=("gandalf", "rewardkit"),
        default="gandalf",
        help="In-environment Harbor verifier task shape.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Generated dataset root. Defaults separately for each verifier kind.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="Optional imported trial archive holding the selected rollouts' agent artifacts.",
    )
    parser.add_argument(
        "--execution-environment",
        choices=("docker", "daytona"),
        default="docker",
        help="Generate local Docker tasks or direct Daytona-snapshot tasks.",
    )
    parser.add_argument(
        "--daytona-snapshot",
        help="Immutable Daytona snapshot containing the BVB MCP and shared tools.",
    )
    parser.add_argument(
        "--daytona-storage-mb",
        type=int,
        default=20_480,
        help="Disk reserved for each direct Daytona task sandbox.",
    )
    parser.add_argument("--verifier-model", default=None)
    parser.add_argument("--rewardkit-agent", default=DEFAULT_REWARDKIT_AGENT, choices=("claude-code",))
    parser.add_argument(
        "--rewardkit-judge-timeout-sec",
        type=int,
        default=DEFAULT_REWARDKIT_JUDGE_TIMEOUT_SEC,
    )
    parser.add_argument(
        "--harbor-verifier-timeout-sec",
        type=float,
        default=DEFAULT_REWARDKIT_HARBOR_TIMEOUT_SEC,
        help="RewardKit-only outer verifier timeout.",
    )
    args = parser.parse_args(argv)
    try:
        release, _ = validate_release(args.release_dir, require_tools=True)
        output_root = args.output_dir or (
            DEFAULT_REWARDKIT_OUTPUT_DIR
            if args.verifier_kind == "rewardkit"
            else DEFAULT_OUTPUT_DIR
        )
        output_dir = output_root / COHORT
        tasks = release.select_tasks(COHORT, args.task_ids)
        adapter_class = RewardKitBVBAdapter if args.verifier_kind == "rewardkit" else BVBAdapter
        verifier_model = args.verifier_model or (
            DEFAULT_REWARDKIT_VERIFIER_MODEL
            if args.verifier_kind == "rewardkit"
            else DEFAULT_VERIFIER_MODEL
        )
        adapter_kwargs = {
            "output_dir": output_dir,
            # The MCP resolves individual files directly from the archives in the release.
            "tool_root": release.root,
            "verifier_model": verifier_model,
            "artifact_root": args.artifact_root,
            "execution_environment": args.execution_environment,
            "daytona_snapshot": args.daytona_snapshot,
            "daytona_storage_mb": args.daytona_storage_mb,
        }
        if args.verifier_kind == "rewardkit":
            adapter_kwargs.update(
                {
                    "rewardkit_agent": args.rewardkit_agent,
                    "rewardkit_judge_timeout_sec": args.rewardkit_judge_timeout_sec,
                    "harbor_verifier_timeout_sec": args.harbor_verifier_timeout_sec,
                }
            )
        results = adapter_class(
            release,
            **adapter_kwargs,
        ).generate(tasks)
    except ReleaseError as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "cohort": COHORT,
                "verifier_kind": args.verifier_kind,
                "output_dir": str(output_dir),
                "tasks": len(results),
                "execution_environment": args.execution_environment,
                "daytona_snapshot": args.daytona_snapshot,
                "daytona_storage_mb": args.daytona_storage_mb,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
