#!/usr/bin/env python3
"""Run a no-LLM BVB generated-task integrity smoke check."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.bvb.config import DEFAULT_RELEASE_DIR
from adapters.bvb.contract import REPLAY_VIEW, validate_runtime
from adapters.bvb.release import BVBRelease, ReleaseError, sha256_file, tree_digest


def check_task(release: BVBRelease, generated_dir: Path, task_id: str) -> dict:
    task = release.tasks[task_id]
    if not task.is_adjudicated:
        raise ReleaseError(f"{task_id} is not adjudicated")
    task_dir = generated_dir / task_id
    generated = json.loads((task_dir / "bvb-task-manifest.json").read_text())
    if generated["canonical_rubric_sha256"] != task.manifest["evaluation_rubric"]["sha256"]:
        raise ReleaseError(f"{task_id}: generated task canonical-rubric digest mismatch")
    if generated["canonical_rubric_count"] != len(task.criteria):
        raise ReleaseError(f"{task_id}: generated task canonical-rubric count mismatch")
    if generated["mcp_tree_sha256"] != tree_digest(task_dir / "environment" / "mcp-server"):
        raise ReleaseError(f"{task_id}: generated MCP tree digest mismatch")
    if generated["instruction_sha256"] != sha256_file(task_dir / "instruction.md"):
        raise ReleaseError(f"{task_id}: generated instruction digest mismatch")
    if generated["input_tree_sha256"] != tree_digest(task_dir / "environment" / "input"):
        raise ReleaseError(f"{task_id}: generated input tree digest mismatch")
    if generated["verifier_workdir"] != REPLAY_VIEW:
        raise ReleaseError(f"{task_id}: generated verifier workdir mismatch")
    with tempfile.TemporaryDirectory() as temp:
        validate_runtime(task_dir / "tests", task_dir / "solution/workspace",
                         task_dir / "solution/trajectory.json", Path(temp) / "audit.json")
    return {
        "task_id": task_id,
        "criteria": len(task.criteria),
        "trajectory_provenance": generated["trajectory_provenance"],
        "workspace_provenance": generated["workspace_provenance"],
        "status": "passed",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE_DIR)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--task-id", help="Check only this task (default: every task in the cohort).")
    args = parser.parse_args()
    try:
        release = BVBRelease.open(args.release_dir)
        task_ids = [args.task_id] if args.task_id else release.task_ids(release.cohort)
        criteria = 0
        for task_id in task_ids:
            result = check_task(release, args.generated_dir, task_id)
            criteria += result["criteria"]
            print(json.dumps(result, sort_keys=True), flush=True)
        print(json.dumps({"tasks": len(task_ids), "criteria": criteria, "status": "passed"}, sort_keys=True))
        return 0
    except (OSError, KeyError, json.JSONDecodeError, ReleaseError) as error:
        print(f"BVB smoke failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
