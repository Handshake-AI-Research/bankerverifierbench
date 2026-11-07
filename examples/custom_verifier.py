"""Template for evaluating your own verifier on BankerVerifierBench.

Replace `judge` with your verifier, then score the predictions it writes:

    python examples/custom_verifier.py --output outputs/my-verifier.jsonl
    uv run bvb score --predictions outputs/my-verifier.jsonl

Only task inputs are read: the instruction, the canonical rubric and the selected rollout.
The human labels in human-adjudication/ stay unread. Standard library only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_RELEASE = Path(__file__).resolve().parents[1] / "bvb-data" / "BankerVerifierBench"


def judge(task_id: str, instruction: str, criterion: dict, rollout: Path) -> str:
    """Return "MET" or "UNMET" for one rubric criterion.

    `criterion` holds criterion_text, category and weight. `rollout` is the selected agent
    rollout, which holds only agent/workspace/ (the final workspace; submitted files are in
    banker_workspace/deliverables/) and agent/trajectory.json (the agent's messages, tool
    calls and tool outputs). The README's "Rollout evidence" section describes both, and
    when source data is needed. This placeholder accepts every criterion.
    """
    return "MET"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--output", type=Path, required=True, help="Predictions JSONL to write.")
    parser.add_argument("--repeats", type=int, default=1, help="Independent runs per task (the paper uses 3).")
    parser.add_argument("--tasks", help="Comma-separated task IDs (default: all 24).")
    args = parser.parse_args()

    release = args.release_dir
    manifest = [json.loads(line) for line in (release / "manifests/task-manifest.jsonl").read_text().splitlines() if line]
    wanted = set(args.tasks.split(",")) if args.tasks else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as out:
        for task in manifest:
            task_id = task["task_id"]
            if wanted and task_id not in wanted:
                continue
            instruction = (release / "task-source" / task_id / "instruction.md").read_text()
            rubric = json.loads((release / task["evaluation_rubric"]["path"]).read_text())
            rollout = release / task["label_target_rollout"]["candidate_path"]
            for repeat in range(1, args.repeats + 1):
                for criterion in rubric["criteria"]:
                    out.write(json.dumps(dict(
                        task_id=task_id,
                        evaluation_criterion_index=criterion["evaluation_criterion_index"],
                        criterion_text_sha256=criterion["criterion_text_sha256"],
                        repeat=repeat,
                        decision=judge(task_id, instruction, criterion, rollout),
                    )) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
