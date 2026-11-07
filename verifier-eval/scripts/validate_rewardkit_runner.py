#!/usr/bin/env python3
"""No-LLM contract checks for the staged RewardKit output translator."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from verifier_runner.inputs import InputBundle
from verifier_runner.runners import rewardkit


def _bundle(trial_dir: Path, rubric_n: int) -> InputBundle:
    rubric = [
        {"criterion": f"criterion {index}", "weight": 1}
        for index in range(rubric_n)
    ]
    return InputBundle(
        cohort="bvb",
        task_id="btb-contract",
        gt_task_id="btb-contract",
        rubric=rubric,
        instructions="instructions",
        judge_guidance=None,
        task_dir=trial_dir,
        ground_truth_path=trial_dir,
        release_root=trial_dir,
        canonical_rubric_sha256="contract",
        workspace_dir=None,
        trajectory_path=None,
        trial_dir=trial_dir,
        trajectory_is_placeholder=False,
        trajectory_provenance="authentic_agent_trajectory",
    )


def _trial(details: dict) -> Path:
    trial = Path(tempfile.mkdtemp(prefix="rewardkit-contract-"))
    verifier_dir = trial / "verifier"
    verifier_dir.mkdir()
    (verifier_dir / "reward.json").write_text('{"rubric": 0.5}')
    (verifier_dir / "reward-details.json").write_text(json.dumps(details))
    return trial


def test_normal_output() -> None:
    details = {
        "rubric": {
            "judge": {"model": "anthropic/claude-sonnet-4-6"},
            "criteria": [
                {
                    "description": "criterion 0",
                    "value": 1.0,
                    "raw": 1,
                    "reasoning": "present",
                },
                {
                    "description": "criterion 1",
                    "value": 0.0,
                    "raw": 0,
                    "reasoning": "missing",
                },
            ],
        }
    }
    verdict = rewardkit.invoke(_bundle(_trial(details), 2))
    assert verdict.model == "anthropic/claude-sonnet-4-6"
    assert [criterion.satisfied for criterion in verdict.criteria] == [True, False]


def test_empty_workspace() -> None:
    verdict = rewardkit.invoke(
        _bundle(_trial({"reason": "empty_workspace", "rewards": {}}), 3)
    )
    assert len(verdict.criteria) == 3
    assert all(not criterion.satisfied for criterion in verdict.criteria)


def test_criterion_count_fails_closed() -> None:
    details = {
        "rubric": {
            "criteria": [
                {"description": "criterion 0", "value": 1.0},
            ],
        }
    }
    try:
        rewardkit.invoke(_bundle(_trial(details), 2))
    except ValueError as error:
        assert "criterion count mismatch" in str(error)
    else:
        raise AssertionError("criterion-count mismatch did not fail closed")


def main() -> int:
    for check in (
        test_normal_output,
        test_empty_workspace,
        test_criterion_count_fails_closed,
    ):
        check()
        print(f"ok: {check.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
