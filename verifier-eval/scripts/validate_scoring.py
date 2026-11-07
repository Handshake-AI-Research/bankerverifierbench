#!/usr/bin/env python3
"""No-LLM BVB scoring and fail-closed contract checks."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "verifier-eval")]

from adapters.bvb.config import COHORT, DEFAULT_RELEASE_DIR
from adapters.bvb.contract import input_identity
from adapters.bvb.release import BVBRelease
from verifier_runner.scoring import load_ground_truth, score_verifier
from verifier_runner.verdict import CriterionVerdict, Verdict


def perfect_verdicts(release: BVBRelease) -> list[Verdict]:
    verdicts = []
    for task_id in release.task_ids(COHORT):
        task = release.tasks[task_id]
        criteria = [
            CriterionVerdict(
                index=label["evaluation_criterion_index"],
                evaluation_criterion_index=label["evaluation_criterion_index"],
                source_criterion_index=label["source_criterion_index"],
                criterion_text_sha256=label["criterion_text_sha256"],
                satisfied=label["decision"] == "MET",
            )
            for label in load_ground_truth(release, task_id)
        ]
        verdicts.append(
            Verdict(
                verifier="gandalf",
                model="test",
                task_id=task_id,
                gt_task_id=task_id,
                cohort=COHORT,
                canonical_rubric_sha256=task.manifest["evaluation_rubric"]["sha256"],
                input_identity=input_identity(release, task),
                criteria=criteria,
            )
        )
    return verdicts


def main() -> int:
    release_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RELEASE_DIR
    release = BVBRelease.open(release_dir)
    verdicts = perfect_verdicts(release)
    expected = int(release.release["counts"]["scoring_eligible_criteria"])
    score = score_verifier(verdicts, release_dir, cohort=COHORT)
    if score.metrics.counts.total != expected or score.metrics.f1 != 1.0:
        raise AssertionError(f"Unexpected perfect score: {score.metrics}")
    malformed = verdicts.copy()
    malformed[0] = malformed[0].model_copy(update={"criteria": malformed[0].criteria[:-1]})
    try:
        score_verifier(malformed, release_dir, cohort=COHORT)
    except ValueError:
        pass
    else:
        raise AssertionError("Missing criterion did not fail closed")
    try:
        score_verifier(verdicts[1:], release_dir, cohort=COHORT)
    except ValueError:
        pass
    else:
        raise AssertionError("Missing task did not fail closed")
    print(f"BVB scoring validation passed: {len(verdicts)} tasks, {expected:,} criteria.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
