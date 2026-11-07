import csv
import gzip
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from adapters.bvb.config import DEFAULT_RELEASE_DIR
from adapters.bvb.release import BVBRelease
from adapters.bvb.score import ScoreError, load_predictions, load_verdicts, score

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "results" / "paper"
RELEASE_DIR = Path(os.environ.get("BVB_TEST_RELEASE", DEFAULT_RELEASE_DIR))


def bundle_predictions(condition: str) -> dict:
    data = json.loads(gzip.decompress((BUNDLE / "experiment-data.json.gz").read_bytes()))
    digest = {(r[0], r[1]): r[7] for r in data["rubric"]}
    return {(task, index, repeat): (digest[task, index], predicted)
            for cid, task, index, _, *predictions in data["criteria"] if cid == condition
            for repeat, predicted in enumerate(predictions, 1) if predicted is not None}


@unittest.skipUnless((RELEASE_DIR / "release.json").exists(), "Run `make fetch` to download the release")
class ScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.release = BVBRelease.open(RELEASE_DIR)
        with (BUNDLE / "conditions.csv").open(newline="") as stream:
            cls.paper = {row["id"]: row for row in csv.DictReader(stream)}

    def test_published_predictions_reproduce_their_paper_row(self):
        result = score(self.release, bundle_predictions("M01"))
        self.assertTrue(result["complete"])
        for key in ("f1", "precision", "recall", "accuracy", "overall_mae", "majority_f1"):
            self.assertAlmostEqual(result[key], float(self.paper["M01"][key]), places=12, msg=key)

    def test_example_verifier_scores_as_the_always_met_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "predictions.jsonl"
            subprocess.run([sys.executable, str(ROOT / "examples/custom_verifier.py"),
                            "--release-dir", str(RELEASE_DIR), "--output", str(output)],
                           check=True, capture_output=True)
            result = score(self.release, load_predictions(output))
        baseline = json.loads((BUNDLE / "baselines.json").read_text())["always-met"]
        self.assertTrue(result["complete"])
        self.assertEqual(result["counts"], baseline["counts"])
        self.assertAlmostEqual(result["overall_mae"], baseline["overall_mae"], places=12)

    def test_run_verdicts_score_like_their_predictions(self):
        from adapters.bvb.contract import input_identity
        sys.path.insert(0, str(ROOT / "verifier-eval"))
        from verifier_runner.verdict import CriterionVerdict, Verdict

        predictions = bundle_predictions("M13")
        by_run = {}
        for (task, index, repeat), (digest, decision) in predictions.items():
            by_run.setdefault((task, repeat), []).append(CriterionVerdict(
                index=index, evaluation_criterion_index=index, criterion_text_sha256=digest,
                satisfied=decision == "MET"))
        with tempfile.TemporaryDirectory() as tmp:
            for (task, repeat), criteria in by_run.items():
                release_task = self.release.tasks[task]
                criteria = [
                    criterion.model_copy(update={
                        "source_criterion_index": release_task.criteria[criterion.index].source_index,
                    })
                    for criterion in criteria
                ]
                verdict = Verdict(verifier="archipelago", model="gemini/gemini-3-flash-preview",
                                  task_id=task, cohort=self.release.cohort,
                                  canonical_rubric_sha256=release_task.manifest["evaluation_rubric"]["sha256"],
                                  input_identity=input_identity(self.release, release_task),
                                  criteria=criteria).model_dump_json()
                matrix = Path(tmp, "matrix", f"repeat-{repeat}", "verdicts", f"{task}.json")
                batching = Path(tmp, "batching", "results", "b16", f"repeat-{repeat}", task, "verdict.json")
                for path in (matrix, batching):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(verdict)
            self.assertEqual(load_verdicts(Path(tmp, "matrix"), self.release), predictions)
            self.assertEqual(load_verdicts(Path(tmp, "batching"), self.release), predictions)
            with self.assertRaisesRegex(ScoreError, "duplicate verdict"):
                load_verdicts(Path(tmp), self.release)
        self.assertAlmostEqual(score(self.release, predictions)["f1"], float(self.paper["M13"]["f1"]), places=12)

    def test_identity_and_coverage_are_enforced(self):
        predictions = bundle_predictions("M17")
        key = next(iter(predictions))
        wrong_digest = {**predictions, key: ("0" * 64, predictions[key][1])}
        with self.assertRaisesRegex(ScoreError, "does not match the release"):
            score(self.release, wrong_digest)
        with self.assertRaisesRegex(ScoreError, "decides"):
            score(self.release, {k: v for k, v in predictions.items() if k != key})
        one_task = {k: v for k, v in predictions.items() if k[0] == key[0]}
        self.assertFalse(score(self.release, one_task)["complete"])

    def test_native_verdicts_are_strictly_typed_and_bound_to_one_configuration(self):
        from adapters.bvb.contract import input_identity
        sys.path.insert(0, str(ROOT / "verifier-eval"))
        from verifier_runner.verdict import CriterionVerdict, Verdict

        def payload(task_id, *, model="gemini/gemini-3-flash-preview"):
            task = self.release.tasks[task_id]
            return Verdict(
                verifier="archipelago",
                model=model,
                task_id=task_id,
                cohort=self.release.cohort,
                canonical_rubric_sha256=task.manifest["evaluation_rubric"]["sha256"],
                input_identity=input_identity(self.release, task),
                criteria=[
                    CriterionVerdict(
                        index=c.evaluation_index,
                        evaluation_criterion_index=c.evaluation_index,
                        source_criterion_index=c.source_index,
                        criterion_text_sha256=c.text_sha256,
                        satisfied=True,
                    )
                    for c in task.criteria
                ],
            ).model_dump()

        def write(root, task_id, repeat, body):
            path = root / f"repeat-{repeat}" / "verdicts" / f"{task_id}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(body))

        task_a, task_b = sorted(self.release.tasks)[:2]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            typed = payload(task_a)
            typed["criteria"][0]["satisfied"] = "false"
            write(root, task_a, 1, typed)
            with self.assertRaisesRegex(ScoreError, "invalid Verdict schema"):
                load_verdicts(root, self.release)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            excluded = payload(task_a)
            excluded["criteria"][0]["excluded"] = True
            write(root, task_a, 1, excluded)
            with self.assertRaisesRegex(ScoreError, "excluded criterion"):
                load_verdicts(root, self.release)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            excluded = payload(task_a)
            excluded["scoring_eligible"] = False
            write(root, task_a, 1, excluded)
            with self.assertRaisesRegex(ScoreError, "excluded verdict"):
                load_verdicts(root, self.release)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root, task_a, 1, payload(task_a))
            write(root, task_b, 2, payload(task_b, model="different-model"))
            with self.assertRaisesRegex(ScoreError, "mix verifier, model, or cohort"):
                load_verdicts(root, self.release)


if __name__ == "__main__":
    unittest.main()
