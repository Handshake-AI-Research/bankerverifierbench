"""Enforce the fixed dataset's one-rollout-per-task release boundary."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from adapters.bvb.release import BVBRelease, ReleaseError


class RolloutSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        selected = "rollout-candidates/btb-example/glm5"
        (self.root / selected / "agent/workspace").mkdir(parents=True)
        (self.root / selected / "agent/workspace/report.txt").write_text("answer")
        (self.root / selected / "agent/trajectory.json").write_text('{"steps": [1]}')
        (self.root / "manifests").mkdir()
        self.candidate = dict(task_id="btb-example", path=selected, model="glm5", availability="available")
        self.task = SimpleNamespace(selected_rollout=SimpleNamespace(candidate_path=selected, model="glm5"),
                                    manifest=dict(rollout_candidates=[dict(self.candidate)]))
        self.release = BVBRelease(self.root, dict(counts=dict(rollout_candidate_slots=1, available_rollout_candidates=1)),
                                  {"btb-example": self.task}, {}, {})
        self.write_candidate()

    def write_candidate(self):
        (self.root / "manifests/rollout-candidates.jsonl").write_text(json.dumps(self.candidate) + "\n")

    def test_exact_selection_is_accepted(self):
        self.release.validate_rollout_selection()

    def test_prior_grades_in_selected_directory_are_rejected(self):
        extra = self.root / self.candidate["path"] / "verifier/reward.json"
        extra.parent.mkdir()
        extra.write_text('{"reward": 1}')
        with self.assertRaisesRegex(ReleaseError, "only workspace and trajectory"):
            self.release.validate_rollout_selection()

    def test_unlisted_extra_directory_is_rejected(self):
        (self.root / "rollout-candidates/btb-example/other").mkdir()
        with self.assertRaisesRegex(ReleaseError, "directories"):
            self.release.validate_rollout_selection()

    def test_nested_alternate_candidate_is_rejected(self):
        self.task.manifest["rollout_candidates"].append(dict(path="other"))
        with self.assertRaisesRegex(ReleaseError, "Task manifest"):
            self.release.validate_rollout_selection()

    def test_unavailable_or_wrong_model_is_rejected(self):
        for field, value in [("availability", "unavailable"), ("model", "other")]:
            with self.subTest(field=field):
                original = self.candidate[field]
                self.candidate[field] = value
                self.write_candidate()
                with self.assertRaisesRegex(ReleaseError, "unavailable"):
                    self.release.validate_rollout_selection()
                self.candidate[field] = original


if __name__ == "__main__":
    unittest.main()
