import unittest
from types import SimpleNamespace

from scripts.bvb_public_results import _validate_release_binding


class AnnotationBindingTests(unittest.TestCase):
    def setUp(self):
        self.release = SimpleNamespace(tasks={"btb-example": SimpleNamespace(
            manifest={"evaluation_rubric": {"sha256": "b" * 64}},
            selected_rollout=SimpleNamespace(trial_name="selected-trial"))})
        self.data = dict(runs=[dict(task="btb-example", condition="M01", selected_rollout_trial="selected-trial",
                                    canonical_rubric_sha256="b" * 64)])

    def test_runs_bound_to_the_release_are_accepted(self):
        _validate_release_binding(self.data, self.release)

    def test_changed_public_rubric_is_rejected(self):
        self.data["runs"][0]["canonical_rubric_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "Run evidence identity"):
            _validate_release_binding(self.data, self.release)

    def test_other_rollout_is_rejected(self):
        self.data["runs"][0]["selected_rollout_trial"] = "other-trial"
        with self.assertRaisesRegex(ValueError, "Run evidence identity"):
            _validate_release_binding(self.data, self.release)

    def test_unknown_task_is_rejected(self):
        self.data["runs"][0]["task"] = "btb-unknown"
        with self.assertRaisesRegex(ValueError, "unknown task"):
            _validate_release_binding(self.data, self.release)


if __name__ == "__main__":
    unittest.main()
