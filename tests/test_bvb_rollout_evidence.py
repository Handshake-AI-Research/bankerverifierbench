"""Prevent archived grades and logs from returning to public verifier evidence."""
import tempfile
import unittest
from pathlib import Path
from adapters.bvb.release import ReleaseError, validate_rollout_evidence


class RolloutEvidenceTests(unittest.TestCase):
    def test_only_workspace_and_trajectory_are_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "agent/workspace/banker_workspace/deliverables").mkdir(parents=True)
            (root / "agent/workspace/banker_workspace/deliverables/report.txt").write_text("answer")
            (root / "agent/trajectory.json").write_text('{"steps": [1]}')
            validate_rollout_evidence(root)
            for relative in ("verifier/reward.json", "result.json", "trial.log", "config.json",
                             "deliverables/report.txt", "agent/opencode.txt", "agent/replay-provenance.json"):
                with self.subTest(path=relative):
                    extra = root / relative
                    extra.parent.mkdir(parents=True, exist_ok=True)
                    extra.write_text("archived generation data")
                    with self.assertRaisesRegex(ReleaseError, "only workspace and trajectory"):
                        validate_rollout_evidence(root)
                    extra.unlink()
            (root / "agent/trajectory.json").unlink()
            with self.assertRaises(ReleaseError):
                validate_rollout_evidence(root)
