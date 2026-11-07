import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class VerifierRunnerMainTests(unittest.TestCase):
    def test_handled_score_failure_propagates_its_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {**os.environ, "PYTHONPATH": str(ROOT / "verifier-eval")}
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "verifier_runner",
                    "--results-dir",
                    tmp,
                    "score",
                    "--run-id",
                    "missing",
                ],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn("ERROR: run dir not found", result.stderr)


if __name__ == "__main__":
    unittest.main()
