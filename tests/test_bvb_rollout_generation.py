import unittest

from scripts.bvb_rollout_generation import DEFAULT_OUTPUT, check


class RolloutGenerationTests(unittest.TestCase):
    def test_summary_recomputes_from_rollouts(self):
        result = check(DEFAULT_OUTPUT)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["rollouts"], 24)


if __name__ == "__main__":
    unittest.main()
