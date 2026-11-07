import unittest

from scripts.bvb_public_results import DEFAULT_BUNDLE, check


class PaperResultsTests(unittest.TestCase):
    def test_bundle_recomputes_from_its_predictions(self):
        result = check(DEFAULT_BUNDLE)
        self.assertEqual(result["status"], "passed")
        self.assertEqual((result["tasks"], result["criteria"]), (24, 2464))


if __name__ == "__main__":
    unittest.main()
