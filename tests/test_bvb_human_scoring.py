"""Tests for verifier scoring against human adjudication."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from score_bvb_verifier_against_humans import (
    _confusion,
    _majority,
    _metric_dict,
    _weighted_accuracy,
)


class HumanScoringTests(unittest.TestCase):
    def test_confusion_uses_unmet_as_positive_class(self) -> None:
        counts, predicted_unmet = _confusion(
            [False, False, True, True],
            [False, True, False, True],
        )
        self.assertEqual(
            (counts.tp, counts.fp, counts.fn, counts.tn),
            (1, 1, 1, 1),
        )
        self.assertEqual(predicted_unmet, 2)
        metrics = _metric_dict(counts, predicted_unmet)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["balanced_accuracy"], 0.5)
        self.assertEqual(metrics["f1_unmet"], 0.5)
        self.assertEqual(metrics["f1_met"], 0.5)
        self.assertEqual(metrics["macro_f1"], 0.5)

    def test_majority_and_weighted_accuracy(self) -> None:
        self.assertTrue(_majority([True, False, True]))
        self.assertFalse(_majority([False, True, False]))
        with self.assertRaisesRegex(ValueError, "odd number"):
            _majority([True, False])
        self.assertEqual(
            _weighted_accuracy(
                [True, False, True],
                [True, True, True],
                [10.0, 5.0, 1.0],
            ),
            11 / 16,
        )


if __name__ == "__main__":
    unittest.main()
