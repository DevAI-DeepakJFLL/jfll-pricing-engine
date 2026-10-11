"""
tests/test_evaluator.py
Unit tests verifying compute_evaluation_metrics calculation and integrity.
"""

import unittest
import numpy as np
import pandas as pd
import os
import sys

# Add project root and scripts directory to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from scripts.evaluate_honest_benchmark import compute_evaluation_metrics


class TestEvaluatorMetrics(unittest.TestCase):
    """Test suite for evaluation metric calculations."""

    def test_compute_evaluation_metrics_exact_formulas(self):
        """Verifies exact metric formula outputs against hand-computed values."""
        actual = pd.Series([10.0, 5.0, 8.0])
        pred = pd.Series([12.0, 4.0, 8.0])
        buy = pd.Series([1000.0, 2000.0, 1000.0])

        metrics = compute_evaluation_metrics(actual, pred, buy)

        # MAE: mean(|12-10|, |4-5|, |8-8|) = (2 + 1 + 0) / 3 = 1.0
        self.assertAlmostEqual(metrics["mae"], 1.0, places=6)

        # Bias: mean(12-10, 4-5, 8-8) = (2 - 1 + 0) / 3 = 1/3 = 0.333333...
        self.assertAlmostEqual(metrics["bias"], 1.0 / 3.0, places=6)

        # Median Error: median(2, 1, 0) = 1.0
        self.assertAlmostEqual(metrics["median_error"], 1.0, places=6)

        # Rupee-weighted MAE: sum(|pred - actual|/100 * buy) / sum(buy) * 100
        # (2/100*1000 + 1/100*2000 + 0/100*1000) / 4000 * 100 = 40 / 4000 * 100 = 1.0
        self.assertAlmostEqual(metrics["rupee_weighted_mae"], 1.0, places=6)

        # Total Rupee Gap: sum((pred - actual)/100 * buy)
        # (2/100*1000 + (-1)/100*2000 + 0/100*1000) = 20 - 20 + 0 = 0.0
        self.assertAlmostEqual(metrics["total_rupee_gap"], 0.0, places=6)

        # Corridor Hit Rate: None if bounds not provided
        self.assertIsNone(metrics["corridor_hit_rate"])

    def test_compute_evaluation_metrics_with_corridor(self):
        """Verifies corridor hit rate calculation against lower and upper margin bounds."""
        actual = pd.Series([10.0, 5.0, 8.0])
        pred = pd.Series([12.0, 4.0, 8.0])
        buy = pd.Series([1000.0, 2000.0, 1000.0])
        min_margin = pd.Series([8.0, 4.0, 9.0])
        max_margin = pd.Series([12.0, 6.0, 11.0])

        # Row 0: 8.0 <= 10.0 <= 12.0 -> Hit (True)
        # Row 1: 4.0 <= 5.0 <= 6.0   -> Hit (True)
        # Row 2: 9.0 <= 8.0 <= 11.0  -> Miss (False, actual < min)
        # Expected hit rate = 2 / 3 = 0.666667
        metrics = compute_evaluation_metrics(
            actual, pred, buy, min_margin=min_margin, max_margin=max_margin
        )
        self.assertIsNotNone(metrics["corridor_hit_rate"])
        self.assertAlmostEqual(metrics["corridor_hit_rate"], 2.0 / 3.0, places=6)

    def test_compute_evaluation_metrics_rupee_weighting_divergence(self):
        """Verifies that large-volume shipments properly steer rupee-weighted MAE."""
        # Unweighted MAE is 5.0%, but error is exclusively on small shipment (buy=100)
        # Large shipment (buy=900) has 0% error. Rupee-weighted MAE should be 1.0%.
        actual = pd.Series([10.0, 10.0])
        pred = pd.Series([20.0, 10.0])
        buy = pd.Series([100.0, 900.0])

        metrics = compute_evaluation_metrics(actual, pred, buy)
        self.assertAlmostEqual(metrics["mae"], 5.0, places=6)
        self.assertAlmostEqual(metrics["rupee_weighted_mae"], 1.0, places=6)
        self.assertAlmostEqual(metrics["total_rupee_gap"], 10.0, places=6)

    def test_compute_evaluation_metrics_input_types(self):
        """Verifies functions accept lists, numpy arrays, or pandas Series interchangeably."""
        actual_list = [10.0, 15.0]
        pred_arr = np.array([12.0, 14.0])
        buy_s = pd.Series([500.0, 500.0])

        metrics = compute_evaluation_metrics(actual_list, pred_arr, buy_s)
        self.assertAlmostEqual(metrics["mae"], 1.5, places=6)
        self.assertAlmostEqual(metrics["bias"], 0.5, places=6)

    def test_compute_evaluation_metrics_validation_errors(self):
        """Verifies proper ValueError exceptions on invalid inputs."""
        # Mismatched lengths
        with self.assertRaises(ValueError):
            compute_evaluation_metrics([1.0, 2.0], [1.0], [100.0, 200.0])

        with self.assertRaises(ValueError):
            compute_evaluation_metrics([1.0, 2.0], [1.0, 2.0], [100.0])

        # Empty inputs
        with self.assertRaises(ValueError):
            compute_evaluation_metrics([], [], [])

        # Mismatched corridor bounds
        with self.assertRaises(ValueError):
            compute_evaluation_metrics(
                [1.0, 2.0], [1.0, 2.0], [100.0, 200.0], min_margin=[1.0], max_margin=[2.0, 3.0]
            )


if __name__ == "__main__":
    unittest.main()
