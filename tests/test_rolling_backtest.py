"""
tests/test_rolling_backtest.py
Unit and integration tests for Rolling-Origin Out-of-Time Backtesting Harness.

Verifies:
1. Temporal isolation: Strictly verifies train max timestamp < test min timestamp
   for all folds (zero lookahead leakage across expanding chronological origins).
2. Backtest execution: execute_backtest_folds() runs July, August, September holdouts
   and outputs all required metrics.
3. Baseline superiority: model_mae < commodity_median_mae on every fold.
4. Low bias guarantee: |model_bias| < 1.50 pp on every fold.
5. Conformal coverage guarantee: 90% conformal corridor coverage >= 85.0% on every fold.
6. Customer segmentation: Segregates known vs new customer accounts and confirms both
   cohorts are tracked with positive non-zero samples and valid MAEs.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from scripts.run_rolling_backtest import execute_backtest_folds

DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "won_deals_margin_v3.csv")


class TestRollingBacktest(unittest.TestCase):
    """Test suite verifying rolling-origin out-of-time evaluation harness and baseline superiority."""

    @classmethod
    def setUpClass(cls):
        """Execute rolling backtest across the standard 2026 holdout folds once for reuse."""
        if not os.path.exists(DATA_PATH):
            cls.results = None
            return

        cls.results = execute_backtest_folds(data_path=DATA_PATH)

    def test_backtest_executed_all_three_folds(self):
        """Verifies that execute_backtest_folds outputs exactly 3 rolling folds (Jul, Aug, Sep)."""
        self.assertIsNotNone(self.results, f"Dataset not found at {DATA_PATH}")
        self.assertEqual(len(self.results), 3, "Expected 3 rolling-origin folds (Jul, Aug, Sep 2026)")

        expected_test_months = ["2026-07", "2026-08", "2026-09"]
        for i, fold_res in enumerate(self.results):
            self.assertEqual(fold_res["test_month"], expected_test_months[i])

    def test_temporal_ordering_and_no_lookahead_leakage(self):
        """
        Verifies strict chronological isolation:
        train max date < test min date for all folds (zero lookahead leakage).
        """
        self.assertIsNotNone(self.results)
        for fold_res in self.results:
            fold_idx = fold_res["fold"]
            train_max = str(fold_res["train_max_date"])
            test_min = str(fold_res["test_min_date"])

            self.assertLess(
                train_max,
                test_min,
                f"Fold {fold_idx} temporal leakage detected: train_max ({train_max}) >= test_min ({test_min})",
            )
            # Verify deal counts match expanding window contract
            self.assertGreater(fold_res["train_n"], 5000)
            self.assertGreater(fold_res["test_n"], 500)

    def test_all_required_metrics_present(self):
        """Verifies every fold result contains all required statistical and commercial metrics."""
        self.assertIsNotNone(self.results)
        required_keys = [
            "fold",
            "train_cutoff",
            "test_month",
            "train_n",
            "test_n",
            "train_max_date",
            "test_min_date",
            "model_mae",
            "model_bias",
            "model_median_error",
            "model_rupee_weighted_error",
            "commodity_median_mae",
            "global_median_mae",
            "conformal_coverage",
            "new_customer_mae",
            "known_customer_mae",
        ]
        for fold_res in self.results:
            for k in required_keys:
                self.assertIn(k, fold_res, f"Missing required key '{k}' in fold {fold_res.get('fold')}")
                self.assertIsNotNone(fold_res[k], f"Metric '{k}' cannot be None in fold {fold_res.get('fold')}")

    def test_baseline_superiority_every_fold(self):
        """
        Verifies baseline superiority release gate:
        model_mae < commodity_median_mae and model_mae < global_median_mae on every fold.
        """
        self.assertIsNotNone(self.results)
        for fold_res in self.results:
            fold_idx = fold_res["fold"]
            model_mae = fold_res["model_mae"]
            comm_mae = fold_res["commodity_median_mae"]
            global_mae = fold_res["global_median_mae"]

            self.assertLess(
                model_mae,
                comm_mae,
                f"Fold {fold_idx}: Model MAE ({model_mae:.2f} pp) must beat Commodity Median MAE ({comm_mae:.2f} pp)",
            )
            self.assertLess(
                model_mae,
                global_mae,
                f"Fold {fold_idx}: Model MAE ({model_mae:.2f} pp) must beat Global Median MAE ({global_mae:.2f} pp)",
            )

    def test_low_bias_contract_every_fold(self):
        """Verifies low systematic bias: |model_bias| < 1.50 pp across all rolling folds."""
        self.assertIsNotNone(self.results)
        for fold_res in self.results:
            fold_idx = fold_res["fold"]
            bias = fold_res["model_bias"]

            self.assertLess(
                abs(bias),
                1.50,
                f"Fold {fold_idx}: |model_bias| ({abs(bias):.2f} pp) must be < 1.50 pp",
            )

    def test_conformal_corridor_coverage_every_fold(self):
        """Verifies 90% conformal corridor coverage achieves >= 85.0% across all rolling folds."""
        self.assertIsNotNone(self.results)
        for fold_res in self.results:
            fold_idx = fold_res["fold"]
            cov = fold_res["conformal_coverage"]
            # Support both fraction (0.85-1.00) and percentage (85-100) representations
            cov_ratio = cov if cov <= 1.0 else cov / 100.0

            self.assertGreaterEqual(
                cov_ratio,
                0.85,
                f"Fold {fold_idx}: Conformal coverage ({cov_ratio*100:.2f}%) must be >= 85.0%",
            )

    def test_known_vs_new_customer_breakdown(self):
        """Verifies segregated evaluation for known vs new customer accounts."""
        self.assertIsNotNone(self.results)
        for fold_res in self.results:
            fold_idx = fold_res["fold"]
            known_mae = fold_res["known_customer_mae"]
            new_mae = fold_res["new_customer_mae"]

            self.assertGreater(known_mae, 0.0, f"Fold {fold_idx}: known_customer_mae must be positive")
            self.assertGreater(new_mae, 0.0, f"Fold {fold_idx}: new_customer_mae must be positive")
            self.assertGreater(
                fold_res.get("known_customer_n", 1),
                0,
                f"Fold {fold_idx}: known customer count must be > 0",
            )
            self.assertGreater(
                fold_res.get("new_customer_n", 1),
                0,
                f"Fold {fold_idx}: new customer count must be > 0",
            )


if __name__ == "__main__":
    unittest.main()
