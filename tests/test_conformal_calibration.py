"""
tests/test_conformal_calibration.py
Unit and integration tests for Calibrated Split-Conformal Prediction Corridors.

Verifies:
1. Finite-sample distribution-free coverage guarantee (>= 89% for alpha=0.10)
   on normal and skewed nonconformity scores.
2. ConformalCorridorCalibrator lifecycle: fit(), get_half_width(), predict_corridor(),
   and evaluate_coverage().
3. Empirical Bayes shrinkage for sparse segments (n=2) vs dense segments (n=500).
4. Out-of-time empirical coverage on honest benchmark (tests/data/benchmark_won_out_of_time.csv)
   achieving >= 88.0% coverage, resolving legacy Issue M5 (where legacy hit rate was 22-32%).
5. Serialization and deserialization roundtrip parity across save() and load().
"""

import os
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.conformal_corridors import (
    calibrate_conformal_band,
    ConformalCorridorCalibrator,
)
from src.quantile_model import MarginQuantileModel, PIT_FEATURE_COLS

DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "won_deals_margin_v3.csv")
BENCHMARK_PATH = os.path.join(BASE_DIR, "tests", "data", "benchmark_won_out_of_time.csv")


class TestConformalCalibration(unittest.TestCase):
    """Test suite verifying finite-sample split-conformal prediction corridor guarantees."""

    def setUp(self):
        np.random.seed(42)

    def test_finite_sample_coverage_guarantee_normal(self):
        """Verifies distribution-free coverage guarantee on Gaussian errors (alpha=0.10 => >= 89%)."""
        n_cal = 1000
        n_test = 10000
        # Calibration errors from N(0, 3^2)
        cal_residuals = np.random.normal(0, 3.0, size=n_cal)
        half_width = calibrate_conformal_band(cal_residuals, alpha=0.10)
        self.assertGreater(half_width, 0.0)

        # Finite-sample guarantee on calibration sample: ceil((n+1)*0.9)/n >= 0.90
        sample_coverage = float(np.mean(np.abs(cal_residuals) <= half_width))
        self.assertGreaterEqual(sample_coverage, 0.90)

        # Independent test samples from identical distribution
        test_residuals = np.random.normal(0, 3.0, size=n_test)
        coverage = float(np.mean(np.abs(test_residuals) <= half_width))
        self.assertGreaterEqual(
            coverage,
            0.89,
            f"Conformal coverage on Gaussian errors ({coverage*100:.2f}%) must be >= 89.0%",
        )

    def test_finite_sample_coverage_guarantee_skewed(self):
        """Verifies distribution-free coverage on highly skewed asymmetric errors (Gamma distribution)."""
        n_cal = 1000
        n_test = 5000
        # Calibration errors from Gamma(shape=2, scale=3) centered
        cal_residuals = np.random.gamma(shape=2.0, scale=3.0, size=n_cal) - 6.0
        half_width = calibrate_conformal_band(cal_residuals, alpha=0.10)
        self.assertGreater(half_width, 0.0)

        test_residuals = np.random.gamma(shape=2.0, scale=3.0, size=n_test) - 6.0
        coverage = float(np.mean(np.abs(test_residuals) <= half_width))
        self.assertGreaterEqual(
            coverage,
            0.89,
            f"Conformal coverage on skewed errors ({coverage*100:.2f}%) must be >= 89.0%",
        )

    def test_calibrator_fit_and_predict_corridor(self):
        """Tests basic ConformalCorridorCalibrator fit, corridor prediction, and bounds sanity."""
        y_true = np.array([10.0, 15.0, 12.0, 20.0, 8.0, 14.0, 18.0, 11.0, 13.0, 16.0])
        y_pred = np.array([9.5, 14.0, 12.5, 18.0, 8.5, 13.0, 17.5, 11.5, 12.0, 15.0])
        tiers = np.array(["Tier_A"] * 5 + ["Tier_B"] * 5)
        known = np.array([True] * 6 + [False] * 4)

        calibrator = ConformalCorridorCalibrator(alpha=0.10, k_prior=20.0)
        self.assertFalse(calibrator.is_fitted)

        calibrator.fit(y_true, y_pred, weight_tiers=tiers, is_known_customer=known)
        self.assertTrue(calibrator.is_fitted)
        self.assertEqual(calibrator.calibration_n, 10)
        self.assertGreater(calibrator.global_half_width, 0.0)
        self.assertIn("Tier_A", calibrator.slab_half_widths)
        self.assertIn("Tier_B", calibrator.slab_half_widths)
        self.assertIn("known", calibrator.customer_type_half_widths)
        self.assertIn("new", calibrator.customer_type_half_widths)

        # Test predict_corridor
        pred_margin = 14.0
        low, high, hw = calibrator.predict_corridor(pred_margin, weight_tier="Tier_A", is_known_customer=True)
        self.assertEqual(round(high - low, 6), round(2 * hw, 6))
        self.assertLess(low, pred_margin)
        self.assertGreater(high, pred_margin)
        self.assertEqual(hw, calibrator.get_half_width(weight_tier="Tier_A", is_known_customer=True))

    def test_empirical_bayes_shrinkage_small_vs_large_segments(self):
        """Verifies Empirical Bayes shrinkage toward global for sparse segments and data-dominance for large ones."""
        # Global calibration set of 200 items with std ~ 4.0
        n_global = 200
        y_true_g = np.random.normal(15.0, 4.0, size=n_global)
        y_pred_g = np.full(n_global, 15.0)

        # Segment A: tiny segment (n=2) with huge residual (e.g. 25.0)
        y_true_small = np.array([40.0, 40.0])
        y_pred_small = np.array([15.0, 15.0])
        tiers_small = np.array(["Sparse_Tier", "Sparse_Tier"])

        # Segment B: large segment (n=300) with small residual (std ~ 1.5)
        n_large = 300
        y_true_large = np.random.normal(15.0, 1.5, size=n_large)
        y_pred_large = np.full(n_large, 15.0)
        tiers_large = np.array(["Dense_Tier"] * n_large)

        y_true_all = np.concatenate([y_true_g, y_true_small, y_true_large])
        y_pred_all = np.concatenate([y_pred_g, y_pred_small, y_pred_large])
        tiers_all = np.concatenate([np.array(["Global"] * n_global), tiers_small, tiers_large])

        k_prior = 20.0
        calibrator = ConformalCorridorCalibrator(alpha=0.10, k_prior=k_prior)
        calibrator.fit(y_true_all, y_pred_all, weight_tiers=tiers_all)

        hw_global = calibrator.global_half_width
        hw_sparse = calibrator.get_half_width("Sparse_Tier")
        hw_dense = calibrator.get_half_width("Dense_Tier")

        # Raw sparse quantile would be ~25.0
        raw_sparse_hw = calibrate_conformal_band(y_true_small - y_pred_small, alpha=0.10)
        # With n=2 and k=20, weight on raw is 2/(2+20) = 0.091, weight on global is 20/22 = 0.909
        expected_sparse_hw = (2 / 22) * raw_sparse_hw + (20 / 22) * hw_global
        self.assertAlmostEqual(hw_sparse, expected_sparse_hw, places=4)
        # Sparse tier must shrink toward global: closer to global than to raw
        self.assertLess(abs(hw_sparse - hw_global), abs(hw_sparse - raw_sparse_hw))

        # Dense tier: with n=300 and k=20, weight on raw is 300/320 = 0.9375
        raw_dense_hw = calibrate_conformal_band(y_true_large - y_pred_large, alpha=0.10)
        expected_dense_hw = (300 / 320) * raw_dense_hw + (20 / 320) * hw_global
        self.assertAlmostEqual(hw_dense, expected_dense_hw, places=4)
        # Dense tier should be very close to raw empirical quantile
        self.assertLess(abs(hw_dense - raw_dense_hw), 0.5)

    def test_evaluate_coverage_metrics(self):
        """Verifies evaluate_coverage returns structured coverage and breakdown metrics."""
        y_true = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0])
        y_pred = np.array([10.2, 10.8, 12.1, 12.9, 14.3, 14.8, 16.2, 16.9])
        tiers = np.array(["T1", "T1", "T1", "T1", "T2", "T2", "T2", "T2"])

        calibrator = ConformalCorridorCalibrator(alpha=0.10)
        calibrator.fit(y_true, y_pred, weight_tiers=tiers)

        metrics = calibrator.evaluate_coverage(y_true, y_pred, weight_tiers=tiers)
        self.assertIn("overall_coverage", metrics)
        self.assertIn("coverage_rate", metrics)
        self.assertIn("mean_half_width", metrics)
        self.assertIn("per_slab_coverage", metrics)
        self.assertIn("T1", metrics["per_slab_coverage"])
        self.assertIn("T2", metrics["per_slab_coverage"])
        self.assertGreaterEqual(metrics["overall_coverage"], 0.89)

    def test_serialization_roundtrip(self):
        """Verifies save() and load() preserve state and produce bit-identical corridor bands."""
        y_true = np.random.normal(12.0, 3.0, size=100)
        y_pred = np.random.normal(12.0, 2.5, size=100)
        tiers = np.random.choice(["Tier_1", "Tier_2", "Tier_3"], size=100)
        known = np.random.choice([True, False], size=100)

        calibrator = ConformalCorridorCalibrator(alpha=0.10, k_prior=15.0)
        calibrator.fit(y_true, y_pred, weight_tiers=tiers, is_known_customer=known)

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            calibrator.save(tmp_path)
            loaded = ConformalCorridorCalibrator.load(tmp_path)

            self.assertTrue(loaded.is_fitted)
            self.assertEqual(loaded.alpha, calibrator.alpha)
            self.assertEqual(loaded.k_prior, calibrator.k_prior)
            self.assertEqual(loaded.calibration_n, calibrator.calibration_n)
            self.assertEqual(loaded.global_half_width, calibrator.global_half_width)
            self.assertEqual(loaded.slab_half_widths, calibrator.slab_half_widths)
            self.assertEqual(loaded.customer_type_half_widths, calibrator.customer_type_half_widths)

            # Compare corridor prediction outputs
            test_preds = [8.0, 12.5, 18.0]
            for p in test_preds:
                for t in ["Tier_1", "Tier_2", "Tier_3", "Unknown"]:
                    for k in [True, False]:
                        orig_res = calibrator.predict_corridor(p, weight_tier=t, is_known_customer=k)
                        load_res = loaded.predict_corridor(p, weight_tier=t, is_known_customer=k)
                        self.assertEqual(orig_res, load_res)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_real_out_of_time_holdout_coverage_resolves_issue_m5(self):
        """
        Integration test proving calibrated split-conformal corridors achieve >= 88.0%
        empirical coverage on the out-of-time holdout (Aug-Sep 2026), resolving Issue M5
        where legacy fixed-spread corridors collapsed to 22-32% coverage.
        """
        if not (os.path.exists(DATA_PATH) and os.path.exists(BENCHMARK_PATH)):
            self.skipTest("Data files not available for out-of-time integration test")

        df_all = pd.read_csv(DATA_PATH)
        bench_df = pd.read_csv(BENCHMARK_PATH)

        # Enrich benchmark with point-in-time features from won deals reference
        missing_pit = [col for col in PIT_FEATURE_COLS if col not in bench_df.columns]
        if missing_pit and "Inquiry Number" in bench_df.columns:
            pit_ref = pd.read_csv(DATA_PATH, usecols=["Inquiry Number"] + PIT_FEATURE_COLS)
            bench_df = bench_df.merge(pit_ref, on="Inquiry Number", how="left")

        # Split data temporally:
        # Train on pre-July 2026 deals (Jan-Jun 2026)
        train_df = df_all[df_all["Year_Month"] < "2026-07"].copy()
        # Calibrate conformal corridors on July 2026 validation split
        cal_df = df_all[df_all["Year_Month"] == "2026-07"].copy()

        # Train Stage 1A MarginQuantileModel
        model = MarginQuantileModel()
        model.fit(train_df, train_df["Margin_Percentage"])

        # Predict margins on calibration set
        cal_preds = model.predict(cal_df)
        cal_actual = cal_df["Margin_Percentage"].values
        cal_slabs = cal_df["Weight_Tier"].values
        cal_known = (cal_df["cust_pit_won_count_180d"] > 0).values

        # Calibrate ConformalCorridorCalibrator (alpha=0.10 => 90% nominal coverage)
        calibrator = ConformalCorridorCalibrator(alpha=0.10, k_prior=20.0)
        calibrator.fit(
            y_true=cal_actual,
            y_pred=cal_preds,
            weight_tiers=cal_slabs,
            is_known_customer=cal_known,
        )

        # Evaluate on the honest out-of-time benchmark (August-September 2026)
        bench_preds = model.predict(bench_df)
        bench_actual = bench_df["Margin_Percentage"].values
        bench_slabs = bench_df["Weight_Tier"].values

        eval_results = calibrator.evaluate_coverage(
            y_true=bench_actual,
            y_pred=bench_preds,
            weight_tiers=bench_slabs,
        )

        empirical_coverage = eval_results["coverage_rate"]
        mean_half_width = eval_results["mean_half_width"]

        # Conformal guarantee verification:
        # 1. Real holdout coverage must be >= 88.0% (target 90%)
        self.assertGreaterEqual(
            empirical_coverage,
            0.88,
            f"Calibrated conformal corridor out-of-time coverage ({empirical_coverage*100:.2f}%) must be >= 88.0%",
        )

        # 2. Solves Issue M5: drastically outperforms legacy corridor hit rate (legacy was 22.5% - 31.6%)
        legacy_reported_max_rate = 0.35
        self.assertGreater(
            empirical_coverage,
            legacy_reported_max_rate,
            f"Conformal coverage ({empirical_coverage*100:.2f}%) must far exceed legacy corridor ({legacy_reported_max_rate*100:.1f}%)",
        )

        # 3. Half-width is realistic for air export freight (between 7.0 and 17.0 pp)
        self.assertGreaterEqual(mean_half_width, 7.0)
        self.assertLessEqual(mean_half_width, 17.0)


if __name__ == "__main__":
    unittest.main()
