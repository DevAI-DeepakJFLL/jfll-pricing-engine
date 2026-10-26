"""
tests/test_quantile_model.py
Unit and integration tests for MarginQuantileModel.

Verifies:
1. Strict mathematical quantile ordering (P10 <= P50 <= P90) across all predictions.
2. Monotonic decreasing constraints on Total_Buy_INR and Chargeable_Weight_Kg.
3. Serialization and deserialization roundtrip parity.
4. Out-of-time benchmark MAE beats commercial baselines (MAE <= 5.50 pp).
5. predict() parity with P50 median quantile.
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

from src.quantile_model import MarginQuantileModel, FEATURE_COLS

DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "won_deals_margin_v3.csv")
BENCHMARK_PATH = os.path.join(BASE_DIR, "tests", "data", "benchmark_won_out_of_time.csv")


class TestQuantileModel(unittest.TestCase):
    """Test suite verifying MarginQuantileModel mathematical and empirical integrity."""

    @classmethod
    def setUpClass(cls):
        """Train standard model on Jan-Jul 2026 won deals for test reuse."""
        if os.path.exists(DATA_PATH):
            df_all = pd.read_csv(DATA_PATH)
            cls.train_df = df_all[df_all["Year_Month"] < "2026-08"].copy()
            cls.model = MarginQuantileModel()
            cls.model.fit(cls.train_df, cls.train_df["Margin_Percentage"])
        else:
            cls.train_df = None
            cls.model = None

        if os.path.exists(BENCHMARK_PATH):
            bench_df = pd.read_csv(BENCHMARK_PATH)
            # Decoupled test enrichment: merge point-in-time features from won deals reference
            if os.path.exists(DATA_PATH):
                from src.quantile_model import PIT_FEATURE_COLS
                missing_pit = [col for col in PIT_FEATURE_COLS if col not in bench_df.columns]
                if missing_pit and "Inquiry Number" in bench_df.columns:
                    pit_ref = pd.read_csv(DATA_PATH, usecols=["Inquiry Number"] + PIT_FEATURE_COLS)
                    bench_df = bench_df.merge(pit_ref, on="Inquiry Number", how="left")
            if "Unit_Buy_INR" not in bench_df.columns and "Total_Buy_INR" in bench_df.columns:
                bench_df["Unit_Buy_INR"] = bench_df["Total_Buy_INR"] / bench_df["Chargeable_Weight_Kg"].clip(lower=0.1)
            cls.bench_df = bench_df
        else:
            cls.bench_df = None

    def test_p10_p50_p90_ordering_synthetic(self):
        """Verifies P10 <= P50 <= P90 on synthetic matrix inputs."""
        model = MarginQuantileModel()
        X_synth = np.random.rand(100, 5)
        y_synth = np.random.rand(100) * 20.0 + 5.0
        model.fit(X_synth, y_synth)

        p10, p50, p90 = model.predict_quantiles(X_synth[:20])
        self.assertEqual(len(p10), 20)
        self.assertEqual(len(p50), 20)
        self.assertEqual(len(p90), 20)
        self.assertTrue((p10 <= p50).all(), "P10 must be <= P50 for all samples")
        self.assertTrue((p50 <= p90).all(), "P50 must be <= P90 for all samples")

    def test_p10_p50_p90_ordering_real_data(self):
        """Verifies strict quantile ordering on real out-of-time benchmark records."""
        self.assertIsNotNone(self.model, "Model must be trained on won_deals_margin_v3")
        self.assertIsNotNone(self.bench_df, "Benchmark dataset must exist")

        p10, p50, p90 = self.model.predict_quantiles(self.bench_df)
        self.assertEqual(len(p10), len(self.bench_df))
        self.assertTrue(
            (p10 <= p50 + 1e-9).all(),
            "Real-data P10 must be <= P50 across all holdout predictions",
        )
        self.assertTrue(
            (p50 <= p90 + 1e-9).all(),
            "Real-data P50 must be <= P90 across all holdout predictions",
        )

    def test_monotonicity_on_buy_cost_and_weight(self):
        """Proves higher buy cost and higher weight do not increase margin percentage."""
        self.assertIsNotNone(self.model, "Model must be trained")
        self.assertIsNotNone(self.bench_df, "Benchmark dataset must exist")

        sample = self.bench_df.iloc[:50].copy()
        base_preds = self.model.predict(sample)

        # 1. Monotonicity on Total_Buy_INR: increasing buy cost must not increase margin
        high_buy_sample = sample.copy()
        high_buy_sample["Total_Buy_INR"] = high_buy_sample["Total_Buy_INR"] + 50000.0
        high_buy_preds = self.model.predict(high_buy_sample)
        buy_violations = (high_buy_preds > base_preds + 1e-5).sum()
        self.assertEqual(
            buy_violations,
            0,
            f"Violations detected where higher Total_Buy_INR increased margin: {buy_violations}",
        )

        # 2. Monotonicity on Chargeable_Weight_Kg: increasing weight must not increase margin
        high_wt_sample = sample.copy()
        high_wt_sample["Chargeable_Weight_Kg"] = high_wt_sample["Chargeable_Weight_Kg"] + 500.0
        high_wt_preds = self.model.predict(high_wt_sample)
        wt_violations = (high_wt_preds > base_preds + 1e-5).sum()
        self.assertEqual(
            wt_violations,
            0,
            f"Violations detected where higher Chargeable_Weight_Kg increased margin: {wt_violations}",
        )

        # 3. Monotonicity on Unit_Buy_INR: increasing freight rate per kg must not increase margin
        high_unit_sample = sample.copy()
        high_unit_sample["Unit_Buy_INR"] = high_unit_sample["Unit_Buy_INR"] + 100.0
        high_unit_preds = self.model.predict(high_unit_sample)
        unit_violations = (high_unit_preds > base_preds + 1e-5).sum()
        self.assertEqual(
            unit_violations,
            0,
            f"Violations detected where higher Unit_Buy_INR increased margin: {unit_violations}",
        )

    def test_serialization_deserialization_roundtrip(self):
        """Verifies exact prediction parity across save() and load() roundtrip."""
        self.assertIsNotNone(self.model, "Model must be trained")
        self.assertIsNotNone(self.bench_df, "Benchmark dataset must exist")

        sample = self.bench_df.iloc[:20].copy()
        orig_p10, orig_p50, orig_p90 = self.model.predict_quantiles(sample)
        orig_predict = self.model.predict(sample)

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            self.model.save(tmp_path)
            loaded_model = MarginQuantileModel.load(tmp_path)

            loaded_p10, loaded_p50, loaded_p90 = loaded_model.predict_quantiles(sample)
            loaded_predict = loaded_model.predict(sample)

            np.testing.assert_allclose(orig_p10, loaded_p10, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(orig_p50, loaded_p50, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(orig_p90, loaded_p90, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(orig_predict, loaded_predict, rtol=1e-5, atol=1e-5)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_predict_equals_p50(self):
        """Verifies predict() is an exact alias for the P50 median target quote."""
        self.assertIsNotNone(self.model, "Model must be trained")
        self.assertIsNotNone(self.bench_df, "Benchmark dataset must exist")

        sample = self.bench_df.iloc[:30].copy()
        _, p50, _ = self.model.predict_quantiles(sample)
        direct_pred = self.model.predict(sample)

        np.testing.assert_allclose(direct_pred, p50, rtol=1e-5, atol=1e-5)

    def test_out_of_time_mae_beats_baselines(self):
        """Verifies that P50 out-of-time MAE beats naive baselines (MAE <= 5.50 pp)."""
        self.assertIsNotNone(self.model, "Model must be trained")
        self.assertIsNotNone(self.bench_df, "Benchmark dataset must exist")

        p50_preds = self.model.predict(self.bench_df)
        actual = self.bench_df["Margin_Percentage"].values
        mae = float(np.mean(np.abs(p50_preds - actual)))

        # Target threshold: MAE <= 5.50% (true P50 achieves ~5.36% / ~5.48% with monotonic tree bounding, beats 5.98% production & 7.00% commodity median)
        self.assertLessEqual(
            mae,
            5.50,
            f"Quantile P50 MAE ({mae:.3f}%) must be <= 5.50%",
        )
        # Outperforms commodity median baseline (7.00%)
        self.assertLess(
            mae,
            7.00,
            f"Quantile P50 MAE ({mae:.3f}%) must outperform Commodity Median (7.00%)",
        )
        # Outperforms legacy production framework (5.98%)
        self.assertLess(
            mae,
            5.98,
            f"Quantile P50 MAE ({mae:.3f}%) must outperform legacy production framework (5.98%)",
        )


if __name__ == "__main__":
    unittest.main()
