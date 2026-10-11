"""
tests/test_training_evaluation_harness.py
Unit and integration tests for Model Inference & Margin Gap Evaluation Harness.
Verifies gap calculation identity, MAE/RMSE math, end-to-end evaluation on 500 samples,
and segment aggregations (by_cell, by_commodity, by_budget, overall).
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.evaluate_training_samples import (
    compute_margin_gap,
    compute_mae,
    compute_rmse,
    compute_segment_metrics,
    evaluate_model_on_training_samples,
)
from src.engine_v3 import PricingEngineV3

SAMPLES_CSV_PATH = os.path.join(PROJECT_ROOT, "tests", "data", "training_eval_samples_500.csv")


class TestTrainingEvaluationHarness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Shared PricingEngineV3 instance for fast test execution
        cls.engine = PricingEngineV3()
        cls.samples_df = pd.read_csv(SAMPLES_CSV_PATH)

    def test_margin_gap_identity(self):
        """Verifies Gap calculation identity: Gap = Model - Actual."""
        # Over-quoting case: Model 18.5%, Actual 15.0% -> Gap +3.5 pp
        self.assertAlmostEqual(compute_margin_gap(18.5, 15.0), 3.5, places=5)
        # Under-quoting case: Model 10.0%, Actual 14.2% -> Gap -4.2 pp
        self.assertAlmostEqual(compute_margin_gap(10.0, 14.2), -4.2, places=5)
        # Exact match case: Model 12.0%, Actual 12.0% -> Gap 0.0 pp
        self.assertAlmostEqual(compute_margin_gap(12.0, 12.0), 0.0, places=5)

    def test_mae_and_rmse_math_synthetic(self):
        """Verifies MAE and RMSE math on synthetic inputs with hand-calculated expected values."""
        gaps = [3.0, -4.0, 0.0, -1.0, 2.0]
        # Absolute errors: 3, 4, 0, 1, 2 -> sum = 10 -> MAE = 10 / 5 = 2.0
        expected_mae = 2.0
        # Squared errors: 9, 16, 0, 1, 4 -> sum = 30 -> MSE = 6.0 -> RMSE = sqrt(6) ~ 2.4494897
        expected_rmse = float(np.sqrt(6.0))

        calc_mae = compute_mae(gaps)
        calc_rmse = compute_rmse(gaps)

        self.assertAlmostEqual(calc_mae, expected_mae, places=5)
        self.assertAlmostEqual(calc_rmse, expected_rmse, places=5)

    def test_segment_metrics_aggregation_synthetic(self):
        """Verifies compute_segment_metrics computes correct aggregation fields on synthetic DataFrame."""
        syn_df = pd.DataFrame({
            "Actual_Margin": [10.0, 20.0, 15.0, 25.0],
            "Model_Margin": [12.0, 18.0, 15.0, 28.0],
            "Margin_Gap": [2.0, -2.0, 0.0, 3.0],
            "Absolute_Error": [2.0, 2.0, 0.0, 3.0],
            "Squared_Error": [4.0, 4.0, 0.0, 9.0],
            "In_Corridor": [True, True, False, True],
        })

        metrics = compute_segment_metrics(syn_df)
        self.assertEqual(metrics["n_records"], 4)
        self.assertAlmostEqual(metrics["actual_margin_mean"], 17.5, places=4)
        self.assertAlmostEqual(metrics["model_margin_mean"], 18.25, places=4)
        self.assertAlmostEqual(metrics["margin_gap_mean"], 0.75, places=4)
        self.assertAlmostEqual(metrics["mae"], 1.75, places=4)
        self.assertAlmostEqual(metrics["rmse"], float(np.sqrt(17.0 / 4.0)), places=4)
        self.assertAlmostEqual(metrics["corridor_coverage"], 0.75, places=4)

    def test_evaluate_model_on_500_training_samples(self):
        """Integration test: runs engine on 500 stratified training samples and verifies completeness."""
        evaluated_df, aggregations = evaluate_model_on_training_samples(
            samples_df=self.samples_df,
            engine=self.engine,
            strategy="balanced",
        )

        # 1. Verify record count and row alignment
        self.assertEqual(len(evaluated_df), 500)
        self.assertEqual(len(evaluated_df), len(self.samples_df))

        # 2. Verify all required evaluation columns exist with 0 missing values
        required_cols = [
            "Actual_Margin",
            "Model_Margin",
            "Margin_Gap",
            "Absolute_Error",
            "Squared_Error",
            "Corridor_Min",
            "Corridor_Max",
            "In_Corridor",
            "Win_Probability",
            "Quoted_Sell_Price_INR",
            "Cost_Floor_Price_INR",
            "Limiting_Factor",
        ]
        for col in required_cols:
            self.assertIn(col, evaluated_df.columns, f"Missing evaluation column '{col}'")
            missing = evaluated_df[col].isna().sum()
            self.assertEqual(missing, 0, f"Column '{col}' has {missing} NaN values; 0 expected")

        # 3. Verify gap identities and corridor logic on evaluated rows
        for idx, row in evaluated_df.iterrows():
            expected_gap = row["Model_Margin"] - row["Actual_Margin"]
            self.assertAlmostEqual(
                row["Margin_Gap"],
                expected_gap,
                places=4,
                msg=f"Row {idx} margin gap identity mismatch",
            )
            self.assertAlmostEqual(
                row["Absolute_Error"],
                abs(expected_gap),
                places=4,
                msg=f"Row {idx} absolute error mismatch",
            )
            expected_in_corridor = (row["Corridor_Min"] <= row["Actual_Margin"] <= row["Corridor_Max"])
            self.assertEqual(
                bool(row["In_Corridor"]),
                expected_in_corridor,
                msg=f"Row {idx} corridor inclusion mismatch",
            )

        # 4. Verify aggregation structures
        self.assertIn("by_cell", aggregations)
        self.assertIn("by_commodity", aggregations)
        self.assertIn("by_budget", aggregations)
        self.assertIn("overall", aggregations)

        # Verify by_cell has all 25 segments (5 commodities x 5 budget tiers)
        by_cell = aggregations["by_cell"]
        self.assertEqual(len(by_cell), 25, f"Expected 25 cells, got {len(by_cell)}")
        for cell_key, metrics in by_cell.items():
            self.assertEqual(metrics["n_records"], 20, f"Cell {cell_key} n_records != 20")
            for metric_key in [
                "actual_margin_mean",
                "model_margin_mean",
                "margin_gap_mean",
                "mae",
                "rmse",
                "corridor_coverage",
            ]:
                self.assertIn(metric_key, metrics, f"Missing metric {metric_key} in cell {cell_key}")
                self.assertFalse(
                    np.isnan(metrics[metric_key]),
                    f"Metric {metric_key} in cell {cell_key} is NaN",
                )

        # Verify by_commodity has 5 commodities, each with n_records = 100
        by_comm = aggregations["by_commodity"]
        self.assertEqual(len(by_comm), 5)
        for comm, metrics in by_comm.items():
            self.assertEqual(metrics["n_records"], 100)

        # Verify by_budget has 5 budget tiers, each with n_records = 100
        by_budget = aggregations["by_budget"]
        self.assertEqual(len(by_budget), 5)
        for budget, metrics in by_budget.items():
            self.assertEqual(metrics["n_records"], 100)

        # Verify overall
        overall = aggregations["overall"]
        self.assertEqual(overall["n_records"], 500)
        self.assertFalse(np.isnan(overall["mae"]))
        self.assertFalse(np.isnan(overall["rmse"]))

    def test_generated_csv_file_if_present(self):
        """Verifies tests/data/training_eval_results_500.csv integrity if already generated."""
        results_path = os.path.join(PROJECT_ROOT, "tests", "data", "training_eval_results_500.csv")
        if os.path.exists(results_path):
            res_df = pd.read_csv(results_path)
            self.assertEqual(len(res_df), 500)
            self.assertIn("Margin_Gap", res_df.columns)
            self.assertIn("Actual_Margin", res_df.columns)
            self.assertIn("Model_Margin", res_df.columns)
            self.assertIn("In_Corridor", res_df.columns)


if __name__ == "__main__":
    unittest.main()

