"""
tests/test_training_eval_sampling.py
Unit tests for the Stratified Training Evaluation Sampling Engine (500 records).
Validates 5x5 segment stratification (25 segments x 20 records = 500 rows),
monotonicity of Total_Buy_INR across budget levels, and zero missing critical values.
"""

import os
import unittest
import pandas as pd

from scripts.generate_training_eval_samples import (
    generate_stratified_training_samples,
    DEFAULT_COMMODITIES,
    BUDGET_TIERS,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "Air_Export_Pricing_Combined_ML.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "tests", "data", "training_eval_samples_500.csv")

CRITICAL_COLUMNS = [
    "Total_Buy_INR",
    "Chargeable_Weight_Kg",
    "Origin_Port",
    "Destination_Port",
    "Company_Group",
    "Commodity_Group",
    "Margin_Percentage",
]


class TestTrainingEvalSampling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Generate samples for test suite
        cls.df = generate_stratified_training_samples(
            data_path=DATA_PATH,
            commodities=DEFAULT_COMMODITIES,
            n_per_cell=20,
            seed=42,
        )

    def test_total_row_count(self):
        """Assert exactly 500 total rows (25 segments x 20 records)."""
        self.assertEqual(len(self.df), 500)

    def test_unique_commodities(self):
        """Assert exactly 5 unique commodity groups matching expected categories."""
        unique_commodities = sorted(self.df["Commodity_Group"].unique())
        self.assertEqual(len(unique_commodities), 5)
        self.assertEqual(unique_commodities, sorted(DEFAULT_COMMODITIES))

    def test_budget_segments_per_commodity(self):
        """Assert exactly 5 budget segments present for every commodity."""
        for comm in DEFAULT_COMMODITIES:
            sub = self.df[self.df["Commodity_Group"] == comm]
            segments = set(sub["Budget_Segment"].unique())
            self.assertEqual(
                segments,
                set(BUDGET_TIERS),
                f"Missing budget segments for commodity {comm}: {set(BUDGET_TIERS) - segments}",
            )

    def test_cell_counts(self):
        """Assert exactly 20 rows for every single (Commodity_Group, Budget_Segment) combination."""
        counts = self.df.groupby(["Commodity_Group", "Budget_Segment"], observed=True).size()
        self.assertEqual(len(counts), 25, "Expected exactly 25 segment combinations")
        for (comm, budget), count in counts.items():
            self.assertEqual(
                count,
                20,
                f"Cell ({comm}, {budget}) has count {count}, expected 20",
            )

    def test_monotonic_increase_of_buy_inr(self):
        """Assert that for every commodity, mean Total_Buy_INR strictly increases across budget tiers."""
        for comm in DEFAULT_COMMODITIES:
            sub = self.df[self.df["Commodity_Group"] == comm]
            means = sub.groupby("Budget_Segment", observed=True)["Total_Buy_INR"].mean()
            mean_values = [means[tier] for tier in BUDGET_TIERS]
            for i in range(len(mean_values) - 1):
                self.assertLess(
                    mean_values[i],
                    mean_values[i + 1],
                    f"Commodity '{comm}': Mean Total_Buy_INR at '{BUDGET_TIERS[i]}' ({mean_values[i]:.2f}) "
                    f"is not strictly less than '{BUDGET_TIERS[i+1]}' ({mean_values[i+1]:.2f})",
                )

    def test_no_missing_critical_fields(self):
        """Assert no missing values in critical inference columns."""
        for col in CRITICAL_COLUMNS:
            self.assertIn(col, self.df.columns, f"Column '{col}' must be present in sample dataset")
            missing_count = self.df[col].isna().sum()
            self.assertEqual(
                missing_count,
                0,
                f"Column '{col}' has {missing_count} missing values; 0 expected",
            )

    def test_generated_csv_file_if_present(self):
        """If tests/data/training_eval_samples_500.csv exists, verify it passes all checks."""
        if os.path.exists(OUTPUT_PATH):
            saved_df = pd.read_csv(OUTPUT_PATH)
            self.assertEqual(len(saved_df), 500)
            self.assertEqual(saved_df["Commodity_Group"].nunique(), 5)
            self.assertEqual(saved_df["Budget_Segment"].nunique(), 5)


if __name__ == "__main__":
    unittest.main()
