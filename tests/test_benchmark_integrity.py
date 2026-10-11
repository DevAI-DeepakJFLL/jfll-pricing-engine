"""
tests/test_benchmark_integrity.py
Verification test suite for honest out-of-sample Won benchmark dataset.
Guarantees zero label leakage (Won-only), out-of-time holdout (Aug-Sep 2026),
no duplicate quotes, and strictly valid economic transactions.
"""

import os
import unittest
import pandas as pd

BENCHMARK_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests",
    "data",
    "benchmark_won_out_of_time.csv",
)

REQUIRED_COLUMNS = [
    "Origin_Port",
    "Destination_Port",
    "Commodity_Group",
    "Total_Buy_INR",
    "Total_Sell_INR",
    "Margin_Amount_INR",
    "Margin_Percentage",
    "Chargeable_Weight_Kg",
    "Customer_Company",
    "Year_Month",
    "is_won",
]


class TestBenchmarkIntegrity(unittest.TestCase):
    def setUp(self):
        self.benchmark_path = BENCHMARK_PATH

    def test_benchmark_file_exists(self):
        self.assertTrue(
            os.path.exists(self.benchmark_path),
            f"Benchmark file must exist at {self.benchmark_path}",
        )

    def test_benchmark_has_no_lost_or_duplicate_rows(self):
        self.assertTrue(
            os.path.exists(self.benchmark_path),
            "Benchmark file must exist before testing row integrity",
        )
        df = pd.read_csv(self.benchmark_path)

        # 1. Minimum sample size
        self.assertGreaterEqual(
            len(df), 200, "Must have at least 200 holdout cases for statistical power"
        )

        # 2. Label purity: Strictly Won quotes
        self.assertTrue(
            (df["is_won"] == 1).all(),
            "Benchmark must strictly contain Won deals (is_won == 1)",
        )

        # 3. Deduplication: No duplicate quote signatures
        duplicates = df.duplicated(
            subset=["Total_Buy_INR", "Chargeable_Weight_Kg", "Customer_Company"]
        )
        self.assertFalse(
            duplicates.any(),
            f"Benchmark must contain zero duplicate rows. Found {duplicates.sum()} duplicates",
        )

        # 4. Out-of-time window: August and September 2026 only
        self.assertTrue(
            (df["Year_Month"] >= "2026-08").all(),
            "Must be strictly out-of-time holdout (Year_Month >= '2026-08')",
        )

        # 5. Financial sanity: Realized buy cost > 500 INR and positive margin
        self.assertTrue(
            (df["Total_Buy_INR"] > 500).all(),
            "All benchmark transactions must have Total_Buy_INR > 500",
        )
        self.assertTrue(
            (df["Total_Sell_INR"] > df["Total_Buy_INR"]).all(),
            "All Won deals must have Total_Sell_INR > Total_Buy_INR (realized positive margin)",
        )

        # 6. Physical sanity: Positive chargeable weight
        self.assertTrue(
            (df["Chargeable_Weight_Kg"] > 0).all(),
            "Chargeable_Weight_Kg must be positive",
        )

    def test_benchmark_required_columns_present(self):
        self.assertTrue(
            os.path.exists(self.benchmark_path),
            "Benchmark file must exist before checking columns",
        )
        df = pd.read_csv(self.benchmark_path)
        for col in REQUIRED_COLUMNS:
            self.assertIn(
                col, df.columns, f"Required column '{col}' missing from benchmark"
            )


if __name__ == "__main__":
    unittest.main()
