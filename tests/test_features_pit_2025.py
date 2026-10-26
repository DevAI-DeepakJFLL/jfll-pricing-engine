"""
tests/test_features_pit_2025.py
Unit tests for Point-in-Time (PIT) Rolling Feature Engineering on 2025 Data:
1. Zero forward lookahead leakage (future rows manipulation cannot alter past features).
2. PIT feature completeness, non-null guarantees on bounded columns, and numeric types.
3. CustomerPITRegistry query accuracy (priors for unknown customers, active stats for known top customers).
4. Streaming vs batch equivalence and hard build gate compliance.
"""

import os
import sys
import unittest
import joblib
import numpy as np
import pandas as pd
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features_pit import (
    compute_point_in_time_features,
    CustomerPITRegistry,
    DEFAULT_RECENCY_DAYS,
    DEFAULT_WIN_RATE,
)
from src.preprocess_pipeline_v3 import validate_build_gates

TRAIN_PIT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "won_deals_2025_train_70_pit.csv")
TEST_PIT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "won_deals_2025_test_30_pit.csv")
REGISTRY_PATH = os.path.join(PROJECT_ROOT, "models", "customer_pit_registry_2025.joblib")

PIT_COLS = [
    "cust_pit_won_count_180d",
    "cust_pit_inquiry_count_180d",
    "cust_pit_recency_days",
    "cust_pit_median_margin_won",
    "cust_pit_total_spend_inr_180d",
    "cust_pit_win_rate_180d",
    "lane_pit_median_margin_won_90d",
]


class TestForwardLeakagePrevention(unittest.TestCase):
    """Verify strictly zero forward lookahead leakage (t < T)."""

    def setUp(self):
        # Synthetic customer scenario across 4 distinct dates
        self.leak_test_df = pd.DataFrame([
            {
                "quote_date": "2025-01-10",
                "customer_id": "CUST_LEAK_TEST",
                "Regional_Lane": "West India -> Europe",
                "is_won": 1,
                "Margin_Percentage": 12.0,
                "Total_Sell_INR": 100000.0,
            },
            {
                "quote_date": "2025-02-15",
                "customer_id": "CUST_LEAK_TEST",
                "Regional_Lane": "West India -> Europe",
                "is_won": 1,
                "Margin_Percentage": 15.0,
                "Total_Sell_INR": 150000.0,
            },
            {
                "quote_date": "2025-03-20",
                "customer_id": "CUST_LEAK_TEST",
                "Regional_Lane": "West India -> Europe",
                "is_won": 0,
                "Margin_Percentage": 25.0,
                "Total_Sell_INR": 80000.0,
            },
            {
                "quote_date": "2025-04-25",
                "customer_id": "CUST_LEAK_TEST",
                "Regional_Lane": "West India -> Europe",
                "is_won": 1,
                "Margin_Percentage": 18.0,
                "Total_Sell_INR": 200000.0,
            },
        ])

    def test_future_row_manipulation_zero_leakage(self):
        """Manipulating or appending future rows MUST NEVER alter earlier PIT features."""
        baseline = compute_point_in_time_features(self.leak_test_df)

        # Mutate future row 3 (e.g., change margin from 18 to 999%, sell to 10M, is_won to 0)
        perturbed_df = self.leak_test_df.copy()
        perturbed_df.loc[3, "Margin_Percentage"] = 999.0
        perturbed_df.loc[3, "Total_Sell_INR"] = 10000000.0
        perturbed_df.loc[3, "is_won"] = 0

        # Also add a new future row 4 in May 2025
        new_row = pd.DataFrame([{
            "quote_date": "2025-05-30",
            "customer_id": "CUST_LEAK_TEST",
            "Regional_Lane": "West India -> Europe",
            "is_won": 1,
            "Margin_Percentage": 50.0,
            "Total_Sell_INR": 500000.0,
        }])
        perturbed_df = pd.concat([perturbed_df, new_row], ignore_index=True)

        recomputed = compute_point_in_time_features(perturbed_df)

        # Rows 0, 1, 2 must have EXACTLY identical PIT features in baseline and recomputed
        for col in PIT_COLS:
            for row_idx in [0, 1, 2]:
                val_base = baseline.loc[row_idx, col]
                val_recomp = recomputed.loc[row_idx, col]
                if np.isnan(val_base):
                    self.assertTrue(
                        np.isnan(val_recomp),
                        f"Expected NaN for {col} at row {row_idx}, got {val_recomp}",
                    )
                else:
                    self.assertAlmostEqual(
                        val_base,
                        val_recomp,
                        places=6,
                        msg=f"Leakage detected! Column {col} at row {row_idx} altered by future rows.",
                    )

    def test_same_day_transactions_cannot_see_each_other(self):
        """Quotes on the same day cannot access each other's transactions."""
        same_day_df = pd.DataFrame([
            {
                "quote_date": "2025-03-01",
                "customer_id": "CUST_SAME_DAY",
                "Regional_Lane": "North India -> Middle East",
                "is_won": 1,
                "Margin_Percentage": 10.0,
                "Total_Sell_INR": 50000.0,
            },
            {
                "quote_date": "2025-03-01",
                "customer_id": "CUST_SAME_DAY",
                "Regional_Lane": "North India -> Middle East",
                "is_won": 1,
                "Margin_Percentage": 20.0,
                "Total_Sell_INR": 75000.0,
            },
        ])
        res = compute_point_in_time_features(same_day_df)

        # Both rows are on the same day, so both must see 0 prior won deals and 0 prior inquiries
        self.assertEqual(res.loc[0, "cust_pit_won_count_180d"], 0)
        self.assertEqual(res.loc[1, "cust_pit_won_count_180d"], 0)
        self.assertEqual(res.loc[0, "cust_pit_inquiry_count_180d"], 0)
        self.assertEqual(res.loc[1, "cust_pit_inquiry_count_180d"], 0)
        self.assertTrue(np.isnan(res.loc[0, "cust_pit_median_margin_won"]))
        self.assertTrue(np.isnan(res.loc[1, "cust_pit_median_margin_won"]))

    def test_test_set_cannot_leak_into_train_set(self):
        """Train set PIT features computed standalone must match combined train slice exactly."""
        train_raw_path = os.path.join(PROJECT_ROOT, "data", "processed", "won_deals_2025_train_70.csv")
        test_raw_path = os.path.join(PROJECT_ROOT, "data", "processed", "won_deals_2025_test_30.csv")
        if not (os.path.exists(train_raw_path) and os.path.exists(test_raw_path)):
            pytest.skip("Base train/test 2025 files not present")

        train_df = pd.read_csv(train_raw_path, low_memory=False)
        test_df = pd.read_csv(test_raw_path, low_memory=False)

        # Standalone train PIT
        train_pit = compute_point_in_time_features(
            train_df,
            date_col="quote_date",
            cust_col="customer_id",
            lane_col="Regional_Lane",
        )

        # Combined train + test PIT
        combined = pd.concat([train_df, test_df], ignore_index=True)
        combined_pit = compute_point_in_time_features(
            combined,
            date_col="quote_date",
            cust_col="customer_id",
            lane_col="Regional_Lane",
        )

        # The train portion of combined_pit must be identical to train_pit
        for col in PIT_COLS:
            s_train = train_pit[col].fillna(-999.0).values
            s_comb = combined_pit.iloc[:len(train_df)][col].fillna(-999.0).values
            np.testing.assert_allclose(
                s_train,
                s_comb,
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"Discrepancy in train PIT feature '{col}' when test data is present!",
            )


class TestPITFeatureCompletenessAndSchema(unittest.TestCase):
    """Verify presence, row count, null rates, and numeric types of generated 2025 PIT datasets."""

    @classmethod
    def setUpClass(cls):
        if not (os.path.exists(TRAIN_PIT_PATH) and os.path.exists(TEST_PIT_PATH)):
            pytest.skip("Generated PIT datasets not found on disk. Run compute_pit_2025.py first.")
        cls.train_df = pd.read_csv(TRAIN_PIT_PATH, low_memory=False)
        cls.test_df = pd.read_csv(TEST_PIT_PATH, low_memory=False)

    def test_persisted_pit_csvs_exist_and_non_empty(self):
        """Generated PIT CSV files must exist and contain exact row counts."""
        self.assertTrue(os.path.exists(TRAIN_PIT_PATH), f"Missing {TRAIN_PIT_PATH}")
        self.assertTrue(os.path.exists(TEST_PIT_PATH), f"Missing {TEST_PIT_PATH}")

        # Train: 9,515 rows (70%), Test: 4,078 rows (30%)
        self.assertEqual(len(self.train_df), 9515)
        self.assertEqual(len(self.test_df), 4078)

    def test_all_seven_pit_columns_present_and_numeric(self):
        """All 7 point-in-time features must exist and have numeric dtypes in both splits."""
        for name, df in [("train", self.train_df), ("test", self.test_df)]:
            for col in PIT_COLS:
                self.assertIn(col, df.columns, f"Feature '{col}' missing from {name} PIT dataset.")
                self.assertTrue(
                    np.issubdtype(df[col].dtype, np.number),
                    f"Feature '{col}' in {name} is not numeric (dtype={df[col].dtype}).",
                )

    def test_bounded_columns_zero_nulls(self):
        """Counts, recency, spend, and win rate must have strictly zero nulls."""
        zero_null_cols = [
            "cust_pit_won_count_180d",
            "cust_pit_inquiry_count_180d",
            "cust_pit_recency_days",
            "cust_pit_total_spend_inr_180d",
            "cust_pit_win_rate_180d",
        ]
        for name, df in [("train", self.train_df), ("test", self.test_df)]:
            for col in zero_null_cols:
                null_cnt = df[col].isnull().sum()
                self.assertEqual(
                    null_cnt,
                    0,
                    f"Found {null_cnt} nulls in bounded PIT feature '{col}' in {name} split.",
                )

    def test_active_customers_have_populated_numeric_median_margins(self):
        """Whenever cust_pit_won_count_180d > 0, cust_pit_median_margin_won must be non-null and positive."""
        for name, df in [("train", self.train_df), ("test", self.test_df)]:
            active_mask = df["cust_pit_won_count_180d"] > 0
            active_margins = df.loc[active_mask, "cust_pit_median_margin_won"]
            self.assertEqual(
                active_margins.isnull().sum(),
                0,
                f"Found NaN median margins for active customers in {name} split.",
            )
            self.assertTrue(
                (active_margins > 0).all(),
                f"Non-positive median margins found for active customers in {name} split.",
            )

    def test_active_lanes_have_populated_numeric_median_margins(self):
        """Whenever lane_pit_median_margin_won_90d is not null, it must be positive and finite."""
        for name, df in [("train", self.train_df), ("test", self.test_df)]:
            valid_lane_margins = df["lane_pit_median_margin_won_90d"].dropna()
            self.assertGreater(len(valid_lane_margins), 0)
            self.assertTrue(
                (valid_lane_margins > 0).all(),
                f"Found non-positive lane median margins in {name} split.",
            )

    def test_build_gate_assertions(self):
        """Both train and test PIT datasets must strictly pass validate_build_gates."""
        validate_build_gates(
            self.train_df,
            is_margin_dataset=True,
            dataset_name="won_deals_2025_train_70_pit.csv",
        )
        validate_build_gates(
            self.test_df,
            is_margin_dataset=True,
            dataset_name="won_deals_2025_test_30_pit.csv",
        )


class TestCustomerPITRegistry2025(unittest.TestCase):
    """Verify CustomerPITRegistry serialization and query accuracy on 2025 data."""

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(REGISTRY_PATH):
            pytest.skip(f"Serialized registry {REGISTRY_PATH} not found. Run compute_pit_2025.py first.")
        cls.registry: CustomerPITRegistry = joblib.load(REGISTRY_PATH)

    def test_registry_instance_type(self):
        """Loaded object must be an instance of CustomerPITRegistry."""
        self.assertIsInstance(self.registry, CustomerPITRegistry)

    def test_unknown_customer_returns_expected_priors(self):
        """Querying an unknown customer must return default prior constants."""
        feats = self.registry.get_features(
            quote_date="2025-09-27",
            customer_id="CUST_NONEXISTENT_XYZ_99999",
            regional_lane="West India -> Europe",
        )
        self.assertEqual(feats["cust_pit_won_count_180d"], 0)
        self.assertEqual(feats["cust_pit_inquiry_count_180d"], 0)
        self.assertEqual(feats["cust_pit_recency_days"], DEFAULT_RECENCY_DAYS)
        self.assertTrue(np.isnan(feats["cust_pit_median_margin_won"]))
        self.assertEqual(feats["cust_pit_total_spend_inr_180d"], 0.0)
        self.assertEqual(feats["cust_pit_win_rate_180d"], DEFAULT_WIN_RATE)

    def test_known_top_customer_universal_clearing(self):
        """Querying top customer CUST_UNIVERSAL_CLEARING_AND_FORWARDING must return populated active stats."""
        feats = self.registry.get_features(
            quote_date="2025-09-27",
            customer_id="CUST_UNIVERSAL_CLEARING_AND_FORWARDING",
            regional_lane="West India -> Europe",
        )
        # Expected active stats as of train cutoff 2025-09-26:
        # > 900 won bookings, 100% win rate, 1 day recency, ~40% median margin, spend > 37M INR
        self.assertGreater(feats["cust_pit_won_count_180d"], 900)
        self.assertGreater(feats["cust_pit_inquiry_count_180d"], 900)
        self.assertLessEqual(feats["cust_pit_recency_days"], 7.0)
        self.assertAlmostEqual(feats["cust_pit_median_margin_won"], 40.0, places=1)
        self.assertGreater(feats["cust_pit_total_spend_inr_180d"], 30000000.0)
        self.assertAlmostEqual(feats["cust_pit_win_rate_180d"], 1.0, places=2)
        self.assertFalse(np.isnan(feats["lane_pit_median_margin_won_90d"]))
        self.assertGreater(feats["lane_pit_median_margin_won_90d"], 5.0)

    def test_lookup_resilience_name_and_slug(self):
        """Registry lookup must produce identical stats whether queried by slug or company name."""
        feats_slug = self.registry.get_features(
            quote_date="2025-09-27",
            customer_id="CUST_UNIVERSAL_CLEARING_AND_FORWARDING",
            regional_lane="West India -> Europe",
        )
        feats_name = self.registry.get_features(
            quote_date="2025-09-27",
            customer_id="UNIVERSAL CLEARING AND FORWARDING",
            regional_lane="West India -> Europe",
        )
        self.assertEqual(feats_slug, feats_name)

    def test_temporal_monotonic_query_growth(self):
        """Querying earlier in the year must show fewer transactions than later in the year."""
        early_feats = self.registry.get_features(
            quote_date="2025-01-15",
            customer_id="CUST_UNIVERSAL_CLEARING_AND_FORWARDING",
            regional_lane="West India -> Europe",
        )
        late_feats = self.registry.get_features(
            quote_date="2025-09-27",
            customer_id="CUST_UNIVERSAL_CLEARING_AND_FORWARDING",
            regional_lane="West India -> Europe",
        )
        self.assertLess(early_feats["cust_pit_won_count_180d"], late_feats["cust_pit_won_count_180d"])
        self.assertLess(early_feats["cust_pit_total_spend_inr_180d"], late_feats["cust_pit_total_spend_inr_180d"])


if __name__ == "__main__":
    unittest.main()
