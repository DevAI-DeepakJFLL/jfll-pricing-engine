"""
tests/test_data_validation.py
Comprehensive automated test suite for validated data engineering pipeline (V3)
and hard build gate assertions (Fix 1, Fix 2, Issue D4).

Tests:
1. Output datasets exist:
   - data/processed/Air_Export_Pricing_Cleaned_V3.csv
   - data/processed/won_deals_margin_v3.csv
   - data/processed/all_inquiries_win_rate_v3.csv
2. Date consistency checks pass:
   - Quote_Month == pd.to_datetime(quote_date).dt.month
   - Year_Month == pd.to_datetime(quote_date).dt.strftime("%Y-%m")
   - Month_Name == pd.to_datetime(quote_date).dt.month_name()
   - Zero occurrences of the 422 June/September mismatches (Issue D4)
3. Margin identity holds strictly:
   - abs(Margin_Percentage - (Total_Sell_INR - Total_Buy_INR) / Total_Buy_INR * 100) <= 0.05
4. No Buy_Cost_Estimated == 1 rows in won_deals_margin_v3.csv
5. Separate clean splits:
   - won_deals_margin_v3 has 100% is_won == 1 with zero default-margin Lost rows
   - all_inquiries_win_rate_v3 has genuine worked quotes without unworked auto-quotes
6. Customer entity normalization complete (customer_id non-null, valid CUST_ prefix)
7. Point-in-time features present and valid
8. Hard build assertions fail loudly on synthetic violations
"""

import os
import unittest
import numpy as np
import pandas as pd

from src.preprocess_pipeline_v3 import validate_build_gates, run_pipeline_v3


CLEANED_V3_PATH = "data/processed/Air_Export_Pricing_Cleaned_V3.csv"
WON_DEALS_PATH = "data/processed/won_deals_margin_v3.csv"
ALL_INQUIRIES_PATH = "data/processed/all_inquiries_win_rate_v3.csv"


class TestDataPipelineV3Outputs(unittest.TestCase):
    """Verifies that the generated production datasets exist and meet structural gates."""

    def test_output_datasets_exist(self):
        """Pipeline must construct and save all three specified processed datasets."""
        self.assertTrue(
            os.path.exists(CLEANED_V3_PATH),
            f"Master Cleaned V3 dataset must exist at {CLEANED_V3_PATH}",
        )
        self.assertTrue(
            os.path.exists(WON_DEALS_PATH),
            f"Won deals margin dataset must exist at {WON_DEALS_PATH}",
        )
        self.assertTrue(
            os.path.exists(ALL_INQUIRIES_PATH),
            f"All inquiries win rate dataset must exist at {ALL_INQUIRIES_PATH}",
        )

    def test_date_consistency_and_d4_repair(self):
        """Date consistency: Quote_Month == quote_date month, and zero June/Sep mismatches."""
        for path in [CLEANED_V3_PATH, WON_DEALS_PATH, ALL_INQUIRIES_PATH]:
            df = pd.read_csv(path, low_memory=False)
            self.assertIn("quote_date", df.columns, f"quote_date column missing in {path}")
            parsed_dates = pd.to_datetime(df["quote_date"])

            # 1. Quote_Month strictly matches parsed quote_date month
            month_match = (df["Quote_Month"] == parsed_dates.dt.month).all()
            self.assertTrue(month_match, f"Quote_Month mismatch with quote_date in {path}")

            # 2. Year_Month strictly matches YYYY-MM
            ym_match = (df["Year_Month"] == parsed_dates.dt.strftime("%Y-%m")).all()
            self.assertTrue(ym_match, f"Year_Month mismatch with quote_date in {path}")

            # 3. Month_Name strictly matches month name
            mname_match = (df["Month_Name"] == parsed_dates.dt.month_name()).all()
            self.assertTrue(mname_match, f"Month_Name mismatch with quote_date in {path}")

            # 4. Explicit verification of Issue D4 repair:
            # No rows with Quote_Month == 9 should have Year_Month == '2026-06' or Month_Name == 'June'
            d4_mismatches = df[
                (df["Quote_Month"] == 9)
                & ((df["Year_Month"] == "2026-06") | (df["Month_Name"] == "June"))
            ]
            self.assertEqual(
                len(d4_mismatches),
                0,
                f"Found {len(d4_mismatches)} un-repaired June/September mismatches in {path}",
            )

    def test_margin_identity_holds_strictly(self):
        """Margin identity: abs(Margin_Percentage - (Sell - Buy)/Buy * 100) <= 0.05."""
        for path in [CLEANED_V3_PATH, WON_DEALS_PATH, ALL_INQUIRIES_PATH]:
            df = pd.read_csv(path, low_memory=False)
            calc_margin = (
                (df["Total_Sell_INR"] - df["Total_Buy_INR"])
                / df["Total_Buy_INR"]
                * 100.0
            )
            margin_diff = (df["Margin_Percentage"] - calc_margin).abs()
            violating_rows = (margin_diff > 0.05).sum()
            self.assertEqual(
                violating_rows,
                0,
                f"Found {violating_rows} rows violating margin identity in {path}",
            )

    def test_won_deals_margin_purity(self):
        """won_deals_margin_v3 has 100% is_won == 1 with zero default-margin Lost rows."""
        df = pd.read_csv(WON_DEALS_PATH)
        # Must be non-empty
        self.assertGreater(len(df), 5000, "Won deals dataset has too few rows")

        # 100% won
        self.assertTrue(
            (df["is_won"] == 1).all(),
            "won_deals_margin_v3 must contain ONLY is_won == 1 deals",
        )
        self.assertTrue(
            (df["Quote Status"] == "Won").all(),
            "won_deals_margin_v3 Quote Status must be strictly 'Won'",
        )

        # Economic sanity
        self.assertTrue(
            (df["Total_Buy_INR"] > 500.0).all(),
            "Total_Buy_INR must be > 500 for all won deals",
        )
        self.assertTrue(
            (df["Total_Sell_INR"] > df["Total_Buy_INR"]).all(),
            "Total_Sell_INR must be strictly greater than Total_Buy_INR",
        )

        # No default-margin lost rows (default 2% gross margin = 2.04% markup)
        # Note: if any row had is_won == 0, that was already asserted above, but check status labels explicitly
        self.assertNotIn("Lost", df["Quote Status"].unique())

    def test_no_estimated_buy_costs_in_margin_training(self):
        """No Buy_Cost_Estimated == 1 rows in won_deals_margin_v3.csv."""
        df = pd.read_csv(WON_DEALS_PATH)
        if "Buy_Cost_Estimated" in df.columns:
            leaked_estimates = (df["Buy_Cost_Estimated"] == 1).sum()
            self.assertEqual(
                leaked_estimates,
                0,
                f"Found {leaked_estimates} estimated buy cost rows in won_deals_margin_v3.csv",
            )

    def test_all_inquiries_win_rate_properties(self):
        """all_inquiries_win_rate_v3 contains genuine worked inquiries (excluding unworked auto-quotes)."""
        df = pd.read_csv(ALL_INQUIRIES_PATH)
        self.assertGreater(len(df), 5000, "Inquiries win rate dataset has too few rows")

        # Contains both outcomes
        unique_outcomes = set(df["is_won"].unique())
        self.assertIn(1, unique_outcomes)
        self.assertIn(0, unique_outcomes)

        # Unworked lost quotes (instant turnaround < 1h and default gross margin ~2%) must be excluded
        if "Quote_Turnaround_Hours" in df.columns:
            gm = (df["Total_Sell_INR"] - df["Total_Buy_INR"]) / df["Total_Sell_INR"]
            unworked_lost = (
                (df["is_won"] == 0)
                & (df["Quote_Turnaround_Hours"] < 1.0)
                & ((gm - 0.02).abs() < 2e-4)
            )
            self.assertEqual(
                unworked_lost.sum(),
                0,
                "Found unworked auto-quote Lost rows in all_inquiries_win_rate_v3",
            )

    def test_customer_normalization_complete(self):
        """All datasets must have a valid, non-null customer_id with CUST_ prefix."""
        for path in [CLEANED_V3_PATH, WON_DEALS_PATH, ALL_INQUIRIES_PATH]:
            df = pd.read_csv(path, low_memory=False)
            self.assertIn("customer_id", df.columns, f"customer_id missing in {path}")
            self.assertFalse(df["customer_id"].isnull().any(), f"Null customer_id in {path}")
            self.assertFalse((df["customer_id"] == "").any(), f"Empty customer_id in {path}")
            self.assertFalse(
                (df["customer_id"] == "CUST_UNKNOWN").any(),
                f"Unmapped CUST_UNKNOWN in {path}",
            )
            # Prefix check
            has_prefix = df["customer_id"].str.startswith("CUST_").all()
            self.assertTrue(has_prefix, f"customer_id lacking CUST_ prefix in {path}")

    def test_point_in_time_features_present_and_valid(self):
        """All 7 point-in-time features must be present in the output datasets."""
        pit_cols = [
            "cust_pit_won_count_180d",
            "cust_pit_inquiry_count_180d",
            "cust_pit_recency_days",
            "cust_pit_median_margin_won",
            "cust_pit_total_spend_inr_180d",
            "cust_pit_win_rate_180d",
            "lane_pit_median_margin_won_90d",
        ]
        for path in [CLEANED_V3_PATH, WON_DEALS_PATH, ALL_INQUIRIES_PATH]:
            df = pd.read_csv(path, low_memory=False)
            for col in pit_cols:
                self.assertIn(col, df.columns, f"PIT feature '{col}' missing in {path}")
            # Counts and spend must have zero nulls
            self.assertEqual(df["cust_pit_won_count_180d"].isnull().sum(), 0)
            self.assertEqual(df["cust_pit_inquiry_count_180d"].isnull().sum(), 0)
            self.assertEqual(df["cust_pit_total_spend_inr_180d"].isnull().sum(), 0)
            self.assertEqual(df["cust_pit_win_rate_180d"].isnull().sum(), 0)


class TestBuildGateAssertions(unittest.TestCase):
    """Tests that hard build validation assertions fail loudly when violated."""

    def setUp(self):
        # Create a valid baseline DataFrame
        self.valid_df = pd.DataFrame([
            {
                "Inquiry Number": "RFQ26010001",
                "quote_date": "2026-01-15",
                "Quote_Month": 1,
                "Year_Month": "2026-01",
                "Month_Name": "January",
                "Total_Buy_INR": 10000.0,
                "Total_Sell_INR": 11000.0,
                "Margin_Percentage": 10.0,
                "is_won": 1,
                "Quote Status": "Won",
                "customer_id": "CUST_TEST_ACCOUNT",
                "Buy_Cost_Estimated": 0,
            }
        ])

    def test_valid_df_passes_gates(self):
        """Valid data must pass build gates without error."""
        try:
            validate_build_gates(self.valid_df, is_margin_dataset=True)
        except Exception as e:
            self.fail(f"validate_build_gates raised unexpected exception: {e}")

    def test_gate_fails_on_date_mismatch(self):
        """Assertion 1: Pipeline must raise ValueError on date mismatches."""
        bad_df = self.valid_df.copy()
        # Create June vs September mismatch (Issue D4)
        bad_df.loc[0, "Quote_Month"] = 9
        bad_df.loc[0, "Year_Month"] = "2026-06"
        bad_df.loc[0, "Month_Name"] = "June"

        with self.assertRaises(ValueError) as ctx:
            validate_build_gates(bad_df, is_margin_dataset=True)
        self.assertIn("date mismatch", str(ctx.exception).lower())

    def test_gate_fails_on_margin_identity_violation(self):
        """Assertion 2: Pipeline must raise ValueError when margin identity violated."""
        bad_df = self.valid_df.copy()
        # Actual margin is (11000 - 10000) / 10000 * 100 = 10.0%, set to 15.0%
        bad_df.loc[0, "Margin_Percentage"] = 15.0

        with self.assertRaises(ValueError) as ctx:
            validate_build_gates(bad_df, is_margin_dataset=True)
        self.assertIn("margin identity", str(ctx.exception).lower())

    def test_gate_fails_on_estimated_buy_cost_leak(self):
        """Assertion 3: Pipeline must raise ValueError when Buy_Cost_Estimated == 1 leaks into margin data."""
        bad_df = self.valid_df.copy()
        bad_df.loc[0, "Buy_Cost_Estimated"] = 1

        with self.assertRaises(ValueError) as ctx:
            validate_build_gates(bad_df, is_margin_dataset=True)
        self.assertIn("estimated buy cost", str(ctx.exception).lower())

    def test_gate_fails_on_unnormalized_customer(self):
        """Assertion 4: Pipeline must raise ValueError when customer_id is null, empty, or unmapped."""
        # Test null customer_id
        bad_null = self.valid_df.copy()
        bad_null.loc[0, "customer_id"] = None
        with self.assertRaises(ValueError) as ctx:
            validate_build_gates(bad_null, is_margin_dataset=True)
        self.assertIn("customer", str(ctx.exception).lower())

        # Test CUST_UNKNOWN
        bad_unknown = self.valid_df.copy()
        bad_unknown.loc[0, "customer_id"] = "CUST_UNKNOWN"
        with self.assertRaises(ValueError) as ctx:
            validate_build_gates(bad_unknown, is_margin_dataset=True)
        self.assertIn("customer", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
