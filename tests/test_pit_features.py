"""
tests/test_pit_features.py
Unit tests for point-in-time rolling customer and lane feature engineering
with zero lookahead leakage and real-time registry consistency.
"""

import unittest
import numpy as np
import pandas as pd

from src.features_pit import compute_point_in_time_features, CustomerPITRegistry


class TestPITFeatures(unittest.TestCase):
    def setUp(self):
        # Base synthetic scenario
        self.sample_df = pd.DataFrame([
            {
                "quote_date": "2026-01-10",
                "Customer_Company": "CUST_A",
                "Regional_Lane": "West India -> Europe",
                "is_won": 1,
                "Margin_Percentage": 10.0,
                "Total_Sell_INR": 100000.0,
            },
            {
                "quote_date": "2026-02-15",
                "Customer_Company": "CUST_A",
                "Regional_Lane": "West India -> Europe",
                "is_won": 1,
                "Margin_Percentage": 8.0,
                "Total_Sell_INR": 150000.0,
            },
            {
                "quote_date": "2026-03-20",
                "Customer_Company": "CUST_A",
                "Regional_Lane": "West India -> Europe",
                "is_won": 0,
                "Margin_Percentage": 12.0,
                "Total_Sell_INR": 80000.0,
            },
            {
                "quote_date": "2026-04-25",
                "Customer_Company": "CUST_A",
                "Regional_Lane": "West India -> North America",
                "is_won": 1,
                "Margin_Percentage": 14.0,
                "Total_Sell_INR": 200000.0,
            },
        ])

    def test_zero_self_leakage(self):
        """Row N must never have access to its own outcome, margin, or spend."""
        res = compute_point_in_time_features(self.sample_df)

        # Row 0 (first quote for CUST_A): must have 0 won, 0 inquiries, 999 recency, NaN median margin, 0 spend
        self.assertEqual(res.loc[0, "cust_pit_won_count_180d"], 0)
        self.assertEqual(res.loc[0, "cust_pit_inquiry_count_180d"], 0)
        self.assertEqual(res.loc[0, "cust_pit_recency_days"], 999.0)
        self.assertTrue(np.isnan(res.loc[0, "cust_pit_median_margin_won"]))
        self.assertEqual(res.loc[0, "cust_pit_total_spend_inr_180d"], 0.0)
        self.assertEqual(res.loc[0, "cust_pit_win_rate_180d"], 0.50)
        self.assertTrue(np.isnan(res.loc[0, "lane_pit_median_margin_won_90d"]))

    def test_zero_lookahead_leakage(self):
        """Row N must never have access to any future row N+k transactions."""
        res = compute_point_in_time_features(self.sample_df)

        # Row 1 (2026-02-15) must ONLY see Row 0 (2026-01-10)
        self.assertEqual(res.loc[1, "cust_pit_won_count_180d"], 1)
        self.assertEqual(res.loc[1, "cust_pit_inquiry_count_180d"], 1)
        # Recency: 2026-02-15 - 2026-01-10 = 36 days
        self.assertEqual(res.loc[1, "cust_pit_recency_days"], 36.0)
        self.assertAlmostEqual(res.loc[1, "cust_pit_median_margin_won"], 10.0)
        self.assertAlmostEqual(res.loc[1, "cust_pit_total_spend_inr_180d"], 100000.0)
        self.assertAlmostEqual(res.loc[1, "cust_pit_win_rate_180d"], 1.0)
        # Lane: Row 0 was West India -> Europe with 10%
        self.assertAlmostEqual(res.loc[1, "lane_pit_median_margin_won_90d"], 10.0)

        # Row 2 (2026-03-20) must see Row 0 and Row 1, but NOT Row 2 itself (lost) or Row 3 (future)
        self.assertEqual(res.loc[2, "cust_pit_won_count_180d"], 2)
        self.assertEqual(res.loc[2, "cust_pit_inquiry_count_180d"], 2)
        # Recency: 2026-03-20 - 2026-02-15 = 33 days
        self.assertEqual(res.loc[2, "cust_pit_recency_days"], 33.0)
        # Median margin of Row 0 (10.0) and Row 1 (8.0) is 9.0
        self.assertAlmostEqual(res.loc[2, "cust_pit_median_margin_won"], 9.0)
        self.assertAlmostEqual(res.loc[2, "cust_pit_total_spend_inr_180d"], 250000.0)
        self.assertAlmostEqual(res.loc[2, "cust_pit_win_rate_180d"], 1.0)
        self.assertAlmostEqual(res.loc[2, "lane_pit_median_margin_won_90d"], 9.0)

    def test_duplicate_timestamps_determinism(self):
        """Duplicate timestamps on the same day must yield identical features regardless of order."""
        df_dups = pd.DataFrame([
            {
                "quote_date": "2026-01-10",
                "Customer_Company": "CUST_B",
                "Regional_Lane": "North India -> Europe",
                "is_won": 1,
                "Margin_Percentage": 10.0,
                "Total_Sell_INR": 100000.0,
            },
            {
                "quote_date": "2026-03-15",
                "Customer_Company": "CUST_B",
                "Regional_Lane": "North India -> Europe",
                "is_won": 1,
                "Margin_Percentage": 12.0,
                "Total_Sell_INR": 200000.0,
            },
            {
                "quote_date": "2026-03-15",
                "Customer_Company": "CUST_B",
                "Regional_Lane": "North India -> Europe",
                "is_won": 0,
                "Margin_Percentage": 5.0,
                "Total_Sell_INR": 50000.0,
            },
        ])

        # Forward order
        res1 = compute_point_in_time_features(df_dups)
        # Both quotes on 2026-03-15 must see ONLY the 2026-01-10 transaction
        self.assertEqual(res1.loc[1, "cust_pit_won_count_180d"], 1)
        self.assertEqual(res1.loc[2, "cust_pit_won_count_180d"], 1)
        self.assertEqual(res1.loc[1, "cust_pit_inquiry_count_180d"], 1)
        self.assertEqual(res1.loc[2, "cust_pit_inquiry_count_180d"], 1)
        self.assertEqual(res1.loc[1, "cust_pit_recency_days"], res1.loc[2, "cust_pit_recency_days"])
        self.assertEqual(res1.loc[1, "cust_pit_median_margin_won"], res1.loc[2, "cust_pit_median_margin_won"])
        self.assertEqual(res1.loc[1, "cust_pit_total_spend_inr_180d"], res1.loc[2, "cust_pit_total_spend_inr_180d"])
        self.assertEqual(res1.loc[1, "lane_pit_median_margin_won_90d"], res1.loc[2, "lane_pit_median_margin_won_90d"])

        # Reversed order for duplicate rows
        df_shuffled = df_dups.iloc[[0, 2, 1]].copy()
        res2 = compute_point_in_time_features(df_shuffled)
        self.assertEqual(res2.loc[1, "cust_pit_won_count_180d"], 1)
        self.assertEqual(res2.loc[2, "cust_pit_won_count_180d"], 1)
        self.assertEqual(res2.loc[1, "cust_pit_inquiry_count_180d"], 1)
        self.assertEqual(res2.loc[2, "cust_pit_inquiry_count_180d"], 1)

    def test_rolling_180d_windows(self):
        """Transactions older than 180 days must roll out of 180d window features."""
        df_long = pd.DataFrame([
            {
                "quote_date": "2025-01-01",
                "Customer_Company": "CUST_C",
                "Regional_Lane": "West India -> ME",
                "is_won": 1,
                "Margin_Percentage": 10.0,
                "Total_Sell_INR": 100000.0,
            },
            {
                "quote_date": "2025-08-01",  # 212 days later (older than 180d)
                "Customer_Company": "CUST_C",
                "Regional_Lane": "West India -> ME",
                "is_won": 1,
                "Margin_Percentage": 12.0,
                "Total_Sell_INR": 200000.0,
            },
        ])
        res = compute_point_in_time_features(df_long)
        # At 2025-08-01, 2025-01-01 is 212 days old (> 180d)
        self.assertEqual(res.loc[1, "cust_pit_won_count_180d"], 0)
        self.assertEqual(res.loc[1, "cust_pit_inquiry_count_180d"], 0)
        self.assertEqual(res.loc[1, "cust_pit_total_spend_inr_180d"], 0.0)
        self.assertEqual(res.loc[1, "cust_pit_win_rate_180d"], 0.50)
        # But recency and lifetime median margin still capture the 2025-01-01 won deal
        self.assertEqual(res.loc[1, "cust_pit_recency_days"], 212.0)
        self.assertEqual(res.loc[1, "cust_pit_median_margin_won"], 10.0)

    def test_lane_median_margin_won_90d(self):
        """Lane won median margin must strictly consider trailing 90 days on that lane."""
        df_lane = pd.DataFrame([
            {
                "quote_date": "2026-01-01",
                "Customer_Company": "CUST_X",
                "Regional_Lane": "South India -> Europe",
                "is_won": 1,
                "Margin_Percentage": 8.0,
                "Total_Sell_INR": 50000.0,
            },
            {
                "quote_date": "2026-02-01",
                "Customer_Company": "CUST_Y",
                "Regional_Lane": "South India -> Europe",
                "is_won": 1,
                "Margin_Percentage": 12.0,
                "Total_Sell_INR": 60000.0,
            },
            {
                "quote_date": "2026-02-15",
                "Customer_Company": "CUST_Z",
                "Regional_Lane": "South India -> APAC",  # different lane
                "is_won": 1,
                "Margin_Percentage": 20.0,
                "Total_Sell_INR": 70000.0,
            },
            {
                "quote_date": "2026-03-01",
                "Customer_Company": "CUST_W",
                "Regional_Lane": "South India -> Europe",
                "is_won": 1,
                "Margin_Percentage": 10.0,
                "Total_Sell_INR": 80000.0,
            },
            {
                "quote_date": "2026-04-20",  # 109 days after 2026-01-01 (> 90d), 78d after 2026-02-01
                "Customer_Company": "CUST_V",
                "Regional_Lane": "South India -> Europe",
                "is_won": 0,
                "Margin_Percentage": 9.0,
                "Total_Sell_INR": 90000.0,
            },
        ])
        res = compute_point_in_time_features(df_lane)

        # At 2026-03-01 (row 3): trailing 90d on South India -> Europe has row 0 (8.0) and row 1 (12.0) -> median 10.0
        self.assertAlmostEqual(res.loc[3, "lane_pit_median_margin_won_90d"], 10.0)

        # At 2026-04-20 (row 4): row 0 (2026-01-01) is 109d ago (> 90d, rolled out).
        # Trailing 90d has row 1 (2026-02-01, 12.0) and row 3 (2026-03-01, 10.0). Median = 11.0.
        self.assertAlmostEqual(res.loc[4, "lane_pit_median_margin_won_90d"], 11.0)

    def test_customer_pit_registry_parity(self):
        """CustomerPITRegistry streaming updates must yield identical features to batch processing."""
        registry = CustomerPITRegistry()

        # Stream row by row
        streaming_records = []
        for _, row in self.sample_df.iterrows():
            feats = registry.get_features(
                quote_date=row["quote_date"],
                customer_id=row["Customer_Company"],
                regional_lane=row["Regional_Lane"],
            )
            streaming_records.append(feats)
            registry.update(
                quote_date=row["quote_date"],
                customer_id=row["Customer_Company"],
                is_won=row["is_won"],
                margin=row["Margin_Percentage"],
                sell_inr=row["Total_Sell_INR"],
                regional_lane=row["Regional_Lane"],
            )

        stream_df = pd.DataFrame(streaming_records)
        batch_df = compute_point_in_time_features(self.sample_df)

        feature_cols = [
            "cust_pit_won_count_180d",
            "cust_pit_inquiry_count_180d",
            "cust_pit_recency_days",
            "cust_pit_median_margin_won",
            "cust_pit_total_spend_inr_180d",
            "cust_pit_win_rate_180d",
            "lane_pit_median_margin_won_90d",
        ]

        for col in feature_cols:
            np.testing.assert_array_almost_equal(
                stream_df[col].values,
                batch_df[col].values,
                err_msg=f"Mismatch in feature '{col}' between streaming registry and batch function.",
            )

    def test_customer_pit_registry_from_dataframe(self):
        """CustomerPITRegistry.from_dataframe correctly warms state."""
        registry = CustomerPITRegistry.from_dataframe(self.sample_df)

        # Query a new quote after all records
        feats = registry.get_features(
            quote_date="2026-05-01",
            customer_id="CUST_A",
            regional_lane="West India -> Europe",
        )
        self.assertEqual(feats["cust_pit_won_count_180d"], 3)
        self.assertEqual(feats["cust_pit_inquiry_count_180d"], 4)
        # Recency: 2026-05-01 - 2026-04-25 = 6 days
        self.assertEqual(feats["cust_pit_recency_days"], 6.0)
        # Median won margin for CUST_A: [10.0, 8.0, 14.0] -> 10.0
        self.assertAlmostEqual(feats["cust_pit_median_margin_won"], 10.0)
        # Total spend: 100000 + 150000 + 200000 = 450000
        self.assertAlmostEqual(feats["cust_pit_total_spend_inr_180d"], 450000.0)
        # Win rate: 3 / 4 = 0.75
        self.assertAlmostEqual(feats["cust_pit_win_rate_180d"], 0.75)
        # Lane West India -> Europe trailing 90d won margins:
        # 2026-02-15 (75d prior, 8.0). (2026-01-10 is 111d prior, > 90d).
        self.assertAlmostEqual(feats["lane_pit_median_margin_won_90d"], 8.0)

    def test_column_name_flexibility(self):
        """compute_point_in_time_features supports custom column names."""
        alt_df = pd.DataFrame([
            {
                "timestamp": "2026-01-15",
                "account_id": "CLIENT_1",
                "lane_name": "DEL -> LHR",
                "won_flag": 1,
                "margin": 15.0,
                "revenue": 50000.0,
            },
            {
                "timestamp": "2026-02-15",
                "account_id": "CLIENT_1",
                "lane_name": "DEL -> LHR",
                "won_flag": 1,
                "margin": 11.0,
                "revenue": 75000.0,
            },
        ])
        res = compute_point_in_time_features(
            alt_df,
            date_col="timestamp",
            cust_col="account_id",
            lane_col="lane_name",
        )
        self.assertEqual(res.loc[0, "cust_pit_won_count_180d"], 0)
        self.assertEqual(res.loc[1, "cust_pit_won_count_180d"], 1)
        self.assertAlmostEqual(res.loc[1, "cust_pit_median_margin_won"], 15.0)
        self.assertAlmostEqual(res.loc[1, "cust_pit_total_spend_inr_180d"], 50000.0)


    def test_empty_dataframe(self):
        """Empty DataFrame returns empty with all 7 columns defined."""
        empty_df = pd.DataFrame(columns=["quote_date", "Customer_Company", "Regional_Lane"])
        res = compute_point_in_time_features(empty_df)
        self.assertTrue(res.empty)
        expected_cols = [
            "cust_pit_won_count_180d",
            "cust_pit_inquiry_count_180d",
            "cust_pit_recency_days",
            "cust_pit_median_margin_won",
            "cust_pit_total_spend_inr_180d",
            "cust_pit_win_rate_180d",
            "lane_pit_median_margin_won_90d",
        ]
        for col in expected_cols:
            self.assertIn(col, res.columns)

    def test_customer_with_only_lost_quotes(self):
        """Customer with only lost quotes has 0 won count, 0 spend, 999 recency, NaN median margin, and 0 win rate."""
        df_lost = pd.DataFrame([
            {
                "quote_date": "2026-01-10",
                "Customer_Company": "CUST_LOST",
                "Regional_Lane": "West India -> Europe",
                "is_won": 0,
                "Margin_Percentage": 2.0,
                "Total_Sell_INR": 50000.0,
            },
            {
                "quote_date": "2026-02-10",
                "Customer_Company": "CUST_LOST",
                "Regional_Lane": "West India -> Europe",
                "is_won": 0,
                "Margin_Percentage": 2.0,
                "Total_Sell_INR": 60000.0,
            },
        ])
        res = compute_point_in_time_features(df_lost)
        # Row 1 sees Row 0 (which was lost)
        self.assertEqual(res.loc[1, "cust_pit_won_count_180d"], 0)
        self.assertEqual(res.loc[1, "cust_pit_inquiry_count_180d"], 1)
        self.assertEqual(res.loc[1, "cust_pit_recency_days"], 999.0)
        self.assertTrue(np.isnan(res.loc[1, "cust_pit_median_margin_won"]))
        self.assertEqual(res.loc[1, "cust_pit_total_spend_inr_180d"], 0.0)
        self.assertEqual(res.loc[1, "cust_pit_win_rate_180d"], 0.0)

    def test_shuffled_input_preserves_original_index_alignment(self):
        """compute_point_in_time_features preserves original index alignment even when input is not sorted."""
        # Unsorted order: Row 2, Row 0, Row 3, Row 1
        shuffled = self.sample_df.iloc[[2, 0, 3, 1]].copy()
        res = compute_point_in_time_features(shuffled)

        # In original sample_df:
        # Row 0 (2026-01-10): won 0
        # Row 1 (2026-02-15): won 1
        # Row 2 (2026-03-20): won 2
        # Row 3 (2026-04-25): won 2 (Row 2 was lost)
        self.assertEqual(res.loc[0, "cust_pit_won_count_180d"], 0)
        self.assertEqual(res.loc[1, "cust_pit_won_count_180d"], 1)
        self.assertEqual(res.loc[2, "cust_pit_won_count_180d"], 2)
        self.assertEqual(res.loc[3, "cust_pit_won_count_180d"], 2)

    def test_smoke_on_real_data(self):
        """Smoke test on actual benchmark dataset ensuring mathematical consistency and bounded metrics."""
        import os
        path = "tests/data/benchmark_won_out_of_time.csv"
        if os.path.exists(path):
            df_bench = pd.read_csv(path)
            # Use Year_Month since benchmark records don't have explicit quote_date
            res = compute_point_in_time_features(df_bench, date_col="Year_Month")
            self.assertEqual(len(res), len(df_bench))
            self.assertTrue((res["cust_pit_won_count_180d"] >= 0).all())
            self.assertTrue((res["cust_pit_inquiry_count_180d"] >= 0).all())
            self.assertTrue((res["cust_pit_recency_days"] >= 0).all())
            self.assertTrue((res["cust_pit_total_spend_inr_180d"] >= 0).all())
            self.assertTrue((res["cust_pit_win_rate_180d"] >= 0.0).all())
            self.assertTrue((res["cust_pit_win_rate_180d"] <= 1.0).all())


if __name__ == "__main__":
    unittest.main()

