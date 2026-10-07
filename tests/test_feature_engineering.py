"""
tests/test_feature_engineering.py
Unit tests for airline rate-per-kg, lane rate index, and non-leaking feature transformations.
"""

import unittest
import numpy as np
import pandas as pd
from src.train import (
    compute_rate_features,
    FEATURE_COLS,
    NATIVE_CAT_COLS,
)


class TestFeatureEngineering(unittest.TestCase):
    def test_feature_list_sanitization(self):
        """Verify leaking or gameable features are excluded from production features."""
        self.assertNotIn("quote_revision_count", FEATURE_COLS)
        self.assertNotIn("Booked by branch", FEATURE_COLS)
        self.assertNotIn("Total_Buy_INR", FEATURE_COLS)  # replaced by rate/kg and log wt
        self.assertNotIn("Total_Sell_INR", FEATURE_COLS)
        self.assertNotIn("status_rank", FEATURE_COLS)

        self.assertIn("Buy_Rate_Per_Kg", FEATURE_COLS)
        self.assertIn("Lane_Rate_Index", FEATURE_COLS)
        self.assertIn("Log_Chargeable_Weight", FEATURE_COLS)
        self.assertIn("Regional_Lane", FEATURE_COLS)
        self.assertIn("Commodity_Group", FEATURE_COLS)

    def test_compute_rate_features_calculation(self):
        sample_df = pd.DataFrame([
            {
                "Total_Buy_INR": 50000.0,
                "Chargeable_Weight_Kg": 250.0,
                "Regional_Lane": "West India -> Europe",
                "Destination_Port": "Frankfurt am Main"
            },
            {
                "Total_Buy_INR": 15000.0,
                "Chargeable_Weight_Kg": 100.0,
                "Regional_Lane": "North India -> Middle East",
                "Destination_Port": "Dubai"
            }
        ])

        lane_medians = {
            "West India -> Europe": 200.0,
            "North India -> Middle East": 150.0
        }
        global_median = 180.0
        port_freq = {
            "Frankfurt am Main": 500,
            "Dubai": 1200
        }

        transformed_df = compute_rate_features(
            sample_df,
            lane_medians=lane_medians,
            global_median=global_median,
            port_frequencies=port_freq
        )

        # 50,000 / 250 = 200 Rs/kg -> Lane_Rate_Index = 200 / 200 = 1.0
        self.assertAlmostEqual(transformed_df.loc[0, "Buy_Rate_Per_Kg"], 200.0)
        self.assertAlmostEqual(transformed_df.loc[0, "Lane_Rate_Index"], 1.0)
        self.assertAlmostEqual(transformed_df.loc[0, "Log_Chargeable_Weight"], np.log1p(250.0))
        self.assertEqual(transformed_df.loc[0, "Dest_Port_Freq"], 500.0)

        # 15,000 / 100 = 150 Rs/kg -> Lane_Rate_Index = 150 / 150 = 1.0
        self.assertAlmostEqual(transformed_df.loc[1, "Buy_Rate_Per_Kg"], 150.0)
        self.assertAlmostEqual(transformed_df.loc[1, "Lane_Rate_Index"], 1.0)
        self.assertAlmostEqual(transformed_df.loc[1, "Log_Chargeable_Weight"], np.log1p(100.0))
        self.assertEqual(transformed_df.loc[1, "Dest_Port_Freq"], 1200.0)

    def test_native_categorical_cardinality_constraint(self):
        """HistGradientBoosting requires native categorical features to have <= 255 categories."""
        # Destination_Port has > 300 categories, so it must be frequency encoded rather than in NATIVE_CAT_COLS
        self.assertNotIn("Destination_Port", NATIVE_CAT_COLS)
        self.assertIn("Dest_Port_Freq", FEATURE_COLS)
        self.assertIn("Regional_Lane", NATIVE_CAT_COLS)
        self.assertIn("Commodity_Group", NATIVE_CAT_COLS)


if __name__ == "__main__":
    unittest.main()
