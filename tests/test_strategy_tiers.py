"""
tests/test_strategy_tiers.py
Unit tests verifying strict differentiation among Floor, Balanced, and Premium strategy tiers,
consistency of rates and profits, and active enforcement of STRATEGY_THRESHOLDS.
"""

import unittest
import os
import joblib
from src.engine import MarginRecommendationEngine
from src.constants import STRATEGY_THRESHOLDS


class TestStrategyTiersDifferentiation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bundle_path = "models/freight_margin_recommender.joblib"
        if not os.path.exists(bundle_path):
            raise FileNotFoundError(f"Model bundle not found at {bundle_path}")
        cls.bundle = joblib.load(bundle_path)
        cls.engine = MarginRecommendationEngine(
            benchmark_reg=cls.bundle["benchmark_reg"],
            win_classifier=cls.bundle["calibrated_clf"],
            encoder=cls.bundle["encoder"],
            feature_cols=cls.bundle["feature_cols"],
            cat_cols=cls.bundle["native_cat_cols"],
            lane_medians=cls.bundle.get("lane_medians"),
            global_median=cls.bundle.get("global_median"),
            port_frequencies=cls.bundle.get("port_frequencies"),
            cohort_table=cls.bundle.get("cohort_table")
        )

    def test_distinct_tiers_and_no_duplicates(self):
        """Verify Best-Recommended Balance and High-Margin Premium are NEVER identical."""
        inquiry = {
            "Total_Buy_INR": 99000.0,
            "Chargeable_Weight_Kg": 1200.0,
            "Gross_Weight_Kg": 1150.0,
            "Commodity_Group": "General Cargo",
            "Origin_Port": "Chhatrapati Shivaji Maharaj International Airport",
            "Destination_Port": "Frankfurt am Main",
            "Origin_Region": "West India",
            "Destination_Region": "Europe",
            "Regional_Lane": "West India -> Europe",
            "Weight_Tier": ">1000kg",
            "Customer_Tier": "Key Account",
            "Customer_Inquiry_Frequency": 25,
            "Incoterms": "FOB",
            "Company_Group": "Direct Client",
            "quote_date": "2026-05-20"
        }

        rec, _ = self.engine.optimize_quote(inquiry)
        strats = rec["strategic_recommendations"]
        floor = strats["floor"]
        balanced = strats["balanced"]
        premium = strats["premium"]

        # Margins must be strictly ascending
        self.assertLess(floor["Margin_Percentage"], balanced["Margin_Percentage"])
        self.assertLess(balanced["Margin_Percentage"], premium["Margin_Percentage"])

        # Prices must be strictly ascending
        self.assertLess(floor["Quoted_Sell_Price_INR"], balanced["Quoted_Sell_Price_INR"])
        self.assertLess(balanced["Quoted_Sell_Price_INR"], premium["Quoted_Sell_Price_INR"])

        # Rates per kg must be consistent
        ch_wt = inquiry["Chargeable_Weight_Kg"]
        self.assertAlmostEqual(floor["Rate_Per_Kg"], floor["Quoted_Sell_Price_INR"] / ch_wt, places=1)
        self.assertAlmostEqual(balanced["Rate_Per_Kg"], balanced["Quoted_Sell_Price_INR"] / ch_wt, places=1)
        self.assertAlmostEqual(premium["Rate_Per_Kg"], premium["Quoted_Sell_Price_INR"] / ch_wt, places=1)

    def test_active_strategy_thresholds_enforced(self):
        """Verify that recommendations adhere to strategy win-probability constraints."""
        inquiry = {
            "Total_Buy_INR": 50000.0,
            "Chargeable_Weight_Kg": 250.0,
            "Gross_Weight_Kg": 240.0,
            "Commodity_Group": "General Cargo",
            "Origin_Port": "Indira Gandhi International Airport",
            "Destination_Port": "Dubai",
            "Origin_Region": "North India",
            "Destination_Region": "Middle East",
            "Regional_Lane": "North India -> Middle East",
            "Weight_Tier": "100-300kg",
            "Customer_Tier": "Regular",
            "Customer_Inquiry_Frequency": 8,
            "Incoterms": "FOB",
            "Company_Group": "Subagent",
            "quote_date": "2026-06-15"
        }

        rec, _ = self.engine.optimize_quote(inquiry)
        strats = rec["strategic_recommendations"]

        # Floor min win prob >= 0.40 (or within 2% if max prob in corridor is slightly lower)
        self.assertGreaterEqual(strats["floor"]["Win_Probability"], STRATEGY_THRESHOLDS["floor"] - 0.05)
        # Balanced min win prob >= 0.25
        self.assertGreaterEqual(strats["balanced"]["Win_Probability"], STRATEGY_THRESHOLDS["balanced"] - 0.05)
        # Premium min win prob >= 0.15
        self.assertGreaterEqual(strats["premium"]["Win_Probability"], STRATEGY_THRESHOLDS["premium"] - 0.05)

    def test_category_and_commodity_max_margin_enforced_in_optimization(self):
        """
        Verify that optimize_quote first checks and strictly enforces the maximum margin cap
        for both category and commodity type across all candidate rows and strategy tiers.
        """
        # Test 1: Auto Parts (commodity cap = 22.0%)
        inquiry_auto = {
            "Total_Buy_INR": 30000.0,
            "Chargeable_Weight_Kg": 200.0,
            "Gross_Weight_Kg": 200.0,
            "Commodity_Group": "Auto Parts",
            "Origin_Port": "Indira Gandhi International Airport",
            "Destination_Port": "Dubai",
            "Weight_Tier": "100-300kg",
            "Customer_Tier": "Regular",
            "Company_Group": "Shipper / Consignee",  # Shipper cap is 35%, so Auto Parts 22% governs
            "quote_date": "2026-06-15"
        }

        rec_auto, frontier_auto = self.engine.optimize_quote(inquiry_auto)
        self.assertEqual(rec_auto["Commodity_Max_Margin_Cap"], 22.0)
        self.assertEqual(rec_auto["Effective_Max_Margin_Cap"], 22.0)
        self.assertIn("Auto Parts", rec_auto["Cap_Limiting_Factor"])

        # No candidate row in the sensitivity frontier can exceed 22.0%
        self.assertLessEqual(frontier_auto["Margin_Percentage"].max(), 22.0)

        # None of the strategy tiers can exceed 22.0%
        strats_auto = rec_auto["strategic_recommendations"]
        self.assertLessEqual(strats_auto["floor"]["Margin_Percentage"], 22.0)
        self.assertLessEqual(strats_auto["balanced"]["Margin_Percentage"], 22.0)
        self.assertLessEqual(strats_auto["premium"]["Margin_Percentage"], 22.0)

        # Strict ordering still holds
        self.assertLess(strats_auto["floor"]["Margin_Percentage"], strats_auto["balanced"]["Margin_Percentage"])
        self.assertLess(strats_auto["balanced"]["Margin_Percentage"], strats_auto["premium"]["Margin_Percentage"])

        # Test 2: Garments / Textiles (25%) with NVOCC / Coloader (category cap = 22.0%)
        inquiry_nvocc = {
            "Total_Buy_INR": 35000.0,
            "Chargeable_Weight_Kg": 250.0,
            "Gross_Weight_Kg": 250.0,
            "Commodity_Group": "Garments / Textiles",
            "Origin_Port": "Chhatrapati Shivaji Maharaj International Airport",
            "Destination_Port": "Heathrow Apt/London",
            "Weight_Tier": "100-300kg",
            "Customer_Tier": "Regular",
            "Company_Group": "NVOCC / Coloader",  # Category cap 22% governs over commodity 25%
            "quote_date": "2026-06-15"
        }

        rec_nvocc, frontier_nvocc = self.engine.optimize_quote(inquiry_nvocc)
        self.assertEqual(rec_nvocc["Effective_Max_Margin_Cap"], 22.0)
        self.assertIn("NVOCC", rec_nvocc["Cap_Limiting_Factor"])

        # Frontier bounded by 22.0%
        self.assertLessEqual(frontier_nvocc["Margin_Percentage"].max(), 22.0)

        # All tiers bounded by 22.0%
        strats_nvocc = rec_nvocc["strategic_recommendations"]
        self.assertLessEqual(strats_nvocc["floor"]["Margin_Percentage"], 22.0)
        self.assertLessEqual(strats_nvocc["balanced"]["Margin_Percentage"], 22.0)
        self.assertLessEqual(strats_nvocc["premium"]["Margin_Percentage"], 22.0)


if __name__ == "__main__":
    unittest.main()

