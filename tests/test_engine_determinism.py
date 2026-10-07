"""
tests/test_engine_determinism.py
Unit tests for deterministic quote dates and out-of-window alerting in MarginRecommendationEngine.
"""

import unittest
import joblib
import os
from src.engine import MarginRecommendationEngine


class TestEngineDeterminism(unittest.TestCase):
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

    def test_quote_date_determinism(self):
        inquiry = {
            "Total_Buy_INR": 75000.0,
            "Chargeable_Weight_Kg": 350.0,
            "Gross_Weight_Kg": 320.0,
            "Commodity_Group": "General Cargo",
            "Origin_Port": "Chhatrapati Shivaji Maharaj International Airport",
            "Destination_Port": "Frankfurt am Main",
            "Origin_Region": "West India",
            "Destination_Region": "Europe",
            "Regional_Lane": "West India -> Europe",
            "Weight_Tier": "300-500kg",
            "Customer_Tier": "Regular",
            "Customer_Inquiry_Frequency": 10,
            "Incoterms": "FOB",
            "Company_Group": "Direct Client",
            "quote_date": "2026-05-15"
        }

        rec1, _ = self.engine.optimize_quote(inquiry)
        rec2, _ = self.engine.optimize_quote(inquiry)

        self.assertEqual(rec1["Margin_Percentage"], rec2["Margin_Percentage"])
        self.assertEqual(rec1["Quoted_Sell_Price_INR"], rec2["Quoted_Sell_Price_INR"])
        self.assertEqual(rec1["Win_Probability"], rec2["Win_Probability"])

    def test_out_of_window_alerting_q4(self):
        # Quote in October (Q4), outside historical training window (Jan-Sep)
        inquiry_q4 = {
            "Total_Buy_INR": 60000.0,
            "Chargeable_Weight_Kg": 200.0,
            "Gross_Weight_Kg": 180.0,
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
            "Company_Group": "Direct Client",
            "quote_date": "2026-10-20"
        }

        rec, _ = self.engine.optimize_quote(inquiry_q4)
        self.assertTrue(rec.get("is_out_of_window", False))
        self.assertEqual(rec.get("confidence"), "Low")
        self.assertIn("out_of_window_warning", rec)
        self.assertIn("Q4", rec["out_of_window_warning"])

    def test_in_window_date_normal_confidence(self):
        inquiry_in_window = {
            "Total_Buy_INR": 60000.0,
            "Chargeable_Weight_Kg": 200.0,
            "Gross_Weight_Kg": 180.0,
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
            "Company_Group": "Direct Client",
            "quote_date": "2026-06-10"
        }

        rec, _ = self.engine.optimize_quote(inquiry_in_window)
        self.assertFalse(rec.get("is_out_of_window", False))
        self.assertEqual(rec.get("confidence"), "High")


if __name__ == "__main__":
    unittest.main()
