"""
tests/test_cohort_lookup.py
Unit tests for cohort margin quantile lookups and hard cost floor enforcement in MarginRecommendationEngine.
"""

import unittest
import os
import joblib
from src.engine import MarginRecommendationEngine
from src.constants import (
    MIN_MARGIN_PER_AWB_INR,
    MIN_MARGIN_PER_KG_INR,
    OPS_HANDLING_FEE_AWB_INR,
    compute_cost_floor_price
)


class TestCohortLookupAndCostFloor(unittest.TestCase):
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

    def test_cohort_quantiles_lookup(self):
        q = self.engine.get_cohort_quantiles(
            regional_lane="West India -> Europe",
            weight_tier="300-500kg",
            commodity_group="General Cargo"
        )
        self.assertIn("p25", q)
        self.assertIn("p50", q)
        self.assertIn("p75", q)
        self.assertIn("p90", q)
        self.assertLessEqual(q["p25"], q["p50"])
        self.assertLessEqual(q["p50"], q["p75"])
        self.assertLessEqual(q["p75"], q["p90"])

    def test_cost_floor_enforcement_on_small_shipment(self):
        # 50 kg shipment with small buy cost of Rs 4,000
        # Absolute floors:
        # Min AWB: Rs 1500
        # Ops fee: Rs 850
        # Min per kg: 50 * 3 = Rs 150
        # Total minimum margin rupee: 1500 + 850 + 150 = Rs 2500
        # Cost floor price: Rs 4000 + 2500 = Rs 6500 (62.5% markup!)
        inquiry_small = {
            "Total_Buy_INR": 4000.0,
            "Chargeable_Weight_Kg": 50.0,
            "Gross_Weight_Kg": 45.0,
            "Commodity_Group": "General Cargo",
            "Origin_Port": "Chhatrapati Shivaji Maharaj International Airport",
            "Destination_Port": "Dubai",
            "Origin_Region": "West India",
            "Destination_Region": "Middle East",
            "Regional_Lane": "West India -> Middle East",
            "Weight_Tier": "45-100kg",
            "Customer_Tier": "Spot / One-Off",
            "Customer_Inquiry_Frequency": 1,
            "Incoterms": "FOB",
            "Company_Group": "Subagent",
            "quote_date": "2026-05-10"
        }

        min_floor_price = compute_cost_floor_price(4000.0, 50.0, "General Cargo")
        rec, results_df = self.engine.optimize_quote(inquiry_small)

        # All 3 strategy cards must satisfy selling_price >= min_floor_price
        for key in ["floor", "balanced", "premium"]:
            card = rec["strategic_recommendations"][key]
            self.assertGreaterEqual(
                card["Quoted_Sell_Price_INR"],
                min_floor_price - 0.01,
                f"Strategy '{key}' quoted price {card['Quoted_Sell_Price_INR']} violated cost floor {min_floor_price}"
            )


if __name__ == "__main__":
    unittest.main()
