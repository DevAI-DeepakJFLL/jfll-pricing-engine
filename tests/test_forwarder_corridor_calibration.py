import unittest
from src.engine_v3 import PricingEngineV3


class TestForwarderCorridorCalibration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = PricingEngineV3()

    def test_corridor_is_tight_operational_band_not_wide_floor(self):
        """Negotiation corridor must be a tight operational band around Target,

        not an unworkable 10%+ chasm spanning down to the emergency floor.
        """
        inq = {
            "origin": "BOM",
            "destination": "DXB",
            "airline": "EK",
            "chargeable_weight": 150.0,
            "buy": 150.0 * 65.0,
            "unit_buy": 65.0,
            "density_ratio": 1.0,
            "customer_type": "Freight Forwarder",
            "commodity": "General Cargo",
            "payment_term": "Prepaid",
            "quote_revision_count": 1,
        }
        res = self.engine.recommend_quote(inq)
        corr = res["negotiation_corridor"]
        strat = res["strategic_recommendations"]

        target_m = corr["target_margin_pct"]
        min_m = corr["min_margin_pct"]
        band_width = corr["band_width_pct"]

        # 1. Bandwidth must be realistic for air freight operations (<= 4.5 pp)
        self.assertLessEqual(
            band_width,
            4.5,
            f"Corridor bandwidth {band_width}pp is too wide for air freight forwarding!",
        )

        # 2. Corridor must tightly bracket target (not skewed 9% downwards)
        downward_cushion = target_m - min_m
        self.assertLessEqual(
            downward_cushion,
            3.0,
            f"Downward cushion {downward_cushion}pp is too wide; corridor should be an opening band!",
        )

        # 3. Floor strategy remains available as the emergency walk-away limit
        floor_m = strat["floor"]["Margin_Percentage"]
        self.assertLessEqual(
            floor_m,
            min_m + 0.05,
            "Floor strategy should be at or below corridor min as walk-away limit",
        )

    def test_premium_strategy_is_commercially_capped_for_forwarding(self):
        """Premium strategy on general cargo / forwarders must be realistic,

        not inflated to 24%+ where no freight deal ever clears.
        """
        inq = {
            "origin": "BOM",
            "destination": "FRA",
            "airline": "LH",
            "chargeable_weight": 650.0,
            "buy": 650.0 * 180.0,
            "unit_buy": 180.0,
            "density_ratio": 1.0,
            "customer_type": "Subagent",
            "commodity": "General Cargo",
            "payment_term": "Prepaid",
            "quote_revision_count": 1,
        }
        res = self.engine.recommend_quote(inq)
        strat = res["strategic_recommendations"]

        bal_m = strat["balanced"]["Margin_Percentage"]
        prem_m = strat["premium"]["Margin_Percentage"]

        # On subagent heavy cargo, premium spread should be restrained (<= 4.5 pp above balanced)
        spread = prem_m - bal_m
        self.assertLessEqual(
            spread,
            4.5,
            f"Premium spread (+{spread:.2f}pp) is unrealistic for subagent air freight!",
        )

    def test_per_kg_metrics_present_in_corridor_and_strategies(self):
        """Forwarders calculate in ₹/kg. Verify per-kg rates and markups are present."""
        inq = {
            "origin": "DEL",
            "destination": "LHR",
            "airline": "AI",
            "chargeable_weight": 320.0,
            "buy": 320.0 * 150.0,
            "unit_buy": 150.0,
            "density_ratio": 1.0,
            "customer_type": "Shipper / Consignee",
            "commodity": "Pharmaceuticals",
            "payment_term": "Prepaid",
            "quote_revision_count": 1,
        }
        res = self.engine.recommend_quote(inq)
        corr = res["negotiation_corridor"]
        strat = res["strategic_recommendations"]

        self.assertIn("target_rate_per_kg", corr)
        self.assertIn("min_rate_per_kg", corr)
        self.assertIn("max_rate_per_kg", corr)

        for key in ["floor", "balanced", "premium"]:
            self.assertIn("Rate_Per_Kg", strat[key])
            self.assertIn("Margin_Per_Kg", strat[key])
            self.assertGreater(strat[key]["Rate_Per_Kg"], 0)


if __name__ == "__main__":
    unittest.main()
