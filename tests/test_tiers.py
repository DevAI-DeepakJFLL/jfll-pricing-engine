"""
tests/test_tiers.py
Unit tests for weight tiers, account tiers, and weight-break arbitrage logic.
"""

import unittest
from src.tiers import (
    get_weight_tier,
    map_account_tier,
    get_next_weight_break,
    check_weight_break_arbitrage,
    IATA_WEIGHT_BREAKS,
)


class TestTiersAndArbitrage(unittest.TestCase):
    def test_weight_tier_boundaries(self):
        self.assertEqual(get_weight_tier(10.0), "<45kg")
        self.assertEqual(get_weight_tier(44.9), "<45kg")
        self.assertEqual(get_weight_tier(45.0), "45-100kg")
        self.assertEqual(get_weight_tier(99.9), "45-100kg")
        self.assertEqual(get_weight_tier(100.0), "100-300kg")
        self.assertEqual(get_weight_tier(299.9), "100-300kg")
        self.assertEqual(get_weight_tier(300.0), "300-500kg")
        self.assertEqual(get_weight_tier(500.0), "500-1000kg")
        self.assertEqual(get_weight_tier(1000.0), ">1000kg")
        self.assertEqual(get_weight_tier(2500.0), ">1000kg")

    def test_account_tier_boundaries(self):
        self.assertEqual(map_account_tier(0), "Spot / One-Off")
        self.assertEqual(map_account_tier(1), "Spot / One-Off")
        self.assertEqual(map_account_tier(2), "Occasional")
        self.assertEqual(map_account_tier(5), "Occasional")
        self.assertEqual(map_account_tier(6), "Regular")
        self.assertEqual(map_account_tier(19), "Regular")
        self.assertEqual(map_account_tier(20), "Key Account")
        self.assertEqual(map_account_tier(150), "Key Account")

    def test_get_next_weight_break(self):
        self.assertEqual(get_next_weight_break(30.0), 45.0)
        self.assertEqual(get_next_weight_break(45.0), 100.0)
        self.assertEqual(get_next_weight_break(92.0), 100.0)
        self.assertEqual(get_next_weight_break(100.0), 300.0)
        self.assertEqual(get_next_weight_break(350.0), 500.0)
        self.assertEqual(get_next_weight_break(750.0), 1000.0)
        self.assertIsNone(get_next_weight_break(1200.0))

    def test_check_weight_break_arbitrage_found(self):
        # 92 kg @ Rs 150/kg = Rs 13,800.
        # Next slab is 100 kg. If rate is Rs 130/kg, 100 * 130 = Rs 13,000.
        # Arbitrage saves Rs 800.
        result = check_weight_break_arbitrage(
            chargeable_wt=92.0,
            current_total_cost=13800.0,
            next_slab_rate_per_kg=130.0,
        )
        self.assertTrue(result["has_arbitrage"])
        self.assertEqual(result["original_chargeable_wt"], 92.0)
        self.assertEqual(result["recommended_chargeable_wt"], 100.0)
        self.assertEqual(result["arbitrage_cost"], 13000.0)
        self.assertEqual(result["savings"], 800.0)
        self.assertAlmostEqual(result["savings_pct"], (800.0 / 13800.0) * 100.0, places=2)

    def test_check_weight_break_arbitrage_none(self):
        # 60 kg @ Rs 150/kg = Rs 9,000.
        # Next slab 100 kg @ Rs 130/kg = Rs 13,000 > Rs 9,000.
        result = check_weight_break_arbitrage(
            chargeable_wt=60.0,
            current_total_cost=9000.0,
            next_slab_rate_per_kg=130.0,
        )
        self.assertFalse(result["has_arbitrage"])
        self.assertEqual(result["recommended_chargeable_wt"], 60.0)
        self.assertEqual(result["savings"], 0.0)

    def test_check_weight_break_arbitrage_with_slab_rates_dict(self):
        slab_rates = {
            0.0: 220.0,
            45.0: 180.0,
            100.0: 150.0,
            300.0: 120.0,
            500.0: 100.0,
            1000.0: 85.0,
        }
        # 280 kg @ 100-300kg rate (Rs 150/kg) = Rs 42,000.
        # Next slab is 300 kg @ Rs 120/kg = Rs 36,000.
        # Savings = Rs 6,000.
        result = check_weight_break_arbitrage(
            chargeable_wt=280.0,
            current_total_cost=42000.0,
            slab_rates=slab_rates,
        )
        self.assertTrue(result["has_arbitrage"])
        self.assertEqual(result["recommended_chargeable_wt"], 300.0)
        self.assertEqual(result["arbitrage_cost"], 36000.0)
        self.assertEqual(result["savings"], 6000.0)

    def test_check_weight_break_arbitrage_over_1000kg(self):
        # Over top break: no next break exists.
        result = check_weight_break_arbitrage(
            chargeable_wt=1500.0,
            current_total_cost=120000.0,
            next_slab_rate_per_kg=75.0,
        )
        self.assertFalse(result["has_arbitrage"])
        self.assertEqual(result["recommended_chargeable_wt"], 1500.0)


if __name__ == "__main__":
    unittest.main()
