"""
tests/test_policy_guardrails.py
Comprehensive unit tests for post-model commercial guardrails, operational floors,
margin caps, risk buffers, and governance approvals. (Issue M1, Fix 4)
"""

import dataclasses
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from policy_guardrails import GuardrailResult, apply_commercial_guardrails
from constants import (
    MIN_MARGIN_PER_AWB_INR,
    MIN_MARGIN_PER_KG_INR,
    OPS_HANDLING_FEE_AWB_INR,
    COMMODITY_MAX_MARGIN_CAPS,
    CLIENT_CATEGORY_MAX_MARGIN_CAPS,
)


class TestPolicyGuardrails(unittest.TestCase):
    """Test suite for post-model commercial guardrail decoupling."""

    def test_cost_floor_small_shipment(self):
        """
        Requirement 1: Small shipment (<= 45 kg) clamped to minimum AWB floor (₹1,500)
        and ops fee, preventing near-cost or sub-cost quotes on low-weight cargo.
        """
        buy_cost = 5000.0
        chargeable_wt = 15.0
        raw_margin_pct = 2.0  # Naive 2% markup would be only ₹100, breaching ₹1,500 AWB floor

        res = apply_commercial_guardrails(
            raw_margin_pct=raw_margin_pct,
            buy_cost=buy_cost,
            chargeable_wt=chargeable_wt,
            commodity="General Cargo",
            client_cat="Subagent",
        )

        self.assertTrue(res.is_floor_clamped)
        self.assertFalse(res.is_cap_clamped)
        self.assertEqual(res.limiting_factor, "cost_floor_awb")
        self.assertGreaterEqual(res.margin_amount_inr, MIN_MARGIN_PER_AWB_INR)
        self.assertAlmostEqual(res.sell_price_inr, res.cost_floor_price_inr, places=2)
        self.assertAlmostEqual(res.final_margin_pct, res.cost_floor_margin_pct, places=2)
        self.assertGreater(res.final_margin_pct, raw_margin_pct)

    def test_unit_margin_floor_heavy_shipment(self):
        """
        Requirement 2: Bulk 2,000 kg shipment respects ₹3/kg operational unit margin floor.
        """
        buy_cost = 400000.0
        chargeable_wt = 2000.0
        raw_margin_pct = 0.5  # 0.5% would yield ₹2,000 (only ₹1/kg), breaching ₹3/kg floor

        res = apply_commercial_guardrails(
            raw_margin_pct=raw_margin_pct,
            buy_cost=buy_cost,
            chargeable_wt=chargeable_wt,
            commodity="General Cargo",
            client_cat="Subagent",
        )

        self.assertTrue(res.is_floor_clamped)
        self.assertFalse(res.is_cap_clamped)
        self.assertEqual(res.limiting_factor, "cost_floor_per_kg")
        # Operational unit margin must satisfy ₹3/kg
        unit_margin_inr = res.margin_amount_inr / chargeable_wt
        self.assertGreaterEqual(unit_margin_inr, MIN_MARGIN_PER_KG_INR)
        self.assertGreaterEqual(res.margin_amount_inr, MIN_MARGIN_PER_KG_INR * chargeable_wt)
        self.assertAlmostEqual(res.sell_price_inr, res.cost_floor_price_inr, places=2)

    def test_commodity_cap_clamping(self):
        """
        Requirement 3: High quote on Garments clamped to 25% commodity ceiling.
        """
        buy_cost = 50000.0
        chargeable_wt = 200.0
        raw_margin_pct = 32.0  # Higher than Garments cap of 25.0%

        res = apply_commercial_guardrails(
            raw_margin_pct=raw_margin_pct,
            buy_cost=buy_cost,
            chargeable_wt=chargeable_wt,
            commodity="Garments / Textiles",
            client_cat="Shipper / Consignee",  # Shipper cap is 35.0%, so Garments (25.0%) governs
        )

        self.assertFalse(res.is_floor_clamped)
        self.assertTrue(res.is_cap_clamped)
        self.assertEqual(res.limiting_factor, "commodity_cap")
        self.assertEqual(res.effective_cap_pct, 25.0)
        self.assertEqual(res.final_margin_pct, 25.0)
        self.assertEqual(res.sell_price_inr, round(buy_cost * 1.25, 2))

    def test_client_category_cap_clamping(self):
        """
        Requirement 4: High quote on GSA client clamped to 20% client category ceiling.
        """
        buy_cost = 50000.0
        chargeable_wt = 200.0
        raw_margin_pct = 26.0  # Higher than GSA cap of 20.0%

        res = apply_commercial_guardrails(
            raw_margin_pct=raw_margin_pct,
            buy_cost=buy_cost,
            chargeable_wt=chargeable_wt,
            commodity="General Cargo",  # General Cargo cap is 30.0%, so GSA (20.0%) governs
            client_cat="GSA",
        )

        self.assertFalse(res.is_floor_clamped)
        self.assertTrue(res.is_cap_clamped)
        self.assertEqual(res.limiting_factor, "client_category_cap")
        self.assertEqual(res.effective_cap_pct, 20.0)
        self.assertEqual(res.final_margin_pct, 20.0)
        self.assertEqual(res.sell_price_inr, round(buy_cost * 1.20, 2))

    def test_commodity_risk_buffer(self):
        """
        Requirement 5: Dangerous Goods floor is strictly higher than General Cargo floor
        for identical weight and buy cost due to mandatory +6.0% risk buffer.
        """
        buy_cost = 200000.0
        chargeable_wt = 1000.0
        raw_margin_pct = 1.0

        res_gen = apply_commercial_guardrails(
            raw_margin_pct=raw_margin_pct,
            buy_cost=buy_cost,
            chargeable_wt=chargeable_wt,
            commodity="General Cargo",
            client_cat="Subagent",
        )

        res_dg = apply_commercial_guardrails(
            raw_margin_pct=raw_margin_pct,
            buy_cost=buy_cost,
            chargeable_wt=chargeable_wt,
            commodity="Dangerous Goods",
            client_cat="Subagent",
        )

        self.assertGreater(res_dg.cost_floor_price_inr, res_gen.cost_floor_price_inr)
        self.assertGreater(res_dg.cost_floor_margin_pct, res_gen.cost_floor_margin_pct)
        self.assertTrue(res_dg.is_floor_clamped)
        self.assertEqual(res_dg.limiting_factor, "cost_floor_risk")

    def test_approval_flags(self):
        """
        Requirement 6: Verifies approval triggers for:
        - Special cargo: Dangerous Goods, Valuables, Live Animals.
        - Conflicting bounds: Cost floor > Cap.
        - Clean quote without approvals.
        """
        # 1. Dangerous Goods approval
        res_dg = apply_commercial_guardrails(
            raw_margin_pct=15.0,
            buy_cost=100000.0,
            chargeable_wt=500.0,
            commodity="Dangerous Goods",
            client_cat="Subagent",
        )
        self.assertIn("special cargo: Dangerous Goods", res_dg.approval_required)

        # 2. Valuables approval
        res_val = apply_commercial_guardrails(
            raw_margin_pct=15.0,
            buy_cost=100000.0,
            chargeable_wt=500.0,
            commodity="Valuables",
            client_cat="Subagent",
        )
        self.assertIn("special cargo: Valuables", res_val.approval_required)

        # 3. Live Animals approval
        res_live = apply_commercial_guardrails(
            raw_margin_pct=15.0,
            buy_cost=100000.0,
            chargeable_wt=500.0,
            commodity="Live Animals",
            client_cat="Subagent",
        )
        self.assertIn("special cargo: Live Animals", res_live.approval_required)

        # 4. Standard cargo - zero approvals
        res_clean = apply_commercial_guardrails(
            raw_margin_pct=15.0,
            buy_cost=100000.0,
            chargeable_wt=500.0,
            commodity="General Cargo",
            client_cat="Subagent",
        )
        self.assertEqual(res_clean.approval_required, ())

        # 5. Conflicting bounds: Small shipment with tight cap (Shipping Line cap = 18%)
        # Small shipment floor requires ~76% margin, exceeding 18% cap
        res_conflict = apply_commercial_guardrails(
            raw_margin_pct=15.0,
            buy_cost=2000.0,
            chargeable_wt=10.0,
            commodity="General Cargo",
            client_cat="Shipping Line",
        )
        self.assertGreater(res_conflict.cost_floor_margin_pct, res_conflict.effective_cap_pct)
        self.assertIn("cost floor above cap", res_conflict.approval_required)
        # Operational floor must take precedence over cap to prevent financial loss
        self.assertTrue(res_conflict.is_floor_clamped)
        self.assertFalse(res_conflict.is_cap_clamped)
        self.assertEqual(res_conflict.final_margin_pct, res_conflict.cost_floor_margin_pct)

    def test_immutability_and_serialization(self):
        """
        Requirement 7: GuardrailResult is strictly immutable and safely serializable to dict and JSON.
        """
        res = apply_commercial_guardrails(
            raw_margin_pct=12.5,
            buy_cost=50000.0,
            chargeable_wt=200.0,
            commodity="Dangerous Goods",
            client_cat="Subagent",
        )

        # Immutability verification
        with self.assertRaises(dataclasses.FrozenInstanceError):
            res.final_margin_pct = 99.0  # type: ignore

        with self.assertRaises(dataclasses.FrozenInstanceError):
            res.approval_required = ()  # type: ignore

        self.assertIsInstance(res.approval_required, tuple)

        # Serialization to dict
        d = res.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["raw_margin_pct"], 12.5)
        self.assertEqual(d["final_margin_pct"], res.final_margin_pct)
        self.assertEqual(d["approval_required"], ("special cargo: Dangerous Goods",))

        # Parity with dataclasses.asdict
        as_dict_d = dataclasses.asdict(res)
        self.assertEqual(d, as_dict_d)

        # JSON roundtrip
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        self.assertEqual(parsed["final_margin_pct"], res.final_margin_pct)
        self.assertEqual(parsed["approval_required"], ["special cargo: Dangerous Goods"])

    def test_unclamped_within_bounds(self):
        """Verify normal quote within bounds retains exact raw margin and sets limiting_factor to none."""
        res = apply_commercial_guardrails(
            raw_margin_pct=15.0,
            buy_cost=50000.0,
            chargeable_wt=200.0,
            commodity="General Cargo",
            client_cat="Subagent",
        )
        self.assertFalse(res.is_floor_clamped)
        self.assertFalse(res.is_cap_clamped)
        self.assertEqual(res.final_margin_pct, 15.0)
        self.assertEqual(res.limiting_factor, "none")
        self.assertEqual(res.approval_required, ())


if __name__ == "__main__":
    unittest.main()
