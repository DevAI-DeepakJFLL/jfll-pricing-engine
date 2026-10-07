"""
tests/test_guardrails.py
Unit tests verifying hard commercial guardrails, minimum ₹ floors,
risk buffers, and version consistency. (Category 1, Task 1.1)
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from constants import (
    MIN_MARGIN_PER_AWB_INR,
    MIN_MARGIN_PER_KG_INR,
    OPS_HANDLING_FEE_AWB_INR,
    COMMODITY_RISK_BUFFERS,
    COMMODITY_MAX_MARGIN_CAPS,
    CLIENT_CATEGORY_MAX_MARGIN_CAPS,
    MODEL_VERSION,
    STRATEGY_THRESHOLDS,
    MIN_CANDIDATE_MARGIN,
    MAX_CANDIDATE_MARGIN,
    compute_cost_floor_price,
    get_commodity_max_margin,
    get_client_category_max_margin,
    get_category_commodity_max_margin
)


class TestConstantsAndGuardrails(unittest.TestCase):
    def test_guardrail_floors_exist(self):
        """Verify absolute rupee floors per AWB and per kg are set to safe commercial minimums."""
        self.assertGreaterEqual(MIN_MARGIN_PER_AWB_INR, 1500.0)
        self.assertGreaterEqual(MIN_MARGIN_PER_KG_INR, 3.0)
        self.assertGreaterEqual(OPS_HANDLING_FEE_AWB_INR, 850.0)

    def test_commodity_risk_buffers(self):
        """Verify specialized commodity risk buffers exist and are calibrated."""
        self.assertIn("dangerous_goods", COMMODITY_RISK_BUFFERS)
        self.assertIn("perishable_cold_chain", COMMODITY_RISK_BUFFERS)
        self.assertIn("pharmaceuticals", COMMODITY_RISK_BUFFERS)
        self.assertGreater(COMMODITY_RISK_BUFFERS["dangerous_goods"], 0.04)
        self.assertGreater(COMMODITY_RISK_BUFFERS["pharmaceuticals"], 0.03)

    def test_version_string_consistent(self):
        """Ensure single source of truth for versioning."""
        self.assertTrue(MODEL_VERSION.startswith("v2.3"))
        self.assertEqual(STRATEGY_THRESHOLDS["floor"], 0.40)
        self.assertEqual(STRATEGY_THRESHOLDS["balanced"], 0.25)
        self.assertEqual(STRATEGY_THRESHOLDS["premium"], 0.15)

    def test_cost_floor_small_shipment(self):
        """
        On a small shipment (45 kg, buy ₹10,000):
        The percentage markup (1% = ₹100) must NOT breach the ₹1,500/AWB and ₹3/kg floor.
        """
        buy = 10000.0
        ch_wt = 45.0
        floor_price = compute_cost_floor_price(buy, ch_wt, "general_cargo")
        margin_inr = floor_price - buy
        # Must cover at least ₹1500 + ₹3/kg * 45kg = ₹1635
        self.assertGreaterEqual(margin_inr, 1500.0 + (3.0 * ch_wt))
        self.assertGreater(margin_inr, 100.0)

    def test_cost_floor_large_shipment_with_dg_buffer(self):
        """
        On a large DG shipment (1,000 kg, buy ₹200,000):
        Must cover DG risk buffer + ops fee.
        """
        buy = 200000.0
        ch_wt = 1000.0
        floor_price = compute_cost_floor_price(buy, ch_wt, "dangerous_goods")
        margin_inr = floor_price - buy
        # DG risk buffer is 6% = ₹12,000, plus ops fee
        self.assertGreaterEqual(margin_inr, buy * COMMODITY_RISK_BUFFERS["dangerous_goods"])

    def test_category_and_commodity_max_margin_caps_exist(self):
        """Verify maximum margin caps exist for all canonical commodities and client categories."""
        # Commodities
        self.assertIn("General Cargo", COMMODITY_MAX_MARGIN_CAPS)
        self.assertIn("Garments / Textiles", COMMODITY_MAX_MARGIN_CAPS)
        self.assertIn("Auto Parts", COMMODITY_MAX_MARGIN_CAPS)
        self.assertIn("Perishable Foodstuff", COMMODITY_MAX_MARGIN_CAPS)
        self.assertIn("Pharmaceuticals", COMMODITY_MAX_MARGIN_CAPS)
        self.assertIn("Courier", COMMODITY_MAX_MARGIN_CAPS)
        self.assertIn("Dangerous Goods", COMMODITY_MAX_MARGIN_CAPS)

        # Commercial cap calibration checks
        self.assertLessEqual(COMMODITY_MAX_MARGIN_CAPS["Auto Parts"], 25.0)
        self.assertLessEqual(COMMODITY_MAX_MARGIN_CAPS["Garments / Textiles"], 25.0)
        self.assertGreater(COMMODITY_MAX_MARGIN_CAPS["Dangerous Goods"], 35.0)
        self.assertGreater(COMMODITY_MAX_MARGIN_CAPS["Valuables"], 40.0)

        # Client categories
        self.assertIn("Shipper / Consignee", CLIENT_CATEGORY_MAX_MARGIN_CAPS)
        self.assertIn("Subagent", CLIENT_CATEGORY_MAX_MARGIN_CAPS)
        self.assertIn("NVOCC / Coloader", CLIENT_CATEGORY_MAX_MARGIN_CAPS)
        self.assertLessEqual(CLIENT_CATEGORY_MAX_MARGIN_CAPS["NVOCC / Coloader"], 25.0)
        self.assertLessEqual(CLIENT_CATEGORY_MAX_MARGIN_CAPS["GSA"], 22.0)

    def test_commodity_max_margin_resolver(self):
        """Verify get_commodity_max_margin properly resolves canonical names and aliases."""
        self.assertEqual(get_commodity_max_margin("General Cargo"), 30.0)
        self.assertEqual(get_commodity_max_margin("Auto Parts"), 22.0)
        self.assertEqual(get_commodity_max_margin("Garments / Textiles"), 25.0)
        self.assertEqual(get_commodity_max_margin("pharma"), 35.0)
        self.assertEqual(get_commodity_max_margin("perishable_cold_chain"), 26.0)
        self.assertEqual(get_commodity_max_margin("hazmat"), 42.0)
        self.assertEqual(get_commodity_max_margin("Unknown Nonexistent"), 28.0)

    def test_client_category_max_margin_resolver(self):
        """Verify get_client_category_max_margin properly resolves client categories."""
        self.assertEqual(get_client_category_max_margin("Shipper / Consignee"), 35.0)
        self.assertEqual(get_client_category_max_margin("Subagent"), 28.0)
        self.assertEqual(get_client_category_max_margin("NVOCC / Coloader"), 22.0)
        self.assertEqual(get_client_category_max_margin("Overseas Agent"), 24.0)
        self.assertEqual(get_client_category_max_margin("gsa"), 20.0)

    def test_combined_category_commodity_max_margin(self):
        """Verify get_category_commodity_max_margin takes the governing minimum cap."""
        # Garments (25%) + Direct Shipper (35%) -> Governed by Garments (25%)
        res1 = get_category_commodity_max_margin("Garments / Textiles", "Shipper / Consignee")
        self.assertEqual(res1["effective_max_margin"], 25.0)
        self.assertIn("Commodity", res1["limiting_factor"])

        # Pharma (35%) + NVOCC / Coloader (22%) -> Governed by NVOCC (22%)
        res2 = get_category_commodity_max_margin("Pharmaceuticals", "NVOCC / Coloader")
        self.assertEqual(res2["effective_max_margin"], 22.0)
        self.assertIn("Client Category", res2["limiting_factor"])


if __name__ == "__main__":
    unittest.main()

