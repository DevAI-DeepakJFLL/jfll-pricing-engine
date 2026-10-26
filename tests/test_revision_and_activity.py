"""
tests/test_revision_and_activity.py
Unit tests verifying commercial logistics logic for:
1. Customer Activity Tier yield modulation (Key Account volume discount vs Spot markup).
2. Quote Revision Concession Ladder (Rev 1 baseline, Rev 2 counter-offer concession, Rev 3+ BAFO floor defense).
3. Heavy Cargo Premium Calibration (commercially bounded high-yield quote, preventing unquoteable outliers).
4. REST API /api/quote end-to-end integration with revision and frequency parameters.
"""

import sys
import os
import unittest
import pytest

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_TESTS_DIR)
_SRC_DIR = os.path.join(_ROOT_DIR, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from src.engine_v3 import PricingEngineV3
from src.tiers import map_account_tier
from src.app import app


class TestRevisionAndActivityLogistics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = PricingEngineV3().load()
        cls.client = app.test_client()

    def test_account_tier_mapping(self):
        """Verify frequency correctly maps to air freight customer relationship tiers."""
        self.assertEqual(map_account_tier(25), "Key Account")
        self.assertEqual(map_account_tier(20), "Key Account")
        self.assertEqual(map_account_tier(10), "Regular")
        self.assertEqual(map_account_tier(6), "Regular")
        self.assertEqual(map_account_tier(3), "Occasional")
        self.assertEqual(map_account_tier(2), "Occasional")
        self.assertEqual(map_account_tier(1), "Spot / One-Off")
        self.assertEqual(map_account_tier(0), "Spot / One-Off")

    def test_customer_activity_margin_modulation(self):
        """
        Verify that Customer Activity directly modulates recommended margin:
        - Key Account receives volume incentive discount.
        - Spot / One-Off is assessed a risk premium for transactional overhead and collection risk.
        - Regular represents the baseline standard rate.
        """
        base_inq = {
            "origin": "Indira Gandhi International Airport",
            "dest": "Heathrow Apt/London",
            "buy": 100000.0,
            "gross_wt": 450.0,
            "chargeable_wt": 450.0,
            "client": "Shipper / Consignee",
            "commodity": "General Cargo",
            "strategy": "balanced",
            "revision": 1,
        }

        # Key Account (25)
        inq_key = dict(base_inq, frequency=25)
        res_key = self.engine.recommend_quote(inq_key)
        m_key = res_key["optimal"]["Margin_Percentage"]
        self.assertEqual(res_key["optimal"]["Account_Activity_Tier"], "Key Account")

        # Regular (10)
        inq_reg = dict(base_inq, frequency=10)
        res_reg = self.engine.recommend_quote(inq_reg)
        m_reg = res_reg["optimal"]["Margin_Percentage"]
        self.assertEqual(res_reg["optimal"]["Account_Activity_Tier"], "Regular")

        # Spot / One-Off (1)
        inq_spot = dict(base_inq, frequency=1)
        res_spot = self.engine.recommend_quote(inq_spot)
        m_spot = res_spot["optimal"]["Margin_Percentage"]
        self.assertEqual(res_spot["optimal"]["Account_Activity_Tier"], "Spot / One-Off")

        # Commercial order verification: Key Account margin < Regular margin < Spot margin
        self.assertLess(m_key, m_reg, f"Key Account margin ({m_key}%) must be < Regular ({m_reg}%)")
        self.assertLess(m_reg, m_spot, f"Regular margin ({m_reg}%) must be < Spot ({m_spot}%)")
        self.assertGreaterEqual(m_spot - m_reg, 1.0, "Spot risk premium should be at least +1.0 pp above regular")

    def test_quote_revision_concession_ladder(self):
        """
        Verify the forwarder multi-round negotiation ladder:
        - Rev 1: Initial Standard Quote (Full Balanced Benchmark).
        - Rev 2: Counter-Offer Pushback (Concession towards negotiation floor).
        - Rev 3+: BAFO / Tender Defense (Target pinned to Floor to defend booking).
        """
        base_inq = {
            "origin": "Indira Gandhi International Airport",
            "dest": "Heathrow Apt/London",
            "buy": 100000.0,
            "gross_wt": 450.0,
            "chargeable_wt": 450.0,
            "client": "Shipper / Consignee",
            "commodity": "General Cargo",
            "frequency": 10,
            "strategy": "balanced",
        }

        # Rev 1
        res1 = self.engine.recommend_quote(dict(base_inq, revision=1))
        m1 = res1["optimal"]["Margin_Percentage"]

        # Rev 2
        res2 = self.engine.recommend_quote(dict(base_inq, revision=2))
        m2 = res2["optimal"]["Margin_Percentage"]
        self.assertIn("Rev 2 Counter-Offer", res2["optimal"]["Revision_Guidance"])
        self.assertGreater(res2["optimal"]["Revision_Concession_Pct"], 0.0)

        # Rev 3
        res3 = self.engine.recommend_quote(dict(base_inq, revision=3))
        m3 = res3["optimal"]["Margin_Percentage"]
        self.assertIn("BAFO", res3["optimal"]["Revision_Guidance"])

        # Monotonic concession: Rev 1 > Rev 2 >= Rev 3
        self.assertGreater(m1, m2, f"Rev 1 margin ({m1}%) must be > Rev 2 margin ({m2}%)")
        self.assertGreaterEqual(m2, m3, f"Rev 2 margin ({m2}%) must be >= Rev 3 margin ({m3}%)")
        self.assertEqual(m3, res3["strategic_recommendations"]["floor"]["Margin_Percentage"])

    def test_heavy_cargo_premium_calibration(self):
        """
        Verify that Premium Strategy on heavy cargo (>300kg and >500kg) is commercially calibrated
        and does not produce unrealistic >15% margin outliers.
        """
        inquiry_450kg = {
            "origin": "Indira Gandhi International Airport",
            "dest": "Heathrow Apt/London",
            "buy": 100000.0,
            "gross_wt": 450.0,
            "chargeable_wt": 450.0,
            "client": "Shipper / Consignee",
            "commodity": "General Cargo",
            "frequency": 10,
            "revision": 1,
        }
        res = self.engine.recommend_quote(inquiry_450kg)
        strats = res["strategic_recommendations"]
        m_floor = strats["floor"]["Margin_Percentage"]
        m_bal = strats["balanced"]["Margin_Percentage"]
        m_prem = strats["premium"]["Margin_Percentage"]

        # Strict ordering
        self.assertLess(m_floor, m_bal)
        self.assertLess(m_bal, m_prem)

        # Heavy cargo commercial sanity: Premium on 450kg should be <= 10.0%, not 18.57%
        self.assertLessEqual(m_prem, 10.0, f"Premium margin on 450kg ({m_prem}%) should be <= 10.0%")
        self.assertGreaterEqual(m_prem, m_bal + 1.0, f"Premium margin must be at least +1.0 pp above balanced")

    def test_api_quote_revision_and_activity_endpoint(self):
        """
        Verify end-to-end /api/quote REST API response with revision and frequency.
        """
        payload = {
            "origin": "Indira Gandhi International Airport",
            "dest": "Heathrow Apt/London",
            "buy": 100000.0,
            "gross_wt": 450.0,
            "chargeable_wt": 450.0,
            "client": "Shipper / Consignee",
            "commodity": "General Cargo",
            "frequency": 1,  # Spot One-Off
            "revision": 2,   # Counter-Offer
        }

        resp = self.client.post("/api/quote", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()

        self.assertIn("optimal", data)
        self.assertIn("negotiation_corridor", data)
        self.assertEqual(data["optimal"]["Account_Activity_Tier"], "Spot / One-Off")
        self.assertEqual(data["optimal"]["Quote_Revision_Count"], 2)
        self.assertIn("Rev 2 Counter-Offer", data["optimal"]["Revision_Guidance"])
        self.assertIn("revision_guidance", data["negotiation_corridor"])


if __name__ == "__main__":
    unittest.main()
