"""
tests/test_api_v3_quote.py
Integration tests for Unified Serving Engine V3 via Flask REST API (/api/quote).

Verifies:
1. Full REST API quote contract parity and schema validation.
2. Calibrated conformal corridor with confidence_coverage: "90%".
3. Active win probabilities (float in (0, 1), not None, with win_prob_floor > win_prob_balanced > win_prob_premium).
4. Distinct strategic tiers (margin_floor < margin_balanced < margin_premium).
5. Commercial guardrails enforcement (AWB floor, DG special cargo approval).
6. Input validation errors (negative buy, invalid weight).
7. Sensitivity ladder (15 steps with active win probabilities).
"""

import unittest
import json
from src.app import app
from src.constants import MODEL_VERSION


class TestApiV3QuoteIntegration(unittest.TestCase):
    """Test suite validating production REST API served by PricingEngineV3."""

    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_quote_endpoint_returns_200_and_schema(self):
        """Verifies /api/quote returns HTTP 200 and complete response JSON contract schema."""
        payload = {
            "buy": 125000,
            "gross_wt": 450,
            "chargeable_wt": 450,
            "origin": "Indira Gandhi International Airport",
            "dest": "Heathrow Apt/London",
            "client": "Shipper / Consignee",
            "commodity": "General Cargo",
            "frequency": 10,
            "revision": 1,
            "strategy": "balanced"
        }
        res = self.client.post("/api/quote", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        # Top-level contract validation
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["model_version"], MODEL_VERSION)
        self.assertIn("optimal", data)
        self.assertIn("strategic_recommendations", data)
        self.assertIn("negotiation_corridor", data)
        self.assertIn("sensitivity", data)
        self.assertIn("confidence", data)
        self.assertIn(data["confidence"], ["High", "Low"])
        self.assertIn("warnings", data)
        self.assertIsInstance(data["warnings"], list)
        self.assertIn("signals", data)
        self.assertIn("cohort_band_per_kg", data)
        self.assertIn("pricing_mode", data)
        self.assertIn("weight_break_arbitrage", data)
        self.assertIn("competitor_intelligence", data)
        self.assertIn("history_logged", data)

        # Strategic recommendations structure
        strat = data["strategic_recommendations"]
        for tier_key in ["floor", "balanced", "premium"]:
            self.assertIn(tier_key, strat)
            card = strat[tier_key]
            self.assertIn("Margin_Percentage", card)
            self.assertIn("Gross_Margin_Percentage", card)
            self.assertIn("Quoted_Sell_Price_INR", card)
            self.assertIn("Margin_Amount_INR", card)
            self.assertIn("Rate_Per_Kg", card)
            self.assertIn("Win_Probability", card)
            self.assertIn("Expected_Profit_INR", card)
            self.assertIn("Strategy_Name", card)
            self.assertIn("Badge_Label", card)

    def test_conformal_corridor_contract(self):
        """Verifies calibrated conformal corridor returns 90% confidence coverage and valid bounds."""
        payload = {
            "buy": 80000,
            "gross_wt": 320,
            "chargeable_wt": 320,
            "origin": "Bangalore",
            "dest": "Frankfurt am Main",
            "client": "Subagent",
            "commodity": "Engineering & Machinery",
            "strategy": "balanced"
        }
        res = self.client.post("/api/quote", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        corr = data["negotiation_corridor"]
        self.assertIsNotNone(corr)
        self.assertEqual(corr["confidence_coverage"], "90%")
        self.assertIn("conformal_half_width_pct", corr)
        self.assertGreater(corr["conformal_half_width_pct"], 0.0)

        # Bounds ordering
        self.assertLessEqual(corr["min_margin_pct"], corr["target_margin_pct"])
        self.assertLessEqual(corr["target_margin_pct"], corr["max_margin_pct"])

        # Price contract
        min_sell = corr.get("min_sell_price_inr") or corr.get("min_sell_inr")
        max_sell = corr.get("max_sell_price_inr") or corr.get("max_sell_inr")
        self.assertIsNotNone(min_sell)
        self.assertIsNotNone(max_sell)
        self.assertLessEqual(min_sell, max_sell)

    def test_active_win_probabilities(self):
        """Verifies active win probabilities are floats in (0, 1), not None, with strict tier monotonicity."""
        payload = {
            "buy": 150000,
            "gross_wt": 500,
            "chargeable_wt": 500,
            "origin": "Chhatrapati Shivaji Maharaj International Airport",
            "dest": "Dubai",
            "client": "Shipper / Consignee",
            "commodity": "Pharmaceuticals",
            "strategy": "balanced"
        }
        res = self.client.post("/api/quote", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        strat = data["strategic_recommendations"]
        floor_win = strat["floor"]["Win_Probability"]
        bal_win = strat["balanced"]["Win_Probability"]
        prem_win = strat["premium"]["Win_Probability"]

        # 1. Elimination of win_probability=None (Issue M7)
        self.assertIsNotNone(floor_win)
        self.assertIsNotNone(bal_win)
        self.assertIsNotNone(prem_win)

        self.assertIsInstance(floor_win, float)
        self.assertIsInstance(bal_win, float)
        self.assertIsInstance(prem_win, float)

        # 2. Probability bounds in (0, 1)
        self.assertGreater(floor_win, 0.0)
        self.assertLess(floor_win, 1.0)
        self.assertGreater(bal_win, 0.0)
        self.assertLess(bal_win, 1.0)
        self.assertGreater(prem_win, 0.0)
        self.assertLess(prem_win, 1.0)

        # 3. Monotonic decreasing win probability with margin: P(Win|Floor) > P(Win|Balanced) > P(Win|Premium)
        self.assertGreater(floor_win, bal_win)
        self.assertGreater(bal_win, prem_win)

        # 4. Realistic calibration zones
        self.assertGreaterEqual(floor_win, 0.55)
        self.assertLessEqual(floor_win, 0.85)
        self.assertGreaterEqual(bal_win, 0.35)
        self.assertLessEqual(bal_win, 0.55)
        self.assertGreaterEqual(prem_win, 0.15)
        self.assertLessEqual(prem_win, 0.35)

        # 5. Expected profit computation: Expected_Profit_INR = Margin_Amount_INR * Win_Probability
        for tier in ["floor", "balanced", "premium"]:
            card = strat[tier]
            expected_p = card["Expected_Profit_INR"]
            self.assertIsNotNone(expected_p)
            self.assertGreater(expected_p, 0.0)
            calculated_p = round(card["Margin_Amount_INR"] * card["Win_Probability"], 2)
            self.assertAlmostEqual(expected_p, calculated_p, delta=1.0)

    def test_distinct_strategic_tiers(self):
        """Verifies strictly distinct strategic margin tiers: floor < balanced < premium."""
        payload = {
            "buy": 95000,
            "gross_wt": 250,
            "chargeable_wt": 250,
            "origin": "Indira Gandhi International Airport",
            "dest": "Frankfurt am Main",
            "client": "Overseas Agent",
            "commodity": "Garments / Textiles",
            "strategy": "balanced"
        }
        res = self.client.post("/api/quote", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        strat = data["strategic_recommendations"]
        m_floor = strat["floor"]["Margin_Percentage"]
        m_bal = strat["balanced"]["Margin_Percentage"]
        m_prem = strat["premium"]["Margin_Percentage"]

        self.assertLess(m_floor, m_bal)
        self.assertLess(m_bal, m_prem)

        p_floor = strat["floor"]["Quoted_Sell_Price_INR"]
        p_bal = strat["balanced"]["Quoted_Sell_Price_INR"]
        p_prem = strat["premium"]["Quoted_Sell_Price_INR"]

        self.assertLess(p_floor, p_bal)
        self.assertLess(p_bal, p_prem)

    def test_commercial_guardrail_enforcement(self):
        """Verifies small shipment AWB floor clamping and DG special cargo governance approvals."""
        # Case A: Small shipment (<= 45 kg) clamped to AWB floor
        small_payload = {
            "buy": 5000,
            "gross_wt": 25,
            "chargeable_wt": 25,
            "origin": "Bangalore",
            "dest": "Dubai",
            "client": "Shipper / Consignee",
            "commodity": "General Cargo",
            "strategy": "floor"
        }
        res_small = self.client.post("/api/quote", data=json.dumps(small_payload), content_type="application/json")
        self.assertEqual(res_small.status_code, 200)
        data_small = res_small.get_json()
        card_floor = data_small["strategic_recommendations"]["floor"]
        # Margin amount must protect at least ₹1,500 AWB minimum
        self.assertGreaterEqual(card_floor["Margin_Amount_INR"], 1500.0)

        # Case B: Special Cargo (Dangerous Goods) triggers approval flag
        dg_payload = {
            "buy": 85000,
            "gross_wt": 200,
            "chargeable_wt": 200,
            "origin": "Indira Gandhi International Airport",
            "dest": "Frankfurt am Main",
            "client": "Custom Broker",
            "commodity": "Dangerous Goods",
            "strategy": "balanced"
        }
        res_dg = self.client.post("/api/quote", data=json.dumps(dg_payload), content_type="application/json")
        self.assertEqual(res_dg.status_code, 200)
        data_dg = res_dg.get_json()

        # Approval required flag must appear in warnings and/or optimal quote
        all_warnings_dg = " ".join(data_dg["warnings"]).lower()
        optimal_approvals = " ".join(data_dg["optimal"].get("Approval_Required", [])).lower()
        has_dg_approval = ("special cargo" in all_warnings_dg or "dangerous goods" in all_warnings_dg
                           or "special cargo" in optimal_approvals or "dangerous goods" in optimal_approvals)
        self.assertTrue(has_dg_approval, "Dangerous Goods inquiry must trigger special cargo approval flag.")

    def test_input_validation_errors(self):
        """Verifies HTTP 400 bad request handling for invalid inputs."""
        # 1. Negative buy cost
        res_neg_buy = self.client.post("/api/quote", data=json.dumps({"buy": -500, "gross_wt": 100}), content_type="application/json")
        self.assertEqual(res_neg_buy.status_code, 400)
        self.assertIn("Invalid Airline Buy Cost", res_neg_buy.get_json()["message"])

        # 2. Impossible low gross weight (< 0.5 kg)
        res_low_wt = self.client.post("/api/quote", data=json.dumps({"buy": 10000, "gross_wt": 0.2}), content_type="application/json")
        self.assertEqual(res_low_wt.status_code, 400)

        # 3. Impossible excessive gross weight (> 25,000 kg)
        res_high_wt = self.client.post("/api/quote", data=json.dumps({"buy": 1000000, "gross_wt": 30000}), content_type="application/json")
        self.assertEqual(res_high_wt.status_code, 400)

    def test_sensitivity_ladder_with_win_probabilities(self):
        """Verifies sensitivity grid produces 15 steps with strictly decreasing win probabilities."""
        payload = {
            "buy": 110000,
            "gross_wt": 350,
            "chargeable_wt": 350,
            "origin": "Chennai",
            "dest": "Heathrow Apt/London",
            "client": "NVOCC / Coloader",
            "commodity": "Auto Parts",
            "strategy": "balanced"
        }
        res = self.client.post("/api/quote", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        sens = data["sensitivity"]
        self.assertEqual(len(sens), 15)

        prev_margin = -1e9
        prev_win = 1e9
        for row in sens:
            self.assertIn("Margin_Percentage", row)
            self.assertIn("Quoted_Sell_Price_INR", row)
            self.assertIn("Margin_Amount_INR", row)
            self.assertIn("Win_Probability", row)
            self.assertIn("Expected_Profit_INR", row)

            curr_margin = row["Margin_Percentage"]
            curr_win = row["Win_Probability"]

            self.assertGreaterEqual(curr_margin, prev_margin)
            self.assertLessEqual(curr_win, prev_win)
            self.assertGreater(curr_win, 0.0)
            self.assertLess(curr_win, 1.0)

            prev_margin = curr_margin
            prev_win = curr_win

    def test_edge_case_tiers_clamping_to_cost_floor(self):
        """Verifies strict tier ordering, cap compliance, and monotonic win probabilities when tiers clamp to cost floor."""
        payload = {
            "buy": 300000,
            "gross_wt": 100,
            "chargeable_wt": 100,
            "origin": "Bangalore",
            "dest": "Dubai",
            "client": "Shipper / Consignee",
            "commodity": "Dangerous Goods",
            "strategy": "balanced"
        }
        res = self.client.post("/api/quote", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        strat = data["strategic_recommendations"]
        m_floor = strat["floor"]["Margin_Percentage"]
        m_bal = strat["balanced"]["Margin_Percentage"]
        m_prem = strat["premium"]["Margin_Percentage"]

        # 1. Strict tier differentiation (floor < balanced < premium)
        self.assertLess(m_floor, m_bal)
        self.assertLess(m_bal, m_prem)

        # 2. All tiers within effective cap
        cap = strat["balanced"]["Effective_Max_Margin_Cap"]
        self.assertLessEqual(m_prem, cap)
        self.assertLessEqual(m_bal, cap)
        self.assertLessEqual(m_floor, cap)

        # 3. Monotonic win probabilities: P(Win|Floor) > P(Win|Balanced) > P(Win|Premium)
        w_floor = strat["floor"]["Win_Probability"]
        w_bal = strat["balanced"]["Win_Probability"]
        w_prem = strat["premium"]["Win_Probability"]
        self.assertGreater(w_floor, w_bal)
        self.assertGreater(w_bal, w_prem)

    def test_edge_case_tiers_clamping_to_effective_cap(self):
        """Verifies no tier exceeds effective_cap when predictions clamp at or near ceiling."""
        payload = {
            "buy": 15000,
            "gross_wt": 20,
            "chargeable_wt": 20,
            "origin": "Indira Gandhi International Airport",
            "dest": "Heathrow Apt/London",
            "client": "Courier",
            "commodity": "Courier",
            "strategy": "balanced"
        }
        res = self.client.post("/api/quote", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        strat = data["strategic_recommendations"]
        m_floor = strat["floor"]["Margin_Percentage"]
        m_bal = strat["balanced"]["Margin_Percentage"]
        m_prem = strat["premium"]["Margin_Percentage"]

        # Strict ordering
        self.assertLess(m_floor, m_bal)
        self.assertLess(m_bal, m_prem)

        # Neither premium nor balanced nor floor exceeds effective cap
        cap = strat["balanced"]["Effective_Max_Margin_Cap"]
        self.assertLessEqual(m_prem, cap)
        self.assertLessEqual(m_bal, cap)
        self.assertLessEqual(m_floor, cap)

        # Monotonic win probabilities
        w_floor = strat["floor"]["Win_Probability"]
        w_bal = strat["balanced"]["Win_Probability"]
        w_prem = strat["premium"]["Win_Probability"]
        self.assertGreater(w_floor, w_bal)
        self.assertGreater(w_bal, w_prem)


if __name__ == "__main__":
    unittest.main()

