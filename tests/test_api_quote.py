"""
tests/test_api_quote.py
Integration tests for Flask /api/quote and /api/history endpoints.
Verifies server-side physical validation, HTTP 400 responses, revision defaults,
and markup vs gross margin calculations.
"""

import unittest
import json
from src.app import app
from src.constants import MODEL_VERSION


class TestApiQuoteIntegration(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_api_quote_valid_payload(self):
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
        self.assertEqual(data["status"], "success")
        self.assertIn("optimal", data)
        self.assertIn("strategic_recommendations", data)
        self.assertEqual(data["model_version"], MODEL_VERSION)

        # Check that both Markup on Buy and Gross Margin on Sell are present in cards
        cards = data["strategic_recommendations"]
        self.assertIn("floor", cards)
        self.assertIn("balanced", cards)
        self.assertIn("premium", cards)

        bal = cards["balanced"]
        self.assertIn("Margin_Percentage", bal)
        self.assertIn("Gross_Margin_Percentage", bal)
        self.assertIn("Rate_Per_Kg", bal)

    def test_api_quote_negative_buy_cost_returns_400(self):
        payload = {"buy": -500, "gross_wt": 100}
        res = self.client.post("/api/quote", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertIn("Invalid Airline Buy Cost", data["message"])

    def test_api_quote_impossible_weight_returns_400(self):
        # Gross weight < 0.5 kg
        payload = {"buy": 10000, "gross_wt": 0.1}
        res = self.client.post("/api/quote", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(res.status_code, 400)

        # Weight > 25,000 kg (charter inquiry required)
        payload2 = {"buy": 500000, "gross_wt": 30000}
        res2 = self.client.post("/api/quote", data=json.dumps(payload2), content_type="application/json")
        self.assertEqual(res2.status_code, 400)

    def test_api_quote_volumetric_server_side_calculation(self):
        # 1 box 100x50x40 = 200,000 / 6000 = 33.33 kg vol weight.
        # Gross weight is 20 kg. Chargeable weight must be computed server-side as >= 33.5 kg!
        payload = {
            "buy": 20000,
            "gross_wt": 20,
            "dim_l": 100,
            "dim_w": 50,
            "dim_h": 40,
            "dim_pcs": 1,
            "strategy": "balanced"
        }
        res = self.client.post("/api/quote", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        optimal = data["optimal"]
        # Chargeable weight in rate per kg should be at least 33.5 kg
        ch_wt = optimal["Quoted_Sell_Price_INR"] / optimal["Rate_Per_Kg"]
        self.assertGreaterEqual(ch_wt, 33.0)

    def test_api_clear_history(self):
        res = self.client.post("/api/history/clear")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")

    def test_api_history_outcome(self):
        # First generate a quote to have an entry in history
        quote_payload = {
            "buy": 100000,
            "gross_wt": 400,
            "origin": "Bangalore",
            "dest": "Frankfurt am Main",
            "strategy": "balanced"
        }
        quote_res = self.client.post("/api/quote", data=json.dumps(quote_payload), content_type="application/json")
        self.assertEqual(quote_res.status_code, 200)

        from src.app import HISTORY_DB_PATH
        from src.db import get_history_paginated
        recs = get_history_paginated(HISTORY_DB_PATH)["records"]
        self.assertTrue(len(recs) > 0)
        quote_id = recs[0]["id"]

        # Record Won outcome
        won_payload = {
            "id": quote_id,
            "status": "Won",
            "final_price_inr": 115000
        }
        res_won = self.client.post("/api/history/outcome", data=json.dumps(won_payload), content_type="application/json")
        self.assertEqual(res_won.status_code, 200)

        # Record Lost outcome with invalid reason (should return 400)
        bad_lost_payload = {
            "id": quote_id,
            "status": "Lost",
            "loss_reason": "invalid_reason_string"
        }
        res_bad = self.client.post("/api/history/outcome", data=json.dumps(bad_lost_payload), content_type="application/json")
        self.assertEqual(res_bad.status_code, 400)

        # Record Lost outcome with valid reason
        good_lost_payload = {
            "id": quote_id,
            "status": "Lost",
            "loss_reason": "price",
            "competitor_rate_per_kg": 260.0
        }
        res_lost = self.client.post("/api/history/outcome", data=json.dumps(good_lost_payload), content_type="application/json")
        self.assertEqual(res_lost.status_code, 200)

    def test_api_quote_weight_break_arbitrage(self):
        # 470 kg cargo near 500 kg break
        payload = {
            "buy": 126900,
            "gross_wt": 470,
            "chargeable_wt": 470,
            "origin": "Bangalore",
            "dest": "Frankfurt am Main",
            "next_slab_rate": 240.0
        }
        res = self.client.post("/api/quote", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("weight_break_arbitrage", data)
        arb = data["weight_break_arbitrage"]
        self.assertTrue(arb["has_arbitrage"])
        self.assertEqual(arb["recommended_chargeable_wt"], 500.0)
        self.assertGreater(arb["savings"], 0.0)
        self.assertIsNotNone(arb["advisory"])

    def test_competitor_rate_feedback_loop(self):
        # 1. First quote without competitor rate
        payload = {
            "buy": 125000,
            "gross_wt": 500,
            "chargeable_wt": 500,
            "origin": "Bangalore",
            "dest": "Frankfurt am Main"
        }
        res1 = self.client.post("/api/quote", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(res1.status_code, 200)
        data1 = res1.get_json()
        self.assertIn("competitor_intelligence", data1)

        # 2. Record a lost deal on this lane with competitor rate 210 INR/kg
        from src.app import HISTORY_DB_PATH
        from src.db import get_history_paginated
        recs = get_history_paginated(HISTORY_DB_PATH)["records"]
        self.assertTrue(len(recs) > 0)
        quote_id = recs[0]["id"]
        res_outcome = self.client.post("/api/history/outcome", data=json.dumps({
            "id": quote_id,
            "status": "Lost",
            "loss_reason": "price",
            "competitor_rate_per_kg": 210.0
        }), content_type="application/json")
        self.assertEqual(res_outcome.status_code, 200)

        # 3. Next quote on the same lane should pick up the historical competitor benchmark
        res2 = self.client.post("/api/quote", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(res2.status_code, 200)
        data2 = res2.get_json()
        intel2 = data2["competitor_intelligence"]
        self.assertEqual(intel2["source"], "historical_benchmark")
        self.assertGreaterEqual(intel2["sample_count"], 1)
        self.assertIn("competitive", data2["signals"])
        self.assertIn("lane benchmark", data2["signals"]["competitive"]["why"])

        # 4. Explicit competitor rate overrides benchmark
        payload_explicit = dict(payload)
        payload_explicit["competitor_sell_per_kg"] = 205.0
        res3 = self.client.post("/api/quote", data=json.dumps(payload_explicit), content_type="application/json")
        self.assertEqual(res3.status_code, 200)
        data3 = res3.get_json()
        intel3 = data3["competitor_intelligence"]
        self.assertEqual(intel3["source"], "explicit")
        self.assertEqual(intel3["competitor_rate_per_kg"], 205.0)


if __name__ == "__main__":
    unittest.main()

