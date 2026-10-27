"""
tests/test_api_expanded_inputs.py
Unit and integration tests for Task 6: UI & API Input Expansion in src/app.py.

Verifies:
1. /api/quote with full expanded payload (carrier, customer_company, unit_buy, quote_date, sales_person).
2. /api/quote with legacy minimal payload ensuring 100% backward compatibility.
3. Dual buy rate interactive resolution (unit_buy * chargeable_wt <-> total buy) and error states.
4. Customer intelligence lookup (known customer vs unknown spot account).
5. GET / endpoint rendering expanded UI with new inputs and strict Notion monochrome styling.
"""

import os
import sys
import json
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.app import app
from src.constants import MODEL_VERSION


class TestApiExpandedInputs(unittest.TestCase):
    """Test suite for expanded commercial inputs in Flask API & UI."""

    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_quote_with_full_expanded_payload(self):
        """Test 1: /api/quote with full expanded payload returns 200 OK and expected contract."""
        payload = {
            "carrier": "Qatar Airways",
            "customer_company": "UNIVERSAL CLEARING AND FORWARDING",
            "unit_buy": 240.0,
            "gross_wt": 100.0,
            "chargeable_wt": 100.0,
            "quote_date": "2025-10-15",
            "sales_person": "Mohnish Arora",
            "origin": "Indira Gandhi International Airport",
            "dest": "Heathrow Apt/London",
            "client": "Shipper / Consignee",
            "vertical": "Air Export Forwarding",
            "commodity": "General Cargo",
            "incoterms": "FOB",
            "branch": "Delhi",
            "strategy": "balanced"
        }

        res = self.client.post("/api/quote", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        self.assertEqual(data["status"], "success")
        self.assertEqual(data["model_version"], MODEL_VERSION)
        self.assertIn("optimal", data)

        opt = data["optimal"]
        self.assertIn("Quoted_Sell_Price_INR", opt)
        self.assertIn("Rate_Per_Kg", opt)
        self.assertIn("Margin_Percentage", opt)

        # Verify carrier and customer metadata are preserved
        carrier_val = data.get("carrier") or opt.get("Quoted_Carrier")
        self.assertEqual(carrier_val, "Qatar Airways")

        cust_val = data.get("customer_company") or opt.get("Customer_Name")
        self.assertIn("UNIVERSAL CLEARING", cust_val)

        # Expected total buy calculated from unit_buy * chargeable_wt = 240.0 * 100.0 = 24000.0
        # Optimal sell price should exceed buy cost
        self.assertGreater(opt["Quoted_Sell_Price_INR"], 24000.0)

    def test_quote_with_legacy_minimal_payload(self):
        """Test 2: /api/quote with legacy minimal payload ensuring 100% backward compatibility."""
        legacy_payload = {
            "buy": 125000,
            "gross_wt": 450,
            "origin": "Indira Gandhi International Airport",
            "dest": "Heathrow Apt/London"
        }

        res = self.client.post("/api/quote", data=json.dumps(legacy_payload), content_type="application/json")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        self.assertEqual(data["status"], "success")
        self.assertIn("optimal", data)
        self.assertIn("strategic_recommendations", data)
        self.assertIn("negotiation_corridor", data)

        opt = data["optimal"]
        self.assertGreater(opt["Quoted_Sell_Price_INR"], 125000)

        # Default fallback values for unprovided commercial inputs
        carrier_val = data.get("carrier") or opt.get("Quoted_Carrier")
        self.assertEqual(carrier_val, "Any Airline")

    def test_dual_buy_rate_resolution(self):
        """Test 3: Dual Buy Rate resolution between unit_buy and total buy."""
        # Case A: Only unit_buy is provided, total buy is computed as unit_buy * chargeable_wt
        payload_unit_only = {
            "unit_buy": 200.0,
            "gross_wt": 50.0,
            "chargeable_wt": 50.0,
            "origin": "Bangalore",
            "dest": "Frankfurt am Main"
        }
        res_a = self.client.post("/api/quote", data=json.dumps(payload_unit_only), content_type="application/json")
        self.assertEqual(res_a.status_code, 200)
        data_a = res_a.get_json()
        self.assertEqual(data_a["status"], "success")
        # Total buy is 200 * 50 = 10,000 INR; sell price should exceed 10,000 INR
        self.assertGreater(data_a["optimal"]["Quoted_Sell_Price_INR"], 10000.0)

        # Case B: Both buy and unit_buy provided (explicit total buy takes precedence or confirms)
        payload_both = {
            "buy": 50000.0,
            "unit_buy": 250.0,
            "gross_wt": 200.0,
            "chargeable_wt": 200.0,
            "origin": "Bangalore",
            "dest": "Frankfurt am Main"
        }
        res_b = self.client.post("/api/quote", data=json.dumps(payload_both), content_type="application/json")
        self.assertEqual(res_b.status_code, 200)
        data_b = res_b.get_json()
        self.assertGreater(data_b["optimal"]["Quoted_Sell_Price_INR"], 50000.0)

        # Case C: Neither buy nor unit_buy provided -> 400 Bad Request
        payload_none = {
            "gross_wt": 100.0,
            "origin": "Bangalore",
            "dest": "Frankfurt am Main"
        }
        res_c = self.client.post("/api/quote", data=json.dumps(payload_none), content_type="application/json")
        self.assertEqual(res_c.status_code, 400)
        self.assertIn("error", res_c.get_json()["status"])

        # Case D: Negative unit_buy -> 400 Bad Request
        payload_neg = {
            "unit_buy": -50.0,
            "gross_wt": 100.0,
            "origin": "Bangalore",
            "dest": "Frankfurt am Main"
        }
        res_d = self.client.post("/api/quote", data=json.dumps(payload_neg), content_type="application/json")
        self.assertEqual(res_d.status_code, 400)

    def test_customer_company_intelligence_lookup(self):
        """Test 4: Customer lookup handles both known PIT accounts and new unlisted accounts."""
        # Known customer account from PIT registry
        payload_known = {
            "buy": 100000.0,
            "gross_wt": 300.0,
            "customer_company": "UNIVERSAL CLEARING AND FORWARDING",
            "origin": "Chhatrapati Shivaji Maharaj International Airport",
            "dest": "Dubai"
        }
        res_known = self.client.post("/api/quote", data=json.dumps(payload_known), content_type="application/json")
        self.assertEqual(res_known.status_code, 200)

        # Unknown / spot customer account
        payload_unknown = {
            "buy": 100000.0,
            "gross_wt": 300.0,
            "customer_company": "COMPLETELY UNKNOWN LOGISTICS ENTERPRISE 999",
            "origin": "Chhatrapati Shivaji Maharaj International Airport",
            "dest": "Dubai"
        }
        res_unknown = self.client.post("/api/quote", data=json.dumps(payload_unknown), content_type="application/json")
        self.assertEqual(res_unknown.status_code, 200)

    def test_get_root_renders_expanded_ui(self):
        """Test 5: GET / renders HTML page with active commercial inputs and approved Notion styling."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)

        # 1. Active Customer company input and datalist present
        self.assertIn('id="customer_company"', html)
        self.assertIn('id="customer_accounts_list"', html)
        self.assertIn("UNIVERSAL CLEARING AND FORWARDING", html)

        # 2. Dual buy inputs present (unit_buy and buy)
        self.assertIn('id="unit_buy"', html)
        self.assertIn('id="buy"', html)

        # 3. Target Quote Date present
        self.assertIn('id="quote_date"', html)

        # 4. Customer Activity and Quote Revision present
        self.assertIn('id="frequency"', html)
        self.assertIn('id="revision"', html)

        # 5. Non-impacting fields removed from UI form
        self.assertNotIn('id="carrier"', html)
        self.assertNotIn('id="sales_person"', html)
        self.assertNotIn('id="incoterms"', html)
        self.assertNotIn('id="branch"', html)
        self.assertNotIn('id="vertical"', html)

        # 6. Strict Notion monochrome design validation
        self.assertIn("Outfit", html)
        self.assertIn("Mulish", html)
        self.assertIn("#0f172a", html)
        self.assertIn("#edf1f5", html)

        # 7. Strictly verify forbidden legacy colors are NOT present
        self.assertNotIn("#23c2f2", html)
        self.assertNotIn("#a7cf45", html)


if __name__ == "__main__":
    unittest.main()
