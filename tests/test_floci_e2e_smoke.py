"""
================================================================================
End-to-End Smoke Test for Containerized Air Export Pricing Engine
================================================================================
Validates live HTTP endpoints against a deployed pricing engine instance:
- Verifies UI rendering and header tags
- Verifies /api/quote REST calculations and conformal prediction corridors
- Verifies input validation HTTP 400 error handling
- Verifies /history leads dashboard access
================================================================================
"""

import os
import sys
import json
import unittest
import urllib.request
import urllib.error

TARGET_URL = os.environ.get("PRICING_ENGINE_URL", "http://localhost:5055").rstrip("/")


class TestFlociE2ESmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            req = urllib.request.Request(f"{TARGET_URL}/")
            with urllib.request.urlopen(req, timeout=1.5):
                pass
            print(f"\n[INFO] Running live E2E smoke tests against: {TARGET_URL}")
        except Exception as e:
            raise unittest.SkipTest(f"Live server not reachable at {TARGET_URL}. Skipping E2E smoke tests.")

    def test_01_ui_homepage_renders(self):
        """Test GET / returns 200 and contains signature styling."""
        req = urllib.request.Request(f"{TARGET_URL}/")
        with urllib.request.urlopen(req, timeout=10) as resp:
            self.assertEqual(resp.status, 200)
            content = resp.read().decode("utf-8")
            self.assertIn("Air Freight Margin Recommender", content)
            self.assertIn("Outfit", content)
            self.assertIn("Mulish", content)

    def test_02_quote_api_success(self):
        """Test POST /api/quote generates accurate strategic recommendations."""
        payload = {
            "origin": "Indira Gandhi International Airport",
            "dest": "Heathrow Apt/London",
            "buy": 125000,
            "gross_wt": 450,
            "chargeable_wt": 450,
            "client": "Shipper / Consignee",
            "commodity": "General Cargo",
            "strategy": "balanced"
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{TARGET_URL}/api/quote",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            self.assertEqual(resp.status, 200)
            res = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(res.get("status"), "success")
            self.assertIn("optimal", res)
            self.assertIn("negotiation_corridor", res)
            self.assertIn("strategic_recommendations", res)
            
            optimal = res["optimal"]
            self.assertGreater(optimal["Quoted_Sell_Price_INR"], payload["buy"])
            self.assertGreater(optimal["Margin_Percentage"], 0)

            corridor = res["negotiation_corridor"]
            self.assertLessEqual(corridor["min_margin_pct"], corridor["target_margin_pct"])
            self.assertLessEqual(corridor["target_margin_pct"], corridor["max_margin_pct"])

    def test_03_quote_api_invalid_input_400(self):
        """Test POST /api/quote returns HTTP 400 for invalid inputs."""
        payload = {
            "origin": "Indira Gandhi International Airport",
            "dest": "Heathrow Apt/London",
            "buy": -500, # Invalid negative buy cost
            "chargeable_wt": 100
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{TARGET_URL}/api/quote",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req, timeout=5)
        self.assertEqual(cm.exception.code, 400)

    def test_04_history_page_accessible(self):
        """Test GET /history returns 200."""
        req = urllib.request.Request(f"{TARGET_URL}/history")
        with urllib.request.urlopen(req, timeout=10) as resp:
            self.assertEqual(resp.status, 200)


if __name__ == "__main__":
    unittest.main()
