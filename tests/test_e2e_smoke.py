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
import json
import unittest
import urllib.request
import urllib.error

TARGET_URL = os.environ.get("PRICING_ENGINE_URL", "http://localhost:5050").rstrip("/")


class TestE2ESmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            req = urllib.request.Request(f"{TARGET_URL}/")
            with urllib.request.urlopen(req, timeout=1.5):
                pass
            print(f"\n[INFO] Running live E2E smoke tests against: {TARGET_URL}")
        except Exception:
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
            "buy": 150000.0,
            "chargeable_wt": 500.0,
            "origin": "Chhatrapati Shivaji Maharaj International Airport",
            "dest": "Frankfurt am Main",
            "commodity": "General Cargo",
            "client": "Shipper / Consignee",
            "strategy": "balanced"
        }
        req = urllib.request.Request(
            f"{TARGET_URL}/api/quote",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data.get("status"), "success")
            self.assertIn("optimal", data)
            self.assertIn("strategic_recommendations", data["optimal"])
            self.assertIn("negotiation_corridor", data["optimal"])
            self.assertGreater(data["optimal"]["Rate_Per_Kg"], 300.0)

    def test_03_quote_api_invalid_input_400(self):
        """Test POST /api/quote rejects negative buy costs with HTTP 400."""
        payload = {
            "buy": -500.0,
            "chargeable_wt": 100.0,
            "origin": "Indira Gandhi International Airport",
            "dest": "Dubai International Airport"
        }
        req = urllib.request.Request(
            f"{TARGET_URL}/api/quote",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=10):
                self.fail("Expected HTTP 400 Bad Request")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 400)
            body = json.loads(e.read().decode("utf-8"))
            self.assertTrue("error" in body or body.get("status") == "error")

    def test_04_history_page_accessible(self):
        """Test GET /history renders quote audit dashboard."""
        req = urllib.request.Request(f"{TARGET_URL}/history")
        with urllib.request.urlopen(req, timeout=10) as resp:
            self.assertEqual(resp.status, 200)
            content = resp.read().decode("utf-8")
            self.assertIn("Recommendation History", content)


if __name__ == "__main__":
    unittest.main()
