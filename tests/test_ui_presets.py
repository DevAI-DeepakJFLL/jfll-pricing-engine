"""
tests/test_ui_presets.py
================================================================================
Automated Verification Suite for UI Operational Edge-Case Presets
Validates that all 6 operational presets are present in the frontend template
and execute correctly through the pricing engine and policy guardrails.
================================================================================
"""

import json
import re
import unittest
from src.app import app
from src.security import reset_rate_limits


class TestUIPresets(unittest.TestCase):
    """Verifies that the UI quick-test presets properly model all 6 commercial edge cases."""

    def setUp(self):
        reset_rate_limits()
        self.client = app.test_client()

    def test_presets_rendered_in_html_template(self):
        """Verify that all 6 edge-case presets appear in the rendered UI."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)

        expected_buttons = [
            "preset_btn_pharma_cold_chain",
            "preset_btn_volumetric_high_cube",
            "preset_btn_heavy_consol",
            "preset_btn_offline_transshipment",
            "preset_btn_small_parcel_courier",
            "preset_btn_dangerous_goods",
        ]
        for btn_id in expected_buttons:
            self.assertIn(btn_id, html, f"Preset button #{btn_id} missing from index.html")

        expected_keys = [
            "pharma_cold_chain",
            "volumetric_high_cube",
            "heavy_consol",
            "offline_transshipment",
            "small_parcel_courier",
            "dangerous_goods",
            "pharma_uk",
            "heavy_subagent",
            "courier_us",
        ]
        for key in expected_keys:
            self.assertIn(key, html, f"Preset data key #{key} missing from PRESET_DATA script")

    def test_preset_1_pharma_cold_chain_execution(self):
        """Scenario 1: DEL -> LHR Pharma Cold-Chain with +7.0% GDP risk buffer."""
        payload = {
            "origin": "Indira Gandhi International Airport",
            "dest": "Heathrow Apt/London",
            "customer_company": "SUN PHARMA LABORATORIES",
            "buy": 155000,
            "gross_wt": 450,
            "chargeable_wt": 450,
            "commodity": "Pharmaceuticals",
            "client": "Shipper / Consignee",
            "frequency": 10,
            "revision": 1,
            "is_preset": True,
        }
        res = self.client.post("/api/quote", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("status"), "success")
        optimal = data["optimal"]
        self.assertGreaterEqual(optimal["Margin_Percentage"], 10.0)
        self.assertGreater(optimal["Quoted_Sell_Price_INR"], 155000)

    def test_preset_2_volumetric_high_cube_execution(self):
        """Scenario 2: BOM -> FRA High-Cube Garments (180kg physical vs 640kg volumetric)."""
        payload = {
            "origin": "Chhatrapati Shivaji Maharaj International Airport",
            "dest": "Frankfurt am Main",
            "customer_company": "RAYMOND APPAREL LIMITED",
            "buy": 115200,
            "gross_wt": 180,
            "chargeable_wt": 640,
            "Cargo_Density_Ratio": 3.56,
            "commodity": "Garments / Textiles",
            "client": "Subagent",
            "frequency": 10,
            "revision": 1,
            "is_preset": True,
        }
        res = self.client.post("/api/quote", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("status"), "success")
        optimal = data["optimal"]
        # Volumetric surcharge applied (+3.46%), ensuring margin stays elevated
        self.assertGreaterEqual(optimal["Margin_Percentage"], 8.5)

    def test_preset_3_bulk_heavy_consolidation_execution(self):
        """Scenario 3: BOM -> FRA 7,500kg Bulk Consolidation subject to <= 4.50 INR/kg ceiling."""
        payload = {
            "origin": "Chhatrapati Shivaji Maharaj International Airport",
            "dest": "Frankfurt am Main",
            "customer_company": "UNIVERSAL CLEARING AND FORWARDING",
            "buy": 839325,
            "gross_wt": 7500,
            "chargeable_wt": 7500,
            "commodity": "Engineering & Machinery",
            "client": "Subagent",
            "frequency": 10,
            "revision": 1,
            "is_preset": True,
        }
        res = self.client.post("/api/quote", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("status"), "success")
        optimal = data["optimal"]
        self.assertEqual(optimal.get("Clamped_By"), "heavy_consol_unit_cap")
        self.assertLessEqual(optimal["Margin_Per_Kg"], 4.55)

    def test_preset_4_offline_transshipment_execution(self):
        """Scenario 4: BOM -> EBB (Entebbe) Remote Lane with +4.0% offline transit surcharge."""
        payload = {
            "origin": "Chhatrapati Shivaji Maharaj International Airport",
            "dest": "Entebbe",
            "customer_company": "AFRICA CARGO LINK",
            "buy": 98000,
            "gross_wt": 350,
            "chargeable_wt": 350,
            "commodity": "General Cargo",
            "client": "Subagent",
            "frequency": 10,
            "revision": 1,
            "is_preset": True,
        }
        res = self.client.post("/api/quote", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("status"), "success")
        optimal = data["optimal"]
        self.assertGreaterEqual(optimal["Margin_Percentage"], 9.0)

    def test_preset_5_small_parcel_courier_execution(self):
        """Scenario 5: BLR -> JFK 4.0kg Express Courier with AWB documentation floor fee."""
        payload = {
            "origin": "Bangalore",
            "dest": "John F. Kennedy Apt/New York",
            "customer_company": "BLR LOGISTICS OVERSEAS",
            "buy": 2400,
            "gross_wt": 4.0,
            "chargeable_wt": 4.0,
            "commodity": "Courier",
            "client": "Overseas Agent",
            "frequency": 3,
            "revision": 1,
            "is_preset": True,
        }
        res = self.client.post("/api/quote", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("status"), "success")
        optimal = data["optimal"]
        self.assertEqual(optimal.get("Clamped_By"), "cost_floor_awb")
        self.assertGreater(optimal["Quoted_Sell_Price_INR"], 2400)

    def test_preset_6_dangerous_goods_execution(self):
        """Scenario 6: BOM -> AMS 500kg Lithium Batteries (UN 3480 Class 9) with +6.0% DGR buffer."""
        payload = {
            "origin": "Chhatrapati Shivaji Maharaj International Airport",
            "dest": "Amsterdam",
            "customer_company": "EXIDE ENERGY SOLUTIONS",
            "buy": 140000,
            "gross_wt": 500,
            "chargeable_wt": 500,
            "commodity": "Dangerous Goods",
            "client": "Direct Shipper",
            "frequency": 10,
            "revision": 1,
            "is_preset": True,
        }
        res = self.client.post("/api/quote", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("status"), "success")
        optimal = data["optimal"]
        self.assertGreaterEqual(optimal["Margin_Percentage"], 11.0)


if __name__ == "__main__":
    unittest.main()
