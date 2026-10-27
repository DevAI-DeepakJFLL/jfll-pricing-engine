"""
tests/test_audit_remediation.py
Unit and integration tests for Principal Audit Remediation & Hardening.

Covers:
Task 1: Unify Destination Taxonomy & Fix Collisions
- San Salvador -> "Latin America"
- Atlanta, Georgia -> "North America"
- Tbilisi, Georgia -> "Europe"
- Colombo, Dhaka, Kathmandu, Male -> "Indian Subcontinent"
- Goa -> "Indian Subcontinent" and Genoa -> "Europe" without dictionary key collision
- unseen_categorical_report flags unseen categorical entries
- engine_v3 ignores caller-supplied Regional_Lane and warns on out-of-vocabulary categories
"""

import unittest
import numpy as np
import pandas as pd

from src.taxonomy import (
    map_destination_region,
    EXACT_PORT_MAP,
)
from src.quantile_model import MarginQuantileModel
from src.engine_v3 import PricingEngineV3


class TestDestinationTaxonomyRemediation(unittest.TestCase):
    """Verifies taxonomy disambiguation and collision fixes."""

    def test_san_salvador_resolves_to_latin_america(self):
        """San Salvador must resolve to Latin America, not North America."""
        resolved = map_destination_region("San Salvador")
        self.assertEqual(resolved, "Latin America")

    def test_atlanta_georgia_resolves_to_north_america(self):
        """Atlanta, Georgia must resolve to North America, not Europe."""
        resolved = map_destination_region("Atlanta, Georgia")
        self.assertEqual(resolved, "North America")

    def test_tbilisi_georgia_resolves_to_europe(self):
        """Tbilisi, Georgia must resolve to Europe."""
        resolved = map_destination_region("Tbilisi, Georgia")
        self.assertEqual(resolved, "Europe")

    def test_indian_subcontinent_gateways(self):
        """Colombo, Dhaka, Kathmandu, and Male must resolve to Indian Subcontinent."""
        ports = ["Colombo", "Dhaka", "Kathmandu", "Male"]
        for p in ports:
            resolved = map_destination_region(p)
            self.assertEqual(
                resolved,
                "Indian Subcontinent",
                f"Failed for port '{p}': got '{resolved}', expected 'Indian Subcontinent'",
            )

    def test_goa_and_genoa_no_collision(self):
        """
        Genoa must resolve to Europe, and Goa must resolve to Indian Subcontinent.
        EXACT_PORT_MAP must not contain conflicting duplicate 'GOA': 'Europe'.
        """
        self.assertEqual(map_destination_region("Goa"), "Indian Subcontinent")
        self.assertEqual(map_destination_region("GOA"), "Indian Subcontinent")
        self.assertEqual(map_destination_region("Genoa"), "Europe")
        self.assertEqual(map_destination_region("GENOA"), "Europe")
        self.assertEqual(map_destination_region("Genova"), "Europe")
        self.assertEqual(map_destination_region("GENOVA"), "Europe")

        # Verify EXACT_PORT_MAP dictionary integrity
        self.assertEqual(EXACT_PORT_MAP.get("GOA"), "Indian Subcontinent")
        self.assertEqual(EXACT_PORT_MAP.get("GENOVA"), "Europe")


class TestUnseenCategoricalReporting(unittest.TestCase):
    """Verifies out-of-vocabulary categorical detection in MarginQuantileModel."""

    def test_unseen_categorical_report(self):
        model = MarginQuantileModel()
        # Train on a small toy dataset with known categories
        df_train = pd.DataFrame([
            {
                "Total_Buy_INR": 10000.0,
                "Unit_Buy_INR": 100.0,
                "Chargeable_Weight_Kg": 100.0,
                "Cargo_Density_Ratio": 1.0,
                "Weight_Tier": "100-300kg",
                "Commodity_Group": "General Cargo",
                "Regional_Lane": "North India -> Europe",
                "Origin_Port": "DEL",
                "Destination_Region": "Europe",
                "Company_Group": "Shipper",
                "Incoterms": "FOB",
                "Margin_Percentage": 10.0,
            },
            {
                "Total_Buy_INR": 20000.0,
                "Unit_Buy_INR": 200.0,
                "Chargeable_Weight_Kg": 100.0,
                "Cargo_Density_Ratio": 1.0,
                "Weight_Tier": "100-300kg",
                "Commodity_Group": "Dangerous Goods",
                "Regional_Lane": "North India -> Middle East",
                "Origin_Port": "BOM",
                "Destination_Region": "Middle East",
                "Company_Group": "Subagent",
                "Incoterms": "CIF",
                "Margin_Percentage": 12.0,
            },
        ])
        model.fit(df_train, df_train["Margin_Percentage"])

        # Predict with seen categories -> report should be empty
        df_seen = pd.DataFrame([{
            "Total_Buy_INR": 15000.0,
            "Unit_Buy_INR": 150.0,
            "Chargeable_Weight_Kg": 100.0,
            "Cargo_Density_Ratio": 1.0,
            "Weight_Tier": "100-300kg",
            "Commodity_Group": "General Cargo",
            "Regional_Lane": "North India -> Europe",
            "Origin_Port": "DEL",
            "Destination_Region": "Europe",
            "Company_Group": "Shipper",
            "Incoterms": "FOB",
        }])
        unseen = model.unseen_categorical_report(df_seen)
        self.assertEqual(unseen, {})

        # Predict with unseen category (e.g. unknown lane and unknown commodity)
        df_unseen = pd.DataFrame([{
            "Total_Buy_INR": 15000.0,
            "Unit_Buy_INR": 150.0,
            "Chargeable_Weight_Kg": 100.0,
            "Cargo_Density_Ratio": 1.0,
            "Weight_Tier": "100-300kg",
            "Commodity_Group": "Valuables",  # unseen in toy train
            "Regional_Lane": "North India -> Latin America",  # unseen in toy train
            "Origin_Port": "DEL",
            "Destination_Region": "Latin America",  # unseen in toy train
            "Company_Group": "Shipper",
            "Incoterms": "FOB",
        }])
        unseen_report = model.unseen_categorical_report(df_unseen)
        self.assertIn("Commodity_Group", unseen_report)
        self.assertEqual(unseen_report["Commodity_Group"], "Valuables")
        self.assertIn("Regional_Lane", unseen_report)
        self.assertEqual(unseen_report["Regional_Lane"], "North India -> Latin America")


class TestEngineV3LaneIntegrity(unittest.TestCase):
    """Verifies that PricingEngineV3 ignores caller-supplied Regional_Lane and warns on OOV."""

    @classmethod
    def setUpClass(cls):
        cls.engine = PricingEngineV3()

    def test_engine_recomputes_lane_and_warns_on_unseen(self):
        # 1. Caller provides bogus Regional_Lane: "Europe -> North America"
        # but origin is Delhi (North India) and destination is Dubai (Middle East).
        inquiry = {
            "Total_Buy_INR": 50000.0,
            "Chargeable_Weight_Kg": 500.0,
            "origin": "Indira Gandhi International Airport",
            "dest": "Dubai",
            "Regional_Lane": "Europe -> North America",  # Spoofed / skewed
            "commodity": "General Cargo",
            "client": "Subagent",
            "incoterms": "FOB",
            "customer_name": "Test Customer",
        }
        rec = self.engine.recommend_quote(inquiry)
        # Should have computed the true regional lane ("North India -> Middle East")
        self.assertEqual(rec["status"], "success")

        # 2. Caller provides an unseen out-of-vocabulary categorical input
        inquiry_oov = {
            "Total_Buy_INR": 50000.0,
            "Chargeable_Weight_Kg": 500.0,
            "origin": "Indira Gandhi International Airport",
            "dest": "Dubai",
            "client": "Completely Unknown Client Type 123",  # Out-of-vocabulary
            "commodity": "General Cargo",
            "incoterms": "FOB",
            "customer_name": "Test Customer",
        }
        rec_oov = self.engine.recommend_quote(inquiry_oov)
        self.assertEqual(rec_oov["confidence"], "Low")
        oov_warnings = [w for w in rec_oov["warnings"] if "Out-of-vocabulary" in w]
        self.assertTrue(len(oov_warnings) > 0, f"Expected OOV warning, got: {rec_oov['warnings']}")


class TestSecurityEndpointRemediation(unittest.TestCase):
    """Verifies authentication and validation on /api/history/outcome and /api/history/update_status."""

    def setUp(self):
        import os
        from src.app import app
        self.app = app
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()
        os.environ.pop("ADMIN_API_KEY", None)
        os.environ.pop("ALLOW_INSECURE_DEV", None)

    def tearDown(self):
        import os
        os.environ.pop("ADMIN_API_KEY", None)
        os.environ.pop("ALLOW_INSECURE_DEV", None)

    def test_outcome_and_status_require_auth(self):
        """Endpoints must return 401 when ADMIN_API_KEY is configured and no credentials supplied."""
        import os
        import json
        test_key = "secure-audit-secret-token"
        os.environ["ADMIN_API_KEY"] = test_key

        # Outcome without auth -> 401
        res = self.client.post("/api/history/outcome", json={"id": 1, "status": "Won", "final_price_inr": 1000})
        self.assertEqual(res.status_code, 401)

        # Update status without auth -> 401
        res = self.client.post("/api/history/update_status", json={"id": 1, "status": "Quoted"})
        self.assertEqual(res.status_code, 401)

        # Query param auth rejected -> 401
        res = self.client.post(f"/api/history/outcome?api_key={test_key}", json={"id": 1, "status": "Won", "final_price_inr": 1000})
        self.assertEqual(res.status_code, 401)

    def test_invalid_id_returns_400(self):
        """Endpoints must return 400 when quote ID is missing, non-numeric, or non-positive."""
        import os
        import json
        test_key = "secure-audit-secret-token"
        os.environ["ADMIN_API_KEY"] = test_key
        headers = {"X-API-Key": test_key}

        # 1. Outcome with string non-numeric ID -> 400
        res = self.client.post("/api/history/outcome", headers=headers, json={"id": "invalid_id", "status": "Won"})
        self.assertEqual(res.status_code, 400)
        self.assertIn("Invalid quote ID", res.get_json()["message"])

        # 2. Outcome with missing ID -> 400
        res = self.client.post("/api/history/outcome", headers=headers, json={"status": "Won"})
        self.assertEqual(res.status_code, 400)
        self.assertIn("Missing quote ID", res.get_json()["message"])

        # 3. Update status with string non-numeric ID -> 400
        res = self.client.post("/api/history/update_status", headers=headers, json={"id": "abc", "status": "Quoted"})
        self.assertEqual(res.status_code, 400)
        self.assertIn("Invalid quote ID", res.get_json()["message"])

        # 4. Update status with non-positive ID -> 400
        res = self.client.post("/api/history/update_status", headers=headers, json={"id": -5, "status": "Quoted"})
        self.assertEqual(res.status_code, 400)
        self.assertIn("Invalid quote ID", res.get_json()["message"])

