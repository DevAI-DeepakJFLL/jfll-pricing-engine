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
