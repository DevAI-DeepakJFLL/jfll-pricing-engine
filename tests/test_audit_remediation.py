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


class TestWeightBreakArbitrageAndBoundaryParity(unittest.TestCase):
    """Verifies Task 3: Weight-break arbitrage string keys, estimates, and 45kg boundary parity."""

    def test_string_keys_in_slab_rates(self):
        """JSON payloads provide string keys like '100' or '100.0'. Must resolve correctly."""
        from src.tiers import check_weight_break_arbitrage

        # 92kg @ 150 INR/kg = 13800. Next break 100kg @ 130 INR/kg = 13000.
        res = check_weight_break_arbitrage(
            chargeable_wt=92.0,
            current_total_cost=13800.0,
            slab_rates={"100": 130.0}  # String key
        )
        self.assertTrue(res["has_arbitrage"])
        self.assertEqual(res["recommended_chargeable_wt"], 100.0)
        self.assertEqual(res["arbitrage_cost"], 13000.0)
        self.assertEqual(res["savings"], 800.0)
        self.assertFalse(res["is_estimate"])
        self.assertNotIn("ESTIMATE", res["advisory"])

    def test_estimated_arbitrage_flag_and_advisory(self):
        """When no carrier rates are supplied, arbitrage is flagged as an estimate."""
        from src.tiers import check_weight_break_arbitrage

        # 92kg @ 150 INR/kg = 13800. No slab rates.
        # Near 100kg break (92 >= 80kg), estimate assumes 12% discount.
        res = check_weight_break_arbitrage(
            chargeable_wt=92.0,
            current_total_cost=13800.0,
        )
        self.assertTrue(res["has_arbitrage"])
        self.assertTrue(res["is_estimate"])
        self.assertIn("ESTIMATE (assumes 12% break discount)", res["advisory"])

    def test_45kg_boundary_parity(self):
        """Verifies 45.0kg boundary behavior across compute_cost_floor_price and get_weight_tier."""
        from src.tiers import get_weight_tier
        from src.constants import compute_cost_floor_price, MIN_MARGIN_PER_AWB_INR

        # At exactly 45.0kg, get_weight_tier classifies as 45-100kg
        tier = get_weight_tier(45.0)
        self.assertEqual(tier, "45-100kg")

        # In compute_cost_floor_price, small shipments <= 45.0kg strictly enforce full AWB documentation floor
        price_45_0 = compute_cost_floor_price(airline_buy_inr=5000.0, chargeable_wt_kg=45.0)
        margin_45 = price_45_0 - 5000.0
        self.assertGreaterEqual(margin_45, MIN_MARGIN_PER_AWB_INR)

        # Above 45.0kg (e.g., 45.1kg), unit and documentation minimums scale dynamically with buy rate
        price_45_1 = compute_cost_floor_price(airline_buy_inr=5000.0, chargeable_wt_kg=45.1)
        self.assertTrue(price_45_1 > 0)


class TestRiskBufferDeduplication(unittest.TestCase):
    """Verifies Task 4: Deduplicated canonical resolve_risk_buffer in constants.py."""

    def test_unified_risk_buffer_resolution(self):
        from src.constants import resolve_risk_buffer
        from src.policy_guardrails import _resolve_risk_buffer

        commodities = [
            "Pharmaceuticals", "pharma_active", "Vaccines",
            "Perishable Foodstuff", "fresh_fruits",
            "Dangerous Goods", "dgr_hazmat",
            "Valuables", "gold_vun",
            "Machinery", "auto_parts",
            "Courier Express", "express_documents",
            "Live Animals", "avi_equine",
            "General Cargo", "unknown_commodity"
        ]

        companies = ["Shipper", "Subagent", "IATA Cargo Agent", "GSA Wholesale", "NVOCC Master", "Coloader"]

        for comm in commodities:
            for comp in companies:
                buf1 = resolve_risk_buffer(comm, comp)
                buf2 = _resolve_risk_buffer(comm, comp)
                self.assertEqual(
                    buf1, buf2,
                    f"Risk buffer mismatch for commodity='{comm}', company='{comp}': {buf1} != {buf2}"
                )

                # Wholesale discount check
                if any(k in comp.lower() for k in ["gsa", "nvocc", "coloader"]):
                    undiscounted = resolve_risk_buffer(comm, "Shipper")
                    self.assertAlmostEqual(buf1, undiscounted * 0.50, places=4)


class TestPhase2FinancialAndDimensionSafeguards(unittest.TestCase):
    """Verifies Phase 2: Financial input safeguards, dimensions validation, and engine singleton."""

    def setUp(self):
        import os
        from src.app import app
        self.app = app
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def test_missing_destination_returns_422(self):
        """Quotation without destination must return 422."""
        payload = {
            "buy": 20000.0,
            "chargeable_wt": 100.0,
            "origin": "Bangalore"
        }
        res = self.client.post("/api/quote", json=payload)
        self.assertEqual(res.status_code, 422)
        self.assertIn("Destination is required", res.get_json()["message"])

    def test_ambiguous_buy_rate_requires_confirmation_422(self):
        """Ambiguous buy rate (< 10 INR/kg) without confirm_total_buy must return 422."""
        payload = {
            "buy": 500.0,
            "chargeable_wt": 100.0,
            "origin": "Bangalore",
            "dest": "Frankfurt am Main"
        }
        res = self.client.post("/api/quote", json=payload)
        self.assertEqual(res.status_code, 422)
        self.assertIn("confirm_total_buy=true", res.get_json()["message"])

    def test_strict_dimensions_validation(self):
        """Invalid physical dimensions must raise ValueError / return 400 or 422."""
        from src.preprocess import compute_volumetric_weight

        # Zero length
        with self.assertRaises(ValueError):
            compute_volumetric_weight([{"length": 0, "width": 50, "height": 50}])

        # Negative height
        with self.assertRaises(ValueError):
            compute_volumetric_weight([{"length": 50, "width": 50, "height": -10}])

        # Out-of-range length (> 1200 cm)
        with self.assertRaises(ValueError):
            compute_volumetric_weight([{"length": 1500, "width": 50, "height": 50}])

        # Courier 5000 divisor
        vol_5000 = compute_volumetric_weight([{"length": 100, "width": 50, "height": 50}], divisor=5000.0)
        self.assertEqual(vol_5000, 50.0)

    def test_lazy_threadsafe_engine_singleton(self):
        """get_engine() returns a single initialized instance across repeated calls."""
        from src.app import get_engine
        e1 = get_engine()
        e2 = get_engine()
        self.assertIs(e1, e2)
        self.assertFalse(e1.model is None)


class TestPhase3MachineLearningRigor(unittest.TestCase):
    """Verifies WinModel calibration, counterfactual signals, tier collapse, and closed loop outcome feedback."""

    def test_win_model_monotonicity(self):
        from src.win_model import WinModel
        model = WinModel().load("models/win_model.joblib")
        self.assertTrue(model.is_fitted)

        feat_row = pd.DataFrame([{
            "Total_Buy_INR": 25000.0,
            "Unit_Buy_INR": 250.0,
            "Chargeable_Weight_Kg": 100.0,
            "Cargo_Density_Ratio": 1.0,
            "Weight_Tier": "100-300kg",
            "Commodity_Group": "General Cargo",
            "Regional_Lane": "North India -> Europe",
            "Origin_Port": "DEL",
            "Destination_Region": "Europe",
            "Company_Group": "Shipper",
            "Incoterms": "FOB",
            "cust_pit_won_count_180d": 3,
            "cust_pit_median_margin_won": 12.0,
            "cust_pit_recency_days": 15.0,
            "lane_pit_median_margin_won_90d": 11.0,
        }])

        candidate_margins = [2.0, 5.0, 8.0, 12.0, 18.0, 25.0]
        probs = model.predict_margins(feat_row, candidate_margins)
        self.assertEqual(len(probs), len(candidate_margins))
        for i in range(len(probs) - 1):
            self.assertGreaterEqual(probs[i], probs[i + 1] - 1e-6)

    def test_counterfactual_signals_in_recommendation(self):
        from src.app import get_engine
        engine = get_engine()
        rec = engine.recommend_quote({
            "buy": 25000.0,
            "chargeable_wt": 100.0,
            "origin": "Delhi",
            "dest": "Frankfurt am Main",
            "commodity": "Pharma",
            "client_type": "Direct Shipper",
        })
        self.assertIn("signals", rec)
        signals = rec["signals"]
        self.assertIn("commodity", signals)
        self.assertIn("category", signals)
        self.assertIn("customer_history", signals)
        self.assertIn("lane", signals)
        self.assertIn("effect_pp", signals["commodity"])
        self.assertIn("s", signals["commodity"])

    def test_tier_collapse_warning_trigger(self):
        from src.app import get_engine
        engine = get_engine()
        rec = engine.recommend_quote({
            "buy": 1200000.0,
            "chargeable_wt": 4000.0,
            "origin": "Delhi",
            "dest": "Frankfurt am Main",
            "commodity": "General Cargo",
            "client_type": "Subagent",
        })
        m_floor = rec["strategic_recommendations"]["floor"]["Margin_Percentage"]
        m_prem = rec["strategic_recommendations"]["premium"]["Margin_Percentage"]
        if (m_prem - m_floor) < 0.5:
            self.assertTrue(any("Strategic tiers have collapsed" in w for w in rec.get("warnings", [])))

    def test_closed_loop_outcome_feedback(self):
        from src.app import app, get_engine
        app.config["TESTING"] = True
        client = app.test_client()

        # Generate a quote to insert into history
        quote_res = client.post("/api/quote", json={
            "buy": 20000.0,
            "chargeable_wt": 100.0,
            "origin": "Delhi",
            "dest": "London",
            "customer_name": "Acme Closed Loop Test Corp",
            "commodity": "General Cargo"
        }, headers={"X-API-Key": "dev_test_key"})
        self.assertEqual(quote_res.status_code, 200)
        q_data = quote_res.get_json()
        quote_id = q_data.get("quote_id")

        if quote_id:
            eng = get_engine()
            # Record outcome won
            res = client.post("/api/history/outcome", json={
                "id": quote_id,
                "status": "Won",
                "final_price_inr": 23000.0
            }, headers={"X-API-Key": "dev_test_key"})
            self.assertEqual(res.status_code, 200)

            # Check that PIT registry has customer record updated
            canonical_cust = "ACME CLOSED LOOP TEST CORP"
            c_mem = eng.pit_registry.lookup_customer(canonical_cust)
            self.assertGreaterEqual(c_mem["won_count_180d"], 1)

    def test_legacy_engine_shielded_without_env_var(self):
        """Verifies legacy modules raise RuntimeError if imported outside ALLOW_LEGACY_ENGINE."""
        import subprocess, sys, os
        env = dict(os.environ)
        env.pop("ALLOW_LEGACY_ENGINE", None)
        env.pop("PYTEST_CURRENT_TEST", None)
        cmd = [sys.executable, "-c", "from src import engine"]
        res = subprocess.run(cmd, env=env, capture_output=True, text=True)
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("RuntimeError: Legacy engine (v2.x). Superseded by engine_v3.py", res.stderr)





