"""
tests/test_scenarios_t1_t15.py
================================================================================
Comprehensive verification of all 15 realistic air-export test scenarios
defined in Section 11 of CODEBASE_REVIEW_REPORT.md:

T1:  Rate shock pass-through (BOM->FRA 800 kg, +20% buy: Sell increases, ₹ margin does not fall)
T2:  Weight-break arbitrage (98 kg vs 101 kg DEL->DXB detected and handled)
T3:  Dangerous Goods (DG) markup (risk buffer applied, higher floor)
T4:  Perishable cold-chain / peak week (capacity/temp risk buffer applied)
T5:  Customer retention / account tiering (proper retention guardrails without leakage)
T6:  Small shipment floor (45 kg cargo adheres to >= ₹1,500/AWB commercial minimum)
T7:  Branch & revision invariance (changing branch or revision does not distort price/win prob)
T8:  Temporal determinism (explicit quote_date yields identical results regardless of wall-clock date)
T9:  Q4 peak season out-of-window warning (15 Nov flags out-of-window and lower confidence)
T10: Unseen port fallback (Tashkent falls back to regional cohort without crashing)
T11: Volumetric cargo server-side calculation (3 boxes 120x80x90 cm, chargeable >= gross)
T12: Near-cost bid protection (₹500k buy never priced below ₹ floor)
T13: Distinct Strategy Tiers (floor < balanced < premium strictly differentiated)
T14: Input validation (buy <= 0, impossible weights return HTTP 400)
T15: Audit & Preset Isolation (presets tagged, excluded from KPI stats, soft delete works)
================================================================================
"""

import os
import sys
import unittest
import json
import joblib

# Ensure local imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.constants import (
    MODEL_VERSION,
    MIN_MARGIN_PER_AWB_INR,
    MIN_MARGIN_PER_KG_INR,
    COMMODITY_RISK_BUFFERS,
    compute_cost_floor_price
)
from src.tiers import (
    check_weight_break_arbitrage,
    map_account_tier,
    get_weight_tier
)
from src.preprocess import (
    validate_shipment_physics,
    compute_volumetric_weight,
    normalize_port_name,
    map_commodity_group
)
from src.engine import MarginRecommendationEngine
from src.db import (
    init_history_db,
    log_recommendation,
    get_history_stats,
    clear_history_table,
    update_lead_status
)
from src.app import app


class TestScenariosT1ToT15(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_path = os.path.join(os.path.dirname(__file__), "..", "models", "freight_margin_recommender.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model artifact not found at {model_path}")
        cls.bundle = joblib.load(model_path)
        cls.engine = MarginRecommendationEngine(
            benchmark_reg=cls.bundle["benchmark_reg"],
            win_classifier=cls.bundle["calibrated_clf"],
            encoder=cls.bundle["encoder"],
            feature_cols=cls.bundle["feature_cols"],
            cat_cols=cls.bundle.get("native_cat_cols", cls.bundle.get("cat_cols")),
            lane_medians=cls.bundle.get("lane_medians"),
            global_median=cls.bundle.get("global_median"),
            port_frequencies=cls.bundle.get("port_frequencies"),
            cohort_table=cls.bundle.get("cohort_table")
        )
        app.config["TESTING"] = True
        cls.client = app.test_client()

    # --------------------------------------------------------------------------
    # T1: Rate Shock Pass-Through
    # --------------------------------------------------------------------------
    def test_t1_rate_shock_pass_through(self):
        """BOM->FRA 800 kg: Buy ₹160k vs ₹192k (+20%). Sell increases; sublinear ₹ margin growth (<15%)."""
        base_payload = {
            "buy": 160000.0,
            "chargeable_wt": 800.0,
            "gross_wt": 800.0,
            "origin": "Chhatrapati Shivaji Maharaj International Airport",
            "dest": "Frankfurt am Main",
            "client": "Shipper / Consignee",
            "commodity": "General Cargo",
            "quote_date": "2026-06-15",
            "strategy": "balanced"
        }
        shock_payload = dict(base_payload)
        shock_payload["buy"] = 192000.0  # +20% shock

        res_base = self.client.post("/api/quote", data=json.dumps(base_payload), content_type="application/json")
        res_shock = self.client.post("/api/quote", data=json.dumps(shock_payload), content_type="application/json")
        self.assertEqual(res_base.status_code, 200)
        self.assertEqual(res_shock.status_code, 200)

        opt_base = res_base.get_json()["optimal"]
        opt_shock = res_shock.get_json()["optimal"]

        # Quoted Sell price must strictly increase
        self.assertGreater(opt_shock["Quoted_Sell_Price_INR"], opt_base["Quoted_Sell_Price_INR"])
        # Total Gross Profit in Rupees must increase sublinearly (< 15% on +20% cost shock)
        growth = (opt_shock["Margin_Amount_INR"] / opt_base["Margin_Amount_INR"]) - 1.0
        self.assertGreater(growth, 0.0)
        self.assertLess(growth, 0.15)

    # --------------------------------------------------------------------------
    # T2: Weight-Break Arbitrage
    # --------------------------------------------------------------------------
    def test_t2_weight_break_arbitrage(self):
        """98 kg vs 101 kg DEL->DXB: Slab arbitrage checker detects break-point discount."""
        # Rate at +45kg slab is ₹120/kg -> ₹11,760 for 98 kg.
        # Airline +100kg slab rate is ₹105/kg -> 100 * 105 = ₹10,500.
        arb = check_weight_break_arbitrage(
            chargeable_wt=98.0,
            current_total_cost=11760.0,
            next_slab_rate_per_kg=105.0
        )
        self.assertTrue(arb["has_arbitrage"])
        self.assertEqual(arb["recommended_chargeable_wt"], 100.0)
        self.assertLess(arb["arbitrage_cost"], 11760.0)

        # For 101 kg, next break is 300 kg with hypothetical rate 105: no savings
        arb_above = check_weight_break_arbitrage(
            chargeable_wt=101.0,
            current_total_cost=10605.0,
            next_slab_rate_per_kg=105.0
        )
        self.assertFalse(arb_above["has_arbitrage"])

    # --------------------------------------------------------------------------
    # T3: Dangerous Goods (DG) Markup
    # --------------------------------------------------------------------------
    def test_t3_dangerous_goods_markup(self):
        """DG cargo receives positive risk buffer and higher floor."""
        self.assertIn("Dangerous Goods", COMMODITY_RISK_BUFFERS)
        dg_buffer = COMMODITY_RISK_BUFFERS["Dangerous Goods"]
        self.assertGreaterEqual(dg_buffer, 0.03)  # At least +3% risk buffer (0.06 = 6.0%)

        gen_floor = compute_cost_floor_price(100000.0, 150.0, "General Cargo")
        dg_floor = compute_cost_floor_price(100000.0, 150.0, "Dangerous Goods")
        self.assertGreater(dg_floor, gen_floor)

    # --------------------------------------------------------------------------
    # T4: Perishable Cold-Chain Risk Buffer
    # --------------------------------------------------------------------------
    def test_t4_perishable_cold_chain_buffer(self):
        """Perishable foodstuff receives temp-chain risk buffer."""
        self.assertIn("Perishable Foodstuff", COMMODITY_RISK_BUFFERS)
        perish_buffer = COMMODITY_RISK_BUFFERS["Perishable Foodstuff"]
        self.assertGreater(perish_buffer, 0.0)

        floor_gen = compute_cost_floor_price(150000.0, 1200.0, "General Cargo")
        floor_perish = compute_cost_floor_price(150000.0, 1200.0, "Perishable Foodstuff")
        self.assertGreater(floor_perish, floor_gen)

    # --------------------------------------------------------------------------
    # T5: Customer Retention / Account Tiering
    # --------------------------------------------------------------------------
    def test_t5_customer_retention_tiering(self):
        """Frequent key account receives appropriate commercial retention tiering."""
        tier_high = map_account_tier(120)
        self.assertEqual(tier_high, "Key Account")

        tier_spot = map_account_tier(1)
        self.assertEqual(tier_spot, "Spot / One-Off")

    # --------------------------------------------------------------------------
    # T6: Small Shipment Floor
    # --------------------------------------------------------------------------
    def test_t6_small_shipment_floor(self):
        """Small 45 kg shipment strictly adheres to ₹1,500/AWB commercial floor."""
        buy_cost = 5000.0
        wt = 45.0
        floor_price = compute_cost_floor_price(buy_cost, wt, "General Cargo", min_pct=2.0)
        margin_inr = floor_price - buy_cost
        self.assertGreaterEqual(margin_inr, MIN_MARGIN_PER_AWB_INR)

    # --------------------------------------------------------------------------
    # T7: Branch & Revision Invariance
    # --------------------------------------------------------------------------
    def test_t7_branch_and_revision_invariance(self):
        """Changing branch office or quote revision count does NOT alter price or win prob."""
        p1 = {
            "Total_Buy_INR": 100000.0,
            "Chargeable_Weight_Kg": 300.0,
            "Gross_Weight_Kg": 300.0,
            "Origin_Port": "Indira Gandhi International Airport",
            "Destination_Port": "Heathrow Apt/London",
            "Booked by branch": "Mumbai",
            "quote_revision_count": 1,
            "quote_date": "2026-05-10"
        }
        p2 = dict(p1)
        p2["Booked by branch"] = "Ahmedabad"
        p2["quote_revision_count"] = 5

        opt1, _ = self.engine.optimize_quote(p1)
        opt2, _ = self.engine.optimize_quote(p2)

        self.assertAlmostEqual(opt1["Quoted_Sell_Price_INR"], opt2["Quoted_Sell_Price_INR"], places=1)
        self.assertAlmostEqual(opt1["Win_Probability"], opt2["Win_Probability"], places=3)

    # --------------------------------------------------------------------------
    # T8: Temporal Determinism
    # --------------------------------------------------------------------------
    def test_t8_temporal_determinism(self):
        """Explicit quote_date yields identical deterministic recommendations."""
        payload = {
            "Total_Buy_INR": 85000.0,
            "Chargeable_Weight_Kg": 250.0,
            "Gross_Weight_Kg": 250.0,
            "Origin_Port": "Chhatrapati Shivaji Maharaj International Airport",
            "Destination_Port": "Dubai",
            "quote_date": "2026-04-12"
        }
        res1, _ = self.engine.optimize_quote(payload)
        res2, _ = self.engine.optimize_quote(payload)

        self.assertEqual(res1["Quoted_Sell_Price_INR"], res2["Quoted_Sell_Price_INR"])
        self.assertEqual(res1["Margin_Percentage"], res2["Margin_Percentage"])

    # --------------------------------------------------------------------------
    # T9: Q4 Peak Season Out-of-Window Warning
    # --------------------------------------------------------------------------
    def test_t9_q4_out_of_window_warning(self):
        """Q4 date (15 Nov 2026) flags out-of-window and sets lower confidence."""
        payload = {
            "Total_Buy_INR": 85000.0,
            "Chargeable_Weight_Kg": 250.0,
            "Gross_Weight_Kg": 250.0,
            "quote_date": "2026-11-15"
        }
        opt, _ = self.engine.optimize_quote(payload)
        self.assertTrue(opt.get("Is_Out_Of_Window", False))
        self.assertEqual(opt.get("Confidence_Level"), "Low")
        self.assertIsNotNone(opt.get("Seasonal_Warning"))

    # --------------------------------------------------------------------------
    # T10: Unseen Port Fallback
    # --------------------------------------------------------------------------
    def test_t10_unseen_port_fallback(self):
        """Unseen port (Tashkent / TAS) falls back smoothly without crashing."""
        payload = {
            "Total_Buy_INR": 60000.0,
            "Chargeable_Weight_Kg": 150.0,
            "Gross_Weight_Kg": 150.0,
            "Origin_Port": "Indira Gandhi International Airport",
            "Destination_Port": "Tashkent International Airport",
            "quote_date": "2026-06-01"
        }
        opt, df = self.engine.optimize_quote(payload)
        self.assertIsNotNone(opt)
        self.assertGreater(opt["Quoted_Sell_Price_INR"], 60000.0)
        self.assertFalse(df.empty)

    # --------------------------------------------------------------------------
    # T11: Volumetric Cargo Calculation
    # --------------------------------------------------------------------------
    def test_t11_volumetric_cargo_calculation(self):
        """3 boxes 120x80x90 cm, gross 450 kg -> Chargeable computed server-side."""
        dims = [{"length": 120, "width": 80, "height": 90, "pieces": 3}]
        # 120 * 80 * 90 * 3 / 6000 = 432.0 kg.
        # Since gross is 450 kg, chargeable must be max(450, 432) = 450 kg!
        is_valid, eff_wt, density, err = validate_shipment_physics(450.0, 0.0, dims)
        self.assertTrue(is_valid)
        self.assertEqual(eff_wt, 450.0)

        # Conversely, if gross is 400 kg, chargeable must be 432 kg!
        is_valid2, eff_wt2, density2, err2 = validate_shipment_physics(400.0, 0.0, dims)
        self.assertTrue(is_valid2)
        self.assertEqual(eff_wt2, 432.0)

    # --------------------------------------------------------------------------
    # T12: Near-Cost Bid Protection
    # --------------------------------------------------------------------------
    def test_t12_near_cost_bid_protection(self):
        """₹500,000 buy is protected by minimum commercial floor."""
        buy_cost = 500000.0
        wt = 1200.0
        floor_price = compute_cost_floor_price(buy_cost, wt, "General Cargo", min_pct=2.0)
        margin_inr = floor_price - buy_cost
        # Margin should be at least max(₹1,500, 1200 * ₹3 = ₹3,600, 2% of ₹500k = ₹10,000)
        self.assertGreaterEqual(margin_inr, 10000.0)

    # --------------------------------------------------------------------------
    # T13: Distinct Strategy Tiers
    # --------------------------------------------------------------------------
    def test_t13_distinct_strategy_tiers(self):
        """Top strategic tiers strictly differentiated (floor < balanced < premium)."""
        payload = {
            "Total_Buy_INR": 100000.0,
            "Chargeable_Weight_Kg": 400.0,
            "Gross_Weight_Kg": 400.0,
            "Origin_Port": "Indira Gandhi International Airport",
            "Destination_Port": "Heathrow Apt/London",
            "quote_date": "2026-06-15"
        }
        opt, _ = self.engine.optimize_quote(payload)
        strats = opt["strategic_recommendations"]
        floor = strats["floor"]
        balanced = strats["balanced"]
        premium = strats["premium"]

        # Strictly ascending margin percentage and sell price
        self.assertLess(floor["Margin_Percentage"], balanced["Margin_Percentage"])
        self.assertLess(balanced["Margin_Percentage"], premium["Margin_Percentage"])
        self.assertLess(floor["Quoted_Sell_Price_INR"], balanced["Quoted_Sell_Price_INR"])
        self.assertLess(balanced["Quoted_Sell_Price_INR"], premium["Quoted_Sell_Price_INR"])

    # --------------------------------------------------------------------------
    # T14: Input Validation (HTTP 400)
    # --------------------------------------------------------------------------
    def test_t14_input_validation(self):
        """Invalid inputs (negative buy, weight < 0.5 kg, weight > 25,000 kg) return HTTP 400."""
        # 1. Negative Buy
        r1 = self.client.post("/api/quote", data=json.dumps({"buy": -100, "gross_wt": 50}), content_type="application/json")
        self.assertEqual(r1.status_code, 400)

        # 2. Impossibly small weight
        r2 = self.client.post("/api/quote", data=json.dumps({"buy": 10000, "gross_wt": 0.2}), content_type="application/json")
        self.assertEqual(r2.status_code, 400)

        # 3. Impossibly large weight (charter inquiry)
        r3 = self.client.post("/api/quote", data=json.dumps({"buy": 500000, "gross_wt": 30000}), content_type="application/json")
        self.assertEqual(r3.status_code, 400)

    # --------------------------------------------------------------------------
    # T15: Audit & Preset Isolation
    # --------------------------------------------------------------------------
    def test_t15_audit_and_preset_isolation(self):
        """Presets tagged and excluded from KPI stats; soft delete works correctly."""
        db_path = "/tmp/test_scenarios_audit.db"
        if os.path.exists(db_path):
            os.remove(db_path)
        init_history_db(db_path)

        dummy_payload = {
            "Total_Buy_INR": 50000.0,
            "Chargeable_Weight_Kg": 200.0,
            "Gross_Weight_Kg": 200.0,
            "Origin_Port": "DEL",
            "Destination_Port": "LHR"
        }
        dummy_opt = {
            "Quoted_Sell_Price_INR": 55000.0,
            "Margin_Percentage": 10.0,
            "Margin_Amount_INR": 5000.0,
            "Win_Probability": 0.65,
            "Expected_Profit_INR": 3250.0,
            "Rate_Per_Kg": 275.0,
            "Cost_Floor_Price_INR": 52000.0
        }

        # 1. Log a real customer quote
        log_recommendation(db_path, dummy_payload, dummy_opt, is_preset=False)
        # 2. Log a preset test quote
        log_recommendation(db_path, dummy_payload, dummy_opt, is_preset=True)

        # Stats should only count the non-preset inquiry
        stats = get_history_stats(db_path)
        self.assertEqual(stats["total_count"], 1)

        # Update lead status to Won
        success = update_lead_status(db_path, 1, "Won / Booked")
        self.assertTrue(success)

        stats_won = get_history_stats(db_path)
        self.assertEqual(stats_won["won_count"], 1)
        self.assertAlmostEqual(stats_won["win_rate"], 100.0, places=1)

        # Soft delete
        clear_history_table(db_path, soft=True)
        stats_cleared = get_history_stats(db_path)
        self.assertEqual(stats_cleared["total_count"], 0)


if __name__ == "__main__":
    unittest.main()
