"""
tests/test_db_audit.py
Unit tests for the enriched SQLite audit schema, strategy logging, soft delete, and preset filtering.
"""

import unittest
import os
import sqlite3
from src.db import (
    init_history_db,
    log_recommendation,
    get_history_stats,
    clear_history_table,
    get_all_history,
    iter_all_history
)
from src.constants import MODEL_VERSION


class TestDBAuditEnhancements(unittest.TestCase):
    def setUp(self):
        self.test_db = "data/test_audit_history.db"
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
        init_history_db(self.test_db, force=True)

    def tearDown(self):
        if os.path.exists(self.test_db):
            os.remove(self.test_db)

    def test_schema_has_enriched_columns(self):
        with sqlite3.connect(self.test_db) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(recommendation_history)")
            cols = {row[1] for row in cursor.fetchall()}

        required_cols = {
            "customer_name", "inquiry_ref", "chosen_strategy",
            "is_preset", "rate_per_kg", "cost_floor_price_inr",
            "override_reason", "is_deleted"
        }
        for col in required_cols:
            self.assertIn(col, cols, f"Enriched column '{col}' missing in audit schema")

    def test_log_chosen_strategy_and_customer(self):
        payload = {
            "Origin_Port": "Chhatrapati Shivaji Maharaj International Airport",
            "Destination_Port": "Frankfurt am Main",
            "Total_Buy_INR": 80000.0,
            "Gross_Weight_Kg": 400.0,
            "Chargeable_Weight_Kg": 400.0,
            "Commodity_Group": "Pharmaceuticals",
            "Company_Group": "Shipper / Consignee",
            "Customer_Company": "Sun Pharma Ltd",
            "Inquiry Number": "INQ-2026-9999",
            "Booked by branch": "Mumbai"
        }
        optimal = {
            "Quoted_Sell_Price_INR": 98000.0,
            "Margin_Percentage": 22.5,
            "Margin_Amount_INR": 18000.0,
            "Rate_Per_Kg": 245.0,
            "Win_Probability": 0.35,
            "Expected_Profit_INR": 6300.0,
            "Cost_Floor_Price_INR": 85000.0,
            "Strategy_Key": "premium"
        }

        success = log_recommendation(
            db_path=self.test_db,
            payload=payload,
            optimal=optimal,
            model_version=MODEL_VERSION,
            chosen_strategy="premium",
            is_preset=False
        )
        self.assertTrue(success)

        records = get_all_history(self.test_db)
        self.assertEqual(len(records), 1)
        r = records[0]
        self.assertEqual(r["customer_name"], "Sun Pharma Ltd")
        self.assertEqual(r["inquiry_ref"], "INQ-2026-9999")
        self.assertEqual(r["chosen_strategy"], "premium")
        self.assertEqual(r["rate_per_kg"], 245.0)

    def test_preset_filtering_in_kpi_stats(self):
        # 1 real quote
        payload_real = {"Total_Buy_INR": 50000, "Gross_Weight_Kg": 200, "Chargeable_Weight_Kg": 200}
        opt_real = {"Quoted_Sell_Price_INR": 60000, "Margin_Percentage": 20, "Margin_Amount_INR": 10000, "Expected_Profit_INR": 3000, "Win_Probability": 0.3}
        log_recommendation(self.test_db, payload_real, opt_real, is_preset=False)

        # 3 preset clicks
        for _ in range(3):
            payload_preset = {"Total_Buy_INR": 50000, "Gross_Weight_Kg": 200, "Chargeable_Weight_Kg": 200}
            opt_preset = {"Quoted_Sell_Price_INR": 60000, "Margin_Percentage": 20, "Margin_Amount_INR": 10000, "Expected_Profit_INR": 3000, "Win_Probability": 0.3}
            log_recommendation(self.test_db, payload_preset, opt_preset, is_preset=True)

        stats = get_history_stats(self.test_db)
        # KPI statistics should count only real quotes (1), not inflated by 3 presets
        self.assertEqual(stats["total_quotes"], 1)

    def test_soft_delete(self):
        payload = {"Total_Buy_INR": 50000, "Gross_Weight_Kg": 200, "Chargeable_Weight_Kg": 200}
        opt = {"Quoted_Sell_Price_INR": 60000, "Margin_Percentage": 20, "Margin_Amount_INR": 10000, "Expected_Profit_INR": 3000, "Win_Probability": 0.3}
        log_recommendation(self.test_db, payload, opt)

        # Soft delete
        clear_history_table(self.test_db, soft=True)
        records = get_all_history(self.test_db)
        self.assertEqual(len(records), 0)

    def test_margin_range_logging_with_corridor(self):
        payload = {
            "Customer_Company": "Tata Motors",
            "Total_Buy_INR": 100000,
            "Gross_Weight_Kg": 500,
            "Chargeable_Weight_Kg": 500
        }
        optimal = {
            "Quoted_Sell_Price_INR": 105000.0,
            "Margin_Percentage": 5.0,
            "Margin_Amount_INR": 5000.0,
            "Rate_Per_Kg": 210.0,
            "negotiation_corridor": {
                "min_margin_pct": 4.0,
                "max_margin_pct": 6.0
            }
        }
        log_recommendation(self.test_db, payload, optimal)
        records = get_all_history(self.test_db)
        r = [rec for rec in records if rec.get("customer_name") == "Tata Motors"][0]
        self.assertEqual(r["margin_range"], "4.0% – 6.0%")
        self.assertEqual(r["min_margin_pct"], 4.0)
        self.assertEqual(r["max_margin_pct"], 6.0)

    def test_iter_all_history_generator(self):
        """Verifies iter_all_history yields rows as an iterator matching get_all_history."""
        payload = {
            "Customer_Company": "Sun Pharma",
            "Total_Buy_INR": 200000,
            "Gross_Weight_Kg": 400,
            "Chargeable_Weight_Kg": 400
        }
        optimal = {
            "Quoted_Sell_Price_INR": 215000.0,
            "Margin_Percentage": 7.5,
            "Margin_Amount_INR": 15000.0,
            "Rate_Per_Kg": 537.5,
        }
        log_recommendation(self.test_db, payload, optimal)

        streamed = list(iter_all_history(self.test_db))
        fetched = get_all_history(self.test_db)
        self.assertEqual(len(streamed), len(fetched))
        self.assertEqual(streamed[0]["customer_name"], fetched[0]["customer_name"])
        self.assertEqual(streamed[0]["selling_price_inr"], fetched[0]["selling_price_inr"])


if __name__ == "__main__":
    unittest.main()
