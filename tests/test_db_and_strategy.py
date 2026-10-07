#!/usr/bin/env python3
"""
tests/test_db_and_strategy.py
Unit tests verifying database performance indexes, initialization caching,
and strategy threshold consistency.
"""

import os
import sys
import tempfile
import sqlite3
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from src.constants import STRATEGY_THRESHOLDS, MODEL_VERSION
from src.db import init_history_db, _INITIALIZED_DBS, get_history_stats, get_history_paginated, log_recommendation


class TestDBAndStrategy(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_db = os.path.join(self.temp_dir.name, "test_history.db")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_strategy_thresholds_alignment(self):
        """Verify that STRATEGY_THRESHOLDS supports both modern tier keys and legacy aliases."""
        self.assertIn("floor", STRATEGY_THRESHOLDS)
        self.assertIn("balanced", STRATEGY_THRESHOLDS)
        self.assertIn("premium", STRATEGY_THRESHOLDS)
        self.assertIn("volume", STRATEGY_THRESHOLDS)
        self.assertIn("skimmer", STRATEGY_THRESHOLDS)

        self.assertEqual(STRATEGY_THRESHOLDS["floor"], 0.40)
        self.assertEqual(STRATEGY_THRESHOLDS["volume"], 0.40)
        self.assertEqual(STRATEGY_THRESHOLDS["balanced"], 0.25)
        self.assertEqual(STRATEGY_THRESHOLDS["premium"], 0.15)
        self.assertEqual(STRATEGY_THRESHOLDS["skimmer"], 0.15)

    def test_model_version_format(self):
        """Verify MODEL_VERSION format."""
        self.assertTrue(MODEL_VERSION.startswith("v2.3") or MODEL_VERSION.startswith("v2.2-panindia"))

    def test_db_initialization_caching_and_indexes(self):
        """Verify init_history_db creates all required indexes and caches path."""
        init_history_db(self.test_db, force=True)
        abs_path = os.path.abspath(self.test_db)
        self.assertIn(abs_path, _INITIALIZED_DBS)

        with sqlite3.connect(self.test_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
            indexes = {row[0] for row in cursor.fetchall()}

        self.assertIn("idx_rec_history_time", indexes)
        self.assertIn("idx_rec_history_status", indexes)
        self.assertIn("idx_rec_history_id_desc", indexes)

    def test_db_logging_and_paginated_queries(self):
        """Verify logging recommendations and pagination query paths with indexes."""
        init_history_db(self.test_db)

        payload = {
            "Origin_Port": "Indira Gandhi International Airport",
            "Destination_Port": "Heathrow Apt/London",
            "Total_Buy_INR": 100000.0,
            "Gross_Weight_Kg": 500.0,
            "Chargeable_Weight_Kg": 550.0,
            "Commodity_Group": "General Cargo",
            "Company_Group": "Shipper / Consignee",
            "Booked by branch": "Delhi"
        }
        optimal = {
            "Quoted_Sell_Price_INR": 125000.0,
            "Margin_Percentage": 20.0,
            "Margin_Amount_INR": 25000.0,
            "Win_Probability": 0.85,
            "Expected_Profit_INR": 21250.0
        }

        success = log_recommendation(self.test_db, payload, optimal, model_version=MODEL_VERSION)
        self.assertTrue(success)

        stats = get_history_stats(self.test_db)
        self.assertEqual(stats["total_count"], 1)
        self.assertEqual(stats["won_count"], 0)
        self.assertAlmostEqual(stats["total_pipeline"], 125000.0)

        paginated = get_history_paginated(self.test_db, page=1, per_page=10)
        self.assertEqual(len(paginated["records"]), 1)
        self.assertEqual(paginated["records"][0]["model_version"], MODEL_VERSION)


if __name__ == "__main__":
    unittest.main()
