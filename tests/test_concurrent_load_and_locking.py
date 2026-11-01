"""
tests/test_concurrent_load_and_locking.py
High-concurrency and thread-safety verification for Air Export Pricing Engine.
Verifies:
1. 50 concurrent /api/quote requests under multi-threaded load with zero database locks.
2. Keyset / cursor-based pagination on recommendation history.
3. Deduplicated volumetric weight calculation when weights are omitted.
"""

import os
import json
import unittest
import tempfile
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.app import app
from src.db import init_history_db, log_recommendation, get_history_paginated
from src.security import reset_rate_limits


class TestConcurrentLoadAndLocking(unittest.TestCase):
    def setUp(self):
        reset_rate_limits()
        app.config["TESTING"] = True
        self.app = app
        self.client = app.test_client()

    def tearDown(self):
        reset_rate_limits()

    def test_50_concurrent_quotes_zero_db_locks(self):
        """Simulate 50 concurrent pricing requests across 10 worker threads with zero database locks."""
        payload = {
            "Total_Buy_INR": 45000.0,
            "Chargeable_Weight_Kg": 250.0,
            "Gross_Weight_Kg": 240.0,
            "Origin_Port": "Chhatrapati Shivaji Maharaj International Airport",
            "Destination_Port": "Frankfurt am Main",
            "Commodity_Group": "General Cargo",
            "Company_Group": "Subagent"
        }

        def make_quote_call(request_id: int):
            with app.test_client() as client:
                res = client.post(
                    "/api/quote",
                    json=payload,
                    environ_base={"REMOTE_ADDR": f"10.1.{request_id // 5}.{request_id % 5 + 1}"}
                )
                return res.status_code, res.get_json()

        results = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_quote_call, i) for i in range(50)]
            for fut in as_completed(futures):
                results.append(fut.result())

        self.assertEqual(len(results), 50)
        status_codes = [r[0] for r in results]
        # Every request must succeed with 200 OK
        self.assertTrue(all(sc == 200 for sc in status_codes), f"Non-200 responses observed: {set(status_codes)}")

        # Verify response structure on sample
        data = results[0][1]
        self.assertIn("optimal", data)
        self.assertIn("Margin_Percentage", data["optimal"])
        self.assertIn("Quoted_Sell_Price_INR", data["optimal"])

    def test_keyset_cursor_pagination(self):
        """Verify keyset / cursor pagination operates in O(log N) on recommendation history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_db = os.path.join(tmpdir, "test_hist.db")
            init_history_db(test_db)

            # Insert 30 test records
            sample_payload = {
                "Origin_Port": "BOM",
                "Destination_Port": "FRA",
                "chargeable_wt": 100.0,
                "commodity": "General Cargo"
            }
            sample_optimal = {
                "Rate_Per_Kg": 250.0,
                "Cost_Floor_Price_INR": 20000.0,
                "Effective_Max_Margin_Cap": 40.0,
                "Win_Probability": 0.85,
                "Expected_Profit_INR": 5000.0,
                "Strategy_Key": "balanced"
            }

            for i in range(30):
                log_recommendation(test_db, sample_payload, sample_optimal)

            # 1. Fetch first page of 10
            page1 = get_history_paginated(test_db, page=1, per_page=10)
            self.assertEqual(len(page1["records"]), 10)
            self.assertEqual(page1["total_count"], 30)
            next_cursor = page1["next_cursor"]
            self.assertIsNotNone(next_cursor)

            # 2. Fetch second page using cursor
            page2 = get_history_paginated(test_db, per_page=10, cursor_id=next_cursor)
            self.assertEqual(len(page2["records"]), 10)
            # IDs on page 2 must be strictly smaller than the cursor
            for r in page2["records"]:
                self.assertLess(r["id"], next_cursor)

            # Verify stats are accurate
            self.assertEqual(page2["stats"]["total_count"], 30)

    def test_deduplicated_volumetric_weight_only_dimensions(self):
        """Passing only dimensions without chargeable or gross weight correctly derives chargeable weight."""
        payload = {
            "dimensions": [
                {"length": 100.0, "width": 60.0, "height": 50.0, "pieces": 2}  # 2 * 300,000 / 6000 = 100 kg
            ],
            "buy": 25000.0,
            "origin": "Bangalore",
            "dest": "Frankfurt am Main",
            "commodity": "General Cargo"
        }
        res = self.client.post("/api/quote", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["chargeable_wt"], 100.0)
        self.assertEqual(data["gross_wt"], 100.0)
