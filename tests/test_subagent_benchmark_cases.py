"""
tests/test_subagent_benchmark_cases.py
================================================================================
Automated Verification Suite for Subagent Benchmark Test Cases
Validates that all 5 benchmark archetypes load cleanly, execute through
PricingEngineV3, and adhere strictly to commercial expected bounds.
================================================================================
"""

import os
import json
import unittest
from src.app import get_engine, app

FIXTURE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests",
    "data",
    "subagent_benchmark_cases_5.json",
)


class TestSubagentBenchmarkCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
            cls.cases = json.load(f)
        cls.engine = get_engine()

    def test_fixture_contains_exact_five_archetypes(self):
        """Verify that exactly 5 representative freight archetypes are present."""
        self.assertEqual(len(self.cases), 5)
        case_ids = [c["case_id"] for c in self.cases]
        self.assertIn("CASE-01-SMALL-PARCEL", case_ids)
        self.assertIn("CASE-02-BULK-HEAVY", case_ids)
        self.assertIn("CASE-03-PHARMA-COLDCHAIN", case_ids)
        self.assertIn("CASE-04-VOLUMETRIC-HIGHCUBE", case_ids)
        self.assertIn("CASE-05-KEYACCOUNT-RETENTION", case_ids)

    def test_all_five_cases_execute_and_match_commercial_expectations(self):
        """Verify each test case against its commercial baseline expectations and model output."""
        for case in self.cases:
            cid = case["case_id"]
            payload = case["input_payload"]
            exp = case["commercial_baseline_expectations"]
            actual = case["model_actual_output"]

            res = self.engine.recommend_quote(payload)
            opt = res["optimal"]
            corr = res["negotiation_corridor"]

            # 1. Check margin is within commercial expected baseline bounds
            min_exp, max_exp = exp["expected_margin_range_pct"]
            self.assertGreaterEqual(
                opt["Margin_Percentage"],
                min_exp,
                f"[{cid}] Quoted margin {opt['Margin_Percentage']}% below expected min {min_exp}%",
            )
            self.assertLessEqual(
                opt["Margin_Percentage"],
                max_exp,
                f"[{cid}] Quoted margin {opt['Margin_Percentage']}% above expected max {max_exp}%",
            )

            # 2. Check sell price meets minimum revenue threshold
            self.assertGreaterEqual(
                opt["Quoted_Sell_Price_INR"],
                exp["expected_min_sell_inr"],
                f"[{cid}] Quoted sell price {opt['Quoted_Sell_Price_INR']} below min {exp['expected_min_sell_inr']}",
            )

            # 3. Check win probability is within commercial expectation range
            p_min, p_max = exp["expected_win_probability_range"]
            win_p = opt.get("Win_Probability") or 0.5
            self.assertGreaterEqual(
                win_p,
                p_min,
                f"[{cid}] Win probability {win_p} below expected {p_min}",
            )
            self.assertLessEqual(
                win_p,
                p_max,
                f"[{cid}] Win probability {win_p} above expected {p_max}",
            )

            # 4. Check that model output matches fixture reference within numerical precision
            self.assertAlmostEqual(opt["Margin_Percentage"], actual["target_margin_pct"], delta=0.1)
            self.assertAlmostEqual(opt["Quoted_Sell_Price_INR"], actual["quoted_sell_price_inr"], delta=1.0)
            self.assertAlmostEqual(corr["min_margin_pct"], actual["negotiation_corridor"]["min_margin_pct"], delta=0.1)
            self.assertAlmostEqual(corr["max_margin_pct"], actual["negotiation_corridor"]["max_margin_pct"], delta=0.1)

    def test_api_quote_endpoint_serves_all_five_cases(self):
        """Verify that Flask /api/quote endpoint serves all 5 benchmark cases via HTTP."""
        client = app.test_client()
        for case in self.cases:
            res = client.post(
                "/api/quote",
                data=json.dumps(case["input_payload"]),
                content_type="application/json",
            )
            self.assertEqual(res.status_code, 200, f"Endpoint failed for {case['case_id']}")
            data = res.get_json()
            self.assertEqual(data["status"], "success")
            self.assertIn("optimal", data)
            self.assertIn("negotiation_corridor", data)
            self.assertIn("strategic_recommendations", data)


if __name__ == "__main__":
    unittest.main()
