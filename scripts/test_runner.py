import re
import json
import joblib
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.engine import MarginRecommendationEngine

def run_tests():
    bundle = joblib.load("models/freight_margin_recommender.joblib")
    engine = MarginRecommendationEngine(
        benchmark_reg=bundle["benchmark_reg"],
        win_classifier=bundle["calibrated_clf"],
        encoder=bundle["encoder"],
        feature_cols=bundle["feature_cols"],
        cat_cols=bundle.get("native_cat_cols", bundle.get("cat_cols")),
        lane_medians=bundle.get("lane_medians"),
        global_median=bundle.get("global_median"),
        port_frequencies=bundle.get("port_frequencies"),
        cohort_table=bundle.get("cohort_table")
    )

    with open("margin_engine_test_cases.md") as f:
        text = f.read()

    cases = re.findall(
        r"## Case (\d+).*?\*\*Input\*\*\s*```json\s*(.*?)\s*```.*?\*\*Expected Output\*\*\s*```json\s*(.*?)\s*```",
        text,
        re.DOTALL
    )

    print(f"Loaded {len(cases)} test cases.\n")

    results = []
    for case_num, inp_str, exp_str in cases:
        inp = json.loads(inp_str)
        exp = json.loads(exp_str)

        # Pin reference date (2026-09-23) to prevent month-end temporal drift in regression tests
        inp.setdefault("Quote_Month", 9)
        inp.setdefault("Quote_Day", 23)
        inp.setdefault("Quote_DayOfWeek", 2)
        
        try:
            opt, df = engine.optimize_quote(inp)
            res = {
                "case": int(case_num),
                "status": "PASS",
                "diffs": [],
                "exp": exp,
                "act": {
                    "Benchmark_Margin_pct": round(opt["Benchmark_Margin"], 4),
                    "Recommended_Margin_pct": round(opt["Margin_Percentage"], 2),
                    "Quoted_Sell_Price_INR": round(opt["Quoted_Sell_Price_INR"], 2),
                    "Margin_Amount_INR": round(opt["Margin_Amount_INR"], 2),
                    "Win_Probability": round(opt["Win_Probability"], 4),
                    "Expected_Profit_INR": round(opt["Expected_Profit_INR"], 2)
                }
            }

            for k in ["Benchmark_Margin_pct", "Recommended_Margin_pct", "Quoted_Sell_Price_INR", "Margin_Amount_INR", "Win_Probability", "Expected_Profit_INR"]:
                e_val = exp.get(k)
                a_val = res["act"].get(k)
                
                # Tolerances
                if "pct" in k or "Probability" in k:
                    tol = 0.05  # 5% relative or absolute
                else:
                    tol = max(1.0, e_val * 0.02)

                abs_diff = abs(e_val - a_val)
                if abs_diff > tol:
                    res["diffs"].append((k, e_val, a_val, abs_diff))

            if res["diffs"]:
                res["status"] = "FAIL"

            results.append(res)
        except Exception as e:
            results.append({
                "case": int(case_num),
                "status": "ERROR",
                "error": str(e),
                "exp": exp,
                "act": {}
            })

    for r in results:
        print(f"=== Case {r['case']}: {r['status']} ===")
        if r['status'] == "ERROR":
            print(f"  Error: {r['error']}")
        else:
            for k in ["Benchmark_Margin_pct", "Recommended_Margin_pct", "Quoted_Sell_Price_INR", "Margin_Amount_INR", "Win_Probability", "Expected_Profit_INR"]:
                e_v = r["exp"].get(k)
                a_v = r["act"].get(k)
                diff = a_v - e_v if a_v is not None and e_v is not None else 0
                flag = "FAIL" if any(d[0] == k for d in r["diffs"]) else "OK"
                print(f"  [{flag}] {k:24}: Expected={e_v} | Actual={a_v} (Diff={diff:+.4f})")
        print()

if __name__ == "__main__":
    run_tests()
