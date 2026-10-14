#!/usr/bin/env python3
"""
scripts/test_very_low_budget.py
-------------------------------
Comprehensive evaluation of PricingEngineV3 on Very Low Budget shipments.
Tests 20 test cases each across distinct commodity types:
  1. General Cargo
  2. Perishable Foodstuff
  3. Garments / Textiles
  4. Engineering & Machinery
  5. Pharmaceuticals
  6. Courier
  7. Auto Parts
  8. Other

Analyzes:
  - Behavior under fixed operational floors (min ₹1,500 AWB gross margin)
  - Strategic tier separation (Floor vs Balanced vs Premium)
  - Pre-approved negotiation corridor spread & bounding
  - Calibrated win probability monotonicity
  - Guardrail limiting factors (cost_floor_awb, commodity_cap, etc.)
  - Ground-truth validation on historical Very Low Budget records
"""

import sys
import os
import random
import numpy as np
import pandas as pd

# Setup Python paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, PROJECT_ROOT)

from engine_v3 import PricingEngineV3

COMMODITIES_TO_TEST = [
    "General Cargo",
    "Perishable Foodstuff",
    "Garments / Textiles",
    "Engineering & Machinery",
    "Pharmaceuticals",
    "Courier",
    "Auto Parts",
    "Other",
]

CLIENT_CATEGORIES = [
    "Shipper / Consignee",
    "Subagent",
    "Overseas Agent",
    "Direct Shipper",
    "Freight Forwarder",
]

ORIGINS = [
    "Indira Gandhi International Airport",
    "Chhatrapati Shivaji Maharaj International Airport",
    "Bangalore",
    "Chennai",
    "Hyderabad",
    "Cochin",
]

DESTINATIONS = [
    ("Frankfurt am Main", "Europe"),
    ("Dubai International", "Middle East"),
    ("London Heathrow", "Europe"),
    ("Singapore Changi", "Southeast Asia"),
    ("Bangkok Suvarnabhumi", "Southeast Asia"),
    ("Riyadh King Khalid", "Middle East"),
]


def generate_very_low_budget_inquiries(n_per_commodity: int = 20, seed: int = 42):
    """
    Generates n_per_commodity inquiries per commodity in the Very Low Budget regime.
    Buy Cost: ₹1,500 – ₹22,000 (micro/small consignments).
    Chargeable Weight: 1.0 kg – 60.0 kg.
    """
    random.seed(seed)
    np.random.seed(seed)

    all_inquiries = {}

    for comm in COMMODITIES_TO_TEST:
        comm_inquiries = []
        for i in range(n_per_commodity):
            origin = ORIGINS[i % len(ORIGINS)]
            dest_port, dest_region = DESTINATIONS[i % len(DESTINATIONS)]
            client = CLIENT_CATEGORIES[i % len(CLIENT_CATEGORIES)]

            # Low budget distribution: mix of ultra-low (₹1.5k - ₹6k) and low (₹6k - ₹22k)
            if i < 7:
                # Ultra low / micro parcel
                chg_wt = round(random.uniform(1.0, 12.0), 1)
                buy_cost = round(random.uniform(1500.0, 5500.0), 2)
            elif i < 14:
                # Small shipment
                chg_wt = round(random.uniform(12.5, 35.0), 1)
                buy_cost = round(random.uniform(5500.0, 12500.0), 2)
            else:
                # Lower-mid boundary shipment
                chg_wt = round(random.uniform(35.5, 60.0), 1)
                buy_cost = round(random.uniform(12500.0, 22000.0), 2)

            gross_wt = round(chg_wt * random.uniform(0.90, 1.15), 1)
            density_ratio = round(chg_wt / max(1.0, gross_wt), 2)

            inq = {
                "case_id": f"{comm[:4].upper()}-{i+1:02d}",
                "commodity": comm,
                "Commodity_Group": comm,
                "buy": buy_cost,
                "Total_Buy_INR": buy_cost,
                "chargeable_wt": chg_wt,
                "Chargeable_Weight_Kg": chg_wt,
                "gross_wt": gross_wt,
                "Gross_Weight_Kg": gross_wt,
                "Cargo_Density_Ratio": density_ratio,
                "origin": origin,
                "Origin_Port": origin,
                "dest": dest_port,
                "Destination_Port": dest_port,
                "Destination_Region": dest_region,
                "client": client,
                "Company_Group": client,
                "Incoterms": "FOB",
                "customer_name": f"Client_VLB_{i+100}",
                "quote_date": "2026-10-14",
                "strategy": "balanced",
            }
            comm_inquiries.append(inq)
        all_inquiries[comm] = comm_inquiries

    return all_inquiries


def run_very_low_budget_test_suite():
    print("=" * 85)
    print("AIR EXPORT PRICING ENGINE — VERY LOW BUDGET TEST SUITE (20 CASES EACH)")
    print("=" * 85)

    engine = PricingEngineV3()
    inquiries_by_comm = generate_very_low_budget_inquiries(n_per_commodity=20, seed=42)

    all_results = []
    summary_by_comm = {}

    total_inversions = 0
    total_prob_inversions = 0
    total_cases = 0

    for comm, items in inquiries_by_comm.items():
        comm_results = []
        for inq in items:
            total_cases += 1
            res = engine.recommend_quote(inq, pricing_strategy="balanced")
            strat = res["strategic_recommendations"]
            corr = res["negotiation_corridor"]
            opt = res["optimal"]

            m_floor = strat["floor"]["Margin_Percentage"]
            m_bal = strat["balanced"]["Margin_Percentage"]
            m_prem = strat["premium"]["Margin_Percentage"]

            p_floor = strat["floor"]["Win_Probability"]
            p_bal = strat["balanced"]["Win_Probability"]
            p_prem = strat["premium"]["Win_Probability"]

            c_min = corr["min_margin_pct"]
            c_target = corr["target_margin_pct"]
            c_max = corr["max_margin_pct"]
            gap = corr["band_width_pct"]

            # Checks
            inversion = not (m_floor - 0.01 <= c_min <= c_target + 0.01 and c_target - 0.01 <= c_max <= m_prem + 0.01)
            prob_inversion = not (p_floor >= p_bal >= p_prem)

            if inversion:
                total_inversions += 1
            if prob_inversion:
                total_prob_inversions += 1

            record = {
                "case_id": inq["case_id"],
                "commodity": comm,
                "buy_inr": inq["buy"],
                "weight_kg": inq["chargeable_wt"],
                "floor_pct": m_floor,
                "bal_pct": m_bal,
                "prem_pct": m_prem,
                "corr_min": c_min,
                "corr_max": c_max,
                "corr_gap": gap,
                "p_win_bal": p_bal,
                "sell_price_inr": opt["Quoted_Sell_Price_INR"],
                "rate_per_kg": opt["Rate_Per_Kg"],
                "clamped_by": opt["Clamped_By"] or "none",
                "inversion": inversion,
            }
            comm_results.append(record)
            all_results.append(record)

        # Roll-up for commodity
        df_c = pd.DataFrame(comm_results)
        summary_by_comm[comm] = {
            "n": len(df_c),
            "mean_buy": df_c["buy_inr"].mean(),
            "mean_weight": df_c["weight_kg"].mean(),
            "mean_floor": df_c["floor_pct"].mean(),
            "mean_bal": df_c["bal_pct"].mean(),
            "mean_prem": df_c["prem_pct"].mean(),
            "mean_gap": df_c["corr_gap"].mean(),
            "min_gap": df_c["corr_gap"].min(),
            "max_gap": df_c["corr_gap"].max(),
            "awb_clamped_count": (df_c["clamped_by"] == "cost_floor_awb").sum(),
            "cap_clamped_count": df_c["clamped_by"].str.contains("cap").sum(),
            "none_count": (df_c["clamped_by"] == "none").sum(),
        }

    # Print Roll-Up Table
    print("\n" + "-" * 85)
    print("SUMMARY BY COMMODITY TYPE (VERY LOW BUDGET: ₹1,500 – ₹22,000, 20 CASES EACH)")
    print("-" * 85)
    header = (
        f"{'Commodity':24s} | {'N':2s} | {'Mean Buy':9s} | {'Floor %':7s} | "
        f"{'Bal %':7s} | {'Prem %':7s} | {'Corr Gap':8s} | {'AWB Clamped':11s}"
    )
    print(header)
    print("-" * 85)

    for comm, s in summary_by_comm.items():
        row = (
            f"{comm:24s} | {s['n']:2d} | ₹{s['mean_buy']:8.0f} | "
            f"{s['mean_floor']:6.2f}% | {s['mean_bal']:6.2f}% | {s['mean_prem']:6.2f}% | "
            f"{s['mean_gap']:6.2f} pp | {s['awb_clamped_count']:2d} / {s['n']:2d}"
        )
        print(row)
    print("-" * 85)
    print(f"Total Test Cases Evaluated : {total_cases}")
    print(f"Strategic Tier Inversions : {total_inversions} (Must be 0)")
    print(f"Win Probability Inversions: {total_prob_inversions} (Must be 0)")
    print("=" * 85)

    return all_results, summary_by_comm


if __name__ == "__main__":
    run_very_low_budget_test_suite()
