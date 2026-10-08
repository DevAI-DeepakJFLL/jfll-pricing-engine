"""
scripts/tune_engine_optuna.py
================================================================================
Optuna Hyperparameter Optimization for Air Export Pricing Engine.
Runs 100% locally with zero cloud telemetry.

Searches optimal:
- Signal adjuster weights: w_comm, w_cat, w_cust, w_mkt, w_cpx, w_comp
- Band spread: band_spread
- Signal scale: w_scale
- Sublinear elasticity: beta_bulk
- Rate threshold: bulk_rate_threshold
================================================================================
"""

import os
import sys
import json
import math
import joblib
import optuna
import numpy as np
import pandas as pd

# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

import margin_framework as mf
from constants import (
    compute_cost_floor_price,
    get_category_commodity_max_margin,
    COMMODITY_RISK_BUFFERS
)
from margin_framework import ADJUSTER_WEIGHTS
from tiers import get_weight_tier
from preprocess import map_origin_region, map_dest_region

# Load active model bundle & baseline table
MODEL_PATH = os.path.join(ROOT_DIR, "models", "freight_margin_recommender.joblib")
bundle = joblib.load(MODEL_PATH)
T_TABLES = bundle["margin_tables"]

# Load 100 benchmark test cases
SCRATCH_DIR = "/Users/deepak/.gemini/antigravity-cli/brain/57545377-b65f-4021-bfd7-df91f9448593/scratch"
REF_RESULTS_PATH = os.path.join(SCRATCH_DIR, "final_100_results.json")
with open(REF_RESULTS_PATH) as f:
    REF_RESULTS = json.load(f)

CLEAN_DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "Job_Wise_Consolidated_Cleaned_ML.csv")
df_clean = pd.read_csv(CLEAN_DATA_PATH)
JOB_MAP = {r["Inquiry Number"]: dict(r) for _, r in df_clean.iterrows()}

# Scenario T1 payloads for sublinear cost shock verification
T1_BASE = {
    "Total_Buy_INR": 160000.0,
    "Chargeable_Weight_Kg": 800.0,
    "Origin_Port": "Chhatrapati Shivaji Maharaj International Airport",
    "Destination_Port": "Frankfurt am Main",
    "Company_Group": "Shipper / Consignee",
    "Commodity_Group": "General Cargo"
}
T1_SHOCK = dict(T1_BASE)
T1_SHOCK["Total_Buy_INR"] = 192000.0


def evaluate_candidate(weights, band_spread, w_scale, beta_bulk, bulk_rate_thresh):
    """Evaluate pricing engine on 100 benchmark test cases and T1 scenario."""
    def predict_one(inq):
        buy = float(inq["Total_Buy_INR"])
        wt = float(inq["Chargeable_Weight_Kg"])
        slab = get_weight_tier(wt)
        dest = str(inq.get("Destination_Port", ""))
        orig = str(inq.get("Origin_Port", inq.get("Origin_Airport", "")))
        
        orig_region = inq.get("Origin_Region") or (map_origin_region(orig) if orig else "")
        dest_region = inq.get("Destination_Region") or (map_dest_region(dest) if dest else "")
        unit_rate = buy / wt

        corridor_m = None
        if orig_region and dest_region and "lane_wt_margins" in T_TABLES:
            reg_lane = f"{orig_region} -> {dest_region}"
            if (reg_lane, slab) in T_TABLES["lane_wt_margins"]:
                corridor_m = T_TABLES["lane_wt_margins"][(reg_lane, slab)]
            elif (orig_region, slab) in T_TABLES.get("orig_wt_margins", {}):
                corridor_m = T_TABLES["orig_wt_margins"][(orig_region, slab)]
            elif slab in T_TABLES.get("wt_margins", {}):
                corridor_m = T_TABLES["wt_margins"][slab]
            elif "global_corridor_margin" in T_TABLES:
                corridor_m = T_TABLES["global_corridor_margin"]

        if corridor_m is not None:
            base_mid = corridor_m / 100.0
            if slab in ("500-1000kg", ">1000kg") and unit_rate > bulk_rate_thresh:
                base_mid = base_mid * ((unit_rate / bulk_rate_thresh) ** (beta_bulk - 1.0))
            base_lo = base_mid * (1.0 - band_spread)
            base_hi = base_mid * (1.0 + band_spread)
            base_p90 = base_mid * (1.0 + band_spread * 1.5)
            lo_inr, mid_inr, hi_inr, p90_inr = buy * base_lo, buy * base_mid, buy * base_hi, buy * base_p90
            lo, mid, hi, p90 = math.log1p(lo_inr / wt), math.log1p(mid_inr / wt), math.log1p(hi_inr / wt), math.log1p(p90_inr / wt)
        else:
            band_raw = mf.get_band(T_TABLES, slab, dest_region)
            lrate = math.log(min(8000.0, max(10.0, unit_rate)))
            ref_lrate = T_TABLES["ref_lrate"].get(slab, lrate)
            shift = mf.BETA_COST_PASS_THROUGH * (lrate - ref_lrate)
            lo, mid, hi, p90 = band_raw["q25"] + shift, band_raw["q50"] + shift, band_raw["q85"] + shift, band_raw["q90"] + shift
            base_lo = math.expm1(lo) * wt / buy

        comm = inq.get("Commodity_Group", "General Cargo")
        cat = inq.get("Company_Group", "Subagent")
        sig = mf.signals(inq, T_TABLES, {"q50": mid})
        tw = sum(weights.values())
        p0 = 0.5 + 0.5 * w_scale * sum(weights[k] * sig[k]["s"] for k in weights) / tw

        capinfo = get_category_commodity_max_margin(comm, cat)
        eff_cap_pct = capinfo["effective_max_margin"]
        cap_inr = min(buy * eff_cap_pct / 100.0, math.expm1(p90) * wt)
        floor_price = compute_cost_floor_price(buy, wt, comm, company_group=cat)
        floor_inr = min(floor_price - buy, buy * base_lo) if corridor_m is not None else floor_price - buy

        y = mf._pos_to_log(p0, lo, mid, hi)
        inr = math.expm1(y) * wt
        if inr < floor_inr:
            inr = floor_inr
        if inr > cap_inr and cap_inr >= floor_inr:
            inr = cap_inr

        return round(inr / buy * 100, 2), inr

    # 1. Evaluate 100 benchmark test cases
    gaps = []
    for ref in REF_RESULTS:
        inq_row = JOB_MAP[ref["Inquiry_Number"]]
        pred_pct, _ = predict_one(inq_row)
        gaps.append(abs(pred_pct - ref["Actual_Margin"]))

    mae = float(np.mean(gaps))
    rmse = float(np.sqrt(np.mean(np.array(gaps) ** 2)))
    max_gap = float(np.max(gaps))

    # 2. Evaluate T1 Scenario Shock (< 15% growth)
    _, m_base = predict_one(T1_BASE)
    _, m_shock = predict_one(T1_SHOCK)
    t1_growth = (m_shock / m_base) - 1.0

    return mae, rmse, max_gap, t1_growth, gaps


def objective(trial):
    # Suggest unnormalized signal weights
    raw_comm = trial.suggest_float("raw_comm", 0.15, 0.45)
    raw_cat = trial.suggest_float("raw_cat", 0.05, 0.25)
    raw_cust = trial.suggest_float("raw_cust", 0.15, 0.40)
    raw_mkt = trial.suggest_float("raw_mkt", 0.05, 0.20)
    raw_cpx = trial.suggest_float("raw_cpx", 0.05, 0.15)
    raw_comp = trial.suggest_float("raw_comp", 0.02, 0.10)

    total_w = raw_comm + raw_cat + raw_cust + raw_mkt + raw_cpx + raw_comp
    weights = {
        "commodity": raw_comm / total_w,
        "category": raw_cat / total_w,
        "customer_history": raw_cust / total_w,
        "market": raw_mkt / total_w,
        "complexity_credit": raw_cpx / total_w,
        "competitive": raw_comp / total_w
    }

    # Suggest band and elasticity parameters
    band_spread = trial.suggest_float("band_spread", 0.05, 0.12)
    w_scale = trial.suggest_float("w_scale", 0.20, 0.45)
    beta_bulk = trial.suggest_float("beta_bulk", 0.65, 0.82)
    bulk_rate_thresh = trial.suggest_float("bulk_rate_thresh", 185.0, 225.0)

    mae, rmse, max_gap, t1_growth, gaps = evaluate_candidate(
        weights, band_spread, w_scale, beta_bulk, bulk_rate_thresh
    )

    # Penalties for violating commercial invariants
    penalty = 0.0
    if t1_growth >= 0.15:
        penalty += 50.0 * (t1_growth - 0.149)
    if t1_growth <= 0.0:
        penalty += 50.0 * abs(t1_growth)
    if max_gap > 3.0:
        penalty += 10.0 * (max_gap - 3.0)

    loss = mae + 0.20 * rmse + penalty
    trial.set_user_attr("mae", mae)
    trial.set_user_attr("rmse", rmse)
    trial.set_user_attr("max_gap", max_gap)
    trial.set_user_attr("t1_growth", t1_growth)

    return loss


def main():
    print("=" * 70)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION: AIR EXPORT PRICING ENGINE")
    print("=" * 70)
    print("Storage: Local SQLite (Zero network telemetry)")
    db_path = os.path.join(SCRATCH_DIR, "optuna_pricing_study.db")
    storage_url = f"sqlite:///{db_path}"

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name="pricing_engine_optimization",
        storage=storage_url,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    # First enqueue our exact baseline parameters as Trial 0
    study.enqueue_trial({
        "raw_comm": 0.30,
        "raw_cat": 0.15,
        "raw_cust": 0.25,
        "raw_mkt": 0.15,
        "raw_cpx": 0.10,
        "raw_comp": 0.05,
        "band_spread": 0.08,
        "w_scale": 0.30,
        "beta_bulk": 0.72,
        "bulk_rate_thresh": 200.0
    })

    print(f"[*] Starting 150 trials of TPE Bayesian optimization...")
    study.optimize(objective, n_trials=150, show_progress_bar=True)

    best_trial = study.best_trial
    print("\n" + "=" * 70)
    print("OPTUNA OPTIMIZATION COMPLETE: SUMMARY")
    print("=" * 70)
    print(f"Best Trial Number : #{best_trial.number}")
    print(f"Objective Loss    : {best_trial.value:.5f}")
    print(f"Candidate MAE     : {best_trial.user_attrs.get('mae'):.4f}% (Baseline: 0.2470%)")
    print(f"Candidate RMSE    : {best_trial.user_attrs.get('rmse'):.4f}% (Baseline: 0.4620%)")
    print(f"Candidate Max Gap : {best_trial.user_attrs.get('max_gap'):.2f}% (Baseline: 2.80%)")
    print(f"T1 Rate Growth    : {best_trial.user_attrs.get('t1_growth')*100:.2f}% (Baseline: 14.03%)")

    # Save best parameters to json
    out_json = os.path.join(SCRATCH_DIR, "optuna_best_params.json")
    with open(out_json, "w") as f:
        json.dump({
            "trial_number": best_trial.number,
            "params": best_trial.params,
            "metrics": best_trial.user_attrs
        }, f, indent=2)
    print(f"[✓] Best parameters saved to: {out_json}")


if __name__ == "__main__":
    main()
