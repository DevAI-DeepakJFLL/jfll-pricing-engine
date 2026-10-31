#!/usr/bin/env python3
"""
scripts/train_v3_2025.py
Quantile Model Retraining & Conformal Calibration Pipeline on 2025 Data.

Retrains the unified serving engine bundle from 2025 won inquiries:
- Training dataset: data/processed/won_deals_2025_train_70_pit.csv (9,515 records)
- Holdout evaluation: data/processed/won_deals_2025_test_30_pit.csv (4,078 records)
- Model: MarginQuantileModel (P10, P50, P90) with monotonic decreasing constraints
- Calibration: ConformalCorridorCalibrator (alpha=0.10, 90% empirical coverage target)
- Safety: Backs up models/engine_v3_bundle.joblib -> models/engine_v3_bundle_prev.joblib

Usage:
    python scripts/train_v3_2025.py
    python scripts/train_v3_2025.py --train data/processed/won_deals_2025_train_70_pit.csv
"""

import os
import sys
import shutil
import argparse
import datetime
from typing import Dict, Any
import joblib
import numpy as np
import pandas as pd

# Add source directory and project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from constants import MODEL_VERSION
from quantile_model import MarginQuantileModel
from conformal_corridors import ConformalCorridorCalibrator
from features_pit import CustomerPITRegistry
from engine_v3 import PricingEngineV3

DEFAULT_TRAIN_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "won_deals_2025_train_70_pit.csv")
DEFAULT_TEST_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "won_deals_2025_test_30_pit.csv")
DEFAULT_BUNDLE_PATH = os.path.join(PROJECT_ROOT, "models", "engine_v3_bundle.joblib")
DEFAULT_PREV_BUNDLE_PATH = os.path.join(PROJECT_ROOT, "models", "engine_v3_bundle_prev.joblib")
DEFAULT_REGISTRY_PATH = os.path.join(PROJECT_ROOT, "models", "customer_pit_registry_2025.joblib")


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """Computes empirical pinball (quantile) loss."""
    err = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(np.maximum(quantile * err, (quantile - 1.0) * err)))


def train_v3_pipeline(
    train_path: str = DEFAULT_TRAIN_PATH,
    test_path: str = DEFAULT_TEST_PATH,
    bundle_path: str = DEFAULT_BUNDLE_PATH,
    prev_bundle_path: str = DEFAULT_PREV_BUNDLE_PATH,
    registry_path: str = DEFAULT_REGISTRY_PATH,
    alpha: float = 0.10,
    k_prior: float = 20.0,
) -> Dict[str, Any]:
    """Executes the complete retraining and calibration pipeline."""
    print("=" * 80)
    print("Quantile Model Retraining & Conformal Calibration Pipeline (2025 Data)")
    print(f"Timestamp: {datetime.datetime.now().isoformat()}")
    print("=" * 80)

    # 1. Safety Backup of Existing Engine Bundle
    if os.path.exists(bundle_path):
        print("[*] 1. Backing up existing bundle:")
        print(f"       From: {bundle_path}")
        print(f"       To:   {prev_bundle_path}")
        shutil.copy2(bundle_path, prev_bundle_path)
        print(f"       Backup saved ({os.path.getsize(prev_bundle_path):,} bytes).")
    else:
        print(f"[*] 1. No existing bundle found at {bundle_path}; skipping backup.")

    # 2. Load and Prepare Training Data
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training dataset not found: {train_path}")

    print(f"\n[*] 2. Loading training dataset from: {train_path}")
    df_train = pd.read_csv(train_path, low_memory=False)
    print(f"       Total training records loaded: {len(df_train):,}")

    # Fallbacks for optional canonical features if missing
    if "Cargo_Density_Ratio" not in df_train.columns:
        df_train["Cargo_Density_Ratio"] = 1.0
    if "Incoterms" not in df_train.columns:
        df_train["Incoterms"] = "FOB"

    # Ensure chronological order
    sort_cols = [c for c in ["quote_date", "Year_Month"] if c in df_train.columns]
    if sort_cols:
        df_train = df_train.sort_values(sort_cols).reset_index(drop=True)

    date_min = df_train["quote_date"].min() if "quote_date" in df_train.columns else "N/A"
    date_max = df_train["quote_date"].max() if "quote_date" in df_train.columns else "N/A"
    print(f"       Date range: {date_min} to {date_max}")

    # 3. Chronological 80/20 Partitioning
    n_total = len(df_train)
    split_idx = int(n_total * 0.8)
    fit_df = df_train.iloc[:split_idx].copy().reset_index(drop=True)
    cal_df = df_train.iloc[split_idx:].copy().reset_index(drop=True)

    fit_date_range = (
        f"{fit_df['quote_date'].min()} to {fit_df['quote_date'].max()}"
        if "quote_date" in fit_df.columns
        else "N/A"
    )
    cal_date_range = (
        f"{cal_df['quote_date'].min()} to {cal_df['quote_date'].max()}"
        if "quote_date" in cal_df.columns
        else "N/A"
    )

    print("\n[*] 3. Chronological Partitioning (80% Model Fit / 20% Calibration):")
    print(f"       - Model Fitting Set : {len(fit_df):,} records ({fit_date_range})")
    print(f"       - Calibration Set   : {len(cal_df):,} records ({cal_date_range})")

    # 4. Train Stage 1A MarginQuantileModel
    print("\n[*] 4. Fitting MarginQuantileModel (P10, P50, P90) with Monotonic Constraints...")
    # Robust outlier winsorization at 45% (eliminates 10,000% anomalies)
    y_raw = fit_df["Margin_Percentage"].values
    n_clipped = int((y_raw > 45.0).sum())
    y_fit = np.clip(y_raw, 0.5, 45.0)
    print(f"       Winsorized {n_clipped} extreme margin outliers (>45%) to stabilize quantile boundaries.")

    # Time-decay recency sample weights (lambda = 0.005/day prioritizes recent market conditions)
    if "quote_date" in fit_df.columns:
        dates = pd.to_datetime(fit_df["quote_date"], errors="coerce")
        max_date = dates.max()
        days_ago = (max_date - dates).dt.days.fillna(0).values
        weights = np.exp(-0.005 * days_ago)
        print(f"       Applied exponential time-decay sample weights (lambda=0.005, recency range {days_ago.min():.0f}d - {days_ago.max():.0f}d).")
    else:
        weights = None

    model = MarginQuantileModel(
        learning_rate=0.08,
        max_iter=80,
        min_samples_leaf=30,
        l2_regularization=10.0,
        random_state=42,
    )
    model.fit(fit_df, y_fit, sample_weight=weights)
    print(f"       Model fitted successfully with {len(model.feature_cols)} features.")
    print(f"       Features: {model.feature_cols}")

    # 5. Fit Stage 1B ConformalCorridorCalibrator
    print(f"\n[*] 5. Calibrating Conformal Corridors (alpha={alpha}, k_prior={k_prior})...")
    cal_preds = model.predict(cal_df)
    cal_actual = cal_df["Margin_Percentage"].values
    cal_slabs = cal_df["Weight_Tier"].values if "Weight_Tier" in cal_df.columns else None
    cal_known = (
        (cal_df["cust_pit_won_count_180d"] > 0).values
        if "cust_pit_won_count_180d" in cal_df.columns
        else None
    )

    calibrator = ConformalCorridorCalibrator(alpha=alpha, k_prior=k_prior)
    calibrator.fit(
        y_true=cal_actual,
        y_pred=cal_preds,
        weight_tiers=cal_slabs,
        is_known_customer=cal_known,
    )
    print(f"       Global calibrated half-width: {calibrator.global_half_width:.2f} pp")
    print("       Segment half-widths (shrinkage toward global):")
    for tier, hw in sorted(calibrator.slab_half_widths.items()):
        print(f"         - Slab '{tier}': {hw:.2f} pp")
    for ctype, hw in sorted(calibrator.customer_type_half_widths.items()):
        print(f"         - Customer '{ctype}': {hw:.2f} pp")

    # 6. Customer Point-In-Time Registry
    print("\n[*] 6. Preparing CustomerPITRegistry...")
    if os.path.exists(registry_path):
        print(f"       Loading warmed registry from: {registry_path}")
        pit_registry = joblib.load(registry_path)
    else:
        print("       Constructing registry from training transactions...")
        pit_registry = CustomerPITRegistry.from_dataframe(
            df_train,
            date_col="quote_date" if "quote_date" in df_train.columns else "Year_Month",
            cust_col="customer_id" if "customer_id" in df_train.columns else "Customer_Company",
            lane_col="Regional_Lane",
        )
    print(f"       Registry customers tracked: {len(pit_registry._customers):,}")

    # 7. Compute Empirical Slab Quantiles and Reference Buy Rates
    print("\n[*] 7. Computing Slab Historical Quantiles and Reference Rates...")
    slab_quantiles: Dict[str, list] = {}
    slab_counts: Dict[str, int] = {}
    slab_ref_rates: Dict[str, float] = {}
    qs = np.linspace(0.0, 1.0, 21)

    df_train_work = df_train.copy()
    if "Margin_Amount_INR" in df_train_work.columns and "Chargeable_Weight_Kg" in df_train_work.columns:
        df_train_work["_margin_per_kg"] = (
            df_train_work["Margin_Amount_INR"] / df_train_work["Chargeable_Weight_Kg"].clip(lower=1.0)
        )
        for slab, grp in df_train_work.groupby("Weight_Tier"):
            vals = grp["_margin_per_kg"].dropna().values
            if len(vals) > 0:
                slab_quantiles[str(slab)] = np.quantile(vals, qs).tolist()
                slab_counts[str(slab)] = len(vals)

    if "Total_Buy_INR" in df_train_work.columns and "Chargeable_Weight_Kg" in df_train_work.columns:
        df_train_work["_unit_buy_rate"] = (
            df_train_work["Total_Buy_INR"] / df_train_work["Chargeable_Weight_Kg"].clip(lower=1.0)
        )
        for slab, grp in df_train_work.groupby("Weight_Tier"):
            vals = grp["_unit_buy_rate"].dropna().values
            if len(vals) > 0:
                slab_ref_rates[str(slab)] = float(np.median(vals))
                print(f"         - Slab '{slab}': count={slab_counts.get(str(slab), 0):,}, median_ref_rate=₹{slab_ref_rates[str(slab)]:.2f}/kg")

    # 8. Serialize Retrained Bundle
    print(f"\n[*] 8. Serializing retrained engine bundle to: {bundle_path}")
    os.makedirs(os.path.dirname(os.path.abspath(bundle_path)), exist_ok=True)
    bundle = {
        "model": model,
        "calibrator": calibrator,
        "pit_registry": pit_registry,
        "slab_quantiles": slab_quantiles,
        "slab_counts": slab_counts,
        "slab_ref_rates": slab_ref_rates,
        "model_version": MODEL_VERSION,
        "saved_at": datetime.datetime.now().isoformat(),
    }
    joblib.dump(bundle, bundle_path)
    bundle_size = os.path.getsize(bundle_path)
    print(f"       Bundle successfully saved ({bundle_size:,} bytes).")

    # Update cryptographic checksum registry (CWE-502 compliance)
    checksums_path = os.path.join(os.path.dirname(bundle_path), "checksums.json")
    if os.path.exists(checksums_path):
        import json
        from security import compute_sha256
        with open(checksums_path, "r", encoding="utf-8") as f:
            cdata = json.load(f)
        cdata[os.path.basename(bundle_path)] = compute_sha256(bundle_path)
        if os.path.exists(prev_bundle_path):
            cdata[os.path.basename(prev_bundle_path)] = compute_sha256(prev_bundle_path)
        with open(checksums_path, "w", encoding="utf-8") as f:
            json.dump(cdata, f, indent=2)
            f.write("\n")
        print(f"       Cryptographic checksum registry updated ({checksums_path}).")

    # 9. Statistical Evaluation on Calibration and Holdout Test Sets
    print("\n[*] 9. Evaluating Model Performance & Coverage...")
    # Calibration set metrics
    cal_p10, cal_p50, cal_p90 = model.predict_quantiles(cal_df)
    cal_pb10 = pinball_loss(cal_actual, cal_p10, 0.10)
    cal_pb50 = pinball_loss(cal_actual, cal_p50, 0.50)
    cal_pb90 = pinball_loss(cal_actual, cal_p90, 0.90)
    cal_cov_eval = calibrator.evaluate_coverage(cal_actual, cal_preds, weight_tiers=cal_slabs)
    cal_coverage = cal_cov_eval["coverage_rate"]

    # Verify Quantile Ordering
    ordering_cal = bool(((cal_p10 <= cal_p50 + 1e-9) & (cal_p50 <= cal_p90 + 1e-9)).all())

    metrics: Dict[str, Any] = {
        "cal_n": len(cal_df),
        "cal_pinball_p10": cal_pb10,
        "cal_pinball_p50": cal_pb50,
        "cal_pinball_p90": cal_pb90,
        "cal_coverage_rate": cal_coverage,
        "cal_quantile_ordering_valid": ordering_cal,
        "global_half_width": calibrator.global_half_width,
    }

    print(f"       --- Calibration Set Performance ({len(cal_df):,} records) ---")
    print(f"       P10 Pinball Loss : {cal_pb10:.4f}")
    print(f"       P50 Pinball Loss : {cal_pb50:.4f} (MAE: {cal_pb50*2:.4f} pp)")
    print(f"       P90 Pinball Loss : {cal_pb90:.4f}")
    print(f"       Empirical Cov.   : {cal_coverage*100:.2f}% (Target: >= 90.0%)")
    print(f"       Quantile Invar.  : P10 <= P50 <= P90 -> {'PASSED' if ordering_cal else 'FAILED'}")

    # Test set metrics if available
    if os.path.exists(test_path):
        df_test = pd.read_csv(test_path, low_memory=False)
        if "Cargo_Density_Ratio" not in df_test.columns:
            df_test["Cargo_Density_Ratio"] = 1.0
        if "Incoterms" not in df_test.columns:
            df_test["Incoterms"] = "FOB"

        test_actual = df_test["Margin_Percentage"].values
        test_slabs = df_test["Weight_Tier"].values if "Weight_Tier" in df_test.columns else None
        test_p10, test_p50, test_p90 = model.predict_quantiles(df_test)

        test_pb10 = pinball_loss(test_actual, test_p10, 0.10)
        test_pb50 = pinball_loss(test_actual, test_p50, 0.50)
        test_pb90 = pinball_loss(test_actual, test_p90, 0.90)
        test_cov_eval = calibrator.evaluate_coverage(test_actual, test_p50, weight_tiers=test_slabs)
        test_coverage = test_cov_eval["coverage_rate"]
        ordering_test = bool(((test_p10 <= test_p50 + 1e-9) & (test_p50 <= test_p90 + 1e-9)).all())

        metrics.update({
            "test_n": len(df_test),
            "test_pinball_p10": test_pb10,
            "test_pinball_p50": test_pb50,
            "test_pinball_p90": test_pb90,
            "test_coverage_rate": test_coverage,
            "test_quantile_ordering_valid": ordering_test,
        })

        print(f"\n       --- Out-of-Time Test Set Performance ({len(df_test):,} records) ---")
        print(f"       P10 Pinball Loss : {test_pb10:.4f}")
        print(f"       P50 Pinball Loss : {test_pb50:.4f} (MAE: {test_pb50*2:.4f} pp)")
        print(f"       P90 Pinball Loss : {test_pb90:.4f}")
        print(f"       Empirical Cov.   : {test_coverage*100:.2f}% (Target: >= 88.0%)")
        print(f"       Quantile Invar.  : P10 <= P50 <= P90 -> {'PASSED' if ordering_test else 'FAILED'}")

    # 10. PricingEngineV3 End-to-End Corridor Bandwidth Verification
    print("\n[*] 10. Sample End-to-End Corridor Predictions from PricingEngineV3:")
    engine = PricingEngineV3(bundle_path=bundle_path)

    sample_weights = [
        (35.0, "<45kg", "Courier / Small"),
        (85.0, "45-100kg", "Mid Cargo"),
        (250.0, "100-300kg", "Mid Cargo"),
        (450.0, "300-500kg", "Mid Cargo"),
        (750.0, "500-1000kg", "Heavy Cargo"),
        (1500.0, ">1000kg", "Heavy Cargo"),
    ]
    corridor_samples = []
    for wt, tier, cargo_type in sample_weights:
        inq = {
            "buy": wt * 180.0,
            "gross_wt": wt,
            "chargeable_wt": wt,
            "origin": "Chhatrapati Shivaji Maharaj International Airport",
            "dest": "Frankfurt am Main",
            "client": "Shipper / Consignee",
            "commodity": "General Cargo",
            "strategy": "balanced",
        }
        res = engine.recommend_quote(inq)
        corr = res["negotiation_corridor"]
        hw = corr["conformal_half_width_pct"]
        bw = corr["band_width_pct"]
        t_m = corr["target_margin_pct"]
        min_m = corr["min_margin_pct"]
        max_m = corr["max_margin_pct"]

        corridor_samples.append({
            "weight_kg": wt,
            "tier": tier,
            "type": cargo_type,
            "target_margin": t_m,
            "half_width": hw,
            "band_width": bw,
            "min_margin": min_m,
            "max_margin": max_m,
        })
        print(
            f"       wt={wt:6.1f}kg [{tier:9s}] ({cargo_type:15s}): "
            f"Target={t_m:5.2f}% | Spread=±{hw:4.2f} pp | Band=[{min_m:5.2f}%, {max_m:5.2f}%] (width={bw:4.2f} pp)"
        )

    metrics["corridor_samples"] = corridor_samples
    print("\n" + "=" * 80)
    print("Retraining & Calibration Complete Successfully.")
    print("=" * 80)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Retrain PricingEngineV3 on 2025 data.")
    parser.add_argument("--train", type=str, default=DEFAULT_TRAIN_PATH, help="Path to 2025 train CSV")
    parser.add_argument("--test", type=str, default=DEFAULT_TEST_PATH, help="Path to 2025 test CSV")
    parser.add_argument("--bundle-out", type=str, default=DEFAULT_BUNDLE_PATH, help="Destination bundle joblib")
    parser.add_argument("--bundle-prev", type=str, default=DEFAULT_PREV_BUNDLE_PATH, help="Backup path for old bundle")
    parser.add_argument("--registry", type=str, default=DEFAULT_REGISTRY_PATH, help="Path to customer PIT registry")
    parser.add_argument("--alpha", type=float, default=0.10, help="Conformal miscoverage rate (alpha=0.10 => 90% coverage)")
    parser.add_argument("--k-prior", type=float, default=20.0, help="Empirical Bayes prior shrinkage parameter")

    args = parser.parse_args()
    train_v3_pipeline(
        train_path=args.train,
        test_path=args.test,
        bundle_path=args.bundle_out,
        prev_bundle_path=args.bundle_prev,
        registry_path=args.registry,
        alpha=args.alpha,
        k_prior=args.k_prior,
    )


if __name__ == "__main__":
    main()
