"""
scripts/monitor_market_drift.py
================================================================================
Offline Market & Concept Drift Watchdog for Air Export Pricing Engine.
Runs 100% locally and offline without external network or browser automation.

Monitors:
1. Population Stability Index (PSI) on Unit Buy Cost (INR/Kg).
   - PSI < 0.10: Stable (Market consistent with baseline training regime).
   - 0.10 <= PSI < 0.25: Moderate Shift (Airline rate movements observed).
   - PSI >= 0.25: Significant Regime Shift (Model retraining recommended).
2. Kolmogorov-Smirnov (KS) Two-Sample Test for continuous rate shock detection.
3. Origin Airport Buy Rate Inflation / Deflation shifts.
4. Segment & Commodity Volume Mix Drift.
================================================================================
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy import stats

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

DEFAULT_BASELINE_DATA = os.path.join(ROOT_DIR, "data", "processed", "Air_Export_Pricing_Combined_ML.csv")


def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10) -> float:
    """
    Compute Population Stability Index (PSI) between baseline and evaluation datasets.
    Uses reference quantiles for stable binning.
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    quantiles = np.linspace(0, 100, num_buckets + 1)
    bins = np.percentile(expected, quantiles)
    bins[0] = -np.inf
    bins[-1] = np.inf
    bins = np.unique(bins)

    if len(bins) < 2:
        return 0.0

    e_counts = pd.Series(pd.cut(expected, bins=bins)).value_counts(sort=False)
    a_counts = pd.Series(pd.cut(actual, bins=bins)).value_counts(sort=False)

    # Laplace smoothing to prevent division by zero or infinite log
    e_perc = (e_counts / len(expected)).replace(0, 1e-4)
    a_perc = (a_counts / len(actual)).replace(0, 1e-4)

    psi_val = np.sum((a_perc - e_perc) * np.log(a_perc / e_perc))
    return float(psi_val)


def run_drift_analysis(
    baseline_path: str = DEFAULT_BASELINE_DATA,
    current_path: str = None,
    split_month: str = "2026-08"
):
    print("=" * 78)
    print("AIR EXPORT PRICING ENGINE — OFFLINE MARKET DRIFT WATCHDOG")
    print("=" * 78)

    if current_path and os.path.exists(current_path):
        print(f"[*] Loading Baseline Dataset: {baseline_path}")
        df_base = pd.read_csv(baseline_path)
        print(f"[*] Loading Production Audit Dataset: {current_path}")
        df_curr = pd.read_csv(current_path)
    else:
        print(f"[*] Loading Historical Reference Dataset: {baseline_path}")
        df_all = pd.read_csv(baseline_path)
        if "Year_Month" in df_all.columns:
            df_base = df_all[df_all["Year_Month"] <= split_month].copy()
            df_curr = df_all[df_all["Year_Month"] > split_month].copy()
            print(f"[*] Splitting temporal regimes at '{split_month}':")
            print(f"    - Baseline Regime (<= {split_month}): {len(df_base):,} records")
            print(f"    - Production Regime (> {split_month}) : {len(df_curr):,} records")
        else:
            n_split = int(len(df_all) * 0.8)
            df_base = df_all.iloc[:n_split].copy()
            df_curr = df_all.iloc[n_split:].copy()
            print(f"[*] 80/20 chronological partition: {len(df_base):,} base vs {len(df_curr):,} curr")

    # Compute unit buy rate
    df_base["Unit_Buy_INR_Kg"] = (df_base["Total_Buy_INR"] / df_base["Chargeable_Weight_Kg"].clip(lower=1.0)).clip(10, 8000)
    df_curr["Unit_Buy_INR_Kg"] = (df_curr["Total_Buy_INR"] / df_curr["Chargeable_Weight_Kg"].clip(lower=1.0)).clip(10, 8000)

    # 1. Global PSI on Airline Buy Cost
    psi_buy = calculate_psi(df_base["Unit_Buy_INR_Kg"].values, df_curr["Unit_Buy_INR_Kg"].values)
    ks_buy = stats.ks_2samp(df_base["Unit_Buy_INR_Kg"].dropna(), df_curr["Unit_Buy_INR_Kg"].dropna())

    # 2. Chargeable Weight Distribution PSI
    psi_wt = calculate_psi(df_base["Chargeable_Weight_Kg"].values, df_curr["Chargeable_Weight_Kg"].values)

    # Classify overall market status
    if psi_buy < 0.10:
        status_label = "STABLE"
        status_rec = "Market conditions consistent with trained baseline. Pricing model remains valid."
    elif psi_buy < 0.25:
        status_label = "MODERATE DRIFT"
        status_rec = "Airline carrier rates shifting. Closely monitor lane win-rates."
    else:
        status_label = "REGIME SHIFT"
        status_rec = "Substantial market shift detected. Trigger model retraining pipeline."

    print("\n" + "-" * 78)
    print("1. GLOBAL DISTRIBUTION STABILITY METRICS")
    print("-" * 78)
    print(f"  • Unit Buy Rate (INR/Kg) PSI : {psi_buy:.4f}  [{status_label}]")
    print(f"  • Kolmogorov-Smirnov Stat    : {ks_buy.statistic:.4f}  (p-value: {ks_buy.pvalue:.4e})")
    print(f"  • Chargeable Weight PSI      : {psi_wt:.4f}  [{'STABLE' if psi_wt < 0.10 else 'MODERATE'}]")
    print(f"  • Baseline Median Buy Rate   : ₹{df_base['Unit_Buy_INR_Kg'].median():.2f}/kg")
    print(f"  • Current Median Buy Rate    : ₹{df_curr['Unit_Buy_INR_Kg'].median():.2f}/kg "
          f"({((df_curr['Unit_Buy_INR_Kg'].median() / df_base['Unit_Buy_INR_Kg'].median()) - 1.0)*100:+.1f}%)")

    # 3. Airport-Specific Rate Shifts (BOM, DEL, BLR, CCJ)
    print("\n" + "-" * 78)
    print("2. KEY ORIGIN HUBS RATE VARIATION")
    print("-" * 78)
    print(f"  {'Origin Airport':<35} {'Base ₹/kg':>12} {'Curr ₹/kg':>12} {'Shift %':>10} {'KS p-val':>12}")
    print(f"  {'-'*35} {'-'*12} {'-'*12} {'-'*10} {'-'*12}")

    key_origins = [
        "Chhatrapati Shivaji Maharaj International Airport",
        "Indira Gandhi International Airport",
        "Bangalore",
        "Kozhikode (ex Calicut)",
        "Chennai"
    ]
    for orig in key_origins:
        b_sub = df_base[df_base["Origin_Port"] == orig]["Unit_Buy_INR_Kg"]
        c_sub = df_curr[df_curr["Origin_Port"] == orig]["Unit_Buy_INR_Kg"]
        if len(b_sub) >= 20 and len(c_sub) >= 20:
            b_med = b_sub.median()
            c_med = c_sub.median()
            pct_chg = ((c_med / b_med) - 1.0) * 100
            ks_p = stats.ks_2samp(b_sub, c_sub).pvalue
            short_name = orig[:32] + "..." if len(orig) > 35 else orig
            print(f"  {short_name:<35} {b_med:>11.2f} {c_med:>11.2f} {pct_chg:>+9.1f}% {ks_p:>12.3e}")

    # 4. Volume Mix Drift (Commodities)
    print("\n" + "-" * 78)
    print("3. COMMODITY VOLUME MIX SHIFT")
    print("-" * 78)
    b_comm = df_base["Commodity_Group"].value_counts(normalize=True) * 100
    c_comm = df_curr["Commodity_Group"].value_counts(normalize=True) * 100
    print(f"  {'Commodity Group':<28} {'Base Share':>14} {'Curr Share':>14} {'Diff (pts)':>14}")
    print(f"  {'-'*28} {'-'*14} {'-'*14} {'-'*14}")
    for comm in b_comm.index[:6]:
        b_s = b_comm.get(comm, 0.0)
        c_s = c_comm.get(comm, 0.0)
        print(f"  {comm:<28} {b_s:>13.1f}% {c_s:>13.1f}% {c_s - b_s:>+13.1f}%")

    print("\n" + "=" * 78)
    print(f"WATCHDOG VERDICT: [{status_label}]")
    print(f"GUIDANCE: {status_rec}")
    print("=" * 78 + "\n")

    return {
        "status": status_label,
        "psi_buy_rate": psi_buy,
        "ks_statistic": ks_buy.statistic,
        "ks_pvalue": ks_buy.pvalue,
        "psi_chargeable_weight": psi_wt,
        "guidance": status_rec
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline Market Drift Watchdog")
    parser.add_argument("--baseline", type=str, default=DEFAULT_BASELINE_DATA, help="Path to baseline CSV")
    parser.add_argument("--current", type=str, default=None, help="Path to production batch CSV")
    parser.add_argument("--split_month", type=str, default="2026-08", help="Temporal split Year-Month")
    args = parser.parse_args()

    run_drift_analysis(args.baseline, args.current, args.split_month)
