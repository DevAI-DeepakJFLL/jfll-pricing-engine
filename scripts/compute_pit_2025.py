"""
scripts/compute_pit_2025.py
Point-in-Time (PIT) Rolling Feature Engineering & Registry Generation for 2025 Data.

Computes strictly leak-free point-in-time rolling customer and regional lane features on:
- data/processed/won_deals_2025_train_70.csv -> data/processed/won_deals_2025_train_70_pit.csv
- data/processed/won_deals_2025_test_30.csv  -> data/processed/won_deals_2025_test_30_pit.csv
- models/customer_pit_registry_2025.joblib   (warmed with training data through 2025-09-26)

Key Principles:
1. Zero Temporal Leakage: Every transaction at date T only sees historical records where t < T.
2. 180-Day Continuity: Test split quotes (e.g. October-December 2025) look back up to 180 days
   into the training split without seeing future test quotes.
3. Production Serving: Serializes CustomerPITRegistry for high-throughput runtime inference.
"""

import os
import sys
import argparse
from typing import Dict, Any
import joblib
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features_pit import compute_point_in_time_features, CustomerPITRegistry
from src.preprocess_pipeline_v3 import validate_build_gates

DEFAULT_TRAIN_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "won_deals_2025_train_70.csv")
DEFAULT_TEST_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "won_deals_2025_test_30.csv")
DEFAULT_TRAIN_OUT = os.path.join(PROJECT_ROOT, "data", "processed", "won_deals_2025_train_70_pit.csv")
DEFAULT_TEST_OUT = os.path.join(PROJECT_ROOT, "data", "processed", "won_deals_2025_test_30_pit.csv")
DEFAULT_REGISTRY_OUT = os.path.join(PROJECT_ROOT, "models", "customer_pit_registry_2025.joblib")

PIT_COLS = [
    "cust_pit_won_count_180d",
    "cust_pit_inquiry_count_180d",
    "cust_pit_recency_days",
    "cust_pit_median_margin_won",
    "cust_pit_total_spend_inr_180d",
    "cust_pit_win_rate_180d",
    "lane_pit_median_margin_won_90d",
]


def compute_pit_pipeline(
    train_path: str = DEFAULT_TRAIN_PATH,
    test_path: str = DEFAULT_TEST_PATH,
    train_out_path: str = DEFAULT_TRAIN_OUT,
    test_out_path: str = DEFAULT_TEST_OUT,
    registry_out_path: str = DEFAULT_REGISTRY_OUT,
    include_stratified: bool = False,
) -> Dict[str, Any]:
    """
    Executes the PIT feature computation pipeline on 2025 data.
    """
    print("=" * 80)
    print("Point-in-Time (PIT) Rolling Feature Engineering Pipeline (2025 Data)")
    print("=" * 80)

    # 1. Load Chronological Splits
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training dataset not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Holdout test dataset not found: {test_path}")

    print("[*] 1. Loading splits from:")
    print(f"       Train: {train_path}")
    print(f"       Test:  {test_path}")

    train_df = pd.read_csv(train_path, low_memory=False)
    test_df = pd.read_csv(test_path, low_memory=False)

    print(f"       Train records: {len(train_df):,} | Test records: {len(test_df):,}")

    # Ensure chronological sort ordering
    sort_cols = [c for c in ["quote_date", "Created At"] if c in train_df.columns]
    train_df = train_df.sort_values(by=sort_cols, ascending=True).reset_index(drop=True)
    test_df = test_df.sort_values(by=sort_cols, ascending=True).reset_index(drop=True)

    print(f"       Train range: {train_df['quote_date'].min()} to {train_df['quote_date'].max()}")
    print(f"       Test range:  {test_df['quote_date'].min()} to {test_df['quote_date'].max()}")

    # 2. Compute PIT Features on Training Split (Stand-alone, strictly t < T)
    print("\n[*] 2. Computing leak-free PIT features for Train (70%) split...")
    train_pit = compute_point_in_time_features(
        train_df,
        date_col="quote_date",
        cust_col="customer_id",
        lane_col="Regional_Lane",
    )

    # 3. Compute PIT Features on Holdout Test Split with 180d Lookback into Train Split
    print("\n[*] 3. Computing leak-free PIT features for Holdout Test (30%) split (with 180d lookback)...")
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    combined_pit = compute_point_in_time_features(
        combined_df,
        date_col="quote_date",
        cust_col="customer_id",
        lane_col="Regional_Lane",
    )
    test_pit = combined_pit.iloc[len(train_df):].copy().reset_index(drop=True)

    # 4. Serialize CustomerPITRegistry from Training Split
    print("\n[*] 4. Warming and serializing CustomerPITRegistry from Train (70%) split...")
    registry = CustomerPITRegistry.from_dataframe(
        train_df,
        date_col="quote_date",
        cust_col="customer_id",
        lane_col="Regional_Lane",
    )

    os.makedirs(os.path.dirname(os.path.abspath(registry_out_path)), exist_ok=True)
    joblib.dump(registry, registry_out_path)
    print(f"       Registry successfully serialized to: {registry_out_path}")
    print(f"       Registry size on disk: {os.path.getsize(registry_out_path):,} bytes")

    # 5. Verify Registry Query for Known Top Customer
    test_cust = "CUST_UNIVERSAL_CLEARING_AND_FORWARDING"
    test_date = "2025-09-27"
    test_lane = "West India -> Europe"
    print(f"\n[*] 5. Verifying CustomerPITRegistry lookup for '{test_cust}' on {test_date}...")
    sample_stats = registry.get_features(
        quote_date=test_date,
        customer_id=test_cust,
        regional_lane=test_lane,
    )
    print("       Active Customer Query Output:")
    for k, v in sample_stats.items():
        print(f"         - {k}: {v}")

    # Unknown Customer verification
    unknown_stats = registry.get_features(
        quote_date=test_date,
        customer_id="CUST_NONEXISTENT_CARGO",
        regional_lane="Unknown Lane",
    )
    print("       Unknown Customer Query Output (Priors):")
    for k, v in unknown_stats.items():
        print(f"         - {k}: {v}")

    # 6. Hard Build Gate Validation
    print("\n[*] 6. Validating hard build gates and PIT column integrity...")
    validate_build_gates(
        train_pit,
        is_margin_dataset=True,
        dataset_name=os.path.basename(train_out_path),
    )
    validate_build_gates(
        test_pit,
        is_margin_dataset=True,
        dataset_name=os.path.basename(test_out_path),
    )

    for name, df in [("train", train_pit), ("test", test_pit)]:
        for col in PIT_COLS:
            if col not in df.columns:
                raise ValueError(f"Missing PIT column '{col}' in {name} output.")
            if not np.issubdtype(df[col].dtype, np.number):
                raise TypeError(f"PIT column '{col}' in {name} is not numeric.")
        for bcol in [
            "cust_pit_won_count_180d",
            "cust_pit_inquiry_count_180d",
            "cust_pit_recency_days",
            "cust_pit_total_spend_inr_180d",
            "cust_pit_win_rate_180d",
        ]:
            if df[bcol].isnull().any():
                raise ValueError(f"Found unexpected null values in {name} column '{bcol}'.")

    print("       [✓] Hard build validation gates and PIT schema checks PASSED.")

    # 7. Persist Output CSV Files
    print("\n[*] 7. Persisting PIT datasets to disk...")
    os.makedirs(os.path.dirname(os.path.abspath(train_out_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(test_out_path)), exist_ok=True)

    train_pit.to_csv(train_out_path, index=False)
    test_pit.to_csv(test_out_path, index=False)

    print(f"       - Train PIT: {train_out_path} ({len(train_pit):,} rows)")
    print(f"       - Test PIT:  {test_out_path} ({len(test_pit):,} rows)")

    # Optional: Stratified processing if requested
    strat_results = {}
    if include_stratified:
        train_strat_path = os.path.join(PROJECT_ROOT, "data", "processed", "won_deals_2025_train_70_stratified.csv")
        test_strat_path = os.path.join(PROJECT_ROOT, "data", "processed", "won_deals_2025_test_30_stratified.csv")
        train_strat_out = os.path.join(PROJECT_ROOT, "data", "processed", "won_deals_2025_train_70_stratified_pit.csv")
        test_strat_out = os.path.join(PROJECT_ROOT, "data", "processed", "won_deals_2025_test_30_stratified_pit.csv")

        if os.path.exists(train_strat_path) and os.path.exists(test_strat_path):
            print("\n[*] Processing stratified datasets with PIT features...")
            train_strat_df = pd.read_csv(train_strat_path, low_memory=False)
            test_strat_df = pd.read_csv(test_strat_path, low_memory=False)

            train_strat_pit = compute_point_in_time_features(
                train_strat_df,
                date_col="quote_date",
                cust_col="customer_id",
                lane_col="Regional_Lane",
            )
            comb_strat = pd.concat([train_strat_df, test_strat_df], ignore_index=True)
            comb_strat_pit = compute_point_in_time_features(
                comb_strat,
                date_col="quote_date",
                cust_col="customer_id",
                lane_col="Regional_Lane",
            )
            test_strat_pit = comb_strat_pit.iloc[len(train_strat_df):].copy().reset_index(drop=True)

            validate_build_gates(train_strat_pit, is_margin_dataset=True, dataset_name="train_stratified_pit")
            validate_build_gates(test_strat_pit, is_margin_dataset=True, dataset_name="test_stratified_pit")

            train_strat_pit.to_csv(train_strat_out, index=False)
            test_strat_pit.to_csv(test_strat_out, index=False)
            print(f"       - Stratified Train PIT: {train_strat_out} ({len(train_strat_pit):,} rows)")
            print(f"       - Stratified Test PIT:  {test_strat_out} ({len(test_strat_pit):,} rows)")
            strat_results = {
                "train_strat_count": len(train_strat_pit),
                "test_strat_count": len(test_strat_pit),
            }

    # Summary Statistics
    summary_stats = {
        "train_rows": len(train_pit),
        "test_rows": len(test_pit),
        "train_date_range": (train_pit["quote_date"].min(), train_pit["quote_date"].max()),
        "test_date_range": (test_pit["quote_date"].min(), test_pit["quote_date"].max()),
        "train_active_customer_rows": int((train_pit["cust_pit_won_count_180d"] > 0).sum()),
        "test_active_customer_rows": int((test_pit["cust_pit_won_count_180d"] > 0).sum()),
        "train_median_margin_overall": float(train_pit["cust_pit_median_margin_won"].dropna().median()),
        "test_median_margin_overall": float(test_pit["cust_pit_median_margin_won"].dropna().median()),
        "top_customer_stats": sample_stats,
        "registry_path": registry_out_path,
        **strat_results,
    }

    print("\n" + "=" * 80)
    print("PIT Feature Engineering Summary Statistics:")
    print(f"  Train Split Rows:                {summary_stats['train_rows']:,}")
    print(f"  Test Split Rows:                 {summary_stats['test_rows']:,}")
    print(f"  Train Active Customer Quotes:    {summary_stats['train_active_customer_rows']:,} ({summary_stats['train_active_customer_rows'] / summary_stats['train_rows'] * 100:.1f}%)")
    print(f"  Test Active Customer Quotes:     {summary_stats['test_active_customer_rows']:,} ({summary_stats['test_active_customer_rows'] / summary_stats['test_rows'] * 100:.1f}%)")
    print(f"  Train Median Customer Margin:    {summary_stats['train_median_margin_overall']:.2f}%")
    print(f"  Test Median Customer Margin:     {summary_stats['test_median_margin_overall']:.2f}%")
    print("=" * 80)

    return summary_stats


def main():
    parser = argparse.ArgumentParser(description="Compute Point-in-Time Features for 2025 Data")
    parser.add_argument("--train", type=str, default=DEFAULT_TRAIN_PATH, help="Path to won_deals_2025_train_70.csv")
    parser.add_argument("--test", type=str, default=DEFAULT_TEST_PATH, help="Path to won_deals_2025_test_30.csv")
    parser.add_argument("--train-out", type=str, default=DEFAULT_TRAIN_OUT, help="Path to won_deals_2025_train_70_pit.csv")
    parser.add_argument("--test-out", type=str, default=DEFAULT_TEST_OUT, help="Path to won_deals_2025_test_30_pit.csv")
    parser.add_argument("--registry-out", type=str, default=DEFAULT_REGISTRY_OUT, help="Path to customer_pit_registry_2025.joblib")
    parser.add_argument("--include-stratified", action="store_true", help="Also compute PIT for stratified splits")
    args = parser.parse_args()

    compute_pit_pipeline(
        train_path=args.train,
        test_path=args.test,
        train_out_path=args.train_out,
        test_out_path=args.test_out,
        registry_out_path=args.registry_out,
        include_stratified=args.include_stratified,
    )


if __name__ == "__main__":
    main()
