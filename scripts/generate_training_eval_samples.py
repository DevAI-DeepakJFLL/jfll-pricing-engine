"""
scripts/generate_training_eval_samples.py
Stratified Training Sample Generator for Model Evaluation.

Samples exactly n_per_cell (default 20) records per segment across 5 Commodity Types
and 5 Budget Levels (25 segments x 20 records = 500 records) from the training data.

Budget levels are determined within each commodity using 5 quintiles of Total_Buy_INR:
- Very Low Budget  (0% <= quantile <= 20%)
- Low Budget       (20% < quantile <= 40%)
- Medium Budget    (40% < quantile <= 60%)
- High Budget      (60% < quantile <= 80%)
- Very High Budget (80% < quantile <= 100%)
"""

import os
import sys
import argparse
from typing import List, Optional
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_INPUT_PATH = os.path.join(
    PROJECT_ROOT, "data", "processed", "Air_Export_Pricing_Combined_ML.csv"
)

DEFAULT_OUTPUT_PATH = os.path.join(
    PROJECT_ROOT, "tests", "data", "training_eval_samples_500.csv"
)

DEFAULT_COMMODITIES = [
    "General Cargo",
    "Perishable Foodstuff",
    "Garments / Textiles",
    "Engineering & Machinery",
    "Other",
]

BUDGET_TIERS = [
    "Very Low Budget",
    "Low Budget",
    "Medium Budget",
    "High Budget",
    "Very High Budget",
]


def _resolve_path(path: str) -> str:
    """Resolve file path against CWD and PROJECT_ROOT."""
    if os.path.isabs(path):
        return path
    if os.path.exists(path):
        return os.path.abspath(path)
    root_candidate = os.path.join(PROJECT_ROOT, path)
    if os.path.exists(root_candidate):
        return root_candidate
    return os.path.abspath(path)


def generate_stratified_training_samples(
    data_path: str = "data/processed/Air_Export_Pricing_Combined_ML.csv",
    commodities: Optional[List[str]] = None,
    n_per_cell: int = 20,
    seed: int = 42,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate stratified evaluation sample across commodities and budget tiers.

    Args:
        data_path: Path to processed training dataset CSV.
        commodities: List of commodity groups to include (defaults to 5 standard categories).
        n_per_cell: Number of records to sample per (commodity, budget) cell.
        seed: Random seed for reproducible sampling.
        output_path: Optional destination CSV path to save samples.

    Returns:
        pd.DataFrame containing stratified samples with 'Budget_Segment' column.
    """
    resolved_data_path = _resolve_path(data_path)
    if not os.path.exists(resolved_data_path):
        raise FileNotFoundError(f"Training dataset not found at {resolved_data_path}")

    target_commodities = commodities if commodities is not None else DEFAULT_COMMODITIES

    print(f"[1/4] Reading dataset from: {resolved_data_path}")
    df = pd.read_csv(resolved_data_path)
    print(f"      Loaded {len(df):,} total records.")

    sampled_segments = []

    print(f"[2/4] Stratifying across {len(target_commodities)} commodities and {len(BUDGET_TIERS)} budget tiers...")
    for commodity in target_commodities:
        sub = df[df["Commodity_Group"] == commodity].copy()
        if len(sub) == 0:
            raise ValueError(f"Commodity group '{commodity}' has zero records in dataset.")

        # Compute within-commodity quintiles on Total_Buy_INR
        sub["Budget_Segment"] = pd.qcut(
            sub["Total_Buy_INR"],
            q=5,
            labels=BUDGET_TIERS,
            duplicates="drop",
        )

        # Sample n_per_cell records per budget tier
        for tier in BUDGET_TIERS:
            tier_records = sub[sub["Budget_Segment"] == tier]
            if len(tier_records) < n_per_cell:
                raise ValueError(
                    f"Insufficient records for ({commodity}, {tier}): "
                    f"found {len(tier_records)}, required {n_per_cell}"
                )

            sampled_cell = tier_records.sample(n=n_per_cell, random_state=seed, replace=False)
            sampled_segments.append(sampled_cell)

    sampled_df = pd.concat(sampled_segments, ignore_index=True)
    expected_rows = len(target_commodities) * len(BUDGET_TIERS) * n_per_cell
    print(f"[3/4] Successfully sampled {len(sampled_df):,} records (Expected: {expected_rows:,}).")

    if output_path is not None:
        resolved_out_path = _resolve_path(output_path) if not os.path.isabs(output_path) else output_path
        os.makedirs(os.path.dirname(resolved_out_path), exist_ok=True)
        sampled_df.to_csv(resolved_out_path, index=False)
        print(f"[4/4] Saved stratified samples to: {resolved_out_path}")

    return sampled_df


def main():
    parser = argparse.ArgumentParser(
        description="Generate stratified training evaluation samples (5x5 grid, 500 records)"
    )
    parser.add_argument(
        "--data-path",
        "--input",
        type=str,
        default=DEFAULT_INPUT_PATH,
        help="Path to Air_Export_Pricing_Combined_ML.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to save output CSV (default: tests/data/training_eval_samples_500.csv)",
    )
    parser.add_argument(
        "--n-per-cell",
        type=int,
        default=20,
        help="Number of records to sample per cell (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    args = parser.parse_args()

    try:
        generate_stratified_training_samples(
            data_path=args.data_path,
            commodities=DEFAULT_COMMODITIES,
            n_per_cell=args.n_per_cell,
            seed=args.seed,
            output_path=args.output,
        )
    except Exception as e:
        print(f"Error generating stratified samples: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
