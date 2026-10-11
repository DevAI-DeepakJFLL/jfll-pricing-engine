"""
scripts/evaluate_training_samples.py
Model Inference & Margin Gap Evaluation Harness for Air Export Pricing Engine.

Evaluates the active production pricing engine (PricingEngineV3) on stratified
training samples, calculating:
- Model Margin vs Actual Margin gap (Model - Actual)
- Absolute and Squared Error (MAE, RMSE)
- Conformal Corridor Coverage (In_Corridor)
- Multi-dimensional segment roll-up metrics (by_cell, by_commodity, by_budget, overall)
"""

import os
import sys
import argparse
from typing import Dict, Any, List, Optional, Tuple, Sequence
import numpy as np
import pandas as pd

# Project root setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.engine_v3 import PricingEngineV3

DEFAULT_INPUT_PATH = os.path.join(PROJECT_ROOT, "tests", "data", "training_eval_samples_500.csv")
DEFAULT_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "tests", "data", "training_eval_results_500.csv")


def compute_margin_gap(model_margin: float, actual_margin: float) -> float:
    """
    Computes margin gap identity: Model - Actual.
    Positive value indicates model over-quoting relative to actual.
    Negative value indicates model under-quoting relative to actual.
    """
    return float(model_margin - actual_margin)


def compute_mae(gaps: Sequence[float]) -> float:
    """Computes Mean Absolute Error (MAE) in percentage points."""
    arr = np.asarray(gaps, dtype=float)
    if len(arr) == 0:
        return 0.0
    return float(np.mean(np.abs(arr)))


def compute_rmse(gaps: Sequence[float]) -> float:
    """Computes Root Mean Squared Error (RMSE) in percentage points."""
    arr = np.asarray(gaps, dtype=float)
    if len(arr) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(arr))))


def compute_segment_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes roll-up metrics for a subset of evaluated records:
    n_records, actual_margin_mean, model_margin_mean, margin_gap_mean (bias),
    mae, rmse, and corridor_coverage.
    """
    n = len(df)
    if n == 0:
        return {
            "n_records": 0,
            "actual_margin_mean": 0.0,
            "model_margin_mean": 0.0,
            "margin_gap_mean": 0.0,
            "mae": 0.0,
            "rmse": 0.0,
            "corridor_coverage": 0.0,
        }

    actual_mean = float(df["Actual_Margin"].mean())
    model_mean = float(df["Model_Margin"].mean())
    gap_mean = float(df["Margin_Gap"].mean())
    mae = compute_mae(df["Margin_Gap"].values)
    rmse = compute_rmse(df["Margin_Gap"].values)
    coverage = float(df["In_Corridor"].mean())

    return {
        "n_records": n,
        "actual_margin_mean": round(actual_mean, 4),
        "model_margin_mean": round(model_mean, 4),
        "margin_gap_mean": round(gap_mean, 4),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "corridor_coverage": round(coverage, 4),
    }


def aggregate_evaluation_results(evaluated_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes aggregation roll-ups:
    - by_cell: Dict keyed by (Commodity_Group, Budget_Segment) tuple
    - by_commodity: Dict keyed by Commodity_Group
    - by_budget: Dict keyed by Budget_Segment
    - overall: Metrics across the full dataset
    """
    by_cell: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for (comm, budget), group in evaluated_df.groupby(["Commodity_Group", "Budget_Segment"], observed=True):
        by_cell[(str(comm), str(budget))] = compute_segment_metrics(group)

    by_commodity: Dict[str, Dict[str, Any]] = {}
    for comm, group in evaluated_df.groupby("Commodity_Group", observed=True):
        by_commodity[str(comm)] = compute_segment_metrics(group)

    by_budget: Dict[str, Dict[str, Any]] = {}
    for budget, group in evaluated_df.groupby("Budget_Segment", observed=True):
        by_budget[str(budget)] = compute_segment_metrics(group)

    overall = compute_segment_metrics(evaluated_df)

    return {
        "by_cell": by_cell,
        "by_commodity": by_commodity,
        "by_budget": by_budget,
        "overall": overall,
    }


def evaluate_model_on_training_samples(
    samples_df: pd.DataFrame,
    engine: Optional[PricingEngineV3] = None,
    strategy: str = "balanced",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Runs PricingEngineV3 on samples_df, computes margin gap and error metrics,
    and returns an evaluated DataFrame alongside structured aggregation roll-ups.

    Parameters:
        samples_df: DataFrame containing stratified training samples.
        engine: Optional PricingEngineV3 instance (instantiated if None).
        strategy: Pricing strategy key (default: 'balanced').

    Returns:
        Tuple of (evaluated_df, aggregations_dict)
    """
    if engine is None:
        engine = PricingEngineV3()

    eval_rows: List[Dict[str, Any]] = []

    for _, row in samples_df.iterrows():
        # Construct inquiry payload
        inquiry = {
            "Total_Buy_INR": float(row["Total_Buy_INR"]),
            "Chargeable_Weight_Kg": float(row["Chargeable_Weight_Kg"]),
            "Gross_Weight_Kg": float(row["Gross_Weight_Kg"]),
            "Origin_Port": str(row["Origin_Port"]),
            "Destination_Port": str(row["Destination_Port"]),
            "Company_Group": str(row["Company_Group"]),
            "Commodity_Group": str(row["Commodity_Group"]),
            "Weight_Tier": str(row["Weight_Tier"]),
            "Incoterms": str(row["Incoterms"]),
            "Booked by branch": str(row.get("Booked by branch", "")),
            "Customer_Company": str(row.get("Customer_Company", "")),
            "Origin_Region": str(row.get("Origin_Region", "")),
            "Destination_Region": str(row.get("Destination_Region", "")),
            "Regional_Lane": str(row.get("Regional_Lane", "")),
            "Cargo_Density_Ratio": (
                float(row["Cargo_Density_Ratio"])
                if "Cargo_Density_Ratio" in row and pd.notna(row["Cargo_Density_Ratio"])
                else float(row["Chargeable_Weight_Kg"]) / max(1.0, float(row["Gross_Weight_Kg"]))
            ),
            "quote_date": str(
                row.get("quote_date")
                or (f"{row['Year_Month']}-01" if "Year_Month" in row and pd.notna(row["Year_Month"]) else "2026-01-01")
            )[:10],
        }

        # Request quote from production engine
        quote = engine.recommend_quote(inquiry, pricing_strategy=strategy)

        actual_margin = float(row["Margin_Percentage"])
        model_margin = float(quote["optimal"]["Margin_Percentage"])
        margin_gap = compute_margin_gap(model_margin, actual_margin)
        abs_error = abs(margin_gap)
        sq_error = margin_gap ** 2.0

        corridor_min = float(quote["negotiation_corridor"]["min_margin_pct"])
        corridor_max = float(quote["negotiation_corridor"]["max_margin_pct"])
        in_corridor = bool(corridor_min <= actual_margin <= corridor_max)

        win_prob = float(quote["optimal"].get("Win_Probability", 0.0))
        quoted_sell_inr = float(quote["optimal"]["Quoted_Sell_Price_INR"])
        cost_floor_inr = float(quote["optimal"].get("Cost_Floor_Price_INR", 0.0))

        lim_factor = str(
            quote["optimal"].get("Limiting_Factor")
            or quote["optimal"].get("Clamped_By")
            or "none"
        )

        row_dict = dict(row)
        row_dict.update({
            "Actual_Margin": actual_margin,
            "Model_Margin": model_margin,
            "Margin_Gap": margin_gap,
            "Absolute_Error": abs_error,
            "Squared_Error": sq_error,
            "Corridor_Min": corridor_min,
            "Corridor_Max": corridor_max,
            "In_Corridor": in_corridor,
            "Win_Probability": win_prob,
            "Quoted_Sell_Price_INR": quoted_sell_inr,
            "Cost_Floor_Price_INR": cost_floor_inr,
            "Limiting_Factor": lim_factor,
        })
        eval_rows.append(row_dict)

    evaluated_df = pd.DataFrame(eval_rows)
    aggregations = aggregate_evaluation_results(evaluated_df)

    return evaluated_df, aggregations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Model Inference & Margin Gap Evaluation Harness on Stratified Training Samples"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT_PATH,
        help=f"Input CSV path containing stratified training samples (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output CSV path for evaluation results (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="balanced",
        help="Pricing strategy to evaluate ('balanced', 'floor', 'premium'; default: 'balanced')",
    )

    args = parser.parse_args()

    print(f"Loading samples from: {args.input}")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input samples file not found: {args.input}")

    samples_df = pd.read_csv(args.input)
    print(f"Loaded {len(samples_df)} records across {samples_df['Commodity_Group'].nunique()} commodities.")

    print(f"Initializing PricingEngineV3 and evaluating with strategy='{args.strategy}'...")
    engine = PricingEngineV3()
    evaluated_df, aggregations = evaluate_model_on_training_samples(
        samples_df=samples_df,
        engine=engine,
        strategy=args.strategy,
    )

    # Save output CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    evaluated_df.to_csv(args.output, index=False)
    print(f"Saved {len(evaluated_df)} evaluated records to: {args.output}")

    # Display summary
    overall = aggregations["overall"]
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY (OVERALL)")
    print("=" * 60)
    print(f"Total Evaluated Records: {overall['n_records']}")
    print(f"Actual Margin Mean:      {overall['actual_margin_mean']:.2f}%")
    print(f"Model Margin Mean:       {overall['model_margin_mean']:.2f}%")
    print(f"Mean Margin Gap (Bias):  {overall['margin_gap_mean']:+.2f} pp")
    print(f"MAE:                     {overall['mae']:.2f} pp")
    print(f"RMSE:                    {overall['rmse']:.2f} pp")
    print(f"Corridor Coverage:       {overall['corridor_coverage'] * 100:.1f}%")
    print("=" * 60)
    print(f"Total 25-cell segments evaluated: {len(aggregations['by_cell'])}")


if __name__ == "__main__":
    main()
