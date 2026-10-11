"""
scripts/generate_honest_benchmark.py
Generates the honest out-of-time Won-only benchmark evaluation dataset.

Filtration criteria:
1. Ground truth Won quotes only (`is_won == 1`)
2. Out-of-time holdout temporal window (`Year_Month >= '2026-08'`, i.e. August-September 2026)
3. Deduplicated by quote signature: (`Total_Buy_INR`, `Chargeable_Weight_Kg`, `Customer_Company`)
4. Authentic commercial transactions: `Total_Buy_INR > 500` and realized `Total_Sell_INR > Total_Buy_INR`
5. Physically valid: `Chargeable_Weight_Kg > 0`
"""

import os
import sys
import argparse
import pandas as pd


DEFAULT_INPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "processed",
    "Air_Export_Pricing_Combined_ML.csv",
)

DEFAULT_OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests",
    "data",
    "benchmark_won_out_of_time.csv",
)


def generate_honest_benchmark(
    input_path: str = DEFAULT_INPUT_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Reads processed data, filters honest out-of-time Won transactions, and writes benchmark CSV."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found at {input_path}")

    print(f"[1/5] Reading input dataset from: {input_path}")
    df = pd.read_csv(input_path)
    total_raw = len(df)
    print(f"      Total records loaded: {total_raw:,}")

    # 1. Filter out-of-time Won deals (Aug-Sep 2026)
    print("[2/5] Filtering genuine Won deals in holdout window (Year_Month >= '2026-08', is_won == 1)...")
    won_oot = df[(df["is_won"] == 1) & (df["Year_Month"] >= "2026-08")].copy()
    print(f"      Won out-of-time candidates: {len(won_oot):,}")

    # 2. Deduplicate by quote transaction signature
    print("[3/5] Deduplicating by signature (Total_Buy_INR, Chargeable_Weight_Kg, Customer_Company)...")
    deduped = won_oot.drop_duplicates(
        subset=["Total_Buy_INR", "Chargeable_Weight_Kg", "Customer_Company"]
    ).copy()
    num_duplicates_dropped = len(won_oot) - len(deduped)
    print(f"      Duplicates removed: {num_duplicates_dropped:,} (Remaining: {len(deduped):,})")

    # 3. Clean commercial and physical sanity filters
    print("[4/5] Applying economic sanity filters (Total_Buy_INR > 500, Sell > Buy, Weight > 0)...")
    clean_mask = (
        (deduped["Total_Buy_INR"] > 500)
        & (deduped["Total_Sell_INR"] > deduped["Total_Buy_INR"])
        & (deduped["Chargeable_Weight_Kg"] > 0)
    )
    benchmark_df = deduped[clean_mask].copy()
    print(f"      Sanity verified records: {len(benchmark_df):,}")

    if len(benchmark_df) < 200:
        raise ValueError(
            f"Benchmark dataset has insufficient sample size: {len(benchmark_df)} < 200 required"
        )

    # 4. Save to destination
    print(f"[5/5] Writing {len(benchmark_df):,} records to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    benchmark_df.to_csv(output_path, index=False)

    print("\n--- Benchmark Generation Summary ---")
    print(f"Destination:           {output_path}")
    print(f"Final Count:           {len(benchmark_df):,}")
    print(f"Holdout Months:        {sorted(benchmark_df['Year_Month'].unique())}")
    print(f"Mean Margin %:         {benchmark_df['Margin_Percentage'].mean():.2f}%")
    print(f"Median Margin %:       {benchmark_df['Margin_Percentage'].median():.2f}%")
    print(f"Min / Max Margin %:    {benchmark_df['Margin_Percentage'].min():.2f}% / {benchmark_df['Margin_Percentage'].max():.2f}%")
    print(f"Mean Weight (kg):      {benchmark_df['Chargeable_Weight_Kg'].mean():.1f} kg")
    print(f"Unique Customers:      {benchmark_df['Customer_Company'].nunique():,}")
    print(f"Unique Origin Ports:   {benchmark_df['Origin_Port'].nunique():,}")
    print(f"Unique Dest Ports:     {benchmark_df['Destination_Port'].nunique():,}")
    print("------------------------------------\n")

    return benchmark_df


def main():
    parser = argparse.ArgumentParser(
        description="Generate honest out-of-time Won-only benchmark dataset"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT_PATH,
        help="Path to Air_Export_Pricing_Combined_ML.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to output benchmark_won_out_of_time.csv",
    )
    args = parser.parse_args()

    try:
        generate_honest_benchmark(input_path=args.input, output_path=args.output)
    except Exception as e:
        print(f"Error generating benchmark: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
