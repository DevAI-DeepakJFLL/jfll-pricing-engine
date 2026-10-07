"""
scripts/preprocess_ae_gcr_bombay.py
================================================================================
Data Preprocessing Pipeline: AE GCR Bombay (Jan Till Date) Inquiries
================================================================================
Transforms data/raw/AE_GCR_Bombay_Jan_TillDate.xlsx to mirror the target
format and schema of data/processed/Air_Export_Pricing_Cleaned_ML.csv,
including month-wise temporal features (Month_Name, Year_Month).
"""

import os
import sys
import argparse
from typing import Any
import numpy as np
import pandas as pd

# Support both module and standalone execution
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from tiers import WEIGHT_TIER_BINS, WEIGHT_TIER_LABELS, map_account_tier
from preprocess import (
    normalize_port_name,
    map_origin_region,
    map_dest_region,
    map_commodity_group,
    clean_incoterms,
    parse_currency_field,
    get_air_cargo_season,
    compute_cyclical_month
)


def run_pipeline(
    raw_path: str = "data/raw/AE_GCR_Bombay_Jan_TillDate.xlsx",
    output_ml_path: str = "data/processed/AE_GCR_Bombay_Cleaned_ML.csv",
    output_won_path: str = "data/processed/AE_GCR_Bombay_Won_Benchmark.csv"
):
    print("=" * 80)
    print("AE GCR Bombay (Jan Till Date) Preprocessing Pipeline")
    print("=" * 80)
    print(f"[*] Ingesting raw Excel: {raw_path}")

    # Read 'RFQ Report' sheet
    df = pd.read_excel(raw_path, sheet_name="RFQ Report")
    df.columns = df.columns.str.strip()
    initial_rows = len(df)
    print(f"    Raw records ingested: {initial_rows:,} rows across {len(df.columns)} columns.")

    # 1. Deduplication with hierarchical status-rank
    print("[*] Applying hierarchical status-rank deduplication...")
    quote_counts = df.groupby("Inquiry Number")["Quotation Number"].nunique().to_dict()
    df["quote_revision_count"] = df["Inquiry Number"].map(quote_counts).fillna(1).astype(int)

    status_priority = {"Won": 1, "Approved": 2, "Draft": 3, "Lost": 4, "Hold": 5, "Cancelled": 6}
    df["status_rank"] = df["Quote Status"].map(status_priority).fillna(99).astype(int)

    df_sorted = df.sort_values(
        by=["Inquiry Number", "status_rank", "Quote Created At"],
        ascending=[True, True, False]
    )
    df_dedup = df_sorted.drop_duplicates(subset=["Inquiry Number"], keep="first").copy()
    dropped_dups = initial_rows - len(df_dedup)
    print(f"    Deduplicated on Inquiry Number: {dropped_dups:,} duplicate quotes dropped ({len(df_dedup):,} unique inquiries).")

    # 2. Currency Parsing & Filtering
    print("[*] Parsing pricing fields and removing commercial anomalies...")
    df_dedup["Total_Buy_INR"] = df_dedup["Total Buy"].apply(parse_currency_field)
    df_dedup["Total_Sell_INR"] = df_dedup["Total Sell"].apply(parse_currency_field)

    valid_price_mask = (
        (df_dedup["Total_Buy_INR"] > 500.0) &
        (df_dedup["Total_Sell_INR"] > 500.0) &
        (df_dedup["Total_Sell_INR"] >= df_dedup["Total_Buy_INR"])
    )
    df_priced = df_dedup[valid_price_mask].copy()
    dropped_unpriced = len(df_dedup) - len(df_priced)
    print(f"    Valid commercial prices (Buy>500, Sell>500, Sell>=Buy): {len(df_priced):,} ({dropped_unpriced:,} unbilled/zero dropped).")

    df_priced["Margin_Amount_INR"] = df_priced["Total_Sell_INR"] - df_priced["Total_Buy_INR"]
    df_priced["Margin_Percentage"] = (df_priced["Margin_Amount_INR"] / df_priced["Total_Buy_INR"]) * 100.0

    # Filter margin anomalies (0.2% to 60.0%)
    margin_mask = (df_priced["Margin_Percentage"] >= 0.2) & (df_priced["Margin_Percentage"] <= 60.0)
    df_clean = df_priced[margin_mask].copy()
    dropped_margin = len(df_priced) - len(df_clean)
    print(f"    Filtered margin anomalies (<0.2% or >60%): {dropped_margin:,} records pruned.")
    print(f"    Cleaned commercial inquiry cohort retained: {len(df_clean):,} records.")

    # 3. Feature Engineering: Weights & Densities
    print("[*] Engineering weights, density ratios, and weight tiers...")
    gross = pd.to_numeric(df_clean["Total Gross Wt"], errors="coerce").fillna(0.0)
    ch_wt = pd.to_numeric(df_clean["Ch Wt."], errors="coerce").fillna(0.0)
    gross_clean = np.where(gross > 0, gross, np.where(ch_wt > 0, ch_wt, 1.0))
    ch_wt_clean = np.where(ch_wt > 0, ch_wt, gross_clean)

    df_clean["Gross_Weight_Kg"] = gross_clean
    df_clean["Chargeable_Weight_Kg"] = ch_wt_clean
    df_clean["Cargo_Density_Ratio"] = np.clip(gross_clean / ch_wt_clean, 0.1, 5.0)

    df_clean["Weight_Tier"] = pd.cut(
        df_clean["Chargeable_Weight_Kg"],
        bins=WEIGHT_TIER_BINS,
        labels=WEIGHT_TIER_LABELS,
        right=False
    ).astype(str)

    # 4. Canonical Geographic Hierarchy Engineering
    print("[*] Normalizing trade lanes, canonical ports, and regions...")
    raw_pol = df_clean["Port of Loading"].fillna("Chhatrapati Shivaji Maharaj International Airport").astype(str).str.strip()
    raw_pod = df_clean["Port of Discharge"].fillna("Dubai").astype(str).str.strip()

    df_clean["Origin_Port"] = raw_pol.apply(normalize_port_name)
    df_clean["Destination_Port"] = raw_pod.apply(normalize_port_name)
    df_clean["Origin_Region"] = df_clean["Origin_Port"].apply(map_origin_region)
    df_clean["Destination_Region"] = df_clean["Destination_Port"].apply(map_dest_region)
    df_clean["Regional_Lane"] = df_clean["Origin_Region"] + " -> " + df_clean["Destination_Region"]
    df_clean["Trade_Lane"] = df_clean["Origin_Port"] + " -> " + df_clean["Destination_Port"]

    # 5. Commercial, Customer, & Behavioral Features
    print("[*] Feature engineering customer frequency, tiers, and business attributes...")
    df_clean["Business_Vertical"] = df_clean["Business Vertical"].fillna("Air Export Forwarding").astype(str)
    df_clean["Commodity_Group"] = df_clean["Cargo Description"].apply(map_commodity_group)
    df_clean["Company_Group"] = df_clean["Company Group"].fillna("Subagent").astype(str)
    df_clean["Customer_Company"] = df_clean["Customer Company"].fillna("Unknown_Customer").astype(str)
    df_clean["Incoterms"] = df_clean["Incoterms"].apply(clean_incoterms)

    vol_map = df_clean["Customer_Company"].value_counts().to_dict()
    df_clean["Customer_Inquiry_Frequency"] = df_clean["Customer_Company"].map(vol_map).fillna(1).astype(int)
    df_clean["Customer_Tier"] = df_clean["Customer_Inquiry_Frequency"].apply(map_account_tier)

    # 6. Temporal Features & Month-Wise Categorization
    print("[*] Parsing timestamps and computing month-wise variables...")
    q_dates = pd.to_datetime(df_clean["Quote Created At"], errors="coerce")
    c_dates = pd.to_datetime(df_clean["Created At"], errors="coerce")
    final_dates = q_dates.fillna(c_dates)

    latency = (q_dates - c_dates).dt.total_seconds() / 3600.0
    df_clean["Quote_Turnaround_Hours"] = latency.clip(lower=0.0, upper=168.0).fillna(0.0)

    df_clean["Quote_Month"] = final_dates.dt.month.fillna(6).astype(int)
    df_clean["Quote_DayOfWeek"] = final_dates.dt.dayofweek.fillna(0).astype(int)
    df_clean["Is_Month_End"] = (final_dates.dt.day >= 25).astype(int)
    df_clean["Air_Cargo_Season"] = df_clean["Quote_Month"].apply(get_air_cargo_season)

    sin_cos = [compute_cyclical_month(m) for m in df_clean["Quote_Month"]]
    df_clean["Month_Sin"] = [sc[0] for sc in sin_cos]
    df_clean["Month_Cos"] = [sc[1] for sc in sin_cos]

    # Explicit Month-wise columns
    df_clean["Month_Name"] = final_dates.dt.strftime("%B")
    df_clean["Year_Month"] = final_dates.dt.strftime("%Y-%m")

    # 7. Status & Identifiers
    df_clean["is_won"] = (df_clean["Quote Status"] == "Won").astype(int)
    fallback_status = np.where(df_clean["is_won"] == 1, "Won", "Lost")
    df_clean["Inquiry Status"] = df_clean["Inquiry Status"].fillna(pd.Series(fallback_status, index=df_clean.index)).astype(str)
    df_clean["Quote Status"] = df_clean["Quote Status"].fillna(pd.Series(fallback_status, index=df_clean.index)).astype(str)
    df_clean["Inquiry Number"] = df_clean["Inquiry Number"].astype(str)
    df_clean["Quotation Number"] = df_clean["Quotation Number"].fillna(df_clean["Inquiry Number"]).astype(str)
    df_clean["Booked by branch"] = df_clean["Booked by branch"].fillna("Mumbai").astype(str)

    # 8. Schema Assembly
    output_cols = [
        "Inquiry Number", "Quotation Number", "Inquiry Status", "Quote Status", "is_won",
        "Total_Buy_INR", "Total_Sell_INR", "Margin_Amount_INR", "Margin_Percentage",
        "Chargeable_Weight_Kg", "Gross_Weight_Kg", "Cargo_Density_Ratio", "Weight_Tier",
        "Business_Vertical", "Commodity_Group", "Origin_Port", "Destination_Port",
        "Origin_Region", "Destination_Region", "Regional_Lane", "Trade_Lane",
        "Incoterms", "Company_Group", "Customer_Company", "Customer_Inquiry_Frequency", "Customer_Tier",
        "Booked by branch", "quote_revision_count", "Quote_Turnaround_Hours",
        "Quote_Month", "Quote_DayOfWeek", "Is_Month_End",
        "Month_Sin", "Month_Cos", "Air_Cargo_Season",
        "Month_Name", "Year_Month"
    ]

    final_df = df_clean[output_cols].copy()
    won_df = final_df[final_df["is_won"] == 1].copy()

    # Save to disk
    os.makedirs(os.path.dirname(output_ml_path), exist_ok=True)
    final_df.to_csv(output_ml_path, index=False)
    won_df.to_csv(output_won_path, index=False)

    print("=" * 80)
    print("[✓] Preprocessing Complete! Datasets successfully exported:")
    print(f"    - Full Cleaned ML Dataset : {output_ml_path} ({len(final_df):,} rows)")
    print(f"    - Won Deals Benchmark     : {output_won_path} ({len(won_df):,} rows)")
    print("=" * 80)

    # Statistical Profile
    print("\nDataset Summary Statistics:")
    print(f"    Total Inquiries     : {len(final_df):,}")
    print(f"    Won Inquiries       : {len(won_df):,} ({len(won_df)/len(final_df)*100:.2f}%)")
    print(f"    Lost Inquiries      : {len(final_df)-len(won_df):,} ({(len(final_df)-len(won_df))/len(final_df)*100:.2f}%)")
    print(f"    Mean Buy (INR)      : ₹{final_df['Total_Buy_INR'].mean():,.2f}")
    print(f"    Mean Sell (INR)     : ₹{final_df['Total_Sell_INR'].mean():,.2f}")
    print(f"    Mean Margin %       : {final_df['Margin_Percentage'].mean():.2f}%")
    print(f"    Median Margin %     : {final_df['Margin_Percentage'].median():.2f}%")
    print(f"    Unique Customers    : {final_df['Customer_Company'].nunique():,}")
    print(f"    Unique Trade Lanes  : {final_df['Trade_Lane'].nunique():,}")

    print("\nMonth-Wise Volume & Margin Breakdown:")
    monthly = final_df.groupby(["Year_Month", "Month_Name"]).agg(
        inquiry_count=("Inquiry Number", "count"),
        won_count=("is_won", "sum"),
        win_rate_pct=("is_won", lambda x: x.mean() * 100),
        mean_margin_pct=("Margin_Percentage", "mean"),
        median_margin_pct=("Margin_Percentage", "median"),
        mean_buy_inr=("Total_Buy_INR", "mean")
    ).reset_index()
    print(monthly.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess AE GCR Bombay (Jan Till Date)")
    parser.add_argument("--raw", type=str, default="data/raw/AE_GCR_Bombay_Jan_TillDate.xlsx")
    parser.add_argument("--out_ml", type=str, default="data/processed/AE_GCR_Bombay_Cleaned_ML.csv")
    parser.add_argument("--out_won", type=str, default="data/processed/AE_GCR_Bombay_Won_Benchmark.csv")
    args = parser.parse_args()

    run_pipeline(args.raw, args.out_ml, args.out_won)
