"""
scripts/preprocess_ae_gcr_panindia.py
================================================================================
Data Preprocessing Pipeline: AE GCR PanIndia (Non-Mumbai Branch Ingestion)
================================================================================
Ingests new branch records from data/raw/AE_GCR_PanIndia.xlsx, skips previously
cleaned Mumbai records, applies enterprise deduplication and feature engineering,
and updates the combined multi-branch machine learning dataset.
================================================================================
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


def run_panindia_pipeline(
    raw_path: str = "data/raw/AE_GCR_PanIndia.xlsx",
    bom_cleaned_path: str = "data/processed/AE_GCR_Bombay_Cleaned_ML.csv",
    orig_cleaned_path: str = "data/processed/Air_Export_Pricing_Cleaned_ML.csv",
    out_branches_ml: str = "data/processed/AE_GCR_PanIndia_Branches_Cleaned_ML.csv",
    out_branches_won: str = "data/processed/AE_GCR_PanIndia_Branches_Won_Benchmark.csv",
    out_combined_ml: str = "data/processed/Air_Export_Pricing_Combined_ML.csv",
    out_combined_won: str = "data/processed/Air_Export_Pricing_Combined_Won_Benchmark.csv"
):
    print("=" * 80)
    print("AE GCR PanIndia Branch Ingestion Pipeline")
    print("=" * 80)
    print(f"[*] Ingesting raw PanIndia Excel: {raw_path}")

    # Read 'Sheet1'
    df = pd.read_excel(raw_path, sheet_name="Sheet1")
    df.columns = df.columns.str.strip()
    total_raw = len(df)
    print(f"    Raw records ingested: {total_raw:,} rows across {len(df.columns)} columns.")

    # 1. Filter out Mumbai records (skipping previously cleaned data)
    print("[*] Skipping previously cleaned Mumbai records...")
    df_branches = df[df["Booked by branch"] != "Mumbai"].copy()
    skipped_mumbai = total_raw - len(df_branches)
    print(f"    Skipped {skipped_mumbai:,} Mumbai records.")
    print(f"    Retained {len(df_branches):,} non-Mumbai records across {df_branches['Booked by branch'].nunique()} branches.")

    # Standardize branch aliases (Haryana -> Delhi)
    df_branches["Booked by branch"] = df_branches["Booked by branch"].replace({"Haryana": "Delhi"})

    # 2. Hierarchical status-rank deduplication on Inquiry Number
    print("[*] Applying hierarchical status-rank deduplication...")
    quote_counts = df_branches.groupby("Inquiry Number")["Quotation Number"].nunique().to_dict()
    df_branches["quote_revision_count"] = df_branches["Inquiry Number"].map(quote_counts).fillna(1).astype(int)

    status_priority = {"Won": 1, "Approved": 2, "Draft": 3, "Lost": 4, "Hold": 5, "Cancelled": 6}
    df_branches["status_rank"] = df_branches["Quote Status"].map(status_priority).fillna(99).astype(int)

    df_sorted = df_branches.sort_values(
        by=["Inquiry Number", "status_rank", "Quote Created At"],
        ascending=[True, True, False]
    )
    df_dedup = df_sorted.drop_duplicates(subset=["Inquiry Number"], keep="first").copy()
    dropped_dups = len(df_branches) - len(df_dedup)
    print(f"    Deduplicated: {dropped_dups:,} duplicate quote revisions dropped ({len(df_dedup):,} unique inquiries).")

    # 3. Currency Parsing & Filtering
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
    print(f"    Valid commercial prices (Buy>500, Sell>500, Sell>=Buy): {len(df_priced):,} ({dropped_unpriced:,} unbilled/zero/negative dropped).")

    df_priced["Margin_Amount_INR"] = df_priced["Total_Sell_INR"] - df_priced["Total_Buy_INR"]
    df_priced["Margin_Percentage"] = (df_priced["Margin_Amount_INR"] / df_priced["Total_Buy_INR"]) * 100.0

    # Filter margin anomalies (0.2% to 60.0%)
    margin_mask = (df_priced["Margin_Percentage"] >= 0.2) & (df_priced["Margin_Percentage"] <= 60.0)
    df_clean = df_priced[margin_mask].copy()
    dropped_margin = len(df_priced) - len(df_clean)
    print(f"    Filtered margin anomalies (<0.2% or >60%): {dropped_margin:,} records pruned.")
    print(f"    Cleaned branch inquiry cohort retained: {len(df_clean):,} records.")

    # 4. Feature Engineering: Weights & Densities
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

    # 5. Canonical Geographic Hierarchy Engineering
    print("[*] Normalizing trade lanes, canonical ports, and regions...")
    raw_pol = df_clean["Port of Loading"].fillna("Indira Gandhi International Airport").astype(str).str.strip()
    raw_pod = df_clean["Port of Discharge"].fillna("Dubai").astype(str).str.strip()

    df_clean["Origin_Port"] = raw_pol.apply(normalize_port_name)
    df_clean["Destination_Port"] = raw_pod.apply(normalize_port_name)
    df_clean["Origin_Region"] = df_clean["Origin_Port"].apply(map_origin_region)
    df_clean["Destination_Region"] = df_clean["Destination_Port"].apply(map_dest_region)
    df_clean["Regional_Lane"] = df_clean["Origin_Region"] + " -> " + df_clean["Destination_Region"]
    df_clean["Trade_Lane"] = df_clean["Origin_Port"] + " -> " + df_clean["Destination_Port"]

    # 6. Commercial, Customer, & Behavioral Features
    print("[*] Feature engineering customer frequency, tiers, and business attributes...")
    df_clean["Business_Vertical"] = df_clean["Business Vertical"].fillna("Air Export Forwarding").astype(str)
    df_clean["Commodity_Group"] = df_clean["Cargo Description"].apply(map_commodity_group)
    df_clean["Company_Group"] = df_clean["Company Group"].fillna("Subagent").astype(str)
    df_clean["Customer_Company"] = df_clean["Customer Company"].fillna("Unknown_Customer").astype(str)
    df_clean["Incoterms"] = df_clean["Incoterms"].apply(clean_incoterms)

    vol_map = df_clean["Customer_Company"].value_counts().to_dict()
    df_clean["Customer_Inquiry_Frequency"] = df_clean["Customer_Company"].map(vol_map).fillna(1).astype(int)
    df_clean["Customer_Tier"] = df_clean["Customer_Inquiry_Frequency"].apply(map_account_tier)

    # 7. Temporal Features & Month-Wise Categorization
    print("[*] Parsing timestamps and computing temporal variables...")
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

    df_clean["Month_Name"] = final_dates.dt.strftime("%B")
    df_clean["Year_Month"] = final_dates.dt.strftime("%Y-%m")

    # 8. Status & Identifiers
    df_clean["is_won"] = (df_clean["Quote Status"] == "Won").astype(int)
    fallback_status = np.where(df_clean["is_won"] == 1, "Won", "Lost")
    df_clean["Inquiry Status"] = df_clean["Inquiry Status"].fillna(pd.Series(fallback_status, index=df_clean.index)).astype(str)
    df_clean["Quote Status"] = df_clean["Quote Status"].fillna(pd.Series(fallback_status, index=df_clean.index)).astype(str)
    df_clean["Inquiry Number"] = df_clean["Inquiry Number"].astype(str)
    df_clean["Quotation Number"] = df_clean["Quotation Number"].fillna(df_clean["Inquiry Number"]).astype(str)
    df_clean["Booked by branch"] = df_clean["Booked by branch"].fillna("Delhi").astype(str)

    # 9. Schema Assembly
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

    branches_df = df_clean[output_cols].copy()
    branches_won_df = branches_df[branches_df["is_won"] == 1].copy()

    # Save branch-specific cleaned files
    os.makedirs(os.path.dirname(out_branches_ml), exist_ok=True)
    branches_df.to_csv(out_branches_ml, index=False)
    branches_won_df.to_csv(out_branches_won, index=False)
    print(f"[✓] Saved PanIndia non-Mumbai branch dataset: {out_branches_ml} ({len(branches_df):,} rows)")

    # 10. Merge with previously cleaned Mumbai and historical records
    print("[*] Merging with existing cleaned Mumbai dataset...")
    if not os.path.exists(bom_cleaned_path):
        raise FileNotFoundError(f"Cleaned Bombay dataset not found at: {bom_cleaned_path}")

    bom_df = pd.read_csv(bom_cleaned_path)
    print(f"    Loaded Cleaned Bombay dataset: {len(bom_df):,} rows.")

    # Combine Bombay + New Branches
    combined_parts = [bom_df, branches_df]

    # Optionally include any unique historical records from orig_cleaned_path not in either
    if os.path.exists(orig_cleaned_path):
        orig_df = pd.read_csv(orig_cleaned_path)
        existing_inqs = set(bom_df["Inquiry Number"].astype(str).str.strip()) | set(branches_df["Inquiry Number"].astype(str).str.strip())
        orig_unique = orig_df[~orig_df["Inquiry Number"].astype(str).str.strip().isin(existing_inqs)].copy()
        if len(orig_unique) > 0:
            # Ensure Month_Name and Year_Month exist
            if "Month_Name" not in orig_unique.columns:
                orig_unique["Month_Name"] = "June"
            if "Year_Month" not in orig_unique.columns:
                orig_unique["Year_Month"] = "2026-06"
            orig_unique = orig_unique[output_cols].copy()
            combined_parts.append(orig_unique)
            print(f"    Appended {len(orig_unique):,} unique historical inquiries from {orig_cleaned_path}.")

    final_combined = pd.concat(combined_parts, ignore_index=True)
    # Deduplicate in case of any duplicate inquiry number, preserving earlier (Bombay first)
    final_combined = final_combined.drop_duplicates(subset=["Inquiry Number"], keep="first").copy()
    final_combined_won = final_combined[final_combined["is_won"] == 1].copy()

    # Create backups of existing combined files if present
    for path in [out_combined_ml, out_combined_won]:
        if os.path.exists(path):
            bak_path = path + ".bak"
            try:
                import shutil
                shutil.copyfile(path, bak_path)
            except Exception:
                pass

    final_combined.to_csv(out_combined_ml, index=False)
    final_combined_won.to_csv(out_combined_won, index=False)

    print("=" * 80)
    print("[✓] Combined Multi-Branch Dataset Successfully Exported:")
    print(f"    - Cleaned Combined ML Dataset : {out_combined_ml} ({len(final_combined):,} rows)")
    print(f"    - Cleaned Combined Won Bench  : {out_combined_won} ({len(final_combined_won):,} rows)")
    print("=" * 80)

    print("\nCombined Branch Breakdown:")
    branch_counts = final_combined["Booked by branch"].value_counts()
    for branch, count in branch_counts.items():
        won = (final_combined[final_combined["Booked by branch"] == branch]["is_won"] == 1).sum()
        win_pct = (won / count) * 100.0
        print(f"    {branch:<15} : {count:>6,} records ({won:>5,} Won, {win_pct:>5.1f}% win rate)")

    print(f"\nTotal Dataset: {len(final_combined):,} records ({len(final_combined_won):,} Won, {len(final_combined_won)/len(final_combined)*100:.1f}% win rate)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess AE GCR PanIndia Branch Inquiries")
    parser.add_argument("--raw", type=str, default="data/raw/AE_GCR_PanIndia.xlsx")
    parser.add_argument("--bom_cleaned", type=str, default="data/processed/AE_GCR_Bombay_Cleaned_ML.csv")
    parser.add_argument("--orig_cleaned", type=str, default="data/processed/Air_Export_Pricing_Cleaned_ML.csv")
    parser.add_argument("--out_branches_ml", type=str, default="data/processed/AE_GCR_PanIndia_Branches_Cleaned_ML.csv")
    parser.add_argument("--out_branches_won", type=str, default="data/processed/AE_GCR_PanIndia_Branches_Won_Benchmark.csv")
    parser.add_argument("--out_combined_ml", type=str, default="data/processed/Air_Export_Pricing_Combined_ML.csv")
    parser.add_argument("--out_combined_won", type=str, default="data/processed/Air_Export_Pricing_Combined_Won_Benchmark.csv")
    args = parser.parse_args()

    run_panindia_pipeline(
        raw_path=args.raw,
        bom_cleaned_path=args.bom_cleaned,
        orig_cleaned_path=args.orig_cleaned,
        out_branches_ml=args.out_branches_ml,
        out_branches_won=args.out_branches_won,
        out_combined_ml=args.out_combined_ml,
        out_combined_won=args.out_combined_won
    )
