"""
scripts/ingest_and_split_2025.py
Data Ingestion, Cleaning & 70/30 Split Pipeline for 2025 Inquiries Dataset.

Ingests 'data/Shipmnts_Inquiries Jan Dec 2025.xlsx', performs regex currency cleaning,
enforces commercial economic bounds, applies label purity filtering for won deals,
and generates:
1. data/processed/all_inquiries_2025_cleaned.csv
2. data/processed/won_deals_2025_train_70.csv (chronological 70% train)
3. data/processed/won_deals_2025_test_30.csv (chronological 30% test)
4. data/processed/won_deals_2025_train_70_stratified.csv (stratified by Weight_Tier)
5. data/processed/won_deals_2025_test_30_stratified.csv (stratified by Weight_Tier)
"""

import os
import sys
import re
import argparse
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add root and src to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.customer_normalizer import get_canonical_customer_id, normalize_customer_name
from src.taxonomy import classify_commodity, map_destination_region
from src.tiers import get_weight_tier
from src.preprocess import normalize_port_name, map_origin_region
from src.preprocess_pipeline_v3 import (
    compute_cyclical_month,
    get_air_cargo_season,
    validate_build_gates,
)

DEFAULT_EXCEL_PATH = os.path.join(PROJECT_ROOT, "data", "Shipmnts_Inquiries Jan Dec 2025.xlsx")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

BRANCH_MAP: Dict[str, str] = {
    "BOM - Mumbai - JFL": "Mumbai",
    "AMD - Ahmedabad - JFL": "Ahmedabad",
    "DEL - Delhi - JFL": "Delhi",
    "BLR - Bangalore - JFL": "Bangalore",
    "CCJ - Calicut - JFL": "Calicut",
    "HYD - Hyderabad - JFL": "Hyderabad",
    "COK - Cochin - JFL": "Cochin",
    "TRV - Trivandrum - JFL": "Trivandrum",
    "CCU - Kolkata - JFL": "Kolkata",
    "KNN - Kannur - JFL": "Kannur",
    "MAA - Chennai - JFL": "Chennai",
}

BRANCH_TO_REGION: Dict[str, str] = {
    "Mumbai": "West India",
    "Ahmedabad": "West India",
    "Haryana": "North India",
    "Delhi": "North India",
    "Kolkata": "East India",
    "Bangalore": "South India",
    "Cochin": "South India",
    "Hyderabad": "South India",
    "Chennai": "South India",
    "Calicut": "South India",
    "Trivandrum": "South India",
    "Kannur": "South India",
}

BRANCH_TO_DEFAULT_PORT: Dict[str, str] = {
    "Mumbai": "Chhatrapati Shivaji Maharaj International Airport",
    "Delhi": "Indira Gandhi International Airport",
    "Haryana": "Indira Gandhi International Airport",
    "Ahmedabad": "Ahmedabad",
    "Bangalore": "Bangalore",
    "Hyderabad": "Hyderabad",
    "Cochin": "Cochin",
    "Chennai": "Chennai",
    "Kolkata": "Kolkata",
    "Calicut": "Kozhikode (ex Calicut)",
    "Trivandrum": "Thiruvananthapuram (ex Trivandrum)",
    "Kannur": "Kannur International Airport",
}


def clean_branch(raw_branch: Any) -> str:
    """Normalize booking branch code/name to canonical city name."""
    if pd.isna(raw_branch):
        return "Mumbai"
    s = str(raw_branch).strip()
    if s in BRANCH_MAP:
        return BRANCH_MAP[s]
    parts = s.split(" - ")
    if len(parts) >= 2:
        return parts[1].strip()
    return s


def parse_currency_string(val: Any) -> float:
    """
    Parse monetary amount string with commas, currencies (INR, USD, AED), etc. to pure float.
    Returns np.nan if unparseable or null.
    """
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val) if np.isfinite(val) else np.nan
    s = str(val).strip()
    if not s:
        return np.nan
    cleaned = re.sub(r"[^\d.-]", "", s)
    if not cleaned or cleaned in [".", "-", "-."]:
        return np.nan
    try:
        f = float(cleaned)
        return f if np.isfinite(f) else np.nan
    except ValueError:
        return np.nan


def clean_chargeable_weight(series: pd.Series) -> pd.Series:
    """Ensure numeric chargeable weight rounded to nearest 0.5 kg."""
    numeric = pd.to_numeric(series, errors="coerce")
    return np.round(numeric * 2.0) / 2.0


def standardize_company_group(series: pd.Series) -> pd.Series:
    """
    Standardize Business Received Through -> Company_Group:
    'Direct' -> 'Shipper / Consignee'
    'Subagent' -> 'Subagent'
    'Overseas Nomination' -> 'Overseas Agent'
    """
    mapping = {
        "Direct": "Shipper / Consignee",
        "Subagent": "Subagent",
        "Overseas Nomination": "Overseas Agent",
    }
    return series.map(mapping).fillna("Shipper / Consignee")


def resolve_origin_region(row: pd.Series) -> str:
    """
    Map origin to one of 4 Indian export regions: West India, North India, South India, East India.
    Falls back to booking branch if port is unmapped or null.
    """
    port = row.get("Port Of Loading")
    reg = map_origin_region(port) if pd.notnull(port) else "Other Origin"
    if reg in ["West India", "North India", "South India", "East India"]:
        return reg
    branch = clean_branch(row.get("Booked by branch", ""))
    if branch in BRANCH_TO_REGION:
        return BRANCH_TO_REGION[branch]
    return "West India"


def clean_origin_port(row: pd.Series) -> str:
    """Normalize origin port name to canonical naming with booking branch fallback."""
    port = row.get("Port Of Loading")
    if pd.notnull(port) and str(port).strip():
        norm = normalize_port_name(port)
        if norm != "Unknown":
            return norm
    branch = clean_branch(row.get("Booked by branch"))
    return BRANCH_TO_DEFAULT_PORT.get(branch, "Chhatrapati Shivaji Maharaj International Airport")


def filter_commercial_economic_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce commercial economic bounds:
    Total_Buy_INR > 500.0 and Total_Sell_INR > Total_Buy_INR and 0.5 <= Chargeable_Weight_Kg <= 25000.0
    """
    mask = (
        (df["Total_Buy_INR"] > 500.0)
        & (df["Total_Sell_INR"] > df["Total_Buy_INR"])
        & (df["Chargeable_Weight_Kg"] >= 0.5)
        & (df["Chargeable_Weight_Kg"] <= 25000.0)
    )
    return df[mask].copy()


def recalculate_margins(df: pd.DataFrame) -> pd.DataFrame:
    """Recalculate exact margin amount and percentage."""
    out = df.copy()
    out["Margin_Amount_INR"] = (out["Total_Sell_INR"] - out["Total_Buy_INR"]).round(2)
    out["Margin_Percentage"] = (out["Margin_Amount_INR"] / out["Total_Buy_INR"] * 100.0)
    return out


def engineer_date_features(df: pd.DataFrame, date_col: str = "Created At") -> pd.DataFrame:
    """
    Derive leak-free, consistent date features:
    quote_date, Quote_Month, Year_Month, Month_Name, Quote_DayOfWeek, Is_Month_End,
    Month_Sin, Month_Cos, Air_Cargo_Season.
    """
    out = df.copy()
    parsed_dates = pd.to_datetime(out[date_col], errors="coerce")
    if parsed_dates.isnull().any():
        raise ValueError(f"Found {parsed_dates.isnull().sum()} unparseable dates in {date_col}.")

    out["quote_date"] = parsed_dates.dt.strftime("%Y-%m-%d")
    out["Quote_Month"] = parsed_dates.dt.month.astype(int)
    out["Year_Month"] = parsed_dates.dt.strftime("%Y-%m")
    out["Month_Name"] = parsed_dates.dt.month_name()
    out["Quote_DayOfWeek"] = parsed_dates.dt.dayofweek.astype(int)
    out["Is_Month_End"] = (parsed_dates.dt.day >= 25).astype(int)

    sin_cos = [compute_cyclical_month(m) for m in out["Quote_Month"]]
    out["Month_Sin"] = [sc[0] for sc in sin_cos]
    out["Month_Cos"] = [sc[1] for sc in sin_cos]
    out["Air_Cargo_Season"] = out["Quote_Month"].apply(get_air_cargo_season)
    return out


def clean_and_enrich_inquiries(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning and feature enrichment on 2025 raw inquiries dataset.
    """
    df = df_raw.copy()
    df.columns = [c.strip() for c in df.columns]

    # 1. Financial Columns Currency Parsing
    df["Total_Buy_INR"] = df["Total Buy"].apply(parse_currency_string)
    df["Total_Sell_INR"] = df["Total Sell"].apply(parse_currency_string)
    if "Unit Freight Buy" in df.columns:
        df["Unit_Buy_INR"] = df["Unit Freight Buy"].apply(parse_currency_string)
    if "Unit Freight Sell" in df.columns:
        df["Unit_Sell_INR"] = df["Unit Freight Sell"].apply(parse_currency_string)

    # 2. Chargeable Weight Normalization
    df["Chargeable_Weight_Kg"] = clean_chargeable_weight(df["Chargeable Weight"])

    # 3. Commercial Economic Filter
    df = filter_commercial_economic_bounds(df)

    # 4. Strict Margin Identity Recalculation
    df = recalculate_margins(df)

    # 5. Customer Entity Normalization
    df["customer_id"] = df["Customer Company"].apply(get_canonical_customer_id)
    df["Customer_Company"] = df["Customer Company"].apply(normalize_customer_name)

    # 6. Origin Port & Region
    df["Origin_Port"] = df.apply(clean_origin_port, axis=1)
    df["Origin_Region"] = df.apply(resolve_origin_region, axis=1)

    # 7. Destination Port & Region
    df["Destination_Port"] = df["Port Of Discharge"].fillna("Unknown").astype(str).str.strip()
    df["Destination_Region"] = df["Destination_Port"].apply(map_destination_region)
    df["Regional_Lane"] = df["Origin_Region"] + " -> " + df["Destination_Region"]
    df["Trade_Lane"] = df["Origin_Port"] + " -> " + df["Destination_Port"]

    # 8. Commodity Taxonomy
    if "Commodity Type" in df.columns:
        df["Commodity_Group"] = df["Commodity Type"].apply(lambda x: classify_commodity(str(x)))
    else:
        df["Commodity_Group"] = "General Cargo"

    # 9. Company Group Standardization
    if "Business Received Through" in df.columns:
        df["Company_Group"] = standardize_company_group(df["Business Received Through"])
    else:
        df["Company_Group"] = "Shipper / Consignee"

    # 10. Weight Tier
    df["Weight_Tier"] = df["Chargeable_Weight_Kg"].apply(get_weight_tier)

    # 11. Date Engineering
    df = engineer_date_features(df, date_col="Created At")

    # 12. Operational Attributes & Turnaround
    if "Quote Created At" in df.columns and "Created At" in df.columns:
        created = pd.to_datetime(df["Created At"], errors="coerce")
        quote_created = pd.to_datetime(df["Quote Created At"], errors="coerce")
        turnaround = (quote_created - created).dt.total_seconds() / 3600.0
        df["Quote_Turnaround_Hours"] = turnaround.clip(lower=0.05).fillna(1.0).round(2)
    else:
        df["Quote_Turnaround_Hours"] = 1.0

    # 13. Status Flags
    df["is_won"] = (df["Quote Status"] == "Won").astype(int)
    df["Buy_Cost_Estimated"] = 0

    # 14. Carrier & Staff Attributes
    if "Quoted Carrier" in df.columns:
        df["Quoted Carrier"] = df["Quoted Carrier"].fillna("Unknown Airline").astype(str).str.strip()
    if "Sales Person" in df.columns:
        df["Sales Person"] = df["Sales Person"].fillna("Unassigned").astype(str).str.strip()
    if "Booked by branch" in df.columns:
        df["Booked by branch"] = df["Booked by branch"].apply(clean_branch)

    return df.reset_index(drop=True)


def chronological_split_70_30(
    df: pd.DataFrame, train_pct: float = 0.70
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological 70/30 split ordered monotonically by quote_date ascending.
    """
    sort_cols = [c for c in ["quote_date", "Created At"] if c in df.columns]
    sorted_df = df.sort_values(by=sort_cols, ascending=True).reset_index(drop=True)
    n_train = int(len(sorted_df) * train_pct)
    train_df = sorted_df.iloc[:n_train].copy().reset_index(drop=True)
    test_df = sorted_df.iloc[n_train:].copy().reset_index(drop=True)
    return train_df, test_df


def stratified_split_70_30(
    df: pd.DataFrame, test_size: float = 0.30, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified 70/30 split using Weight_Tier.
    """
    if "Weight_Tier" not in df.columns:
        df["Weight_Tier"] = df["Chargeable_Weight_Kg"].apply(get_weight_tier)
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["Weight_Tier"],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def run_ingest_and_split(
    excel_path: str = DEFAULT_EXCEL_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> Dict[str, Any]:
    """
    Orchestrate full ingestion, cleaning, validation, and split generation.
    """
    print("=" * 80)
    print("2025 Inquiries Ingestion & 70/30 Split Pipeline")
    print("=" * 80)

    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Input Excel file not found at: {excel_path}")

    print(f"[*] Loading raw dataset from: {excel_path} (Sheet: 'Data')...")
    df_raw = pd.read_excel(excel_path, sheet_name="Data")
    print(f"    Raw records loaded: {len(df_raw):,} rows, {len(df_raw.columns)} columns")

    print("[*] Performing financial currency parsing, entity resolution & date engineering...")
    df_cleaned = clean_and_enrich_inquiries(df_raw)
    print(f"    Commercially valid inquiries retained: {len(df_cleaned):,} rows")

    # Label Purity Gate: Won Deals
    print("[*] Applying label purity gate (Quote Status == 'Won') for margin dataset...")
    won_deals_df = df_cleaned[df_cleaned["Quote Status"] == "Won"].copy().reset_index(drop=True)
    print(f"    Won deals (100% won, zero rejected quotes): {len(won_deals_df):,} rows")

    # Chronological Split
    print("[*] Generating chronological 70/30 split...")
    won_train_70, won_test_30 = chronological_split_70_30(won_deals_df, train_pct=0.70)
    print(f"    Chronological Train (70%): {len(won_train_70):,} rows ({won_train_70['quote_date'].min()} to {won_train_70['quote_date'].max()})")
    print(f"    Chronological Test (30%):  {len(won_test_30):,} rows ({won_test_30['quote_date'].min()} to {won_test_30['quote_date'].max()})")

    # Stratified Split
    print("[*] Generating stratified 70/30 split (stratify=Weight_Tier)...")
    won_train_strat, won_test_strat = stratified_split_70_30(won_deals_df, test_size=0.30, random_state=42)
    print(f"    Stratified Train (70%):    {len(won_train_strat):,} rows")
    print(f"    Stratified Test (30%):     {len(won_test_strat):,} rows")

    # Validate Hard Build Gates
    print("[*] Enforcing hard build gates via validate_build_gates...")
    validate_build_gates(df_cleaned, is_margin_dataset=False, dataset_name="all_inquiries_2025_cleaned.csv")
    validate_build_gates(won_train_70, is_margin_dataset=True, dataset_name="won_deals_2025_train_70.csv")
    validate_build_gates(won_test_30, is_margin_dataset=True, dataset_name="won_deals_2025_test_30.csv")
    validate_build_gates(won_train_strat, is_margin_dataset=True, dataset_name="won_deals_2025_train_70_stratified.csv")
    validate_build_gates(won_test_strat, is_margin_dataset=True, dataset_name="won_deals_2025_test_30_stratified.csv")
    print("    [✓] All 5 datasets passed validate_build_gates assertions successfully.")

    # Save to disk
    os.makedirs(output_dir, exist_ok=True)
    all_inq_path = os.path.join(output_dir, "all_inquiries_2025_cleaned.csv")
    train_path = os.path.join(output_dir, "won_deals_2025_train_70.csv")
    test_path = os.path.join(output_dir, "won_deals_2025_test_30.csv")
    train_strat_path = os.path.join(output_dir, "won_deals_2025_train_70_stratified.csv")
    test_strat_path = os.path.join(output_dir, "won_deals_2025_test_30_stratified.csv")

    print(f"[*] Persisting datasets to: {output_dir}...")
    df_cleaned.to_csv(all_inq_path, index=False)
    won_train_70.to_csv(train_path, index=False)
    won_test_30.to_csv(test_path, index=False)
    won_train_strat.to_csv(train_strat_path, index=False)
    won_test_strat.to_csv(test_strat_path, index=False)

    print(f"    - Cleaned Inquiries:     {all_inq_path} ({len(df_cleaned):,} rows)")
    print(f"    - Won Deals Train (70%): {train_path} ({len(won_train_70):,} rows)")
    print(f"    - Won Deals Test (30%):  {test_path} ({len(won_test_30):,} rows)")
    print(f"    - Stratified Train (70%):{train_strat_path} ({len(won_train_strat):,} rows)")
    print(f"    - Stratified Test (30%): {test_strat_path} ({len(won_test_strat):,} rows)")
    print("=" * 80)
    print("Pipeline Execution Completed Successfully.")
    print("=" * 80)

    return {
        "all_inquiries_count": len(df_cleaned),
        "won_deals_count": len(won_deals_df),
        "train_70_count": len(won_train_70),
        "test_30_count": len(won_test_30),
        "train_strat_count": len(won_train_strat),
        "test_strat_count": len(won_test_strat),
        "train_date_range": (won_train_70["quote_date"].min(), won_train_70["quote_date"].max()),
        "test_date_range": (won_test_30["quote_date"].min(), won_test_30["quote_date"].max()),
    }


def main():
    parser = argparse.ArgumentParser(description="Ingest and Split 2025 Air Freight Inquiries")
    parser.add_argument("--excel", type=str, default=DEFAULT_EXCEL_PATH, help="Path to raw 2025 Excel file")
    parser.add_argument("--outdir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory for processed CSVs")
    args = parser.parse_args()

    run_ingest_and_split(excel_path=args.excel, output_dir=args.outdir)


if __name__ == "__main__":
    main()
