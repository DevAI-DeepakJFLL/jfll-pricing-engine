"""
scripts/preprocess_job_wise_profitability.py
================================================================================
Data Preprocessing Pipeline: 6-Month Job Wise Profitability Report
================================================================================
Cleans and standardizes raw/Job_Wise_Profitability_Report_20260922.xlsx
to mirror the 32-column ML schema of Air_Export_Pricing_Cleaned_ML.csv.
"""

import os
import sys
import argparse
from typing import Any, Dict
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
    parse_currency_field,
    get_air_cargo_season,
    compute_cyclical_month
)

# Country to Region mapping for comprehensive destination resolution
COUNTRY_TO_REGION: Dict[str, str] = {
    # Middle East
    'United Arab Emirates': 'Middle East', 'Oman': 'Middle East', 'Qatar': 'Middle East',
    'Saudi Arabia': 'Middle East', 'Bahrain': 'Middle East', 'Kuwait': 'Middle East',
    'Jordan': 'Middle East', 'Lebanon': 'Middle East', 'Iraq': 'Middle East',
    'Israel': 'Middle East', 'Yemen': 'Middle East', 'Iran, Islamic Republic of': 'Middle East',
    'Iran': 'Middle East',
    # Europe
    'United Kingdom': 'Europe', 'Germany': 'Europe', 'Netherlands': 'Europe', 'France': 'Europe',
    'Italy': 'Europe', 'Spain': 'Europe', 'Belgium': 'Europe', 'Switzerland': 'Europe',
    'Sweden': 'Europe', 'Poland': 'Europe', 'Austria': 'Europe', 'Czech Republic': 'Europe',
    'Denmark': 'Europe', 'Ireland': 'Europe', 'Norway': 'Europe', 'Portugal': 'Europe',
    'Finland': 'Europe', 'Greece': 'Europe', 'Hungary': 'Europe', 'Romania': 'Europe',
    'Russian Federation': 'Europe', 'Turkey': 'Europe', 'Ukraine': 'Europe', 'Belarus': 'Europe',
    'Bulgaria': 'Europe', 'Croatia': 'Europe', 'Slovakia': 'Europe', 'Slovenia': 'Europe',
    'Lithuania': 'Europe', 'Latvia': 'Europe', 'Estonia': 'Europe', 'Luxembourg': 'Europe',
    'Cyprus': 'Europe', 'Malta': 'Europe', 'Iceland': 'Europe', 'Albania': 'Europe',
    'Bosnia and Herzegovina': 'Europe', 'North Macedonia': 'Europe', 'Serbia': 'Europe',
    'Montenegro': 'Europe', 'Armenia': 'Europe', 'Georgia': 'Europe', 'Azerbaijan': 'Europe',
    # North America
    'United States': 'North America', 'Canada': 'North America', 'Mexico': 'North America',
    'Bermuda': 'North America',
    # Asia-Pacific
    'Singapore': 'Asia-Pacific', 'Thailand': 'Asia-Pacific', 'China': 'Asia-Pacific',
    'Australia': 'Asia-Pacific', 'Sri Lanka': 'Asia-Pacific', 'Malaysia': 'Asia-Pacific',
    'Hong Kong': 'Asia-Pacific', 'Korea, Republic of': 'Asia-Pacific', 'Japan': 'Asia-Pacific',
    'Maldives': 'Asia-Pacific', 'Indonesia': 'Asia-Pacific', 'Viet Nam': 'Asia-Pacific',
    'Vietnam': 'Asia-Pacific', 'Philippines': 'Asia-Pacific', 'New Zealand': 'Asia-Pacific',
    'Taiwan, Province of China': 'Asia-Pacific', 'Taiwan': 'Asia-Pacific',
    'Bangladesh': 'Asia-Pacific', 'Nepal': 'Asia-Pacific', 'Kazakhstan': 'Asia-Pacific',
    'Uzbekistan': 'Asia-Pacific', 'Cambodia': 'Asia-Pacific', 'Myanmar': 'Asia-Pacific',
    'Afghanistan': 'Asia-Pacific', 'Pakistan': 'Asia-Pacific', 'Brunei Darussalam': 'Asia-Pacific',
    'Fiji': 'Asia-Pacific', 'Papua New Guinea': 'Asia-Pacific', 'Mongolia': 'Asia-Pacific',
    'Kyrgyzstan': 'Asia-Pacific', 'Tajikistan': 'Asia-Pacific', 'Turkmenistan': 'Asia-Pacific',
    'Lao People\'s Democratic Republic': 'Asia-Pacific', 'Lao Peoples Democratic Republic': 'Asia-Pacific',
    'Macao': 'Asia-Pacific', 'Solomon Islands': 'Asia-Pacific',
    # Africa
    'Egypt': 'Africa', 'South Africa': 'Africa', 'Kenya': 'Africa', 'Nigeria': 'Africa',
    'Ghana': 'Africa', 'Tanzania, United Republic of': 'Africa', 'Tanzania': 'Africa',
    'Uganda': 'Africa', 'Ethiopia': 'Africa', 'Morocco': 'Africa', 'Algeria': 'Africa',
    'Tunisia': 'Africa', 'Mauritius': 'Africa', 'Seychelles': 'Africa', 'Zambia': 'Africa',
    'Zimbabwe': 'Africa', 'Rwanda': 'Africa', 'Mozambique': 'Africa', 'Angola': 'Africa',
    'Senegal': 'Africa', 'Cote d\'Ivoire': 'Africa', 'Ivory Coast': 'Africa',
    'Cameroon': 'Africa', 'Namibia': 'Africa', 'Botswana': 'Africa', 'Madagascar': 'Africa',
    'Malawi': 'Africa', 'Mali': 'Africa', 'Sudan': 'Africa', 'South Sudan': 'Africa',
    'Congo': 'Africa', 'Congo, The Democratic Republic of the': 'Africa', 'Gabon': 'Africa',
    'Guinea': 'Africa', 'Benin': 'Africa', 'Burkina Faso': 'Africa', 'Burundi': 'Africa',
    'Chad': 'Africa', 'Djibouti': 'Africa', 'Equatorial Guinea': 'Africa', 'Eritrea': 'Africa',
    'Gambia': 'Africa', 'Liberia': 'Africa', 'Mauritania': 'Africa', 'Niger': 'Africa',
    'Sierra Leone': 'Africa', 'Somalia': 'Africa', 'Togo': 'Africa', 'Libya': 'Africa',
    # Latin America & Caribbean
    'Brazil': 'Latin America', 'Argentina': 'Latin America', 'Chile': 'Latin America',
    'Colombia': 'Latin America', 'Peru': 'Latin America', 'Ecuador': 'Latin America',
    'Uruguay': 'Latin America', 'Paraguay': 'Latin America', 'Panama': 'Latin America',
    'Costa Rica': 'Latin America', 'Guatemala': 'Latin America', 'Honduras': 'Latin America',
    'Dominican Republic': 'Latin America', 'Jamaica': 'Latin America', 'Trinidad and Tobago': 'Latin America',
    'Bolivia, Plurinational State of': 'Latin America', 'El Salvador': 'Latin America',
    'Nicaragua': 'Latin America', 'Venezuela, Bolivarian Republic of': 'Latin America',
    'Antigua and Barbuda': 'Latin America', 'Barbados': 'Latin America', 'Curaçao': 'Latin America',
    'Guyana': 'Latin America', 'Suriname': 'Latin America', 'Haiti': 'Latin America',
    'Bahamas': 'Latin America', 'Belize': 'Latin America', 'Saint Lucia': 'Latin America'
}

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


def clean_branch(raw_branch: Any) -> str:
    """Normalize booking branch code/name to canonical city."""
    if pd.isna(raw_branch):
        return "Mumbai"
    s = str(raw_branch).strip()
    if s in BRANCH_MAP:
        return BRANCH_MAP[s]
    parts = s.split(" - ")
    if len(parts) >= 2:
        return parts[1].strip()
    return s


def derive_business_vertical(row: pd.Series) -> str:
    """Map logistics product, service, and commodity attributes to business vertical."""
    service = str(row.get("Service", "")).strip().lower()
    comm_type = str(row.get("Commodity Type", "")).strip().lower()
    product = str(row.get("Product", "")).strip().lower()

    if "clearance" in service:
        return "Export Clearance"
    if "courier" in comm_type or "courier" in product:
        return "Courier"
    if "perishable" in comm_type or any(k in product for k in ["fruit", "meat", "fish", "flower", "fresh", "perishable"]):
        return "Air Perishable"
    return "Air Export Forwarding"


def derive_destination_region(dest_port: str, country_name: Any) -> str:
    """Resolve destination region using port lookup with country-level fallback."""
    reg = map_dest_region(dest_port)
    if reg != "Other Destination":
        return reg
    c_str = str(country_name).strip() if pd.notna(country_name) else ""
    return COUNTRY_TO_REGION.get(c_str, "Other Destination")


def run_pipeline(
    raw_path: str = "data/raw/Job_Wise_Profitability_Report_20260922.xlsx",
    output_ml_path: str = "data/processed/Job_Wise_Pricing_Cleaned_ML.csv",
    output_won_path: str = "data/processed/Job_Wise_Pricing_Won_Benchmark.csv"
):
    print("=" * 80)
    print("Job-Wise Profitability Report Preprocessing Pipeline")
    print("=" * 80)
    print(f"[*] Ingesting raw Excel: {raw_path}")
    
    # Header is at row index 4 (5th row in Excel)
    df = pd.read_excel(raw_path, skiprows=4)
    df.columns = df.columns.str.strip()
    initial_rows = len(df)
    print(f"    Raw records ingested: {initial_rows:,} rows across {len(df.columns)} columns.")

    # 1. Remove summary row & Non-Air-Export rows
    df = df[df["Job #"].notna()].copy()
    valid_jobs_count = len(df)
    print(f"    Removed summary totals: {initial_rows - valid_jobs_count} rows dropped ({valid_jobs_count:,} remaining).")

    ae_mask = df["Logistics Product"].astype(str).str.strip() == "Air Export"
    df_ae = df[ae_mask].copy()
    print(f"    Filtered for 'Air Export' product: {len(df_ae):,} records retained ({len(df) - len(df_ae)} non-AE dropped).")

    # 2. Track revisions & Deduplicate on Job #
    revision_counts = df_ae["Job #"].value_counts().to_dict()
    df_ae["quote_revision_count"] = df_ae["Job #"].map(revision_counts).fillna(1).astype(int)

    status_priority = {
        "Executed": 1,
        "Shipment Planned": 2,
        "Shipment Cancelled": 3,
        "Shipment Back To Town": 4
    }
    df_ae["status_rank"] = df_ae["Status"].map(status_priority).fillna(99).astype(int)

    df_sorted = df_ae.sort_values(by=["Job #", "status_rank", "Income (INR)"], ascending=[True, True, False])
    df_dedup = df_sorted.drop_duplicates(subset=["Job #"], keep="first").copy()
    print(f"    Deduplication on 'Job #': {len(df_ae) - len(df_dedup)} duplicates removed ({len(df_dedup):,} unique jobs).")

    # 3. Currency parsing & Financial validation
    print("[*] Parsing financial pricing fields and validating margins...")
    df_dedup["Total_Buy_INR"] = df_dedup["Expense (INR)"].apply(parse_currency_field)
    df_dedup["Total_Sell_INR"] = df_dedup["Income (INR)"].apply(parse_currency_field)

    valid_price_mask = (
        (df_dedup["Total_Buy_INR"] > 500.0) &
        (df_dedup["Total_Sell_INR"] > 500.0) &
        (df_dedup["Total_Sell_INR"] >= df_dedup["Total_Buy_INR"])
    )
    df_priced = df_dedup[valid_price_mask].copy()
    dropped_unpriced = len(df_dedup) - len(df_priced)
    print(f"    Valid commercial prices (Buy>500, Sell>500, Sell>=Buy): {len(df_priced):,} ({dropped_unpriced:,} dropped).")

    df_priced["Margin_Amount_INR"] = df_priced["Total_Sell_INR"] - df_priced["Total_Buy_INR"]
    df_priced["Margin_Percentage"] = (df_priced["Margin_Amount_INR"] / df_priced["Total_Buy_INR"]) * 100.0

    # Prune non-commercial outlier margins (0.2% to 60.0%)
    margin_mask = (df_priced["Margin_Percentage"] >= 0.2) & (df_priced["Margin_Percentage"] <= 60.0)
    df_clean = df_priced[margin_mask].copy()
    dropped_margin = len(df_priced) - len(df_clean)
    print(f"    Filtered margin anomalies (<0.2% or >60%): {dropped_margin:,} records pruned.")
    print(f"    Total commercial cohort retained: {len(df_clean):,} records.")

    # 4. Feature Engineering: Identifiers & Status
    print("[*] Feature engineering target schema columns...")
    df_clean["Inquiry Number"] = df_clean["Job #"].astype(str).str.strip()
    df_clean["Quotation Number"] = df_clean["MAWB / MBL # / CN #"].fillna(df_clean["Job #"]).astype(str).str.strip()

    is_won = (df_clean["Status"] == "Executed").astype(int)
    df_clean["is_won"] = is_won
    df_clean["Inquiry Status"] = np.where(is_won == 1, "Won", "Lost")
    df_clean["Quote Status"] = np.where(is_won == 1, "Won", "Lost")

    # 5. Feature Engineering: Weights & Densities
    gross = pd.to_numeric(df_clean["Gross Wt."], errors="coerce").fillna(0.0)
    chargeable = pd.to_numeric(df_clean["Ch. Wt."], errors="coerce").fillna(0.0)

    gross_clean = np.where(gross > 0, gross, np.where(chargeable > 0, chargeable, 1.0))
    chargeable_clean = np.where(chargeable > 0, chargeable, gross_clean)

    df_clean["Gross_Weight_Kg"] = gross_clean
    df_clean["Chargeable_Weight_Kg"] = chargeable_clean
    df_clean["Cargo_Density_Ratio"] = np.clip(gross_clean / chargeable_clean, 0.1, 5.0)

    df_clean["Weight_Tier"] = pd.cut(
        df_clean["Chargeable_Weight_Kg"],
        bins=WEIGHT_TIER_BINS,
        labels=WEIGHT_TIER_LABELS,
        right=False
    ).astype(str)

    # 6. Feature Engineering: Geography & Lanes
    raw_pol = df_clean["POL Name"].fillna(df_clean["POL"]).fillna("Chhatrapati Shivaji Maharaj International Airport").astype(str).str.strip()
    raw_pod = df_clean["POD Name"].fillna(df_clean["POD"]).fillna("Dubai").astype(str).str.strip()

    df_clean["Origin_Port"] = raw_pol.apply(normalize_port_name)
    df_clean["Destination_Port"] = raw_pod.apply(normalize_port_name)

    df_clean["Origin_Region"] = df_clean["Origin_Port"].apply(map_origin_region)
    df_clean["Destination_Region"] = [
        derive_destination_region(p, c) for p, c in zip(df_clean["Destination_Port"], df_clean["POD Country Name"])
    ]
    df_clean["Regional_Lane"] = df_clean["Origin_Region"] + " -> " + df_clean["Destination_Region"]
    df_clean["Trade_Lane"] = df_clean["Origin_Port"] + " -> " + df_clean["Destination_Port"]

    # 7. Feature Engineering: Commercial, Customer & Commodity
    df_clean["Business_Vertical"] = df_clean.apply(derive_business_vertical, axis=1)

    # Commodity categorization using description and type
    comm_combined = df_clean["Commodity Description"].fillna("").astype(str) + " " + df_clean["Commodity Type"].fillna("").astype(str)
    df_clean["Commodity_Group"] = comm_combined.apply(map_commodity_group)

    # Incoterms from Freight terms
    mawb_terms = df_clean["Freight Terms of MAWB/MBL"].fillna("").astype(str).str.upper()
    hawb_terms = df_clean["Freight Terms of HAWB/HBL"].fillna("").astype(str).str.upper()
    df_clean["Incoterms"] = np.where(
        mawb_terms.str.contains("COLLECT") | hawb_terms.str.contains("COLLECT"),
        "FOB",
        np.where(mawb_terms.str.contains("PREPAID") | hawb_terms.str.contains("PREPAID"), "CFR", "FOB")
    )

    company_group_valid = ["Shipper / Consignee", "Subagent", "Overseas Agent", "GSA", "NVOCC / Coloader", "Custom Broker"]
    raw_cg = df_clean["Customer group"].fillna("Subagent").astype(str).str.strip()
    df_clean["Company_Group"] = np.where(raw_cg.isin(company_group_valid), raw_cg, "Subagent")

    df_clean["Customer_Company"] = df_clean["Primary Customer"].fillna("Unknown_Customer").astype(str).str.strip()

    # Calculate inquiry frequency within the cleaned cohort
    cust_freq = df_clean["Customer_Company"].value_counts().to_dict()
    df_clean["Customer_Inquiry_Frequency"] = df_clean["Customer_Company"].map(cust_freq).fillna(1).astype(int)
    df_clean["Customer_Tier"] = df_clean["Customer_Inquiry_Frequency"].apply(map_account_tier)

    # 8. Feature Engineering: Temporal & Branch
    df_clean["Booked by branch"] = df_clean["Shp. Branch"].apply(clean_branch)

    # Date parsing
    exec_dates = pd.to_datetime(df_clean["Job Execution/MIS Date"], errors="coerce")
    job_dates = pd.to_datetime(df_clean["Job Date"], errors="coerce")
    final_dates = exec_dates.fillna(job_dates)

    latency_hours = (exec_dates - job_dates).dt.total_seconds() / 3600.0
    df_clean["Quote_Turnaround_Hours"] = latency_hours.clip(lower=0.0, upper=168.0).fillna(0.0)

    df_clean["Quote_Month"] = final_dates.dt.month.fillna(6).astype(int)
    df_clean["Quote_DayOfWeek"] = final_dates.dt.dayofweek.fillna(0).astype(int)
    df_clean["Is_Month_End"] = (final_dates.dt.day >= 25).astype(int)
    df_clean["Air_Cargo_Season"] = df_clean["Quote_Month"].apply(get_air_cargo_season)
    
    sin_cos = [compute_cyclical_month(m) for m in df_clean["Quote_Month"]]
    df_clean["Month_Sin"] = [sc[0] for sc in sin_cos]
    df_clean["Month_Cos"] = [sc[1] for sc in sin_cos]

    # 9. Schema Assembly & Verification
    target_columns = [
        "Inquiry Number", "Quotation Number", "Inquiry Status", "Quote Status", "is_won",
        "Total_Buy_INR", "Total_Sell_INR", "Margin_Amount_INR", "Margin_Percentage",
        "Chargeable_Weight_Kg", "Gross_Weight_Kg", "Cargo_Density_Ratio", "Weight_Tier",
        "Business_Vertical", "Commodity_Group", "Origin_Port", "Destination_Port",
        "Origin_Region", "Destination_Region", "Regional_Lane", "Trade_Lane",
        "Incoterms", "Company_Group", "Customer_Company", "Customer_Inquiry_Frequency", "Customer_Tier",
        "Booked by branch", "quote_revision_count", "Quote_Turnaround_Hours",
        "Quote_Month", "Quote_DayOfWeek", "Is_Month_End",
        "Month_Sin", "Month_Cos", "Air_Cargo_Season"
    ]

    final_df = df_clean[target_columns].copy()
    won_df = final_df[final_df["is_won"] == 1].copy()

    # Save outputs
    os.makedirs(os.path.dirname(output_ml_path), exist_ok=True)
    final_df.to_csv(output_ml_path, index=False)
    won_df.to_csv(output_won_path, index=False)

    print("=" * 80)
    print(f"[✓] Successfully cleaned and exported datasets:")
    print(f"    - Full ML Dataset      : {output_ml_path} ({len(final_df):,} rows)")
    print(f"    - Won Deals Benchmark  : {output_won_path} ({len(won_df):,} rows)")
    print("=" * 80)

    # Summary Statistics
    print("\nSummary Statistics of Cleaned Dataset:")
    print(f"    Total Records       : {len(final_df):,}")
    print(f"    Won Records         : {len(won_df):,} ({(len(won_df)/len(final_df))*100:.2f}%)")
    print(f"    Mean Buy (INR)      : ₹{final_df['Total_Buy_INR'].mean():,.2f}")
    print(f"    Mean Sell (INR)     : ₹{final_df['Total_Sell_INR'].mean():,.2f}")
    print(f"    Mean Margin %       : {final_df['Margin_Percentage'].mean():.2f}%")
    print(f"    Median Margin %     : {final_df['Margin_Percentage'].median():.2f}%")
    print(f"    Mean Ch. Wt (Kg)    : {final_df['Chargeable_Weight_Kg'].mean():,.1f} kg")
    print(f"    Unique Customers    : {final_df['Customer_Company'].nunique():,}")
    print(f"    Unique Trade Lanes  : {final_df['Trade_Lane'].nunique():,}")
    print("\nWeight Tier Distribution:")
    print(final_df['Weight_Tier'].value_counts())
    print("\nRegional Lane Distribution (Top 10):")
    print(final_df['Regional_Lane'].value_counts().head(10))
    print("\nCommodity Group Distribution:")
    print(final_df['Commodity_Group'].value_counts())
    print("\nBooking Branch Distribution:")
    print(final_df['Booked by branch'].value_counts())
    print("\nAir Cargo Season Distribution:")
    print(final_df['Air_Cargo_Season'].value_counts())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess 6-Month Job Wise Profitability Report")
    parser.add_argument("--raw", type=str, default="data/raw/Job_Wise_Profitability_Report_20260922.xlsx")
    parser.add_argument("--out_ml", type=str, default="data/processed/Job_Wise_Pricing_Cleaned_ML.csv")
    parser.add_argument("--out_won", type=str, default="data/processed/Job_Wise_Pricing_Won_Benchmark.csv")
    args = parser.parse_args()

    run_pipeline(args.raw, args.out_ml, args.out_won)
