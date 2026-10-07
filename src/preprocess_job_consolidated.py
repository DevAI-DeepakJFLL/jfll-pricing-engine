"""
================================================================================
src/preprocess_job_consolidated.py
================================================================================
Enterprise Data Preprocessing & Feature Engineering Pipeline:
Operational Job Wise Consolidated Report (Jan Till Date)

Ingests raw TMS operational shipment ledger data from:
  data/raw/Job Wise Consolidated Report Jan Till Date.xlsx

Executes comprehensive data engineering & statistical validation:
  1. Deduplication on Job No (retaining primary record).
  2. Pure Air Export Forwarding filtering (quarantining Ocean, Air Import, General).
  3. Strict Origin & Destination verification:
     - Origin airport must be an authentic Indian commercial gateway.
     - Destination airport must be international (quarantining domestic/unrouted legs).
  4. Physical plausibility & IATA air cargo standards:
     - Chargeable Weight >= Gross Weight.
     - Standard air freight 0.5 kg rounding up (ceil).
     - Commercial weight envelope: 0.5 kg to 25,000 kg.
     - Cargo density ratio clamped in [0.1, 1.0].
  5. Commercial pricing hygiene & rate bounds:
     - Minimum realized commercial revenue (Total_Sell_INR >= 500.0).
     - Commercial airfreight rate floor (Sell_Rate_Per_Kg >= 20.0 INR/kg).
     - Commercial airfreight rate ceiling (Sell_Rate_Per_Kg <= 15,000.0 INR/kg).
  6. Comprehensive geographic & trade corridor resolution (100% international coverage).
  7. Commodity categorization (Dangerous Goods, Pharma, Perishable, Engineering, Garments, General Cargo, Courier).
  8. Centralized weight tiering and customer inquiry/booking frequency tiers.
  9. Temporal feature engineering: circular harmonic seasonality, turnaround hours, month flags.
  10. Trade lane benchmark buy estimation & schema alignment with Air_Export_Pricing_Combined_ML.csv.

Outputs:
  - data/processed/Job_Wise_Consolidated_Cleaned_ML.csv
  - data/processed/Job_Wise_Consolidated_Won_Benchmark.csv
================================================================================
"""

import os
import sys
import argparse
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd

# Support module and standalone execution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tiers import WEIGHT_TIER_BINS, WEIGHT_TIER_LABELS, map_account_tier, get_weight_tier
from preprocess import (
    normalize_port_name,
    map_origin_region,
    map_dest_region,
    is_valid_indian_origin,
    map_commodity_group,
    clean_incoterms,
    get_air_cargo_season,
    compute_cyclical_month,
)


# ------------------------------------------------------------------------------
# GLOBAL COUNTRY TO FREIGHT CORRIDOR CATALOG
# ------------------------------------------------------------------------------
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


def build_benchmark_margin_lookup(
    reference_ml_path: str = "data/processed/Air_Export_Pricing_Combined_ML.csv"
) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float], Dict[str, float], float]:
    """
    Build hierarchical trade-lane margin benchmark lookup tables from the Won deals
    of the verified reference ML dataset.
    """
    if not os.path.exists(reference_ml_path):
        # Default fallback values if reference file is absent
        return {}, {}, {}, 7.03

    ml_df = pd.read_csv(reference_ml_path)
    won = ml_df[ml_df["is_won"] == 1]
    lane_wt = won.groupby(["Regional_Lane", "Weight_Tier"])["Margin_Percentage"].median().to_dict()
    orig_wt = won.groupby(["Origin_Region", "Weight_Tier"])["Margin_Percentage"].median().to_dict()
    wt = won.groupby("Weight_Tier")["Margin_Percentage"].median().to_dict()
    global_median = float(won["Margin_Percentage"].median())
    return lane_wt, orig_wt, wt, global_median


def preprocess_job_consolidated_report(
    input_path: str = "data/raw/Job Wise Consolidated Report Jan Till Date.xlsx",
    output_ml_path: str = "data/processed/Job_Wise_Consolidated_Cleaned_ML.csv",
    output_won_path: Optional[str] = "data/processed/Job_Wise_Consolidated_Won_Benchmark.csv",
    reference_ml_path: str = "data/processed/Air_Export_Pricing_Combined_ML.csv",
    sheet_name: str = "Main Data"
) -> pd.DataFrame:
    """
    Execute end-to-end data cleaning, quality filtering, feature engineering,
    and schema standardization on the raw Job Wise Consolidated Report.
    """
    print("=" * 80)
    print("Job Wise Consolidated Report: Data Preprocessing & Feature Engineering")
    print("=" * 80)
    print(f"[*] Loading raw dataset: {input_path} (Sheet: '{sheet_name}')")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found at: {input_path}")

    df = pd.read_excel(input_path, sheet_name=sheet_name)
    initial_rows = len(df)
    print(f"    Raw records ingested: {initial_rows:,} rows across {len(df.columns)} columns.")

    # 1. Deduplication on Job No
    print("[*] 1. Deduplicating records on primary Job No...")
    dups_count = int(df["Job No"].duplicated().sum())
    df = df.drop_duplicates(subset=["Job No"], keep="first").copy()
    print(f"    Dropped {dups_count} duplicate rows ({len(df):,} unique jobs remaining).")

    # 2. Filter Logistics Product == 'Air Export'
    print("[*] 2. Filtering for 'Air Export' logistics products...")
    before_prod = len(df)
    df = df[df["Logistics Product"].astype(str).str.strip() == "Air Export"].copy()
    dropped_prod = before_prod - len(df)
    print(f"    Retained {len(df):,} Air Export records (quarantined {dropped_prod:,} non-air-export rows).")

    # 3. Origin Airport Validation (Strict Indian Commercial Gateways)
    print("[*] 3. Validating Indian origin commercial gateways...")
    before_orig = len(df)
    norm_pol = df["POL Name"].fillna(df["POL"]).apply(normalize_port_name)
    df["Origin_Port"] = norm_pol
    valid_origin_mask = df["Origin_Port"].apply(is_valid_indian_origin)
    df = df[valid_origin_mask].copy()
    dropped_orig = before_orig - len(df)
    print(f"    Retained {len(df):,} Indian origin records (quarantined {dropped_orig:,} foreign origin records).")

    # 4. International Destination Validation (International Exports)
    print("[*] 4. Validating international export destinations...")
    before_dest = len(df)
    norm_pod = df["POD Name"].fillna(df["POD"]).apply(normalize_port_name)
    pod_country = df["POD Country Name"].fillna("").astype(str).str.strip()
    is_intl = (pod_country != "India") & (pod_country != "") & (pod_country != "(N/A)") & (norm_pod != "(N/A)")
    df = df[is_intl].copy()
    df["Destination_Port"] = df["POD Name"].fillna(df["POD"]).apply(normalize_port_name)
    dropped_dest = before_dest - len(df)
    print(f"    Retained {len(df):,} international export records (quarantined {dropped_dest:,} domestic/unrouted legs).")

    # 5. Physical Plausibility & IATA Airfreight Constraints
    print("[*] 5. Enforcing IATA physical plausibility & weight bounds...")
    before_wt = len(df)
    gross = pd.to_numeric(df["Gross Wt."], errors="coerce").fillna(0.0)
    chargeable = pd.to_numeric(df["Ch. Wt."], errors="coerce").fillna(0.0)

    # IATA rule: Chargeable Weight >= Gross Weight
    eff_ch_wt = np.maximum(gross, chargeable)
    # Standard airfreight 0.5 kg increment rounding up (ceil)
    eff_ch_wt = np.ceil(eff_ch_wt * 2.0) / 2.0

    df["Gross_Weight_Kg"] = gross
    df["Chargeable_Weight_Kg"] = eff_ch_wt
    df["Cargo_Density_Ratio"] = (df["Gross_Weight_Kg"] / df["Chargeable_Weight_Kg"]).clip(lower=0.1, upper=1.0).round(4)

    weight_mask = (df["Chargeable_Weight_Kg"] >= 0.5) & (df["Chargeable_Weight_Kg"] <= 25000.0) & (df["Gross_Weight_Kg"] > 0)
    df = df[weight_mask].copy()
    dropped_wt = before_wt - len(df)
    print(f"    Retained {len(df):,} records within [0.5, 25,000] kg (quarantined {dropped_wt:,} invalid weight records).")

    # 6. Commercial Pricing Hygiene & Rate Bounds
    print("[*] 6. Applying commercial revenue hygiene and rate-per-kg bounds...")
    before_price = len(df)
    income = pd.to_numeric(df["Income (INR)"], errors="coerce").fillna(0.0)
    est_income = pd.to_numeric(df["Est. Income"], errors="coerce").fillna(0.0)
    eff_income = np.where(income > 0, income, est_income)
    df["Total_Sell_INR"] = eff_income
    df["Sell_Rate_Per_Kg"] = (df["Total_Sell_INR"] / df["Chargeable_Weight_Kg"]).round(2)

    # Commercial bounds: Sell >= 500 INR, rate per kg between 20 and 15,000 INR/kg
    price_mask = (df["Total_Sell_INR"] >= 500.0) & (df["Sell_Rate_Per_Kg"] >= 20.0) & (df["Sell_Rate_Per_Kg"] <= 15000.0)
    df = df[price_mask].copy()
    dropped_price = before_price - len(df)
    print(f"    Retained {len(df):,} commercially valid records (quarantined {dropped_price:,} rate/revenue anomalies).")

    # 7. Identifiers & Outcome Status
    print("[*] 7. Feature engineering identifiers and deal outcomes...")
    df["Inquiry Number"] = df["Job No"].astype(str).str.strip()
    df["Quotation Number"] = df["MAWB / MBL # / CN #"].fillna(df["Job No"]).astype(str).str.strip()
    df["is_won"] = 1
    df["Inquiry Status"] = "Won"
    df["Quote Status"] = "Won"
    df["quote_revision_count"] = 1

    # 8. Weight Tiers
    df["Weight_Tier"] = df["Chargeable_Weight_Kg"].apply(get_weight_tier)

    # 9. Geographic Trade Corridors
    print("[*] 8. Resolving canonical regional lanes and trade corridors...")
    df["Origin_Region"] = df["Origin_Port"].apply(map_origin_region)
    df["Destination_Region"] = [
        derive_destination_region(p, c) for p, c in zip(df["Destination_Port"], df["POD Country Name"])
    ]
    df["Regional_Lane"] = df["Origin_Region"] + " -> " + df["Destination_Region"]
    df["Trade_Lane"] = df["Origin_Port"] + " -> " + df["Destination_Port"]

    # 10. Commodity Categorization & Business Verticals
    print("[*] 9. Classifying commodities and business verticals...")
    df["Business_Vertical"] = df.apply(derive_business_vertical, axis=1)
    comm_combined = (
        df["Commodity Description"].fillna("").astype(str) + " " +
        df["Commodity Type"].fillna("").astype(str) + " " +
        df["Product"].fillna("").astype(str)
    )
    df["Commodity_Group"] = comm_combined.apply(map_commodity_group)

    # 11. Incoterms
    mawb_terms = df["Freight Terms of MAWB/MBL"].fillna("").astype(str).str.upper()
    hawb_terms = df["Freight Terms of HAWB/HBL"].fillna("").astype(str).str.upper()
    df["Incoterms"] = np.where(
        mawb_terms.str.contains("COLLECT") | hawb_terms.str.contains("COLLECT"),
        "FOB",
        np.where(mawb_terms.str.contains("PREPAID") | hawb_terms.str.contains("PREPAID"), "CFR", "FOB")
    )

    # 12. Customer Account Intelligence & Tiering
    print("[*] 10. Engineering customer frequency and account tiering...")
    company_group_valid = ["Shipper / Consignee", "Subagent", "Overseas Agent", "GSA", "NVOCC / Coloader", "Custom Broker"]
    raw_cg = df["Customer group"].fillna("Subagent").astype(str).str.strip()
    df["Company_Group"] = np.where(raw_cg.isin(company_group_valid), raw_cg, "Subagent")
    df["Customer_Company"] = df["Customer"].fillna("Unknown_Customer").astype(str).str.strip()

    cust_counts = df["Customer_Company"].value_counts().to_dict()
    df["Customer_Inquiry_Frequency"] = df["Customer_Company"].map(cust_counts).fillna(1).astype(int)
    df["Customer_Tier"] = df["Customer_Inquiry_Frequency"].apply(map_account_tier)

    # 13. Booking Branch Normalization
    df["Booked by branch"] = df["Shp. Branch"].apply(clean_branch)

    # 14. Temporal Engineering & Seasonality
    print("[*] 11. Engineering temporal latency and harmonic cyclical seasonality...")
    exec_dates = pd.to_datetime(df["Job Execution Date"], errors="coerce")
    job_dates = pd.to_datetime(df["Job Date"], errors="coerce")
    final_dates = exec_dates.fillna(job_dates)

    latency_hours = (exec_dates - job_dates).dt.total_seconds() / 3600.0
    df["Quote_Turnaround_Hours"] = latency_hours.clip(lower=0.0, upper=168.0).fillna(0.0).round(2)
    df["Quote_Month"] = final_dates.dt.month.fillna(6).astype(int)
    df["Quote_DayOfWeek"] = final_dates.dt.dayofweek.fillna(0).astype(int)
    df["Is_Month_End"] = (final_dates.dt.day >= 25).astype(int)
    df["Air_Cargo_Season"] = df["Quote_Month"].apply(get_air_cargo_season)
    
    sin_cos = [compute_cyclical_month(m) for m in df["Quote_Month"]]
    df["Month_Sin"] = [sc[0] for sc in sin_cos]
    df["Month_Cos"] = [sc[1] for sc in sin_cos]
    df["Month_Name"] = final_dates.dt.month_name().fillna("June")
    df["Year_Month"] = final_dates.dt.strftime("%Y-%m").fillna("2026-06")

    # 15. Operational Context & Logistics Features
    df["Carrier"] = df["Carrier"].fillna("Unknown Airline").astype(str).str.strip()
    df["Sales_Person"] = df["Sales Person"].fillna("Unassigned").astype(str).str.strip()
    df["Job_Type"] = df["Job Type"].fillna("Air Export Loose Direct").astype(str).str.strip()
    df["Service"] = df["Service"].fillna("Forwarding").astype(str).str.strip()
    df["POL_Code"] = df["POL"].fillna("").astype(str).str.strip()
    df["POD_Code"] = df["POD"].fillna("").astype(str).str.strip()

    # 16. Benchmark Buy Rate & Margin Estimation
    print("[*] 12. Estimating trade-lane benchmark buy costs and margins...")
    lane_wt_margins, orig_wt_margins, wt_margins, global_margin = build_benchmark_margin_lookup(reference_ml_path)

    def get_benchmark_margin(row: pd.Series) -> float:
        k1 = (row["Regional_Lane"], row["Weight_Tier"])
        if k1 in lane_wt_margins:
            return lane_wt_margins[k1]
        k2 = (row["Origin_Region"], row["Weight_Tier"])
        if k2 in orig_wt_margins:
            return orig_wt_margins[k2]
        k3 = row["Weight_Tier"]
        if k3 in wt_margins:
            return wt_margins[k3]
        return global_margin

    bench_margin = df.apply(get_benchmark_margin, axis=1)
    df["Margin_Percentage"] = bench_margin.round(2)
    df["Total_Buy_INR"] = (df["Total_Sell_INR"] / (1.0 + df["Margin_Percentage"] / 100.0)).round(2)
    df["Margin_Amount_INR"] = (df["Total_Sell_INR"] - df["Total_Buy_INR"]).round(2)
    df["Buy_Cost_Estimated"] = 1

    # 17. Target Schema Assembly & Verification
    schema_cols = [
        # Core 37 ML Features (identical order to Air_Export_Pricing_Combined_ML.csv)
        "Inquiry Number", "Quotation Number", "Inquiry Status", "Quote Status", "is_won",
        "Total_Buy_INR", "Total_Sell_INR", "Margin_Amount_INR", "Margin_Percentage",
        "Chargeable_Weight_Kg", "Gross_Weight_Kg", "Cargo_Density_Ratio", "Weight_Tier",
        "Business_Vertical", "Commodity_Group", "Origin_Port", "Destination_Port",
        "Origin_Region", "Destination_Region", "Regional_Lane", "Trade_Lane",
        "Incoterms", "Company_Group", "Customer_Company", "Customer_Inquiry_Frequency",
        "Customer_Tier", "Booked by branch", "quote_revision_count",
        "Quote_Turnaround_Hours", "Quote_Month", "Quote_DayOfWeek", "Is_Month_End",
        "Month_Sin", "Month_Cos", "Air_Cargo_Season", "Month_Name", "Year_Month",
        # Extended Operational Enrichments
        "Sell_Rate_Per_Kg", "Carrier", "Sales_Person", "Job_Type", "Service",
        "POL_Code", "POD_Code", "Buy_Cost_Estimated"
    ]

    clean_df = df[schema_cols].copy()

    # 18. Integrity Assertions
    assert clean_df.isnull().sum().sum() == 0, f"Integrity failure: {clean_df.isnull().sum().to_dict()}"
    assert (clean_df["Gross_Weight_Kg"] <= clean_df["Chargeable_Weight_Kg"] + 0.01).all(), "IATA constraint violated"
    assert (clean_df["Chargeable_Weight_Kg"] >= 0.5).all() and (clean_df["Chargeable_Weight_Kg"] <= 25000.0).all(), "Weight bounds violated"
    assert (clean_df["Total_Sell_INR"] >= clean_df["Total_Buy_INR"]).all(), "Sell price must be >= Buy price"
    assert (clean_df["Destination_Region"] != "Other Destination").all(), "Unmapped destination regions present"

    print("=" * 80)
    print(f"[✓] Data preprocessing complete! Retained {len(clean_df):,} pure commercial records ({len(clean_df)/initial_rows*100:.1f}%).")
    print(f"    - Origin Gateways: {clean_df['Origin_Port'].nunique()} Indian commercial airports")
    print(f"    - Destination Ports: {clean_df['Destination_Port'].nunique()} global commercial airports")
    print(f"    - Regional Corridors: {clean_df['Regional_Lane'].nunique()} lanes (0.0% unmapped)")
    print(f"    - Airline Carriers: {clean_df['Carrier'].nunique()} airlines")
    print(f"    - Customer Accounts: {clean_df['Customer_Company'].nunique()} active commercial shippers")
    print(f"    - Total Realized Revenue: ₹{clean_df['Total_Sell_INR'].sum():,.2f}")
    print(f"    - Mean Sell Rate/Kg: ₹{clean_df['Sell_Rate_Per_Kg'].mean():.2f}/kg (Median: ₹{clean_df['Sell_Rate_Per_Kg'].median():.2f}/kg)")

    # 19. Export Production Datasets
    os.makedirs(os.path.dirname(output_ml_path), exist_ok=True)
    clean_df.to_csv(output_ml_path, index=False)
    print(f"[*] Successfully saved clean ML dataset to: {output_ml_path}")

    if output_won_path:
        os.makedirs(os.path.dirname(output_won_path), exist_ok=True)
        clean_df.to_csv(output_won_path, index=False)
        print(f"[*] Successfully saved won benchmark dataset to: {output_won_path}")

    return clean_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and standardize Job Wise Consolidated Report for Air Export Pricing")
    parser.add_argument("--input", type=str, default="data/raw/Job Wise Consolidated Report Jan Till Date.xlsx")
    parser.add_argument("--output", type=str, default="data/processed/Job_Wise_Consolidated_Cleaned_ML.csv")
    parser.add_argument("--output_won", type=str, default="data/processed/Job_Wise_Consolidated_Won_Benchmark.csv")
    parser.add_argument("--reference", type=str, default="data/processed/Air_Export_Pricing_Combined_ML.csv")
    args = parser.parse_args()

    preprocess_job_consolidated_report(args.input, args.output, args.output_won, args.reference)
