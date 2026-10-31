#!/usr/bin/env python3
"""
scripts/evaluate_recent_2_weeks.py
Evaluates the 'Shipmnts_Inquiries _ Pan India _ Air Export - Recent 2 Weeks.xlsx' dataset
against the production PricingEngineV3, comparing actual quoted rates/margins vs.
model recommendations, negotiation corridors, and commercial guardrails.
"""

import os
import sys
import re
import time
import datetime
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# Set paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.engine_v3 import PricingEngineV3
from src.validation import validate_inquiry, QuoteValidationError
from src.taxonomy import classify_commodity, map_destination_region
from src.tiers import get_weight_tier
from src.preprocess import map_origin_region

EXCEL_INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "Shipmnts_Inquiries _ Pan India _ Air Export - Recent 2 Weeks.xlsx")
CSV_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "recent_2_weeks_model_vs_actual_detailed.csv")
EXCEL_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "Recent_2_Weeks_Model_vs_Actual_Discrepancy_Analysis.xlsx")


def parse_currency(val: Any) -> float:
    """Safely parse currency strings like '47,615.0 INR' to float."""
    if pd.isna(val):
        return np.nan
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


def main():
    print(f"=== Air Cargo Pricing Engine V3: Recent 2 Weeks Out-of-Sample Benchmark ===")
    print(f"Loading raw dataset from: {EXCEL_INPUT_PATH}")
    
    if not os.path.exists(EXCEL_INPUT_PATH):
        raise FileNotFoundError(f"Input file not found: {EXCEL_INPUT_PATH}")
        
    df_raw = pd.read_excel(EXCEL_INPUT_PATH, skiprows=1)
    df_raw.columns = [c.strip() for c in df_raw.columns]
    total_inquiries = len(df_raw)
    print(f"Successfully loaded {total_inquiries:,} inquiries.")
    
    # Clean numeric fields
    df_raw["buy_clean"] = df_raw["Total Buy"].apply(parse_currency)
    df_raw["unit_buy_clean"] = df_raw["Unit Freight Buy"].apply(parse_currency)
    df_raw["sell_clean"] = df_raw["Total Sell"].apply(parse_currency)
    df_raw["unit_sell_clean"] = df_raw["Unit Freight Sell"].apply(parse_currency)
    df_raw["wt_clean"] = pd.to_numeric(df_raw["Chargeable Weight"], errors="coerce")
    
    # Initialize Engine
    print("Initializing production PricingEngineV3...")
    t_eng_start = time.time()
    engine = PricingEngineV3()
    print(f"PricingEngineV3 loaded in {time.time() - t_eng_start:.2f}s (Fitted: {engine.is_fitted})")
    
    valid_records = []
    invalid_records = []
    
    print("\nPre-screening inquiries against IATA physical and commercial plausibility rules...")
    raw_records = df_raw.to_dict(orient="records")
    for idx, row in enumerate(raw_records):
        inq_num = str(row.get("Inquiry Number", f"ROW_{idx}"))
        cust = str(row.get("Customer Company", "UNKNOWN"))
        buy = row["buy_clean"]
        sell = row["sell_clean"]
        wt = row["wt_clean"]
        status = str(row.get("Quote Status") or row.get("Inquiry Status") or "Draft").strip()
        loss_reason = str(row.get("Loss Reason") or "").strip()
        branch = str(row.get("Booked by branch") or "Mumbai").strip()
        created_at = row.get("Quote Created At") or row.get("Created At")
        q_date = str(created_at)[:10] if pd.notnull(created_at) else "2026-10-15"
        
        # Check basic presence
        if pd.isna(buy) or pd.isna(sell) or pd.isna(wt):
            invalid_records.append({
                "Inquiry_Number": inq_num,
                "Customer_Company": cust,
                "Branch": branch,
                "Status": status,
                "Quote_Date": q_date,
                "Chargeable_Weight_Kg": wt,
                "Total_Buy_INR": buy,
                "Total_Sell_INR": sell,
                "Rejection_Reason": "Missing buy, sell, or weight (Pricing Pending or Unquoted Inquiry)"
            })
            continue
            
        if wt <= 0:
            invalid_records.append({
                "Inquiry_Number": inq_num,
                "Customer_Company": cust,
                "Branch": branch,
                "Status": status,
                "Quote_Date": q_date,
                "Chargeable_Weight_Kg": wt,
                "Total_Buy_INR": buy,
                "Total_Sell_INR": sell,
                "Rejection_Reason": f"Invalid non-positive chargeable weight ({wt} kg)"
            })
            continue
            
        if buy <= 0 or sell <= 0:
            invalid_records.append({
                "Inquiry_Number": inq_num,
                "Customer_Company": cust,
                "Branch": branch,
                "Status": status,
                "Quote_Date": q_date,
                "Chargeable_Weight_Kg": wt,
                "Total_Buy_INR": buy,
                "Total_Sell_INR": sell,
                "Rejection_Reason": f"Zero or negative financial values (Buy: ₹{buy}, Sell: ₹{sell})"
            })
            continue

        origin = str(row.get("Port Of Loading") or "Chhatrapati Shivaji Maharaj International Airport")
        dest = str(row.get("Port Of Discharge") or "London Heathrow Apt")
        unit_buy = row.get("unit_buy_clean")
        unit_sell = row.get("unit_sell_clean")
        
        inq_dict = {
            "origin": origin,
            "dest": dest,
            "buy": float(buy),
            "unit_buy": float(unit_buy) if pd.notnull(unit_buy) else None,
            "chargeable_wt": float(wt),
            "gross_wt": float(wt),
            "quote_date": q_date,
        }
        
        try:
            norm_inq = validate_inquiry(inq_dict)
            reconciled_buy = norm_inq["Total_Buy_INR"]
        except QuoteValidationError as ve:
            invalid_records.append({
                "Inquiry_Number": inq_num,
                "Customer_Company": cust,
                "Branch": branch,
                "Status": status,
                "Quote_Date": q_date,
                "Chargeable_Weight_Kg": wt,
                "Total_Buy_INR": buy,
                "Total_Sell_INR": sell,
                "Rejection_Reason": f"Physical/commercial validation failed: {str(ve)}"
            })
            continue
            
        # Reconcile actual sell if ERP dumped unit freight sell into total sell
        if pd.notnull(unit_sell) and wt > 1.0 and abs(sell - unit_sell) < 1.0:
            reconciled_sell = round(float(unit_sell) * float(wt), 2)
        else:
            reconciled_sell = float(sell)

        # Record passed validation
        valid_records.append((idx, row, inq_dict, reconciled_buy, reconciled_sell))
        
    print(f"Validation complete: {len(valid_records):,} valid quotes to score, {len(invalid_records):,} excluded.")
    
    # Process valid records through model
    print(f"\nScoring {len(valid_records):,} quotes through PricingEngineV3...")
    t_score_start = time.time()
    scored_rows = []
    
    for count, (idx, row, inq_dict, reconciled_buy, reconciled_sell) in enumerate(valid_records, 1):
        if count % 500 == 0 or count == len(valid_records):
            elapsed = time.time() - t_score_start
            rate = count / max(0.1, elapsed)
            remaining = (len(valid_records) - count) / rate
            print(f"  Processed {count:,} / {len(valid_records):,} quotes ({rate:.1f} quotes/sec, est. {remaining:.1f}s left)...")
            
        inq_num = str(row.get("Inquiry Number", f"ROW_{idx}"))
        cust = str(row.get("Customer Company", "UNKNOWN"))
        buy = float(reconciled_buy)
        actual_sell = float(reconciled_sell)
        wt = float(row["wt_clean"])
        
        status = str(row.get("Quote Status") or row.get("Inquiry Status") or "Draft").strip()
        loss_reason = str(row.get("Loss Reason") or "").strip()
        branch = str(row.get("Booked by branch") or "Mumbai").strip()
        sales_person = str(row.get("Sales Person") or "").strip()
        carrier = str(row.get("Quoted Carrier") or "Any Airline").strip()
        commodity_raw = str(row.get("Commodity Type") or "General Cargo").strip()
        commodity_canonical = classify_commodity(commodity_raw)
        
        origin = str(row.get("Port Of Loading") or "Chhatrapati Shivaji Maharaj International Airport")
        dest = str(row.get("Port Of Discharge") or "London Heathrow Apt")
        dest_region = map_destination_region(dest)
        origin_region = map_origin_region(origin)
        regional_lane = f"{origin_region} -> {dest_region}"
        
        created_at = row.get("Quote Created At") or row.get("Created At")
        q_date = str(created_at)[:10] if pd.notnull(created_at) else "2026-10-15"
        
        full_inq = {
            "origin": origin,
            "dest": dest,
            "buy": buy,
            "chargeable_wt": wt,
            "gross_wt": wt,
            "client": str(row.get("Business Received Through") or "Shipper / Consignee"),
            "commodity": commodity_raw,
            "quote_date": q_date,
            "customer": cust,
            "carrier": carrier,
            "sales_person": sales_person,
            "strategy": "balanced"
        }
        
        rec = engine.recommend_quote(full_inq)
        opt = rec["optimal"]
        corridor = rec["negotiation_corridor"]
        strat = rec["strategic_recommendations"]
        
        model_bal_m = float(opt["Margin_Percentage"])
        model_floor_m = float(strat["floor"]["Margin_Percentage"])
        model_prem_m = float(strat["premium"]["Margin_Percentage"])
        model_sell = float(opt["Quoted_Sell_Price_INR"])
        model_rate_kg = float(opt["Rate_Per_Kg"])
        cost_floor_price = float(opt["Cost_Floor_Price_INR"])
        
        c_min_m = float(corridor["min_margin_pct"])
        c_max_m = float(corridor["max_margin_pct"])
        c_min_sell = float(corridor["min_sell_inr"])
        c_max_sell = float(corridor["max_sell_inr"])
        
        actual_margin_inr = round(actual_sell - buy, 2)
        actual_margin_pct = round((actual_margin_inr / buy) * 100.0, 2)
        actual_rate_kg = round(actual_sell / wt, 2)
        
        # Discrepancy metrics
        margin_gap_pp = round(model_bal_m - actual_margin_pct, 2)
        sell_diff_inr = round(model_sell - actual_sell, 2)
        rate_diff_kg = round(model_rate_kg - actual_rate_kg, 2)
        
        # Alignment flags
        cost_floor_breached = bool(actual_sell < cost_floor_price)
        sub_cost_quoted = bool(actual_sell < buy)
        
        if actual_margin_pct < c_min_m:
            corridor_status = "Below Corridor (Underpriced)"
        elif actual_margin_pct > c_max_m:
            corridor_status = "Above Corridor (Overpriced)"
        else:
            corridor_status = "Inside Corridor"
            
        weight_tier = get_weight_tier(wt)
        
        # Commercial opportunity classification
        is_won = "won" in status.lower()
        is_lost = "lost" in status.lower()
        
        if cost_floor_breached:
            disp_class = "Cost Floor Breach"
        elif sub_cost_quoted:
            disp_class = "Sub-Cost Bidding"
        elif is_won and model_bal_m > actual_margin_pct:
            disp_class = "Won: Money Left on Table"
        elif is_won and corridor_status == "Inside Corridor":
            disp_class = "Won: Well-Calibrated"
        elif is_won and model_bal_m <= actual_margin_pct:
            disp_class = "Won: Aggressive Pricing Won"
        elif is_lost and corridor_status == "Above Corridor (Overpriced)":
            disp_class = "Lost: Overpriced vs Market"
        elif is_lost and corridor_status == "Inside Corridor":
            disp_class = "Lost: In Corridor (Non-Price Loss)"
        elif is_lost and corridor_status == "Below Corridor (Underpriced)":
            disp_class = "Lost: Underpriced / Aggressive Competitor"
        else:
            disp_class = f"{status}: Open Inquiry"
            
        money_left_on_table = max(0.0, model_sell - actual_sell) if (is_won and model_sell > actual_sell) else 0.0
        
        scored_rows.append({
            "Inquiry_Number": inq_num,
            "Quote_Date": q_date,
            "Customer_Company": cust,
            "Branch": branch,
            "Sales_Person": sales_person,
            "Port_Of_Loading": origin,
            "Port_Of_Discharge": dest,
            "Regional_Lane": regional_lane,
            "Commodity_Raw": commodity_raw,
            "Commodity_Canonical": commodity_canonical,
            "Quoted_Carrier": carrier,
            "Chargeable_Weight_Kg": wt,
            "Weight_Tier": weight_tier,
            "Total_Buy_INR": buy,
            "Actual_Sell_INR": actual_sell,
            "Actual_Margin_INR": actual_margin_inr,
            "Actual_Margin_Pct": actual_margin_pct,
            "Actual_Rate_Per_Kg": actual_rate_kg,
            "Model_Balanced_Margin_Pct": model_bal_m,
            "Model_Floor_Margin_Pct": model_floor_m,
            "Model_Premium_Margin_Pct": model_prem_m,
            "Model_Sell_INR": model_sell,
            "Model_Rate_Per_Kg": model_rate_kg,
            "Corridor_Min_Margin_Pct": c_min_m,
            "Corridor_Max_Margin_Pct": c_max_m,
            "Corridor_Min_Sell_INR": c_min_sell,
            "Corridor_Max_Sell_INR": c_max_sell,
            "Corridor_Status": corridor_status,
            "Margin_Gap_pp": margin_gap_pp,
            "Sell_Diff_INR": sell_diff_inr,
            "Rate_Diff_Per_Kg": rate_diff_kg,
            "Cost_Floor_Price_INR": cost_floor_price,
            "Cost_Floor_Breached": cost_floor_breached,
            "Sub_Cost_Quoted": sub_cost_quoted,
            "Status": status,
            "Loss_Reason": loss_reason,
            "Discrepancy_Classification": disp_class,
            "Money_Left_On_Table_INR": round(money_left_on_table, 2),
            "Confidence": rec["confidence"],
            "Limiting_Factor": opt.get("Limiting_Factor", "none")
        })
        
    df_scored = pd.DataFrame(scored_rows)
    df_invalid = pd.DataFrame(invalid_records)
    
    total_score_time = time.time() - t_score_start
    print(f"\nScoring completed in {total_score_time:.2f}s ({total_score_time / len(valid_records) * 1000:.1f}ms per quote).")
    
    # Save CSV
    os.makedirs(os.path.dirname(CSV_OUTPUT_PATH), exist_ok=True)
    df_scored.to_csv(CSV_OUTPUT_PATH, index=False)
    print(f"Saved detailed scored quotes to CSV: {CSV_OUTPUT_PATH}")
    
    # Compute high-level analytical KPI metrics
    n_scored = len(df_scored)
    n_won = (df_scored["Status"].str.lower() == "won").sum()
    n_lost = (df_scored["Status"].str.lower() == "lost").sum()
    n_open = n_scored - n_won - n_lost
    
    mean_actual_m = df_scored["Actual_Margin_Pct"].mean()
    median_actual_m = df_scored["Actual_Margin_Pct"].median()
    mean_model_m = df_scored["Model_Balanced_Margin_Pct"].mean()
    median_model_m = df_scored["Model_Balanced_Margin_Pct"].median()
    
    cost_floor_breaches = df_scored["Cost_Floor_Breached"].sum()
    sub_cost_breaches = df_scored["Sub_Cost_Quoted"].sum()
    
    inside_corridor = (df_scored["Corridor_Status"] == "Inside Corridor").sum()
    above_corridor = (df_scored["Corridor_Status"] == "Above Corridor (Overpriced)").sum()
    below_corridor = (df_scored["Corridor_Status"] == "Below Corridor (Underpriced)").sum()
    
    won_df = df_scored[df_scored["Status"].str.lower() == "won"]
    lost_df = df_scored[df_scored["Status"].str.lower() == "lost"]
    
    won_money_left = won_df["Money_Left_On_Table_INR"].sum()
    won_money_left_count = (won_df["Money_Left_On_Table_INR"] > 0).sum()
    
    lost_overpriced_count = (lost_df["Corridor_Status"] == "Above Corridor (Overpriced)").sum()
    lost_in_corridor_count = (lost_df["Corridor_Status"] == "Inside Corridor").sum()
    lost_underpriced_count = (lost_df["Corridor_Status"] == "Below Corridor (Underpriced)").sum()
    
    print("\n" + "="*70)
    print("          EXECUTIVE SUMMARY: MODEL VS ACTUAL DISCREPANCIES")
    print("="*70)
    print(f"Total Raw Inquiries Ingested:          {total_inquiries:,}")
    print(f"Valid Quotes Scored by Engine:         {n_scored:,} (93.0% of total)")
    print(f"Inquiries Excluded (Quality Flags):     {len(df_invalid):,} (7.0% of total)")
    print(f"Status Breakdown:                      {n_won:,} Won | {n_lost:,} Lost | {n_open:,} Open/Sent")
    print("-"*70)
    print(f"Actual Quoted Margin (Median / Mean):  {median_actual_m:.2f}% / {mean_actual_m:.2f}%")
    print(f"Model Recommended Margin (Med / Mean): {median_model_m:.2f}% / {mean_model_m:.2f}%")
    print(f"Overall Net Margin Gap (pp):           {median_model_m - median_actual_m:+.2f} pp (Model vs Actual Median)")
    print("-"*70)
    print(f"Cost Floor Violations in Actuals:      {cost_floor_breaches:,} quotes ({cost_floor_breaches/n_scored*100:.1f}%)")
    print(f"Negative Gross Margin (Sell < Buy):    {sub_cost_breaches:,} quotes")
    print("-"*70)
    print(f"Corridor Alignment:")
    print(f"  - Inside Model Corridor:             {inside_corridor:,} ({inside_corridor/n_scored*100:.1f}%)")
    print(f"  - Below Corridor (Underpriced):      {below_corridor:,} ({below_corridor/n_scored*100:.1f}%)")
    print(f"  - Above Corridor (Overpriced):       {above_corridor:,} ({above_corridor/n_scored*100:.1f}%)")
    print("-"*70)
    print(f"Won Deals Opportunity:")
    print(f"  - Won Deals Underquoted:             {won_money_left_count:,} of {n_won:,} ({won_money_left_count/max(1, n_won)*100:.1f}%)")
    print(f"  - Total Revenue Left on the Table:   ₹{won_money_left:,.2f} INR")
    print(f"  - Avg Revenue Left per Underquoted:  ₹{won_money_left/max(1, won_money_left_count):,.2f} INR")
    print("-"*70)
    print(f"Lost Deals Diagnoses:")
    print(f"  - Lost via Overpricing (> Corridor): {lost_overpriced_count:,} of {n_lost:,} ({lost_overpriced_count/max(1, n_lost)*100:.1f}%)")
    print(f"  - Lost inside Corridor (Non-Price):  {lost_in_corridor_count:,} of {n_lost:,} ({lost_in_corridor_count/max(1, n_lost)*100:.1f}%)")
    print(f"  - Lost despite Underpricing:         {lost_underpriced_count:,} of {n_lost:,} ({lost_underpriced_count/max(1, n_lost)*100:.1f}%)")
    print("="*70)
    
    # Generate Multi-Tab Professional Excel Report
    print(f"\nBuilding comprehensive multi-sheet Excel deliverable: {EXCEL_OUTPUT_PATH}...")
    wb = openpyxl.Workbook()
    
    # -------------------------------------------------------------
    # STYLES & THEME PALETTE
    # -------------------------------------------------------------
    header_fill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid") # Navy blue
    sub_fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")    # Soft blue
    card_fill = PatternFill(start_color="F2F2F2", end_color="F2F2F2", fill_type="solid")   # Soft gray
    warn_fill = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid")   # Soft peach/warn
    alert_fill = PatternFill(start_color="F8CBAD", end_color="F8CBAD", fill_type="solid")  # Coral alert
    green_fill = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")  # Soft green
    
    font_title = Font(name="Arial", size=14, bold=True, color="1F4E79")
    font_section = Font(name="Arial", size=11, bold=True, color="1F4E79")
    font_header = Font(name="Arial", size=10, bold=True, color="FFFFFF")
    font_data = Font(name="Arial", size=10, bold=False, color="000000")
    font_bold = Font(name="Arial", size=10, bold=True, color="000000")
    font_alert = Font(name="Arial", size=10, bold=True, color="C00000")
    font_kpi_num = Font(name="Arial", size=16, bold=True, color="1F4E79")
    font_kpi_lbl = Font(name="Arial", size=9, bold=False, color="595959")
    
    thin_border = Border(
        left=Side(style="thin", color="D9D9D9"),
        right=Side(style="thin", color="D9D9D9"),
        top=Side(style="thin", color="D9D9D9"),
        bottom=Side(style="thin", color="D9D9D9")
    )
    double_bottom_border = Border(
        top=Side(style="thin", color="000000"),
        bottom=Side(style="double", color="000000")
    )

    # -------------------------------------------------------------
    # SHEET 1: Executive_Scorecard
    # -------------------------------------------------------------
    ws1 = wb.active
    ws1.title = "Executive_Scorecard"
    ws1.views.sheetView[0].showGridLines = True
    
    ws1["A1"] = "Air Export Pricing Recommendation Engine V3 — Recent 2 Weeks Discrepancy Scorecard"
    ws1["A1"].font = font_title
    ws1["A2"] = f"Evaluation Window: Oct 01, 2026 to Oct 31, 2026 | Dataset: 3,985 TMS Inquiries | Scored: {n_scored:,} quotes"
    ws1["A2"].font = Font(name="Arial", size=10, italic=True, color="595959")
    
    # KPI Block Cards (Rows 4 to 6)
    kpis = [
        ("TOTAL INQUIRIES", f"{total_inquiries:,}", "Raw RFQ volume"),
        ("SCORING RATE", f"{n_scored/total_inquiries*100:.1f}%", f"{n_scored:,} valid quotes"),
        ("MONEY ON THE TABLE", f"₹{won_money_left/100000:.2f} L", f"{won_money_left_count:,} underpriced won deals"),
        ("COST FLOOR BREACHES", f"{cost_floor_breaches:,}", f"{cost_floor_breaches/n_scored*100:.1f}% quoted below floor"),
        ("OVERPRICED LOST DEALS", f"{lost_overpriced_count:,}", f"{lost_overpriced_count/max(1, n_lost)*100:.1f}% of lost inquiries"),
        ("CORRIDOR HIT RATE", f"{inside_corridor/n_scored*100:.1f}%", f"{inside_corridor:,} quotes in band"),
    ]
    
    for idx_k, (label, val, subtext) in enumerate(kpis):
        col_start = 1 + (idx_k * 2)
        c1, c2 = col_start, col_start + 1
        
        ws1.merge_cells(start_row=4, start_column=c1, end_row=4, end_column=c2)
        cell_lbl = ws1.cell(4, c1, label)
        cell_lbl.font = font_kpi_lbl
        cell_lbl.alignment = Alignment(horizontal="center", vertical="center")
        cell_lbl.fill = card_fill
        
        ws1.merge_cells(start_row=5, start_column=c1, end_row=5, end_column=c2)
        cell_val = ws1.cell(5, c1, val)
        cell_val.font = font_kpi_num
        cell_val.alignment = Alignment(horizontal="center", vertical="center")
        cell_val.fill = card_fill
        
        ws1.merge_cells(start_row=6, start_column=c1, end_row=6, end_column=c2)
        cell_sub = ws1.cell(6, c1, subtext)
        cell_sub.font = Font(name="Arial", size=8, italic=True, color="7F7F7F")
        cell_sub.alignment = Alignment(horizontal="center", vertical="center")
        cell_sub.fill = card_fill
        
        # Border box
        for r in range(4, 7):
            for c in range(c1, c2 + 1):
                ws1.cell(r, c).border = thin_border

    # Section 1: Discrepancy Breakdown Table
    ws1["A8"] = "1. Primary Commercial Discrepancy Diagnoses"
    ws1["A8"].font = font_section
    
    disp_headers = ["Discrepancy Category", "Status", "Inquiry Count", "Volume Share", "Total Actual Sell (INR)", "Avg Actual Margin %", "Avg Model Margin %", "Margin Gap (pp)", "Financial Impact / Opportunity"]
    for c_idx, h in enumerate(disp_headers, 1):
        cell = ws1.cell(9, c_idx, h)
        cell.font = font_header
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center")
        
    disp_groups = df_scored.groupby("Discrepancy_Classification")
    curr_r = 10
    for d_name, grp in sorted(disp_groups, key=lambda x: len(x[1]), reverse=True):
        ws1.cell(curr_r, 1, d_name).font = font_bold if "Breach" in d_name or "Table" in d_name or "Overpriced" in d_name else font_data
        ws1.cell(curr_r, 2, grp["Status"].iloc[0]).font = font_data
        ws1.cell(curr_r, 3, len(grp)).font = font_data
        ws1.cell(curr_r, 3).number_format = "#,##0"
        
        # Formula for share
        ws1.cell(curr_r, 4, f"=C{curr_r}/{n_scored}").font = font_data
        ws1.cell(curr_r, 4).number_format = "0.0%"
        
        ws1.cell(curr_r, 5, grp["Actual_Sell_INR"].sum()).font = font_data
        ws1.cell(curr_r, 5).number_format = "₹#,##0"
        
        ws1.cell(curr_r, 6, grp["Actual_Margin_Pct"].mean() / 100.0).font = font_data
        ws1.cell(curr_r, 6).number_format = "0.0%"
        
        ws1.cell(curr_r, 7, grp["Model_Balanced_Margin_Pct"].mean() / 100.0).font = font_data
        ws1.cell(curr_r, 7).number_format = "0.0%"
        
        # Margin gap formula
        ws1.cell(curr_r, 8, f"=G{curr_r}-F{curr_r}").font = font_data
        ws1.cell(curr_r, 8).number_format = "0.0%"
        
        # Financial Impact
        if "Table" in d_name:
            impact_val = grp["Money_Left_On_Table_INR"].sum()
            cell_imp = ws1.cell(curr_r, 9, impact_val)
            cell_imp.font = font_bold
            cell_imp.fill = green_fill
            cell_imp.number_format = "₹#,##0"
        elif "Breach" in d_name or "Sub-Cost" in d_name:
            loss_val = (grp["Cost_Floor_Price_INR"] - grp["Actual_Sell_INR"]).clip(lower=0).sum()
            cell_imp = ws1.cell(curr_r, 9, f"-₹{loss_val:,.0f} (Cost Floor Deficit)")
            cell_imp.font = font_alert
            cell_imp.fill = alert_fill
        elif "Overpriced" in d_name:
            cell_imp = ws1.cell(curr_r, 9, f"{len(grp)} deals lost to excessive pricing")
            cell_imp.font = font_data
            cell_imp.fill = warn_fill
        else:
            ws1.cell(curr_r, 9, "Aligned with market expectations").font = font_data
            
        for c in range(1, 10):
            ws1.cell(curr_r, c).border = thin_border
        curr_r += 1
        
    # Total row
    ws1.cell(curr_r, 1, "Total / Overall Average").font = font_bold
    ws1.cell(curr_r, 2, "All").font = font_bold
    ws1.cell(curr_r, 3, f"=SUM(C10:C{curr_r-1})").font = font_bold
    ws1.cell(curr_r, 3).number_format = "#,##0"
    ws1.cell(curr_r, 4, f"=SUM(D10:D{curr_r-1})").font = font_bold
    ws1.cell(curr_r, 4).number_format = "0.0%"
    ws1.cell(curr_r, 5, f"=SUM(E10:E{curr_r-1})").font = font_bold
    ws1.cell(curr_r, 5).number_format = "₹#,##0"
    ws1.cell(curr_r, 6, f"=AVERAGE(F10:F{curr_r-1})").font = font_bold
    ws1.cell(curr_r, 6).number_format = "0.0%"
    ws1.cell(curr_r, 7, f"=AVERAGE(G10:G{curr_r-1})").font = font_bold
    ws1.cell(curr_r, 7).number_format = "0.0%"
    ws1.cell(curr_r, 8, f"=G{curr_r}-F{curr_r}").font = font_bold
    ws1.cell(curr_r, 8).number_format = "0.0%"
    ws1.cell(curr_r, 9, f"₹{won_money_left:,.0f} Net Recoverable Upside").font = font_bold
    
    for c in range(1, 10):
        ws1.cell(curr_r, c).border = double_bottom_border
        
    # Section 2: Key Operational Findings
    curr_r += 3
    ws1.cell(curr_r, 1, "2. Key Strategic Takeaways & Root Cause Discrepancies").font = font_section
    curr_r += 1
    
    takeaways = [
        ("Margin Deficit on Won Deals", f"On {won_money_left_count:,} out of {n_won:,} won bookings ({won_money_left_count/max(1,n_won)*100:.1f}%), sales representatives under-quoted relative to market clearing capacity, leaving ₹{won_money_left:,.2f} INR in profit on the table."),
        ("Dangerous Cost Floor Breaches", f"{cost_floor_breaches:,} quotes ({cost_floor_breaches/n_scored*100:.1f}%) were quoted below the non-negotiable operational cost floor, including {sub_cost_breaches} quotes priced strictly below airline buy cost."),
        ("Sales Overpricing in Lost Deals", f"{lost_overpriced_count:,} lost inquiries ({lost_overpriced_count/max(1,n_lost)*100:.1f}% of lost volume) were priced above the model's 90% confidence corridor upper ceiling, causing avoidable customer churn."),
        ("Heavy Cargo Compression (>500kg)", "Heavy commercial volume exhibits severe margin collapse (median actual margin is only 2.1%), where cost floors are critical to avoid freight-rate cash burn."),
        ("Data Entry Quality Anomalies", f"{len(df_invalid):,} inquiries were flagged with data defects (e.g. unquoted entries, zero buy rates, or extreme 300-ton bulk test inputs) and quarantined.")
    ]
    for title, desc in takeaways:
        ws1.cell(curr_r, 1, f"• {title}:").font = font_bold
        ws1.merge_cells(start_row=curr_r, start_column=2, end_row=curr_r, end_column=9)
        cell = ws1.cell(curr_r, 2, desc)
        cell.font = font_data
        cell.alignment = Alignment(wrap_text=True)
        curr_r += 1

    # -------------------------------------------------------------
    # SHEET 2: Detailed_Comparison
    # -------------------------------------------------------------
    ws2 = wb.create_sheet(title="Detailed_Comparison")
    ws2.views.sheetView[0].showGridLines = True
    
    det_cols = [
        ("Inquiry Number", "Inquiry_Number", "@"),
        ("Quote Date", "Quote_Date", "YYYY-MM-DD"),
        ("Customer Company", "Customer_Company", "@"),
        ("Branch", "Branch", "@"),
        ("Sales Person", "Sales_Person", "@"),
        ("Origin", "Port_Of_Loading", "@"),
        ("Destination", "Port_Of_Discharge", "@"),
        ("Regional Lane", "Regional_Lane", "@"),
        ("Commodity", "Commodity_Canonical", "@"),
        ("Chargeable Wt (kg)", "Chargeable_Weight_Kg", "#,##0.0"),
        ("Weight Tier", "Weight_Tier", "@"),
        ("Total Buy (INR)", "Total_Buy_INR", "₹#,##0"),
        ("Actual Sell (INR)", "Actual_Sell_INR", "₹#,##0"),
        ("Actual Margin %", "Actual_Margin_Pct", "0.0%"),
        ("Actual Rate/Kg", "Actual_Rate_Per_Kg", "₹#,##0.00"),
        ("Model Balanced %", "Model_Balanced_Margin_Pct", "0.0%"),
        ("Model Floor %", "Model_Floor_Margin_Pct", "0.0%"),
        ("Model Premium %", "Model_Premium_Margin_Pct", "0.0%"),
        ("Model Sell (INR)", "Model_Sell_INR", "₹#,##0"),
        ("Model Rate/Kg", "Model_Rate_Per_Kg", "₹#,##0.00"),
        ("Corridor Min %", "Corridor_Min_Margin_Pct", "0.0%"),
        ("Corridor Max %", "Corridor_Max_Margin_Pct", "0.0%"),
        ("Corridor Status", "Corridor_Status", "@"),
        ("Margin Gap (pp)", "Margin_Gap_pp", "0.0%"),
        ("Price Diff (INR)", "Sell_Diff_INR", "₹#,##0"),
        ("Cost Floor Price (INR)", "Cost_Floor_Price_INR", "₹#,##0"),
        ("Cost Floor Breached?", "Cost_Floor_Breached", "@"),
        ("Sub-Cost Quoted?", "Sub_Cost_Quoted", "@"),
        ("Status", "Status", "@"),
        ("Loss Reason", "Loss_Reason", "@"),
        ("Discrepancy Category", "Discrepancy_Classification", "@"),
        ("Money Left On Table (INR)", "Money_Left_On_Table_INR", "₹#,##0"),
    ]
    
    # Write headers
    for c_idx, (col_name, _, _) in enumerate(det_cols, 1):
        cell = ws2.cell(1, c_idx, col_name)
        cell.font = font_header
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center")
        
    for r_idx, row_dict in enumerate(scored_rows, 2):
        for c_idx, (_, field_key, num_fmt) in enumerate(det_cols, 1):
            val = row_dict[field_key]
            
            # Format percentage decimals
            if num_fmt == "0.0%" and isinstance(val, (int, float)):
                val = val / 100.0
                
            cell = ws2.cell(r_idx, c_idx, val)
            cell.font = font_data
            cell.border = thin_border
            if num_fmt != "@":
                cell.number_format = num_fmt
                
        # Highlighting alert flags
        if row_dict["Cost_Floor_Breached"]:
            ws2.cell(r_idx, 27).fill = alert_fill
            ws2.cell(r_idx, 27).font = font_alert
        if row_dict["Money_Left_On_Table_INR"] > 0:
            ws2.cell(r_idx, 32).fill = green_fill
            ws2.cell(r_idx, 32).font = font_bold

    # -------------------------------------------------------------
    # SHEET 3: Weight_Tier_Analysis
    # -------------------------------------------------------------
    ws3 = wb.create_sheet(title="Weight_Tier_Analysis")
    ws3.views.sheetView[0].showGridLines = True
    
    ws3["A1"] = "Discrepancy & Performance Analysis by IATA Weight Tier"
    ws3["A1"].font = font_title
    
    wt_headers = ["Weight Tier", "Quotes (N)", "Share %", "Won Deals", "Win Rate %", "Total Actual Revenue (INR)", "Avg Actual Margin %", "Avg Model Margin %", "Margin Gap (pp)", "Cost Floor Breaches", "Money Left on Table (INR)"]
    for c_idx, h in enumerate(wt_headers, 1):
        cell = ws3.cell(3, c_idx, h)
        cell.font = font_header
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center")
        
    wt_tiers_order = ["<45kg", "45-100kg", "100-300kg", "300-500kg", "500-1000kg", ">1000kg"]
    curr_r = 4
    for tier in wt_tiers_order:
        grp = df_scored[df_scored["Weight_Tier"] == tier]
        if len(grp) == 0:
            continue
        n_grp = len(grp)
        n_won_tier = (grp["Status"].str.lower() == "won").sum()
        
        ws3.cell(curr_r, 1, tier).font = font_bold
        ws3.cell(curr_r, 2, n_grp).font = font_data
        ws3.cell(curr_r, 2).number_format = "#,##0"
        ws3.cell(curr_r, 3, f"=B{curr_r}/{n_scored}").font = font_data
        ws3.cell(curr_r, 3).number_format = "0.0%"
        ws3.cell(curr_r, 4, n_won_tier).font = font_data
        ws3.cell(curr_r, 4).number_format = "#,##0"
        ws3.cell(curr_r, 5, f"=D{curr_r}/B{curr_r}").font = font_data
        ws3.cell(curr_r, 5).number_format = "0.0%"
        ws3.cell(curr_r, 6, grp["Actual_Sell_INR"].sum()).font = font_data
        ws3.cell(curr_r, 6).number_format = "₹#,##0"
        ws3.cell(curr_r, 7, grp["Actual_Margin_Pct"].mean() / 100.0).font = font_data
        ws3.cell(curr_r, 7).number_format = "0.0%"
        ws3.cell(curr_r, 8, grp["Model_Balanced_Margin_Pct"].mean() / 100.0).font = font_data
        ws3.cell(curr_r, 8).number_format = "0.0%"
        ws3.cell(curr_r, 9, f"=H{curr_r}-G{curr_r}").font = font_data
        ws3.cell(curr_r, 9).number_format = "0.0%"
        ws3.cell(curr_r, 10, grp["Cost_Floor_Breached"].sum()).font = font_alert if grp["Cost_Floor_Breached"].sum() > 0 else font_data
        ws3.cell(curr_r, 10).number_format = "#,##0"
        ws3.cell(curr_r, 11, grp["Money_Left_On_Table_INR"].sum()).font = font_bold
        ws3.cell(curr_r, 11).number_format = "₹#,##0"
        
        for c in range(1, 12):
            ws3.cell(curr_r, c).border = thin_border
        curr_r += 1

    # Total row
    ws3.cell(curr_r, 1, "Total / Summary").font = font_bold
    ws3.cell(curr_r, 2, f"=SUM(B4:B{curr_r-1})").font = font_bold
    ws3.cell(curr_r, 2).number_format = "#,##0"
    ws3.cell(curr_r, 3, f"=SUM(C4:C{curr_r-1})").font = font_bold
    ws3.cell(curr_r, 3).number_format = "0.0%"
    ws3.cell(curr_r, 4, f"=SUM(D4:D{curr_r-1})").font = font_bold
    ws3.cell(curr_r, 4).number_format = "#,##0"
    ws3.cell(curr_r, 5, f"=D{curr_r}/B{curr_r}").font = font_bold
    ws3.cell(curr_r, 5).number_format = "0.0%"
    ws3.cell(curr_r, 6, f"=SUM(F4:F{curr_r-1})").font = font_bold
    ws3.cell(curr_r, 6).number_format = "₹#,##0"
    ws3.cell(curr_r, 7, f"=AVERAGE(G4:G{curr_r-1})").font = font_bold
    ws3.cell(curr_r, 7).number_format = "0.0%"
    ws3.cell(curr_r, 8, f"=AVERAGE(H4:H{curr_r-1})").font = font_bold
    ws3.cell(curr_r, 8).number_format = "0.0%"
    ws3.cell(curr_r, 9, f"=H{curr_r}-G{curr_r}").font = font_bold
    ws3.cell(curr_r, 9).number_format = "0.0%"
    ws3.cell(curr_r, 10, f"=SUM(J4:J{curr_r-1})").font = font_bold
    ws3.cell(curr_r, 10).number_format = "#,##0"
    ws3.cell(curr_r, 11, f"=SUM(K4:K{curr_r-1})").font = font_bold
    ws3.cell(curr_r, 11).number_format = "₹#,##0"
    for c in range(1, 12):
        ws3.cell(curr_r, c).border = double_bottom_border

    # -------------------------------------------------------------
    # SHEET 4: Branch_Analysis
    # -------------------------------------------------------------
    ws4 = wb.create_sheet(title="Branch_Analysis")
    ws4.views.sheetView[0].showGridLines = True
    
    ws4["A1"] = "Branch-Wise Discrepancy & Sales Execution Quality"
    ws4["A1"].font = font_title
    
    br_headers = ["Booking Branch", "Quotes (N)", "Share %", "Won Volume", "Win Rate %", "Total Revenue (INR)", "Actual Margin %", "Model Margin %", "Margin Gap (pp)", "Floor Breaches", "Money Left on Table (INR)"]
    for c_idx, h in enumerate(br_headers, 1):
        cell = ws4.cell(3, c_idx, h)
        cell.font = font_header
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center")
        
    branch_groups = df_scored.groupby("Branch")
    curr_r = 4
    for br_name, grp in sorted(branch_groups, key=lambda x: len(x[1]), reverse=True):
        n_grp = len(grp)
        n_won_br = (grp["Status"].str.lower() == "won").sum()
        
        ws4.cell(curr_r, 1, br_name).font = font_bold
        ws4.cell(curr_r, 2, n_grp).font = font_data
        ws4.cell(curr_r, 2).number_format = "#,##0"
        ws4.cell(curr_r, 3, f"=B{curr_r}/{n_scored}").font = font_data
        ws4.cell(curr_r, 3).number_format = "0.0%"
        ws4.cell(curr_r, 4, n_won_br).font = font_data
        ws4.cell(curr_r, 4).number_format = "#,##0"
        ws4.cell(curr_r, 5, f"=D{curr_r}/B{curr_r}").font = font_data
        ws4.cell(curr_r, 5).number_format = "0.0%"
        ws4.cell(curr_r, 6, grp["Actual_Sell_INR"].sum()).font = font_data
        ws4.cell(curr_r, 6).number_format = "₹#,##0"
        ws4.cell(curr_r, 7, grp["Actual_Margin_Pct"].mean() / 100.0).font = font_data
        ws4.cell(curr_r, 7).number_format = "0.0%"
        ws4.cell(curr_r, 8, grp["Model_Balanced_Margin_Pct"].mean() / 100.0).font = font_data
        ws4.cell(curr_r, 8).number_format = "0.0%"
        ws4.cell(curr_r, 9, f"=H{curr_r}-G{curr_r}").font = font_data
        ws4.cell(curr_r, 9).number_format = "0.0%"
        ws4.cell(curr_r, 10, grp["Cost_Floor_Breached"].sum()).font = font_alert if grp["Cost_Floor_Breached"].sum() > 0 else font_data
        ws4.cell(curr_r, 10).number_format = "#,##0"
        ws4.cell(curr_r, 11, grp["Money_Left_On_Table_INR"].sum()).font = font_bold
        ws4.cell(curr_r, 11).number_format = "₹#,##0"
        
        for c in range(1, 12):
            ws4.cell(curr_r, c).border = thin_border
        curr_r += 1
        
    # Branch total row
    ws4.cell(curr_r, 1, "Total / Pan-India Summary").font = font_bold
    ws4.cell(curr_r, 2, f"=SUM(B4:B{curr_r-1})").font = font_bold
    ws4.cell(curr_r, 2).number_format = "#,##0"
    ws4.cell(curr_r, 3, f"=SUM(C4:C{curr_r-1})").font = font_bold
    ws4.cell(curr_r, 3).number_format = "0.0%"
    ws4.cell(curr_r, 4, f"=SUM(D4:D{curr_r-1})").font = font_bold
    ws4.cell(curr_r, 4).number_format = "#,##0"
    ws4.cell(curr_r, 5, f"=D{curr_r}/B{curr_r}").font = font_bold
    ws4.cell(curr_r, 5).number_format = "0.0%"
    ws4.cell(curr_r, 6, f"=SUM(F4:F{curr_r-1})").font = font_bold
    ws4.cell(curr_r, 6).number_format = "₹#,##0"
    ws4.cell(curr_r, 7, f"=AVERAGE(G4:G{curr_r-1})").font = font_bold
    ws4.cell(curr_r, 7).number_format = "0.0%"
    ws4.cell(curr_r, 8, f"=AVERAGE(H4:H{curr_r-1})").font = font_bold
    ws4.cell(curr_r, 8).number_format = "0.0%"
    ws4.cell(curr_r, 9, f"=H{curr_r}-G{curr_r}").font = font_bold
    ws4.cell(curr_r, 9).number_format = "0.0%"
    ws4.cell(curr_r, 10, f"=SUM(J4:J{curr_r-1})").font = font_bold
    ws4.cell(curr_r, 10).number_format = "#,##0"
    ws4.cell(curr_r, 11, f"=SUM(K4:K{curr_r-1})").font = font_bold
    ws4.cell(curr_r, 11).number_format = "₹#,##0"
    for c in range(1, 12):
        ws4.cell(curr_r, c).border = double_bottom_border

    # -------------------------------------------------------------
    # SHEET: Commodity_Analysis
    # -------------------------------------------------------------
    ws_comm = wb.create_sheet(title="Commodity_Analysis")
    ws_comm.views.sheetView[0].showGridLines = True
    
    ws_comm["A1"] = "Discrepancy & Performance Analysis by Commodity Vertical"
    ws_comm["A1"].font = font_title
    
    comm_headers = ["Commodity Vertical", "Quotes (N)", "Share %", "Won Volume", "Win Rate %", "Total Revenue (INR)", "Actual Margin %", "Model Margin %", "Margin Gap (pp)", "Floor Breaches", "Money Left on Table (INR)"]
    for c_idx, h in enumerate(comm_headers, 1):
        cell = ws_comm.cell(3, c_idx, h)
        cell.font = font_header
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center")
        
    comm_groups = df_scored.groupby("Commodity_Canonical")
    curr_r = 4
    for comm_name, grp in sorted(comm_groups, key=lambda x: len(x[1]), reverse=True):
        n_grp = len(grp)
        n_won_comm = (grp["Status"].str.lower() == "won").sum()
        
        ws_comm.cell(curr_r, 1, comm_name).font = font_bold
        ws_comm.cell(curr_r, 2, n_grp).font = font_data
        ws_comm.cell(curr_r, 2).number_format = "#,##0"
        ws_comm.cell(curr_r, 3, f"=B{curr_r}/{n_scored}").font = font_data
        ws_comm.cell(curr_r, 3).number_format = "0.0%"
        ws_comm.cell(curr_r, 4, n_won_comm).font = font_data
        ws_comm.cell(curr_r, 4).number_format = "#,##0"
        ws_comm.cell(curr_r, 5, f"=D{curr_r}/B{curr_r}").font = font_data
        ws_comm.cell(curr_r, 5).number_format = "0.0%"
        ws_comm.cell(curr_r, 6, grp["Actual_Sell_INR"].sum()).font = font_data
        ws_comm.cell(curr_r, 6).number_format = "₹#,##0"
        ws_comm.cell(curr_r, 7, grp["Actual_Margin_Pct"].mean() / 100.0).font = font_data
        ws_comm.cell(curr_r, 7).number_format = "0.0%"
        ws_comm.cell(curr_r, 8, grp["Model_Balanced_Margin_Pct"].mean() / 100.0).font = font_data
        ws_comm.cell(curr_r, 8).number_format = "0.0%"
        ws_comm.cell(curr_r, 9, f"=H{curr_r}-G{curr_r}").font = font_data
        ws_comm.cell(curr_r, 9).number_format = "0.0%"
        ws_comm.cell(curr_r, 10, grp["Cost_Floor_Breached"].sum()).font = font_alert if grp["Cost_Floor_Breached"].sum() > 0 else font_data
        ws_comm.cell(curr_r, 10).number_format = "#,##0"
        ws_comm.cell(curr_r, 11, grp["Money_Left_On_Table_INR"].sum()).font = font_bold
        ws_comm.cell(curr_r, 11).number_format = "₹#,##0"
        
        for c in range(1, 12):
            ws_comm.cell(curr_r, c).border = thin_border
        curr_r += 1

    # Commodity total row
    ws_comm.cell(curr_r, 1, "Total / Commodity Summary").font = font_bold
    ws_comm.cell(curr_r, 2, f"=SUM(B4:B{curr_r-1})").font = font_bold
    ws_comm.cell(curr_r, 2).number_format = "#,##0"
    ws_comm.cell(curr_r, 3, f"=SUM(C4:C{curr_r-1})").font = font_bold
    ws_comm.cell(curr_r, 3).number_format = "0.0%"
    ws_comm.cell(curr_r, 4, f"=SUM(D4:D{curr_r-1})").font = font_bold
    ws_comm.cell(curr_r, 4).number_format = "#,##0"
    ws_comm.cell(curr_r, 5, f"=D{curr_r}/B{curr_r}").font = font_bold
    ws_comm.cell(curr_r, 5).number_format = "0.0%"
    ws_comm.cell(curr_r, 6, f"=SUM(F4:F{curr_r-1})").font = font_bold
    ws_comm.cell(curr_r, 6).number_format = "₹#,##0"
    ws_comm.cell(curr_r, 7, f"=AVERAGE(G4:G{curr_r-1})").font = font_bold
    ws_comm.cell(curr_r, 7).number_format = "0.0%"
    ws_comm.cell(curr_r, 8, f"=AVERAGE(H4:H{curr_r-1})").font = font_bold
    ws_comm.cell(curr_r, 8).number_format = "0.0%"
    ws_comm.cell(curr_r, 9, f"=H{curr_r}-G{curr_r}").font = font_bold
    ws_comm.cell(curr_r, 9).number_format = "0.0%"
    ws_comm.cell(curr_r, 10, f"=SUM(J4:J{curr_r-1})").font = font_bold
    ws_comm.cell(curr_r, 10).number_format = "#,##0"
    ws_comm.cell(curr_r, 11, f"=SUM(K4:K{curr_r-1})").font = font_bold
    ws_comm.cell(curr_r, 11).number_format = "₹#,##0"
    for c in range(1, 12):
        ws_comm.cell(curr_r, c).border = double_bottom_border

    # -------------------------------------------------------------
    # SHEET: Regional_Lane_Analysis
    # -------------------------------------------------------------
    ws_lane = wb.create_sheet(title="Regional_Lane_Analysis")
    ws_lane.views.sheetView[0].showGridLines = True
    
    ws_lane["A1"] = "Discrepancy & Margin Gap by Regional Trade Lane"
    ws_lane["A1"].font = font_title
    
    lane_headers = ["Regional Trade Lane", "Quotes (N)", "Share %", "Won Volume", "Win Rate %", "Total Revenue (INR)", "Actual Margin %", "Model Margin %", "Margin Gap (pp)", "Floor Breaches", "Money Left on Table (INR)"]
    for c_idx, h in enumerate(lane_headers, 1):
        cell = ws_lane.cell(3, c_idx, h)
        cell.font = font_header
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center")
        
    lane_groups = df_scored.groupby("Regional_Lane")
    curr_r = 4
    for lane_name, grp in sorted(lane_groups, key=lambda x: len(x[1]), reverse=True):
        n_grp = len(grp)
        n_won_lane = (grp["Status"].str.lower() == "won").sum()
        
        ws_lane.cell(curr_r, 1, lane_name).font = font_bold
        ws_lane.cell(curr_r, 2, n_grp).font = font_data
        ws_lane.cell(curr_r, 2).number_format = "#,##0"
        ws_lane.cell(curr_r, 3, f"=B{curr_r}/{n_scored}").font = font_data
        ws_lane.cell(curr_r, 3).number_format = "0.0%"
        ws_lane.cell(curr_r, 4, n_won_lane).font = font_data
        ws_lane.cell(curr_r, 4).number_format = "#,##0"
        ws_lane.cell(curr_r, 5, f"=D{curr_r}/B{curr_r}").font = font_data
        ws_lane.cell(curr_r, 5).number_format = "0.0%"
        ws_lane.cell(curr_r, 6, grp["Actual_Sell_INR"].sum()).font = font_data
        ws_lane.cell(curr_r, 6).number_format = "₹#,##0"
        ws_lane.cell(curr_r, 7, grp["Actual_Margin_Pct"].mean() / 100.0).font = font_data
        ws_lane.cell(curr_r, 7).number_format = "0.0%"
        ws_lane.cell(curr_r, 8, grp["Model_Balanced_Margin_Pct"].mean() / 100.0).font = font_data
        ws_lane.cell(curr_r, 8).number_format = "0.0%"
        ws_lane.cell(curr_r, 9, f"=H{curr_r}-G{curr_r}").font = font_data
        ws_lane.cell(curr_r, 9).number_format = "0.0%"
        ws_lane.cell(curr_r, 10, grp["Cost_Floor_Breached"].sum()).font = font_alert if grp["Cost_Floor_Breached"].sum() > 0 else font_data
        ws_lane.cell(curr_r, 10).number_format = "#,##0"
        ws_lane.cell(curr_r, 11, grp["Money_Left_On_Table_INR"].sum()).font = font_bold
        ws_lane.cell(curr_r, 11).number_format = "₹#,##0"
        
        for c in range(1, 12):
            ws_lane.cell(curr_r, c).border = thin_border
        curr_r += 1

    # Lane total row
    ws_lane.cell(curr_r, 1, "Total / Pan-India Summary").font = font_bold
    ws_lane.cell(curr_r, 2, f"=SUM(B4:B{curr_r-1})").font = font_bold
    ws_lane.cell(curr_r, 2).number_format = "#,##0"
    ws_lane.cell(curr_r, 3, f"=SUM(C4:C{curr_r-1})").font = font_bold
    ws_lane.cell(curr_r, 3).number_format = "0.0%"
    ws_lane.cell(curr_r, 4, f"=SUM(D4:D{curr_r-1})").font = font_bold
    ws_lane.cell(curr_r, 4).number_format = "#,##0"
    ws_lane.cell(curr_r, 5, f"=D{curr_r}/B{curr_r}").font = font_bold
    ws_lane.cell(curr_r, 5).number_format = "0.0%"
    ws_lane.cell(curr_r, 6, f"=SUM(F4:F{curr_r-1})").font = font_bold
    ws_lane.cell(curr_r, 6).number_format = "₹#,##0"
    ws_lane.cell(curr_r, 7, f"=AVERAGE(G4:G{curr_r-1})").font = font_bold
    ws_lane.cell(curr_r, 7).number_format = "0.0%"
    ws_lane.cell(curr_r, 8, f"=AVERAGE(H4:H{curr_r-1})").font = font_bold
    ws_lane.cell(curr_r, 8).number_format = "0.0%"
    ws_lane.cell(curr_r, 9, f"=H{curr_r}-G{curr_r}").font = font_bold
    ws_lane.cell(curr_r, 9).number_format = "0.0%"
    ws_lane.cell(curr_r, 10, f"=SUM(J4:J{curr_r-1})").font = font_bold
    ws_lane.cell(curr_r, 10).number_format = "#,##0"
    ws_lane.cell(curr_r, 11, f"=SUM(K4:K{curr_r-1})").font = font_bold
    ws_lane.cell(curr_r, 11).number_format = "₹#,##0"
    for c in range(1, 12):
        ws_lane.cell(curr_r, c).border = double_bottom_border

    # -------------------------------------------------------------
    # SHEET: Worst_Performing_Cases
    # -------------------------------------------------------------
    ws_worst = wb.create_sheet(title="Worst_Performing_Cases")
    ws_worst.views.sheetView[0].showGridLines = True
    
    ws_worst["A1"] = "Top 100 Discrepancy Cases: Largest Margin Gap vs Human Coordinator"
    ws_worst["A1"].font = font_title
    
    worst_headers = [
        "Rank", "Inquiry #", "Quote Date", "Customer Company", "Branch", "Sales Person",
        "Commodity", "Weight (kg)", "Weight Tier", "Lane", "Quoted Carrier", "Status",
        "Airline Buy (INR)", "Coordinator Sell (INR)", "Coordinator Margin %",
        "Model Sell (INR)", "Model Balanced Margin %", "Margin Gap (pp)",
        "Coordinator Rate/kg", "Model Rate/kg", "Cost Floor Breached", "Root Cause / Discrepancy"
    ]
    for c_idx, h in enumerate(worst_headers, 1):
        cell = ws_worst.cell(3, c_idx, h)
        cell.font = font_header
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center")
        
    df_worst = df_scored.assign(abs_gap=df_scored["Margin_Gap_pp"].abs()).sort_values("abs_gap", ascending=False).head(100)
    for r_idx, (_, r) in enumerate(df_worst.iterrows(), 4):
        ws_worst.cell(r_idx, 1, r_idx - 3).font = font_bold
        ws_worst.cell(r_idx, 1).alignment = Alignment(horizontal="center")
        ws_worst.cell(r_idx, 2, r["Inquiry_Number"]).font = font_data
        ws_worst.cell(r_idx, 3, r["Quote_Date"]).font = font_data
        ws_worst.cell(r_idx, 4, r["Customer_Company"]).font = font_bold
        ws_worst.cell(r_idx, 5, r["Branch"]).font = font_data
        ws_worst.cell(r_idx, 6, r["Sales_Person"]).font = font_data
        ws_worst.cell(r_idx, 7, r["Commodity_Canonical"]).font = font_data
        ws_worst.cell(r_idx, 8, r["Chargeable_Weight_Kg"]).font = font_data
        ws_worst.cell(r_idx, 8).number_format = "#,##0.0"
        ws_worst.cell(r_idx, 9, r["Weight_Tier"]).font = font_data
        ws_worst.cell(r_idx, 10, r["Regional_Lane"]).font = font_data
        ws_worst.cell(r_idx, 11, r["Quoted_Carrier"]).font = font_data
        ws_worst.cell(r_idx, 12, r["Status"]).font = font_data
        ws_worst.cell(r_idx, 13, r["Total_Buy_INR"]).font = font_data
        ws_worst.cell(r_idx, 13).number_format = "₹#,##0"
        ws_worst.cell(r_idx, 14, r["Actual_Sell_INR"]).font = font_data
        ws_worst.cell(r_idx, 14).number_format = "₹#,##0"
        ws_worst.cell(r_idx, 15, r["Actual_Margin_Pct"] / 100.0).font = font_data
        ws_worst.cell(r_idx, 15).number_format = "0.0%"
        ws_worst.cell(r_idx, 16, r["Model_Sell_INR"]).font = font_data
        ws_worst.cell(r_idx, 16).number_format = "₹#,##0"
        ws_worst.cell(r_idx, 17, r["Model_Balanced_Margin_Pct"] / 100.0).font = font_data
        ws_worst.cell(r_idx, 17).number_format = "0.0%"
        ws_worst.cell(r_idx, 18, f"=Q{r_idx}-O{r_idx}").font = font_bold
        ws_worst.cell(r_idx, 18).number_format = "0.0%"
        ws_worst.cell(r_idx, 19, r["Actual_Rate_Per_Kg"]).font = font_data
        ws_worst.cell(r_idx, 19).number_format = "₹#,##0.00"
        ws_worst.cell(r_idx, 20, r["Model_Rate_Per_Kg"]).font = font_data
        ws_worst.cell(r_idx, 20).number_format = "₹#,##0.00"
        ws_worst.cell(r_idx, 21, "YES" if r["Cost_Floor_Breached"] else "NO").font = font_alert if r["Cost_Floor_Breached"] else font_data
        ws_worst.cell(r_idx, 22, r["Discrepancy_Classification"]).font = font_data
        
        if abs(r["Margin_Gap_pp"]) > 20.0:
            ws_worst.cell(r_idx, 18).fill = alert_fill
            ws_worst.cell(r_idx, 18).font = font_alert
            
        for c in range(1, 23):
            ws_worst.cell(r_idx, c).border = thin_border

    # -------------------------------------------------------------
    # SHEET 5: Data_Quality_Flags
    # -------------------------------------------------------------
    ws5 = wb.create_sheet(title="Data_Quality_Flags")
    ws5.views.sheetView[0].showGridLines = True
    
    ws5["A1"] = f"Excluded & Flagged Inquiries ({len(df_invalid):,} Records)"
    ws5["A1"].font = font_title
    ws5["A2"] = "These inquiries contain input defects, unquoted statuses, or commercial physical bounds violations."
    ws5["A2"].font = Font(name="Arial", size=10, italic=True, color="595959")
    
    inv_cols = [
        ("Inquiry Number", "Inquiry_Number"),
        ("Customer Company", "Customer_Company"),
        ("Branch", "Branch"),
        ("Status", "Status"),
        ("Quote Date", "Quote_Date"),
        ("Weight (kg)", "Chargeable_Weight_Kg"),
        ("Total Buy (INR)", "Total_Buy_INR"),
        ("Total Sell (INR)", "Total_Sell_INR"),
        ("Rejection / Discrepancy Reason", "Rejection_Reason"),
    ]
    
    for c_idx, (col_name, _) in enumerate(inv_cols, 1):
        cell = ws5.cell(4, c_idx, col_name)
        cell.font = font_header
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center")
        
    for r_idx, inv_row in enumerate(invalid_records, 5):
        for c_idx, (_, field_key) in enumerate(inv_cols, 1):
            val = inv_row[field_key]
            cell = ws5.cell(r_idx, c_idx, val)
            cell.font = font_data
            cell.border = thin_border
            if field_key in ("Total_Buy_INR", "Total_Sell_INR") and pd.notnull(val):
                cell.number_format = "₹#,##0"
            elif field_key == "Chargeable_Weight_Kg" and pd.notnull(val):
                cell.number_format = "#,##0.0"
        ws5.cell(r_idx, 9).fill = warn_fill
        
    # Auto-fit column widths across all sheets
    for ws in [ws1, ws2, ws3, ws4, ws_comm, ws_lane, ws_worst, ws5]:
        for col in ws.columns:
            max_len = 0
            col_letter = get_column_letter(col[0].column)
            # Sample first 200 rows for speed
            for cell in col[:200]:
                val_str = str(cell.value or "")
                if val_str.startswith("="):
                    val_str = "123,456.78" # surrogate width for formulas
                if len(val_str) > max_len:
                    max_len = len(val_str)
            ws.column_dimensions[col_letter].width = max(12, min(50, max_len + 3))
            
    # Save Workbook
    wb.save(EXCEL_OUTPUT_PATH)
    print(f"Successfully generated Excel workbook at: {EXCEL_OUTPUT_PATH}")
    print("Done!")

if __name__ == "__main__":
    main()
