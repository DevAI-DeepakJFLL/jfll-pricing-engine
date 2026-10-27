"""
tests/test_ingest_and_split_2025.py
Unit tests for 2025 inquiries data ingestion, currency parsing, commercial economic bounds,
label purity, date consistency build gates, and exact 70/30 chronological & stratified splits.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.ingest_and_split_2025 import (
    parse_currency_string,
    clean_chargeable_weight,
    standardize_company_group,
    resolve_origin_region,
    filter_commercial_economic_bounds,
    recalculate_margins,
    chronological_split_70_30,
    stratified_split_70_30,
    clean_and_enrich_inquiries,
)
from src.preprocess_pipeline_v3 import validate_build_gates
from src.tiers import get_weight_tier


class TestCurrencyAndWeightParsing(unittest.TestCase):
    def test_parse_currency_string_valid(self):
        self.assertAlmostEqual(parse_currency_string("30,957.5 INR"), 30957.5)
        self.assertAlmostEqual(parse_currency_string("1,92,600.0 INR"), 192600.0)
        self.assertAlmostEqual(parse_currency_string("610.0 INR"), 610.0)
        self.assertAlmostEqual(parse_currency_string("125000"), 125000.0)
        self.assertAlmostEqual(parse_currency_string("1,500.50 USD"), 1500.50)
        self.assertAlmostEqual(parse_currency_string(25000.75), 25000.75)

    def test_parse_currency_string_null_and_invalid(self):
        self.assertTrue(np.isnan(parse_currency_string(np.nan)))
        self.assertTrue(np.isnan(parse_currency_string(None)))
        self.assertTrue(np.isnan(parse_currency_string("")))
        self.assertTrue(np.isnan(parse_currency_string("   ")))
        self.assertTrue(np.isnan(parse_currency_string("N/A")))

    def test_clean_chargeable_weight_rounding(self):
        weights = pd.Series([45.0, 45.2, 45.3, 45.7, 45.8, 100.1, 0.5, 25000.0])
        cleaned = clean_chargeable_weight(weights)
        expected = [45.0, 45.0, 45.5, 45.5, 46.0, 100.0, 0.5, 25000.0]
        for c, e in zip(cleaned, expected):
            self.assertAlmostEqual(c, e, places=1)


class TestCommercialEconomicFilters(unittest.TestCase):
    def test_filter_economic_bounds(self):
        df = pd.DataFrame({
            "Total_Buy_INR": [600.0, 500.0, 400.0, 10000.0, 10000.0, 5000.0],
            "Total_Sell_INR": [700.0, 600.0, 500.0, 10000.0, 8000.0, 6000.0],
            "Chargeable_Weight_Kg": [10.0, 10.0, 10.0, 50.0, 50.0, 0.2],
        })
        filtered = filter_commercial_economic_bounds(df)
        # Row 0: Valid (Buy=600 > 500, Sell=700 > 600, Wt=10 >= 0.5)
        # Row 1: Invalid (Buy=500 NOT > 500)
        # Row 2: Invalid (Buy=400 <= 500)
        # Row 3: Invalid (Sell=10000 NOT > Buy=10000)
        # Row 4: Invalid (Sell=8000 < Buy=10000)
        # Row 5: Invalid (Wt=0.2 < 0.5)
        self.assertEqual(len(filtered), 1)
        self.assertAlmostEqual(filtered.iloc[0]["Total_Buy_INR"], 600.0)

    def test_recalculate_margins(self):
        df = pd.DataFrame({
            "Total_Buy_INR": [10000.0, 20000.0],
            "Total_Sell_INR": [11500.0, 23000.0],
        })
        recalc = recalculate_margins(df)
        self.assertAlmostEqual(recalc.iloc[0]["Margin_Amount_INR"], 1500.0)
        self.assertAlmostEqual(recalc.iloc[0]["Margin_Percentage"], 15.0)
        self.assertAlmostEqual(recalc.iloc[1]["Margin_Amount_INR"], 3000.0)
        self.assertAlmostEqual(recalc.iloc[1]["Margin_Percentage"], 15.0)


class TestEntityAndLaneStandardization(unittest.TestCase):
    def test_standardize_company_group(self):
        s = pd.Series(["Direct", "Subagent", "Overseas Nomination", "Unknown", None])
        std = standardize_company_group(s)
        self.assertEqual(std.iloc[0], "Shipper / Consignee")
        self.assertEqual(std.iloc[1], "Subagent")
        self.assertEqual(std.iloc[2], "Overseas Agent")
        self.assertEqual(std.iloc[3], "Shipper / Consignee")
        self.assertEqual(std.iloc[4], "Shipper / Consignee")

    def test_resolve_origin_region(self):
        row_bom = pd.Series({"Port Of Loading": "Chhatrapati Shivaji Maharaj International Airport", "Booked by branch": "Mumbai"})
        row_del = pd.Series({"Port Of Loading": "Indira Gandhi International Airport", "Booked by branch": "Delhi"})
        row_hyd = pd.Series({"Port Of Loading": "Hyderabad", "Booked by branch": "Hyderabad"})
        row_ccu = pd.Series({"Port Of Loading": "Kolkata", "Booked by branch": "Kolkata"})
        # Foreign or null port with Indian branch fallback
        row_fb = pd.Series({"Port Of Loading": None, "Booked by branch": "Haryana"})
        row_fb_bom = pd.Series({"Port Of Loading": "Nice", "Booked by branch": "Mumbai"})

        self.assertEqual(resolve_origin_region(row_bom), "West India")
        self.assertEqual(resolve_origin_region(row_del), "North India")
        self.assertEqual(resolve_origin_region(row_hyd), "South India")
        self.assertEqual(resolve_origin_region(row_ccu), "East India")
        self.assertEqual(resolve_origin_region(row_fb), "North India")
        self.assertEqual(resolve_origin_region(row_fb_bom), "West India")


class TestLabelPurityAndBuildGates(unittest.TestCase):
    def test_label_purity_separation(self):
        # Scenario: Inquiry has multiple quotes. The Won quote has Quote Status == 'Won',
        # while alternative rejected carrier quotes have Inquiry Status == 'Won' but Quote Status == 'Lost'.
        mock_df = pd.DataFrame([
            {
                "Inquiry Number": "INQ101", "Quotation Number": "Q1", "Inquiry Status": "Won",
                "Quote Status": "Won", "Created At": "2025-03-10 10:00:00",
                "Customer Company": "ACME LOGISTICS PVT LTD", "Total Buy": "10,000 INR", "Total Sell": "11,500 INR",
                "Unit Freight Buy": "100 INR", "Unit Freight Sell": "115 INR", "Chargeable Weight": 100,
                "Port Of Loading": "Chhatrapati Shivaji Maharaj International Airport", "Port Of Discharge": "Dubai",
                "Quoted Carrier": "Emirates", "Commodity Type": "General Cargo", "Booked by branch": "Mumbai",
                "Business Received Through": "Direct",
            },
            {
                "Inquiry Number": "INQ101", "Quotation Number": "Q2", "Inquiry Status": "Won",
                "Quote Status": "Lost", "Created At": "2025-03-10 10:00:00",
                "Customer Company": "ACME LOGISTICS PVT LTD", "Total Buy": "10,500 INR", "Total Sell": "12,000 INR",
                "Unit Freight Buy": "105 INR", "Unit Freight Sell": "120 INR", "Chargeable Weight": 100,
                "Port Of Loading": "Chhatrapati Shivaji Maharaj International Airport", "Port Of Discharge": "Dubai",
                "Quoted Carrier": "Qatar Airways", "Commodity Type": "General Cargo", "Booked by branch": "Mumbai",
                "Business Received Through": "Direct",
            },
            {
                "Inquiry Number": "INQ102", "Quotation Number": "Q3", "Inquiry Status": "Lost",
                "Quote Status": "Lost", "Created At": "2025-04-12 11:00:00",
                "Customer Company": "BETA FREIGHT", "Total Buy": "20,000 INR", "Total Sell": "22,000 INR",
                "Unit Freight Buy": "200 INR", "Unit Freight Sell": "220 INR", "Chargeable Weight": 100,
                "Port Of Loading": "Indira Gandhi International Airport", "Port Of Discharge": "London",
                "Quoted Carrier": "Air India", "Commodity Type": "General Cargo", "Booked by branch": "Delhi",
                "Business Received Through": "Subagent",
            },
        ])

        cleaned = clean_and_enrich_inquiries(mock_df)
        won_deals = cleaned[cleaned["Quote Status"] == "Won"].copy()

        self.assertEqual(len(cleaned), 3)
        self.assertEqual(len(won_deals), 1)
        self.assertEqual(won_deals.iloc[0]["Quotation Number"], "Q1")
        self.assertEqual(won_deals.iloc[0]["is_won"], 1)

        # Validate build gates on won_deals
        validate_build_gates(won_deals, is_margin_dataset=True, dataset_name="won_deals_test")


class TestSplitLogic(unittest.TestCase):
    def setUp(self):
        # Generate 100 mock won records spanning Jan to Dec 2025 across multiple weight tiers
        records = []
        start_date = pd.Timestamp("2025-01-01")
        for i in range(100):
            d = start_date + pd.Timedelta(days=i * 3.6)
            wt = 20.0 + (i % 6) * 200.0  # Spans <45kg, 100-300kg, 500-1000kg, >1000kg
            buy = 5000.0 + i * 50.0
            sell = buy * 1.15
            records.append({
                "Inquiry Number": f"INQ_{i:04d}",
                "Quotation Number": f"Q_{i:04d}",
                "Quote Status": "Won",
                "is_won": 1,
                "Buy_Cost_Estimated": 0,
                "Created At": str(d),
                "quote_date": d.strftime("%Y-%m-%d"),
                "Quote_Month": int(d.month),
                "Year_Month": d.strftime("%Y-%m"),
                "Month_Name": d.month_name(),
                "Quote_DayOfWeek": int(d.dayofweek),
                "Is_Month_End": int(d.day >= 25),
                "Month_Sin": float(np.sin(2.0 * np.pi * d.month / 12.0)),
                "Month_Cos": float(np.cos(2.0 * np.pi * d.month / 12.0)),
                "Air_Cargo_Season": "Q1_Slack_Fiscal_Close" if d.month <= 3 else "Q2_Perishable_Peak",
                "Total_Buy_INR": buy,
                "Total_Sell_INR": sell,
                "Margin_Amount_INR": round(sell - buy, 2),
                "Margin_Percentage": (sell - buy) / buy * 100.0,
                "Chargeable_Weight_Kg": wt,
                "Weight_Tier": get_weight_tier(wt),
                "customer_id": f"CUST_ACC_{i % 10}",
                "Customer_Company": f"ACCOUNT {i % 10}",
                "Origin_Region": "West India",
                "Destination_Region": "Middle East",
                "Company_Group": "Shipper / Consignee",
            })
        self.mock_won_df = pd.DataFrame(records)

    def test_chronological_split(self):
        train, test = chronological_split_70_30(self.mock_won_df)
        self.assertEqual(len(train), 70)
        self.assertEqual(len(test), 30)

        # Ensure no temporal lookahead leakage
        self.assertLessEqual(train["quote_date"].max(), test["quote_date"].min())

        # Validate build gates on both splits
        validate_build_gates(train, is_margin_dataset=True, dataset_name="train_chronological")
        validate_build_gates(test, is_margin_dataset=True, dataset_name="test_chronological")

    def test_stratified_split(self):
        train, test = stratified_split_70_30(self.mock_won_df, test_size=0.30, random_state=42)
        self.assertEqual(len(train), 70)
        self.assertEqual(len(test), 30)

        # Ensure all weight tiers exist in both sets
        train_tiers = set(train["Weight_Tier"].unique())
        test_tiers = set(test["Weight_Tier"].unique())
        total_tiers = set(self.mock_won_df["Weight_Tier"].unique())
        self.assertEqual(train_tiers, total_tiers)
        self.assertEqual(test_tiers, total_tiers)

        # Validate build gates
        validate_build_gates(train, is_margin_dataset=True, dataset_name="train_stratified")
        validate_build_gates(test, is_margin_dataset=True, dataset_name="test_stratified")


class TestPersistedGeneratedFiles(unittest.TestCase):
    """Verify the generated files on disk conform to all build gates and requirements."""
    def test_persisted_csvs_exist_and_pass_gates(self):
        proc_dir = os.path.join(PROJECT_ROOT, "data", "processed")
        expected_files = {
            "all_inquiries_2025_cleaned.csv": False,
            "won_deals_2025_train_70.csv": True,
            "won_deals_2025_test_30.csv": True,
            "won_deals_2025_train_70_stratified.csv": True,
            "won_deals_2025_test_30_stratified.csv": True,
        }

        for fname, is_margin in expected_files.items():
            fpath = os.path.join(proc_dir, fname)
            self.assertTrue(os.path.exists(fpath), f"Expected file {fpath} does not exist.")
            df = pd.read_csv(fpath, low_memory=False)
            self.assertGreater(len(df), 0, f"File {fname} is empty.")
            # Run build gate assertions
            validate_build_gates(df, is_margin_dataset=is_margin, dataset_name=fname)

            if is_margin:
                self.assertTrue((df["Quote Status"] == "Won").all(), f"Non-Won quote leaked in {fname}")
                self.assertTrue((df["is_won"] == 1).all(), f"is_won != 1 found in {fname}")
                self.assertTrue((df["Total_Buy_INR"] > 500.0).all(), f"Total_Buy_INR <= 500 found in {fname}")
                self.assertTrue((df["Total_Sell_INR"] > df["Total_Buy_INR"]).all(), f"Sell <= Buy found in {fname}")
                self.assertTrue((df["Chargeable_Weight_Kg"] >= 0.5).all(), f"Weight < 0.5 found in {fname}")
                self.assertTrue((df["Chargeable_Weight_Kg"] <= 25000.0).all(), f"Weight > 25000 found in {fname}")
                self.assertTrue(df["Origin_Region"].isin(["West India", "North India", "South India", "East India"]).all())


if __name__ == "__main__":
    unittest.main()

