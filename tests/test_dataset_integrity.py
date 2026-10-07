"""
tests/test_dataset_integrity.py
Unit tests verifying the clean dataset integrity according to commercial airfreight rules.
"""

import unittest
import os
import pandas as pd


class TestDatasetIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_path = "data/processed/Air_Export_Pricing_Combined_ML.csv"
        if not os.path.exists(cls.data_path):
            raise FileNotFoundError(f"Clean dataset missing at {cls.data_path}")
        cls.df = pd.read_csv(cls.data_path)

    def test_pure_quote_status(self):
        """Only resolved Won and Lost quotes should exist in the training dataset."""
        unique_statuses = set(self.df["Quote Status"].dropna().unique())
        self.assertSetEqual(unique_statuses, {"Won", "Lost"})
        # Verify is_won aligns with Quote Status
        won_alignment = (self.df["is_won"] == (self.df["Quote Status"] == "Won").astype(int)).all()
        self.assertTrue(won_alignment)

    def test_physical_plausibility(self):
        """Chargeable weight must be >= gross weight, and within commercial bounds."""
        # Under IATA, chargeable cannot be less than gross weight (allow 0.01 tolerance for float)
        violating_physics = (self.df["Gross_Weight_Kg"] > (self.df["Chargeable_Weight_Kg"] + 0.01)).sum()
        self.assertEqual(violating_physics, 0)

        # Density ratio bounded in [0.1, 1.0]
        self.assertGreaterEqual(self.df["Cargo_Density_Ratio"].min(), 0.10)
        self.assertLessEqual(self.df["Cargo_Density_Ratio"].max(), 1.00)

        # Commercial weight bounds
        self.assertGreaterEqual(self.df["Chargeable_Weight_Kg"].min(), 0.5)
        self.assertLessEqual(self.df["Chargeable_Weight_Kg"].max(), 25000.0)

    def test_commercial_margins_and_prices(self):
        """Total buy > 500, Sell >= Buy, margin % between 0.2% and 60%."""
        self.assertGreater(self.df["Total_Buy_INR"].min(), 500.0)
        self.assertTrue((self.df["Total_Sell_INR"] >= self.df["Total_Buy_INR"]).all())
        self.assertGreaterEqual(self.df["Margin_Percentage"].min(), 0.20)
        self.assertLessEqual(self.df["Margin_Percentage"].max(), 60.00)

    def test_destination_mapping_coverage(self):
        """Unmapped destination regions must be less than 0.1%."""
        unmapped = (self.df["Destination_Region"] == "Other Destination").sum()
        unmapped_pct = (unmapped / len(self.df)) * 100.0
        self.assertLess(unmapped_pct, 0.10)

    def test_origin_regions_clean(self):
        """Origins should be valid Indian regions."""
        other_origins = (self.df["Origin_Region"] == "Other Origin").sum()
        self.assertEqual(other_origins, 0)



class TestJobConsolidatedDatasetIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_path = "data/processed/Job_Wise_Consolidated_Cleaned_ML.csv"
        if not os.path.exists(cls.data_path):
            raise FileNotFoundError(f"Clean dataset missing at {cls.data_path}")
        cls.df = pd.read_csv(cls.data_path)

    def test_pure_quote_status(self):
        """All executed jobs represent resolved Won outcomes."""
        unique_statuses = set(self.df["Quote Status"].dropna().unique())
        self.assertSetEqual(unique_statuses, {"Won"})
        self.assertTrue((self.df["is_won"] == 1).all())

    def test_physical_plausibility(self):
        """Chargeable weight must be >= gross weight, and within commercial bounds."""
        violating_physics = (self.df["Gross_Weight_Kg"] > (self.df["Chargeable_Weight_Kg"] + 0.01)).sum()
        self.assertEqual(violating_physics, 0)
        self.assertGreaterEqual(self.df["Cargo_Density_Ratio"].min(), 0.10)
        self.assertLessEqual(self.df["Cargo_Density_Ratio"].max(), 1.00)
        self.assertGreaterEqual(self.df["Chargeable_Weight_Kg"].min(), 0.5)
        self.assertLessEqual(self.df["Chargeable_Weight_Kg"].max(), 25000.0)

    def test_commercial_margins_and_prices(self):
        """Total sell >= Total buy, sell >= 500, margin % within bounds."""
        self.assertGreaterEqual(self.df["Total_Sell_INR"].min(), 500.0)
        self.assertTrue((self.df["Total_Sell_INR"] >= self.df["Total_Buy_INR"]).all())
        self.assertGreaterEqual(self.df["Margin_Percentage"].min(), 0.20)
        self.assertLessEqual(self.df["Margin_Percentage"].max(), 60.00)

    def test_destination_mapping_coverage(self):
        """Unmapped destination regions must be exactly zero."""
        unmapped = (self.df["Destination_Region"] == "Other Destination").sum()
        self.assertEqual(unmapped, 0)

    def test_origin_regions_clean(self):
        """Origins should be valid Indian regions."""
        other_origins = (self.df["Origin_Region"] == "Other Origin").sum()
        self.assertEqual(other_origins, 0)

    def test_schema_completeness(self):
        """Confirm zero nulls across all 45 production schema columns."""
        self.assertEqual(self.df.isnull().sum().sum(), 0)
        self.assertEqual(self.df.shape[1], 45)


if __name__ == "__main__":
    unittest.main()

