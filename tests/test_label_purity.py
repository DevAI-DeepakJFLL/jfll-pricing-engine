"""
tests/test_label_purity.py
Unit tests for label purity (excluding Draft/Cancelled/Hold), airport origin validation,
and destination region mapping coverage.
"""

import unittest
import pandas as pd
from src.preprocess import (
    filter_pure_resolved_quotes,
    filter_worked_quotes,
    is_valid_indian_origin,
    map_dest_region,
    map_origin_region,
)


class TestLabelPurityAndPortMapping(unittest.TestCase):
    def test_filter_pure_resolved_quotes(self):
        sample_data = pd.DataFrame([
            {"Inquiry Number": "INQ1", "Quote Status": "Won", "is_won": 1, "Total_Buy_INR": 10000},
            {"Inquiry Number": "INQ2", "Quote Status": "Lost", "is_won": 0, "Total_Buy_INR": 12000},
            {"Inquiry Number": "INQ3", "Quote Status": "Draft", "is_won": 0, "Total_Buy_INR": 8000},
            {"Inquiry Number": "INQ4", "Quote Status": "Cancelled", "is_won": 0, "Total_Buy_INR": 9500},
            {"Inquiry Number": "INQ5", "Quote Status": "Hold", "is_won": 0, "Total_Buy_INR": 11000},
            {"Inquiry Number": "INQ6", "Quote Status": "Approved", "is_won": 0, "Total_Buy_INR": 15000},
        ])
        filtered_df = filter_pure_resolved_quotes(sample_data)
        self.assertEqual(len(filtered_df), 2)
        self.assertSetEqual(set(filtered_df["Quote Status"]), {"Won", "Lost"})

    def test_unworked_lost_removed(self):
        sample_df = pd.DataFrame([
            {"Inquiry Number": "W1", "Quote Status": "Won", "is_won": 1, "Total_Buy_INR": 10000, "Total_Sell_INR": 11000, "Quote_Turnaround_Hours": 0.5, "quote_revision_count": 1},
            {"Inquiry Number": "L1", "Quote Status": "Lost", "is_won": 0, "Total_Buy_INR": 10000, "Total_Sell_INR": 10204, "Quote_Turnaround_Hours": 0.3, "quote_revision_count": 1},  # unworked (instant & default GM)
            {"Inquiry Number": "L2", "Quote Status": "Lost", "is_won": 0, "Total_Buy_INR": 10000, "Total_Sell_INR": 11500, "Quote_Turnaround_Hours": 2.5, "quote_revision_count": 2},  # worked lost
        ])
        kept, dropped = filter_worked_quotes(sample_df)
        self.assertTrue((dropped["is_won"] == 0).all())
        self.assertTrue((kept[kept.is_won == 0]["Quote_Turnaround_Hours"] >= 1.0).all())
        self.assertEqual(len(dropped), 1)
        self.assertEqual(len(kept), 2)

    def test_is_valid_indian_origin(self):
        self.assertTrue(is_valid_indian_origin("Indira Gandhi International Airport"))
        self.assertTrue(is_valid_indian_origin("Chhatrapati Shivaji Maharaj International Airport"))
        self.assertTrue(is_valid_indian_origin("Ahmedabad"))
        self.assertTrue(is_valid_indian_origin("Bangalore"))
        self.assertTrue(is_valid_indian_origin("Cochin"))
        self.assertTrue(is_valid_indian_origin("Chennai"))
        self.assertTrue(is_valid_indian_origin("Kolkata"))
        self.assertTrue(is_valid_indian_origin("Hyderabad"))
        self.assertTrue(is_valid_indian_origin("Kozhikode (ex Calicut)"))

        # Non-Indian origins should be invalid / quarantined
        self.assertFalse(is_valid_indian_origin("Decimomannu"))
        self.assertFalse(is_valid_indian_origin("Amalfi"))
        self.assertFalse(is_valid_indian_origin("Alexandroupolis"))
        self.assertFalse(is_valid_indian_origin("Los Angeles"))
        self.assertFalse(is_valid_indian_origin("Maastricht"))

    def test_expanded_destination_regions(self):
        # Key unmapped destinations from review report
        self.assertEqual(map_dest_region("Sir Seewoosagur Ramgoolam Int Apt"), "Africa")
        self.assertEqual(map_dest_region("Durban"), "Africa")
        self.assertEqual(map_dest_region("Mohammed V International Airport"), "Africa")
        self.assertEqual(map_dest_region("Cape Town"), "Africa")

        self.assertEqual(map_dest_region("Bruxelles (Brussel)"), "Europe")
        self.assertEqual(map_dest_region("Praha"), "Europe")
        self.assertEqual(map_dest_region("Lisboa"), "Europe")
        self.assertEqual(map_dest_region("Porto"), "Europe")
        self.assertEqual(map_dest_region("Fiumicino Apt/Roma"), "Europe")
        self.assertEqual(map_dest_region("Otopeni Apt/Bucuresti"), "Europe")
        self.assertEqual(map_dest_region("Dusseldorf"), "Europe")

        self.assertEqual(map_dest_region("Dorval"), "North America")
        self.assertEqual(map_dest_region("Charlotte"), "North America")
        self.assertEqual(map_dest_region("Dulles Int Apt/Washington"), "North America")

        self.assertEqual(map_dest_region("Tashkent"), "Asia-Pacific")
        self.assertEqual(map_dest_region("Kansai Int Apt"), "Asia-Pacific")
        self.assertEqual(map_dest_region("Kabul"), "Asia-Pacific")


if __name__ == "__main__":
    unittest.main()
