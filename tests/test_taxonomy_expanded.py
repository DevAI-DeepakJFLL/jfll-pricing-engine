"""
tests/test_taxonomy_expanded.py
Unit tests for expanded international gateways and carrier tier classifications.
Validates 2025 dataset taxonomy resolution and carrier tiers.
"""

import os
import unittest
import pandas as pd

from src.taxonomy import (
    map_destination_region,
    classify_carrier_tier,
    DESTINATION_REGIONS,
    CARRIER_TIERS,
    CARRIER_TIER_PREMIUM,
    CARRIER_TIER_STANDARD,
    CARRIER_TIER_BUDGET,
)


class TestExpandedGatewaysEurope(unittest.TestCase):
    """Verifies newly mapped European airport gateways and IATA codes."""

    def test_europe_gateways(self):
        ports = [
            ("Sheremetyevo International Airport", "Europe"),
            ("Sheremetyevo", "Europe"),
            ("SVO", "Europe"),
            ("Warszawa", "Europe"),
            ("WAW", "Europe"),
            ("El Prat de Llobregat", "Europe"),
            ("El Prat", "Europe"),
            ("BCN", "Europe"),
            ("Otopeni Apt/Bucuresti", "Europe"),
            ("Otopeni", "Europe"),
            ("OTP", "Europe"),
            ("Hannover", "Europe"),
            ("HAJ", "Europe"),
            ("Koln", "Europe"),
            ("CGN", "Europe"),
            ("Koln/Bonn", "Europe"),
            ("Saint Petersburg (ex Leningrad)", "Europe"),
            ("Saint Petersburg", "Europe"),
            ("LED", "Europe"),
            ("Calden", "Europe"),
            ("KSF", "Europe"),
            ("Tbilisi", "Europe"),
            ("TBS", "Europe"),
            ("Munster", "Europe"),
            ("Bordeaux", "Europe"),
            ("Sevilla", "Europe"),
            ("Dresden", "Europe"),
            ("Nuremberg", "Europe"),
            ("Bilbao", "Europe"),
            ("Basel", "Europe"),
            ("Katowice", "Europe"),
            ("Vilnius", "Europe"),
            ("Riga", "Europe"),
            ("Tallinn", "Europe"),
        ]
        for port, expected in ports:
            resolved = map_destination_region(port)
            self.assertEqual(
                resolved, expected,
                f"Europe mapping failed for port='{port}': got '{resolved}', expected '{expected}'"
            )


class TestExpandedGatewaysAfrica(unittest.TestCase):
    """Verifies newly mapped African airport gateways and IATA codes."""

    def test_africa_gateways(self):
        ports = [
            ("Sir Seewoosagur Ramgoolam Int Apt", "Africa"),
            ("Sir Seewoosagur Ramgoolam", "Africa"),
            ("MRU", "Africa"),
            ("Mohammed V International Airport", "Africa"),
            ("Mohammed V", "Africa"),
            ("CMN", "Africa"),
            ("Lome", "Africa"),
            ("LFW", "Africa"),
            ("Blaise Diagne International Airport", "Africa"),
            ("DSS", "Africa"),
            ("Kinshasa", "Africa"),
            ("FIH", "Africa"),
            ("Antananarivo", "Africa"),
            ("TNR", "Africa"),
            ("Mahe Island Apt", "Africa"),
            ("Mahe Island", "Africa"),
            ("SEZ", "Africa"),
            ("Ouagadougou", "Africa"),
            ("OUA", "Africa"),
            ("Douala", "Africa"),
            ("DLA", "Africa"),
            ("Kigali", "Africa"),
            ("KGL", "Africa"),
            ("Windhoek", "Africa"),
            ("WDH", "Africa"),
            ("Bujumbura", "Africa"),
            ("BJM", "Africa"),
            ("Libreville", "Africa"),
            ("LBV", "Africa"),
        ]
        for port, expected in ports:
            resolved = map_destination_region(port)
            self.assertEqual(
                resolved, expected,
                f"Africa mapping failed for port='{port}': got '{resolved}', expected '{expected}'"
            )


class TestExpandedGatewaysAsiaPacific(unittest.TestCase):
    """Verifies newly mapped Asia-Pacific airport gateways and IATA codes."""

    def test_asia_pacific_gateways(self):
        ports = [
            ("Kansai Int Apt", "Asia-Pacific"),
            ("Kansai", "Asia-Pacific"),
            ("KIX", "Asia-Pacific"),
            ("Tashkent", "Asia-Pacific"),
            ("TAS", "Asia-Pacific"),
            ("Almaty", "Asia-Pacific"),
            ("ALA", "Asia-Pacific"),
            ("Yangon", "Asia-Pacific"),
            ("RGN", "Asia-Pacific"),
            ("Vientiane", "Asia-Pacific"),
            ("VTE", "Asia-Pacific"),
            ("Phnom Penh", "Asia-Pacific"),
            ("PNH", "Asia-Pacific"),
            ("Ulaanbaatar", "Asia-Pacific"),
            ("ULN", "Asia-Pacific"),
            ("Dushanbe", "Asia-Pacific"),
            ("DYU", "Asia-Pacific"),
            ("Bishkek", "Asia-Pacific"),
            ("FRU", "Asia-Pacific"),
            ("Ashgabat", "Asia-Pacific"),
            ("ASB", "Asia-Pacific"),
            ("Fukuoka", "Asia-Pacific"),
            ("FUK", "Asia-Pacific"),
            ("Sapporo", "Asia-Pacific"),
            ("CTS", "Asia-Pacific"),
        ]
        for port, expected in ports:
            resolved = map_destination_region(port)
            self.assertEqual(
                resolved, expected,
                f"Asia-Pacific mapping failed for port='{port}': got '{resolved}', expected '{expected}'"
            )


class TestExpandedGatewaysOtherRegions(unittest.TestCase):
    """Verifies Middle East, Indian Subcontinent, Latin America, and North America."""

    def test_middle_east_gateways(self):
        ports = [
            ("Ben Gurion International Apt", "Middle East"),
            ("Ben Gurion", "Middle East"),
            ("TLV", "Middle East"),
        ]
        for port, expected in ports:
            self.assertEqual(map_destination_region(port), expected)

    def test_indian_subcontinent_gateways(self):
        ports = [
            ("Kabul", "Indian Subcontinent"),
            ("KBL", "Indian Subcontinent"),
        ]
        for port, expected in ports:
            self.assertEqual(map_destination_region(port), expected)

    def test_latin_america_gateways(self):
        ports = [
            ("La Aurora International Airport", "Latin America"),
            ("La Aurora", "Latin America"),
            ("GUA", "Latin America"),
            ("Kingston", "Latin America"),
            ("KIN", "Latin America"),
            ("Grand Cayman", "Latin America"),
            ("GCM", "Latin America"),
            ("Montego Bay", "Latin America"),
            ("MBJ", "Latin America"),
            ("San Pedro Sula", "Latin America"),
            ("SAP", "Latin America"),
        ]
        for port, expected in ports:
            self.assertEqual(map_destination_region(port), expected)

    def test_north_america_gateways(self):
        ports = [
            ("Kansas City", "North America"),
            ("MCI", "North America"),
            ("Austin", "North America"),
            ("AUS", "North America"),
            ("Nashville", "North America"),
            ("BNA", "North America"),
            ("Indianapolis", "North America"),
            ("IND", "North America"),
            ("Raleigh", "North America"),
            ("RDU", "North America"),
            ("Cleveland", "North America"),
            ("CLE", "North America"),
            ("Cincinnati", "North America"),
            ("CVG", "North America"),
        ]
        for port, expected in ports:
            self.assertEqual(map_destination_region(port), expected)


class TestCarrierTaxonomy(unittest.TestCase):
    """Verifies carrier tier constants and classify_carrier_tier function."""

    def test_carrier_tier_constants(self):
        self.assertEqual(CARRIER_TIERS, ["Premium", "Standard", "Budget"])
        self.assertIn("Emirates", CARRIER_TIER_PREMIUM)
        self.assertIn("Qatar Airways", CARRIER_TIER_PREMIUM)
        self.assertIn("Singapore Airlines", CARRIER_TIER_PREMIUM)
        self.assertIn("Cathay Pacific", CARRIER_TIER_PREMIUM)
        self.assertIn("Lufthansa", CARRIER_TIER_PREMIUM)
        self.assertIn("Etihad Airways", CARRIER_TIER_PREMIUM)
        self.assertIn("British Airways", CARRIER_TIER_PREMIUM)

        self.assertIn("Air India", CARRIER_TIER_STANDARD)
        self.assertIn("Ethiopian Airlines", CARRIER_TIER_STANDARD)
        self.assertIn("Air France", CARRIER_TIER_STANDARD)
        self.assertIn("Turkish Airlines", CARRIER_TIER_STANDARD)
        self.assertIn("Saudi Airlines", CARRIER_TIER_STANDARD)
        self.assertIn("SriLankan Airlines", CARRIER_TIER_STANDARD)
        self.assertIn("Gulf Air", CARRIER_TIER_STANDARD)
        self.assertIn("Oman Air", CARRIER_TIER_STANDARD)
        self.assertIn("EgyptAir", CARRIER_TIER_STANDARD)
        self.assertIn("Kuwait Airways", CARRIER_TIER_STANDARD)

        self.assertIn("IndiGo", CARRIER_TIER_BUDGET)
        self.assertIn("SpiceJet", CARRIER_TIER_BUDGET)
        self.assertIn("Air Arabia", CARRIER_TIER_BUDGET)
        self.assertIn("Flydubai", CARRIER_TIER_BUDGET)

    def test_classify_carrier_tier_premium(self):
        premium_cases = [
            "Emirates",
            "Emirates SkyCargo",
            "Qatar Airways",
            "Singapore Airlines",
            "Singapore Airlines Ltd",
            "Cathay Pacific",
            "Lufthansa",
            "Lufthansa Cargo",
            "Etihad Airways",
            "British Airways",
        ]
        for c in premium_cases:
            self.assertEqual(
                classify_carrier_tier(c), "Premium",
                f"Failed for premium carrier '{c}'"
            )

    def test_classify_carrier_tier_standard(self):
        standard_cases = [
            "Air India",
            "Ethiopian Airlines",
            "Air France",
            "Turkish Airlines",
            "Saudi Airlines",
            "Saudi Airlines Cargo",
            "Saudia",
            "SriLankan Airlines",
            "Gulf Air",
            "Oman Air",
            "EgyptAir",
            "Egyptair",
            "Kuwait Airways",
            "Virgin Atlantic",
            "KLM Cargo",
            "Air Canada",
            "Unknown Airline",
        ]
        for c in standard_cases:
            self.assertEqual(
                classify_carrier_tier(c), "Standard",
                f"Failed for standard carrier '{c}'"
            )

    def test_classify_carrier_tier_budget(self):
        budget_cases = [
            "IndiGo",
            "Indigo",
            "SpiceJet",
            "Spice Jet",
            "Air Arabia",
            "Air Arabia Abu Dhabi",
            "Flydubai",
            "Fly Dubai",
        ]
        for c in budget_cases:
            self.assertEqual(
                classify_carrier_tier(c), "Budget",
                f"Failed for budget carrier '{c}'"
            )

    def test_classify_carrier_tier_fallbacks(self):
        self.assertEqual(classify_carrier_tier(None), "Standard")
        self.assertEqual(classify_carrier_tier(""), "Standard")
        self.assertEqual(classify_carrier_tier("   "), "Standard")
        self.assertEqual(classify_carrier_tier("Premium"), "Premium")
        self.assertEqual(classify_carrier_tier("Budget"), "Budget")
        self.assertEqual(classify_carrier_tier("Standard"), "Standard")


class Test2025DatasetUnmappedRate(unittest.TestCase):
    """Verifies that unmapped destination rate on won_deals_2025_train_70.csv is < 0.5%."""

    def test_train_70_unmapped_rate_below_half_percent(self):
        path = "data/processed/won_deals_2025_train_70.csv"
        if not os.path.exists(path):
            self.skipTest(f"Dataset {path} not found")

        df = pd.read_csv(path, low_memory=False)
        dest_col = "Port Of Discharge" if "Port Of Discharge" in df.columns else "Destination_Port"
        mapped = df[dest_col].fillna("").apply(map_destination_region)
        unmapped_count = (mapped == "Other Destination").sum()
        total_count = len(df)
        unmapped_rate = unmapped_count / total_count

        self.assertLess(
            unmapped_rate, 0.005,
            f"Unmapped destination rate {unmapped_rate*100:.3f}% ({unmapped_count}/{total_count}) is >= 0.5%"
        )


if __name__ == "__main__":
    unittest.main()
