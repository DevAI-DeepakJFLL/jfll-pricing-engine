"""
tests/test_customer_normalizer.py
Unit and integration tests for customer entity normalization (Issue D6).
Verifies suffix stripping, punctuation cleanup, alias mapping,
deterministic ID generation, and collapsing >= 43 fragmented entity duplicates.
"""

import os
import unittest
import pandas as pd

from src.customer_normalizer import (
    normalize_customer_name,
    get_canonical_customer_id,
    CUSTOMER_ALIAS_MAP,
)

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "processed",
    "Air_Export_Pricing_Combined_ML.csv",
)


class TestCustomerNormalizer(unittest.TestCase):
    def test_normalizes_corporate_suffixes(self):
        variants = [
            "ALLANASONS PRIVATE LIMITED",
            "ALLANASONS PVT LTD",
            "ALLANASONS PVT. LTD.",
            "ALLANASONS  LIMITED",
            "ALLANASONS LTD",
            "ALLANASONS INC",
            "ALLANASONS LLP",
            "ALLANASONS CORP",
            "ALLANASONS CORPORATION",
            "ALLANASONS FZE",
            "ALLANASONS LLC",
        ]
        results = [normalize_customer_name(v) for v in variants]
        self.assertEqual(len(set(results)), 1)
        self.assertEqual(results[0], "ALLANASONS")

    def test_case_and_whitespace_insensitivity(self):
        variants = [
            "  allanasons private limited  ",
            "Allanasons Pvt Ltd",
            "ALLANASONS   PVT   LTD",
            "\tallanasons\tlimited\n",
        ]
        for v in variants:
            self.assertEqual(normalize_customer_name(v), "ALLANASONS")

    def test_handles_ampersand_and_punctuation(self):
        name1 = "M&M MILITZER AND MUENCH (INDIA) PRIVATE LIMITED"
        name2 = "M&M MILITZER & MUENCH INDIA PVT. LTD."
        self.assertEqual(normalize_customer_name(name1), normalize_customer_name(name2))

        name3 = "Y&H CARGO PVT.LTD"
        name4 = "Y & H CARGO PRIVATE LIMITED"
        self.assertEqual(normalize_customer_name(name3), normalize_customer_name(name4))

    def test_canonical_customer_id_format_and_determinism(self):
        id1 = get_canonical_customer_id("ALLANASONS PRIVATE LIMITED")
        id2 = get_canonical_customer_id("ALLANASONS PVT LTD")
        id3 = get_canonical_customer_id("allanasons ltd")

        self.assertEqual(id1, id2)
        self.assertEqual(id2, id3)
        self.assertEqual(id1, "CUST_ALLANASONS")
        self.assertTrue(id1.startswith("CUST_"))

    def test_canonical_customer_id_edge_cases(self):
        self.assertEqual(get_canonical_customer_id(""), "CUST_UNKNOWN")
        self.assertEqual(get_canonical_customer_id(None), "CUST_UNKNOWN")
        self.assertEqual(get_canonical_customer_id("   "), "CUST_UNKNOWN")

    def test_customer_alias_map_resolves_known_aliases(self):
        self.assertIsInstance(CUSTOMER_ALIAS_MAP, dict)
        self.assertGreater(len(CUSTOMER_ALIAS_MAP), 0)

        # Test specific known alias resolution
        self.assertEqual(
            normalize_customer_name("AD LOYAL LOGISTICS"),
            "ADLOYAL LOGISTICS",
        )
        self.assertEqual(
            normalize_customer_name("ALONSO LOGISTICS HNDIA"),
            "ALONSO LOGISTICS INDIA",
        )
        self.assertEqual(
            normalize_customer_name("JETFREGHT"),
            "JETFREIGHT",
        )

    def test_dataset_entity_reduction_meets_issue_d6_requirement(self):
        self.assertTrue(
            os.path.exists(DATA_PATH),
            f"Dataset not found at {DATA_PATH}",
        )
        df = pd.read_csv(DATA_PATH, low_memory=False)
        self.assertIn("Customer_Company", df.columns)

        raw_customers = df["Customer_Company"].dropna().unique()
        initial_count = len(raw_customers)
        self.assertEqual(
            initial_count,
            1073,
            f"Expected exactly 1,073 unique customer names, found {initial_count}",
        )

        canonical_entities = set(normalize_customer_name(c) for c in raw_customers)
        final_count = len(canonical_entities)
        collapsed = initial_count - final_count

        # Must collapse to <= 1,030 entities (resolving at least 43 duplicates)
        self.assertLessEqual(
            final_count,
            1030,
            f"Expected <= 1,030 distinct entities, but got {final_count} (only {collapsed} duplicates resolved)",
        )
        self.assertGreaterEqual(
            collapsed,
            43,
            f"Expected >= 43 duplicates resolved as per Issue D6, but resolved {collapsed}",
        )


if __name__ == "__main__":
    unittest.main()
