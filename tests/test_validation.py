"""
tests/test_validation.py
Unit tests for strict inquiry validation and normalization (src/validation.py).
"""

import unittest
import datetime
from src.validation import (
    validate_inquiry,
    QuoteValidationError,
    IST,
)


class TestInquiryValidation(unittest.TestCase):
    """Test suite for inquiry validation boundaries, types, and defaults."""

    def test_valid_inquiry_normalization(self):
        """Verifies clean inquiry is normalized with correct fields."""
        raw = {
            "chargeable_wt": "150.5",
            "buy": "30100.0",
            "origin": "Bangalore",
            "dest": "Frankfurt am Main",
            "quote_date": "2026-06-15",
        }
        validated = validate_inquiry(raw)
        self.assertEqual(validated["Chargeable_Weight_Kg"], 150.5)
        self.assertEqual(validated["Total_Buy_INR"], 30100.0)
        self.assertEqual(validated["quote_date"], "2026-06-15")

    def test_unit_buy_rate_resolution(self):
        """When unit buy is supplied without total buy, total buy is computed."""
        raw = {
            "chargeable_wt": 100.0,
            "unit_buy": 200.0,
            "dest": "London",
        }
        validated = validate_inquiry(raw)
        self.assertEqual(validated["Total_Buy_INR"], 20000.0)

    def test_missing_weight_raises_error(self):
        """Missing or null chargeable weight must raise QuoteValidationError."""
        raw = {"buy": 10000.0, "dest": "London"}
        with self.assertRaises(QuoteValidationError) as ctx:
            validate_inquiry(raw)
        self.assertIn("Chargeable weight is required", str(ctx.exception))

    def test_non_numeric_weight_raises_error(self):
        """Non-numeric chargeable weight must raise QuoteValidationError."""
        raw = {"chargeable_wt": "one hundred kg", "buy": 10000.0, "dest": "London"}
        with self.assertRaises(QuoteValidationError) as ctx:
            validate_inquiry(raw)
        self.assertIn("must be numeric", str(ctx.exception))

    def test_zero_or_negative_weight_raises_error(self):
        """Weight <= 0 must raise QuoteValidationError."""
        raw = {"chargeable_wt": 0.0, "buy": 10000.0, "dest": "London"}
        with self.assertRaises(QuoteValidationError) as ctx:
            validate_inquiry(raw)
        self.assertIn("outside acceptable range", str(ctx.exception))

    def test_missing_buy_and_unit_buy_raises_error(self):
        """Missing both buy and unit_buy must raise QuoteValidationError."""
        raw = {"chargeable_wt": 100.0, "dest": "London"}
        with self.assertRaises(QuoteValidationError) as ctx:
            validate_inquiry(raw)
        self.assertIn("Total buy or unit buy rate is required", str(ctx.exception))

    def test_implied_rate_out_of_bounds_raises_error(self):
        """Implied rate < MIN_RATE (10 INR/kg) or > MAX_RATE (8000 INR/kg) must raise QuoteValidationError."""
        # Too low: 500 INR for 100kg -> 5 INR/kg
        low_raw = {"chargeable_wt": 100.0, "buy": 500.0, "dest": "London"}
        with self.assertRaises(QuoteValidationError) as ctx:
            validate_inquiry(low_raw)
        self.assertIn("Implied buy rate", str(ctx.exception))

        # Too high: 900,000 INR for 100kg -> 9000 INR/kg
        high_raw = {"chargeable_wt": 100.0, "buy": 900000.0, "dest": "London"}
        with self.assertRaises(QuoteValidationError) as ctx:
            validate_inquiry(high_raw)
        self.assertIn("Implied buy rate", str(ctx.exception))

    def test_missing_destination_raises_error(self):
        """Missing destination must raise QuoteValidationError."""
        raw = {"chargeable_wt": 100.0, "buy": 15000.0}
        with self.assertRaises(QuoteValidationError) as ctx:
            validate_inquiry(raw)
        self.assertIn("Destination is required", str(ctx.exception))

    def test_unparseable_quote_date_raises_error(self):
        """Invalid date string must raise QuoteValidationError."""
        raw = {
            "chargeable_wt": 100.0,
            "buy": 15000.0,
            "dest": "London",
            "quote_date": "not-a-date"
        }
        with self.assertRaises(QuoteValidationError) as ctx:
            validate_inquiry(raw)
        self.assertIn("YYYY-MM-DD", str(ctx.exception))

    def test_quote_date_outside_window_raises_error(self):
        """Dates too far in the future (> 365 days) must raise QuoteValidationError."""
        far_future = (datetime.datetime.now(IST).date() + datetime.timedelta(days=400)).isoformat()
        raw = {
            "chargeable_wt": 100.0,
            "buy": 15000.0,
            "dest": "London",
            "quote_date": far_future
        }
        with self.assertRaises(QuoteValidationError) as ctx:
            validate_inquiry(raw)
        self.assertIn("outside accepted operational window", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
