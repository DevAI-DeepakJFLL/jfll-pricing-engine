"""
tests/test_training_evaluation_report.py
Unit and integration tests for Training Evaluation Report Generation.
Verifies markdown generation, required columns, presence of all 5 commodities
and 5 budget tiers, exact 25-row matrix table, and file output.
"""

import os
import sys
import unittest
import tempfile
import re

# Project root setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.generate_training_evaluation_report import (
    generate_report,
    DEFAULT_INPUT_PATH,
    COMMODITY_ORDER,
    BUDGET_ORDER,
)

RESULTS_CSV_PATH = os.path.join(PROJECT_ROOT, "tests", "data", "training_eval_results_500.csv")

EXPECTED_COMMODITIES = [
    "General Cargo",
    "Perishable Foodstuff",
    "Garments / Textiles",
    "Engineering & Machinery",
    "Other",
]

EXPECTED_BUDGET_TIERS = [
    "Very Low Budget",
    "Low Budget",
    "Medium Budget",
    "High Budget",
    "Very High Budget",
]

REQUIRED_HEADERS = [
    "Commodity Type",
    "Budget Segment",
    "Actual Quoted Margin",
    "Model-Quoted Margin",
    "Margin Gap",
    "Accuracy (MAE)",
    "RMSE",
    "Corridor Coverage",
]


class TestTrainingEvaluationReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a temporary directory for output files
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.output_path = os.path.join(cls.temp_dir.name, "training_eval_report.md")
        cls.artifact_path = os.path.join(cls.temp_dir.name, "IN_SAMPLE_MODEL_EVALUATION_REPORT.md")
        
        # Generate the report once for assertions
        cls.report_content = generate_report(
            results_csv_path=RESULTS_CSV_PATH,
            output_path=cls.output_path,
            artifact_path=cls.artifact_path,
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_report_generation_creates_non_empty_file(self):
        """Verifies report generation creates non-empty files at target paths."""
        self.assertTrue(os.path.exists(self.output_path), "Output report file was not created.")
        self.assertGreater(os.path.getsize(self.output_path), 0, "Output report file is empty.")
        self.assertTrue(os.path.exists(self.artifact_path), "Artifact report file was not created.")
        self.assertGreater(os.path.getsize(self.artifact_path), 0, "Artifact report file is empty.")
        self.assertIsInstance(self.report_content, str)
        self.assertGreater(len(self.report_content), 500, "Report markdown string is unexpectedly short.")

    def test_all_five_commodities_present(self):
        """Verifies all 5 commodities are present in the report."""
        for commodity in EXPECTED_COMMODITIES:
            self.assertIn(
                commodity,
                self.report_content,
                f"Missing expected commodity: {commodity} in report content."
            )

    def test_all_five_budget_segments_present(self):
        """Verifies all 5 budget segments are present in the report."""
        for budget in EXPECTED_BUDGET_TIERS:
            self.assertIn(
                budget,
                self.report_content,
                f"Missing expected budget segment: {budget} in report content."
            )

    def test_required_table_headers_present(self):
        """Verifies all required column headers appear in the matrix table."""
        for header in REQUIRED_HEADERS:
            self.assertIn(
                header,
                self.report_content,
                f"Required column header '{header}' missing from report."
            )

    def test_exact_25_matrix_rows_present(self):
        """Verifies the 25-cell matrix table contains exactly 25 data rows."""
        # Locate the 25-cell evaluation matrix section
        matrix_match = re.search(
            r"## (?:\d+\.\s*)?(?:Complete 25-Cell Evaluation Matrix|25-Cell Evaluation Matrix).*?\n\s*\|([^\n]+)\|\n\s*\|([^\n]+)\|\n(.*?)(?=\n## |\Z)",
            self.report_content,
            re.DOTALL,
        )
        self.assertIsNotNone(matrix_match, "Could not find 25-cell evaluation matrix table in report.")
        
        table_body = matrix_match.group(3).strip()
        rows = [line.strip() for line in table_body.splitlines() if line.strip().startswith("|")]
        
        self.assertEqual(
            len(rows),
            25,
            f"Expected exactly 25 data rows in matrix table, but found {len(rows)}."
        )

        # Verify each of the 25 (Commodity, Budget) combinations appears in the table rows
        for comm in EXPECTED_COMMODITIES:
            for budget in EXPECTED_BUDGET_TIERS:
                cell_found = any(comm in row and budget in row for row in rows)
                self.assertTrue(
                    cell_found,
                    f"Cell combination ({comm}, {budget}) missing from matrix table rows."
                )

    def test_report_sections_present(self):
        """Verifies all required high-level report sections are present."""
        expected_sections = [
            "Executive Summary",
            "Performance Roll-Up by Commodity Type",
            "Performance Roll-Up by Budget Segment",
            "25-Cell Evaluation Matrix",
            "Operational Insights & Analysis",
        ]
        for section in expected_sections:
            self.assertTrue(
                any(section.lower() in line.lower() for line in self.report_content.splitlines()),
                f"Expected section header '{section}' not found in report.",
            )

    def test_operational_insights_cover_key_mechanisms(self):
        """Verifies operational insights cover AWB cost floors, commodity caps, and margin elasticity."""
        content_lower = self.report_content.lower()
        self.assertIn("floor", content_lower)
        self.assertIn("cap", content_lower)
        self.assertIn("elasticity", content_lower)


if __name__ == "__main__":
    unittest.main()
