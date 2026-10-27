"""
tests/test_evaluation_2025_test_set.py
Unit and integration test suite for the 2025 Out-of-Time Holdout Evaluation Harness.

Verifies:
1. End-to-end execution of evaluation suite with zero exceptions.
2. Empirical Conformal Coverage meets or exceeds statutory threshold (>= 88.0%).
3. Heavy Cargo Bandwidth Discipline (>500 kg): median commercial spread <= 1.80 pp.
4. Mid Cargo Bandwidth Discipline (45-500 kg): median commercial spread <= 3.50 pp.
5. Courier/Small Cargo (<45 kg): strictly capped <= 4.00 pp.
6. Quantile Invariance: P10 <= P50 <= P90 across 100% of quotes.
7. Sub-cost bidding protection: 0 sell quotes below cost floor price.
8. Sub-group partitions consistency (weight tiers, commodities, customer types).
9. Markdown report generation and artifact integrity.
"""

import os
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

# Add source directory and project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine_v3 import PricingEngineV3
from scripts.evaluate_2025_test_set import (
    pinball_loss,
    evaluate_quantiles,
    compute_evaluation_suite,
    generate_markdown_report,
    run_evaluation,
    DEFAULT_TEST_PATH,
    DEFAULT_BUNDLE_PATH,
)


class TestEvaluation2025TestSet(unittest.TestCase):
    """Test suite for the out-of-time holdout benchmark evaluation harness."""

    @classmethod
    def setUpClass(cls):
        """Loads PricingEngineV3 and test dataset once for the test suite."""
        cls.test_path = DEFAULT_TEST_PATH
        cls.bundle_path = DEFAULT_BUNDLE_PATH

        if not os.path.exists(cls.test_path) or not os.path.exists(cls.bundle_path):
            raise FileNotFoundError(
                f"Required test fixtures not found: test={cls.test_path}, bundle={cls.bundle_path}"
            )

        cls.df_test = pd.read_csv(cls.test_path, low_memory=False)
        cls.engine = PricingEngineV3(bundle_path=cls.bundle_path)
        cls.results = compute_evaluation_suite(cls.engine, cls.df_test)

    def test_pinball_loss_unit(self):
        """Verifies asymmetric pinball loss mathematical behavior."""
        y_true = np.array([10.0, 10.0])
        # Overpredict by 2.0 pp
        y_over = np.array([12.0, 12.0])
        # Underpredict by 2.0 pp
        y_under = np.array([8.0, 8.0])

        # For q = 0.50 (symmetric MAE / 2)
        self.assertAlmostEqual(pinball_loss(y_true, y_over, 0.50), 1.0)
        self.assertAlmostEqual(pinball_loss(y_true, y_under, 0.50), 1.0)

        # For q = 0.10: underprediction (err = +2) penalized with 0.10 * 2 = 0.20
        # Overprediction (err = -2) penalized with (0.10 - 1.0) * (-2) = 1.80
        self.assertAlmostEqual(pinball_loss(y_true, y_under, 0.10), 0.20)
        self.assertAlmostEqual(pinball_loss(y_true, y_over, 0.10), 1.80)

        # For q = 0.90: underprediction penalized with 0.90 * 2 = 1.80
        # Overprediction penalized with (0.90 - 1.0) * (-2) = 0.20
        self.assertAlmostEqual(pinball_loss(y_true, y_under, 0.90), 1.80)
        self.assertAlmostEqual(pinball_loss(y_true, y_over, 0.90), 0.20)

    def test_evaluate_quantiles_structure(self):
        """Verifies quantile evaluation dictionary structure and metrics."""
        y_true = np.array([5.0, 10.0, 15.0, 20.0])
        p10 = np.array([3.0, 8.0, 12.0, 16.0])
        p50 = np.array([5.0, 10.0, 15.0, 20.0])
        p90 = np.array([8.0, 13.0, 18.0, 25.0])

        res = evaluate_quantiles(y_true, p10, p50, p90)
        self.assertTrue(res["ordering_valid"])
        self.assertEqual(res["mae"], 0.0)
        self.assertEqual(res["bias"], 0.0)
        self.assertGreater(res["pinball_p10"], 0.0)
        self.assertGreater(res["pinball_p90"], 0.0)

    def test_empirical_conformal_coverage_gate(self):
        """
        Acceptance Gate 1: Empirical Conformal Coverage must be >= 88.0% (target 90%).
        """
        cov_rate = self.results["conformal_coverage_rate"]
        self.assertGreaterEqual(
            cov_rate,
            0.880,
            f"Empirical conformal coverage ({cov_rate * 100:.2f}%) fell below statutory threshold (88.0%)",
        )

    def test_heavy_cargo_bandwidth_discipline_gate(self):
        """
        Acceptance Gate 2: Heavy Cargo (>500 kg) commercial half-width median must be <= 1.80 pp.
        """
        median_hw = self.results["heavy_hw_median"]
        mean_hw = self.results["heavy_hw_mean"]
        self.assertLessEqual(
            median_hw,
            1.80 + 1e-6,
            f"Heavy cargo median commercial spread ({median_hw:.2f} pp) exceeded 1.80 pp",
        )
        self.assertLessEqual(
            mean_hw,
            1.80 + 1e-6,
            f"Heavy cargo mean commercial spread ({mean_hw:.2f} pp) exceeded 1.80 pp",
        )

    def test_mid_cargo_bandwidth_discipline_gate(self):
        """
        Acceptance Gate 3: Mid Cargo (45-500 kg) commercial half-width median must be <= 3.50 pp.
        """
        median_hw = self.results["mid_hw_median"]
        mean_hw = self.results["mid_hw_mean"]
        self.assertLessEqual(
            median_hw,
            3.50 + 1e-6,
            f"Mid cargo median commercial spread ({median_hw:.2f} pp) exceeded 3.50 pp",
        )
        self.assertLessEqual(
            mean_hw,
            3.50 + 1e-6,
            f"Mid cargo mean commercial spread ({mean_hw:.2f} pp) exceeded 3.50 pp",
        )

    def test_courier_cargo_bandwidth_cap_gate(self):
        """
        Acceptance Gate 4: Courier / Small Parcel (<45 kg) commercial half-width must be capped <= 4.00 pp.
        """
        max_hw = self.results["courier_hw_max"]
        self.assertLessEqual(
            max_hw,
            4.00 + 1e-6,
            f"Courier maximum commercial spread ({max_hw:.2f} pp) breached commercial cap (4.00 pp)",
        )

    def test_mathematical_quantile_invariance_gate(self):
        """
        Acceptance Gate 5: P10 <= P50 <= P90 must strictly hold for 100% of quotes.
        """
        q = self.results["quantile_metrics"]
        self.assertTrue(
            q["ordering_valid"],
            "Quantile crossing detected in holdout predictions: P10 <= P50 <= P90 violated",
        )

    def test_sub_cost_bidding_protection_gate(self):
        """
        Acceptance Gate 6: 0 quotes may breach the commercial cost floor price.
        """
        violations = self.results["cost_floor_violations"]
        self.assertEqual(
            violations,
            0,
            f"Detected {violations} quotes violating the commercial cost floor",
        )

    def test_all_acceptance_gates_pass(self):
        """Verifies summary flag confirms all gates passed."""
        self.assertTrue(self.results["all_gates_pass"])
        for gate_key, gate in self.results["gate_results"].items():
            self.assertEqual(gate["status"], "PASS", f"Gate {gate_key} did not PASS")

    def test_subgroup_partitions_consistency(self):
        """Verifies sample count integrity across sub-group breakdowns."""
        total_n = self.results["n_samples"]
        self.assertEqual(total_n, 4078)

        # Weight slab counts sum to total_n
        slab_n_sum = sum(s["n"] for s in self.results["slab_stats"])
        self.assertEqual(slab_n_sum, total_n, "Weight tier count sum mismatch")

        # Commodity counts sum to total_n
        comm_n_sum = sum(c["n"] for c in self.results["comm_stats"])
        self.assertEqual(comm_n_sum, total_n, "Commodity group count sum mismatch")

        # Customer status counts sum to total_n
        cust_n_sum = sum(cu["n"] for cu in self.results["cust_stats"])
        self.assertEqual(cust_n_sum, total_n, "Customer status count sum mismatch")

    def test_report_generation_and_file_creation(self):
        """Verifies report generation produces complete markdown document."""
        report = generate_markdown_report(self.results)
        self.assertIsInstance(report, str)
        self.assertGreater(len(report), 2000)

        required_strings = [
            "# Out-of-Time 30% Holdout Benchmark Evaluation Report",
            "ALL ACCEPTANCE GATES PASSED",
            "Empirical Conformal Coverage",
            "Heavy Cargo Bandwidth Discipline",
            "Commercial Profit & Economic Capture Simulation",
            "Total Airline Buy Cost",
        ]
        for s in required_strings:
            self.assertIn(s, report, f"Missing required string in markdown report: {s}")

    def test_run_evaluation_cli_orchestration(self):
        """Verifies run_evaluation executes successfully with temporary output files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_report = os.path.join(tmpdir, "test_eval_report.md")
            art_report = os.path.join(tmpdir, "test_eval_art.md")

            res = run_evaluation(
                test_path=self.test_path,
                bundle_path=self.bundle_path,
                output_report_path=out_report,
                artifact_paths=[art_report],
            )
            self.assertTrue(res["all_gates_pass"])
            self.assertTrue(os.path.exists(out_report))
            self.assertTrue(os.path.exists(art_report))
            self.assertGreater(os.path.getsize(out_report), 1000)
            self.assertGreater(os.path.getsize(art_report), 1000)


if __name__ == "__main__":
    unittest.main()
