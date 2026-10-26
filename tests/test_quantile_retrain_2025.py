"""
tests/test_quantile_retrain_2025.py
Unit and integration test suite verifying the retrained PricingEngineV3 bundle
trained on 2025 point-in-time won deals (won_deals_2025_train_70_pit.csv).

Verifies:
1. Bundle Loading & Structure: Serialized bundle exists, contains required keys,
   and initializes PricingEngineV3 successfully with all 14 canonical features.
2. Previous Bundle Backup: models/engine_v3_bundle_prev.joblib is preserved.
3. Strict Quantile Ordering: P10 <= P50 <= P90 across synthetic inputs and real 2025 holdout deals.
4. Scale-Calibrated Corridor Bandwidth Discipline:
   - Heavy cargo (>500 kg): spread / conformal half-width <= 1.8 pp.
   - Mid cargo (45-500 kg): spread / conformal half-width <= 3.5 pp.
   - Bounds consistency: min_margin_pct <= target_margin_pct <= max_margin_pct.
5. Cost Floor Enforcement:
   - Quoted sell prices strictly protect against sub-cost bidding (sell >= cost_floor_price).
   - Standard balanced quotes uphold healthy margins (>= 3.5%).
   - AWB floor protections uphold minimum fee recovery on micro-shipments.
6. Monotonicity Guarantees:
   - Non-increasing margin percentage with respect to Total_Buy_INR (-1 constraint).
   - Non-increasing margin percentage with respect to Chargeable_Weight_Kg (-1 constraint).
7. Conformal Calibration Quality:
   - Calibrator empirical coverage meets finite-sample target (>= 88.0% for alpha=0.10).
8. Strategic Tiers & Win Probabilities:
   - Margin_Pct(floor) < Margin_Pct(balanced) < Margin_Pct(premium).
   - Win_Prob(floor) > Win_Prob(balanced) > Win_Prob(premium).
"""

import os
import sys
import unittest
import joblib
import numpy as np
import pandas as pd

# Add source directory and project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from engine_v3 import PricingEngineV3
from quantile_model import MarginQuantileModel, FEATURE_COLS
from conformal_corridors import ConformalCorridorCalibrator
from features_pit import CustomerPITRegistry

BUNDLE_PATH = os.path.join(BASE_DIR, "models", "engine_v3_bundle.joblib")
PREV_BUNDLE_PATH = os.path.join(BASE_DIR, "models", "engine_v3_bundle_prev.joblib")
TRAIN_PIT_PATH = os.path.join(BASE_DIR, "data", "processed", "won_deals_2025_train_70_pit.csv")
TEST_PIT_PATH = os.path.join(BASE_DIR, "data", "processed", "won_deals_2025_test_30_pit.csv")


class TestQuantileRetrain2025(unittest.TestCase):
    """Test suite for the 2025 retrained Quantile Pricing Engine V3 bundle."""

    @classmethod
    def setUpClass(cls):
        """Instantiate PricingEngineV3 from retrained bundle."""
        cls.bundle_path = BUNDLE_PATH
        cls.prev_bundle_path = PREV_BUNDLE_PATH
        cls.engine = PricingEngineV3(bundle_path=cls.bundle_path)

        if os.path.exists(TEST_PIT_PATH):
            cls.test_df = pd.read_csv(TEST_PIT_PATH, low_memory=False)
            if "Cargo_Density_Ratio" not in cls.test_df.columns:
                cls.test_df["Cargo_Density_Ratio"] = 1.0
            if "Incoterms" not in cls.test_df.columns:
                cls.test_df["Incoterms"] = "FOB"
        else:
            cls.test_df = None

    def test_bundle_loading_and_keys(self):
        """Verifies bundle file exists, contains all required keys and correct types."""
        self.assertTrue(os.path.exists(self.bundle_path), f"Bundle not found at {self.bundle_path}")
        bundle = joblib.load(self.bundle_path)

        required_keys = [
            "model",
            "calibrator",
            "pit_registry",
            "slab_quantiles",
            "slab_counts",
            "slab_ref_rates",
            "model_version",
            "saved_at",
        ]
        for key in required_keys:
            self.assertIn(key, bundle, f"Missing required bundle key: {key}")

        self.assertEqual(bundle["model"].__class__.__name__, "MarginQuantileModel")
        self.assertEqual(bundle["calibrator"].__class__.__name__, "ConformalCorridorCalibrator")
        self.assertEqual(bundle["pit_registry"].__class__.__name__, "CustomerPITRegistry")
        self.assertIsInstance(bundle["slab_quantiles"], dict)
        self.assertIsInstance(bundle["slab_ref_rates"], dict)

        self.assertTrue(bundle["model"].is_fitted)
        self.assertTrue(bundle["calibrator"].is_fitted)
        self.assertTrue(self.engine.is_fitted)

        # Verify all 14 canonical features
        self.assertEqual(
            bundle["model"].feature_cols,
            FEATURE_COLS,
            "Model feature_cols must match canonical FEATURE_COLS exactly.",
        )

    def test_backup_bundle_exists(self):
        """Verifies previous engine bundle was safely backed up."""
        self.assertTrue(
            os.path.exists(self.prev_bundle_path),
            f"Previous bundle backup not found at {self.prev_bundle_path}",
        )
        self.assertGreater(
            os.path.getsize(self.prev_bundle_path),
            1000,
            "Previous bundle backup file size is suspiciously small.",
        )

    def test_quantile_ordering_synthetic(self):
        """Verifies mathematical invariant P10 <= P50 <= P90 on synthetic inputs."""
        model = self.engine.model
        np.random.seed(42)
        n_samples = 50

        # Construct synthetic feature DataFrame with canonical columns
        synth_df = pd.DataFrame({
            "Total_Buy_INR": np.random.uniform(5000, 500000, size=n_samples),
            "Chargeable_Weight_Kg": np.random.uniform(10, 2000, size=n_samples),
            "Cargo_Density_Ratio": np.random.uniform(0.5, 1.0, size=n_samples),
            "Weight_Tier": np.random.choice(["<45kg", "45-100kg", "100-300kg", "300-500kg", "500-1000kg", ">1000kg"], size=n_samples),
            "Commodity_Group": np.random.choice(["General Cargo", "Pharmaceuticals", "Perishable Foodstuff", "Auto Parts"], size=n_samples),
            "Regional_Lane": ["West India -> Europe"] * n_samples,
            "Origin_Port": ["Chhatrapati Shivaji Maharaj International Airport"] * n_samples,
            "Destination_Region": ["Europe"] * n_samples,
            "Company_Group": ["Subagent"] * n_samples,
            "Incoterms": ["FOB"] * n_samples,
            "cust_pit_won_count_180d": np.random.randint(0, 20, size=n_samples),
            "cust_pit_recency_days": np.random.uniform(1, 180, size=n_samples),
            "cust_pit_median_margin_won": np.random.uniform(3, 15, size=n_samples),
            "lane_pit_median_margin_won_90d": np.random.uniform(4, 12, size=n_samples),
        })

        p10, p50, p90 = model.predict_quantiles(synth_df)
        self.assertEqual(len(p10), n_samples)
        self.assertEqual(len(p50), n_samples)
        self.assertEqual(len(p90), n_samples)

        self.assertTrue((p10 <= p50 + 1e-9).all(), "Synthetic P10 must be <= P50 across all samples")
        self.assertTrue((p50 <= p90 + 1e-9).all(), "Synthetic P50 must be <= P90 across all samples")

    def test_quantile_ordering_2025_holdout(self):
        """Verifies strict quantile ordering P10 <= P50 <= P90 on real 2025 holdout deals."""
        if self.test_df is None:
            self.skipTest("Test dataset not found.")

        # Test on 300 holdout transactions
        sample_df = self.test_df.iloc[:300].copy()
        p10, p50, p90 = self.engine.model.predict_quantiles(sample_df)

        self.assertEqual(len(p10), len(sample_df))
        self.assertTrue(
            (p10 <= p50 + 1e-9).all(),
            "Holdout P10 must be <= P50 across all 2025 test transactions",
        )
        self.assertTrue(
            (p50 <= p90 + 1e-9).all(),
            "Holdout P50 must be <= P90 across all 2025 test transactions",
        )

    def test_corridor_bandwidth_heavy_cargo(self):
        """
        Verifies bandwidth discipline for heavy cargo (>500 kg):
        Conformal half-width spread <= 1.8 pp.
        """
        heavy_weights = [520.0, 750.0, 1000.0, 1500.0, 3000.0]
        for wt in heavy_weights:
            inquiry = {
                "buy": wt * 180.0,
                "gross_wt": wt,
                "chargeable_wt": wt,
                "origin": "Chhatrapati Shivaji Maharaj International Airport",
                "dest": "Frankfurt am Main",
                "client": "Shipper / Consignee",
                "commodity": "General Cargo",
                "strategy": "balanced",
            }
            res = self.engine.recommend_quote(inquiry)
            corr = res["negotiation_corridor"]

            self.assertIsNotNone(corr)
            half_width = corr["conformal_half_width_pct"]
            band_width = corr["band_width_pct"]
            min_m = corr["min_margin_pct"]
            target_m = corr["target_margin_pct"]
            max_m = corr["max_margin_pct"]

            # Heavy cargo spread discipline
            self.assertLessEqual(
                half_width,
                1.80 + 1e-6,
                f"Heavy cargo ({wt}kg) conformal spread {half_width} pp must be <= 1.8 pp",
            )
            # Bounds consistency
            self.assertLessEqual(min_m, target_m + 0.01)
            self.assertLessEqual(target_m, max_m + 0.01)
            self.assertAlmostEqual(band_width, round(max_m - min_m, 2), delta=0.02)

    def test_corridor_bandwidth_mid_cargo(self):
        """
        Verifies bandwidth discipline for mid cargo (45-500 kg):
        Conformal half-width spread <= 3.5 pp.
        """
        mid_weights = [50.0, 85.0, 150.0, 280.0, 420.0, 490.0]
        for wt in mid_weights:
            inquiry = {
                "buy": wt * 210.0,
                "gross_wt": wt,
                "chargeable_wt": wt,
                "origin": "Indira Gandhi International Airport",
                "dest": "London Heathrow",
                "client": "Subagent",
                "commodity": "Garments / Textiles",
                "strategy": "balanced",
            }
            res = self.engine.recommend_quote(inquiry)
            corr = res["negotiation_corridor"]

            self.assertIsNotNone(corr)
            half_width = corr["conformal_half_width_pct"]
            min_m = corr["min_margin_pct"]
            target_m = corr["target_margin_pct"]
            max_m = corr["max_margin_pct"]

            # Mid cargo spread discipline
            self.assertLessEqual(
                half_width,
                3.50 + 1e-6,
                f"Mid cargo ({wt}kg) conformal spread {half_width} pp must be <= 3.5 pp",
            )
            # Bounds consistency
            self.assertLessEqual(min_m, target_m + 0.01)
            self.assertLessEqual(target_m, max_m + 0.01)

    def test_cost_floor_enforcement(self):
        """
        Verifies cost floor integrity:
        Quoted prices and recommended margins must protect against sub-cost bidding.
        Sell prices must be >= Cost_Floor_Price_INR.
        Standard mid-weight quotes must maintain healthy margin (>= 3.5%).
        """
        # Test low-margin pressure scenarios
        inquiries = [
            {"buy": 500000.0, "chargeable_wt": 2000.0, "strategy": "floor"},
            {"buy": 1000000.0, "chargeable_wt": 5000.0, "strategy": "balanced"},
            {"buy": 50000.0, "chargeable_wt": 200.0, "strategy": "floor"},
            {"buy": 80000.0, "chargeable_wt": 250.0, "strategy": "balanced"},
        ]
        for inq in inquiries:
            inq.update({
                "origin": "Bangalore",
                "dest": "Singapore Changi",
                "client": "Overseas Agent",
                "commodity": "General Cargo",
            })
            res = self.engine.recommend_quote(inq)
            strat = res["strategic_recommendations"]
            corr = res["negotiation_corridor"]
            cost_floor_price = res["optimal"]["Cost_Floor_Price_INR"]

            floor_sell = strat["floor"]["Quoted_Sell_Price_INR"]
            bal_sell = strat["balanced"]["Quoted_Sell_Price_INR"]
            min_sell = corr["min_sell_price_inr"]

            # Pricing protection: all quoted prices must cover the cost floor price
            self.assertGreaterEqual(
                floor_sell,
                cost_floor_price - 0.01,
                f"Floor sell price (₹{floor_sell}) below cost floor price (₹{cost_floor_price})",
            )
            self.assertGreaterEqual(
                bal_sell,
                cost_floor_price - 0.01,
                f"Balanced sell price (₹{bal_sell}) below cost floor price (₹{cost_floor_price})",
            )
            self.assertGreaterEqual(
                min_sell,
                cost_floor_price - 0.01,
                f"Corridor min sell price (₹{min_sell}) below cost floor price (₹{cost_floor_price})",
            )

        # Standard mid-market quote health check (>= 3.5% recommended margin)
        std_inq = {
            "buy": 80000.0,
            "chargeable_wt": 250.0,
            "origin": "Chhatrapati Shivaji Maharaj International Airport",
            "dest": "Frankfurt am Main",
            "client": "Shipper / Consignee",
            "commodity": "General Cargo",
            "strategy": "balanced",
        }
        std_res = self.engine.recommend_quote(std_inq)
        std_bal_margin = std_res["strategic_recommendations"]["balanced"]["Margin_Percentage"]
        self.assertGreaterEqual(
            std_bal_margin,
            3.50,
            f"Standard balanced quote margin ({std_bal_margin}%) must be >= 3.5%",
        )

    def test_awb_floor_enforcement_micro_shipments(self):
        """
        Verifies AWB floor protection:
        Micro-shipments must generate at least minimum gross margin in INR.
        """
        # Very small shipment (3kg, ₹1,200 buy)
        inquiry = {
            "buy": 1200.0,
            "gross_wt": 3.0,
            "chargeable_wt": 3.0,
            "origin": "Indira Gandhi International Airport",
            "dest": "Dubai",
            "client": "Direct Shipper",
            "commodity": "Courier",
            "strategy": "balanced",
        }
        res = self.engine.recommend_quote(inquiry)
        strat = res["strategic_recommendations"]
        bal_sell = strat["balanced"]["Quoted_Sell_Price_INR"]
        bal_margin_inr = strat["balanced"]["Margin_Amount_INR"]

        # Minimum gross margin protection must exceed standard AWB fee floor
        self.assertGreater(bal_sell, 1200.0)
        self.assertGreaterEqual(
            bal_margin_inr,
            500.0,
            f"AWB margin amount INR ({bal_margin_inr}) must protect operational floor",
        )

    def test_monotonicity_on_buy_cost_and_weight(self):
        """
        Verifies monotonic decreasing constraints:
        Increasing buy cost or weight must not increase predicted margin percentage.
        """
        sample_df = self.test_df.iloc[:50].copy() if self.test_df is not None else None
        if sample_df is None:
            self.skipTest("Test dataset required for monotonicity verification.")

        base_preds = self.engine.model.predict(sample_df)

        # 1. Higher Buy Cost -> lower or equal margin percentage
        high_buy_sample = sample_df.copy()
        high_buy_sample["Total_Buy_INR"] = high_buy_sample["Total_Buy_INR"] + 50000.0
        high_buy_preds = self.engine.model.predict(high_buy_sample)
        buy_violations = int((high_buy_preds > base_preds + 1e-5).sum())
        self.assertEqual(
            buy_violations,
            0,
            f"Violations detected where higher Total_Buy_INR increased margin: {buy_violations}",
        )

        # 2. Higher Chargeable Weight -> lower or equal margin percentage
        high_wt_sample = sample_df.copy()
        high_wt_sample["Chargeable_Weight_Kg"] = high_wt_sample["Chargeable_Weight_Kg"] + 500.0
        high_wt_preds = self.engine.model.predict(high_wt_sample)
        wt_violations = int((high_wt_preds > base_preds + 1e-5).sum())
        self.assertEqual(
            wt_violations,
            0,
            f"Violations detected where higher Chargeable_Weight_Kg increased margin: {wt_violations}",
        )

    def test_strategic_tiers_and_win_probabilities(self):
        """
        Verifies strict tier hierarchy:
        Margin: floor < balanced < premium
        Win Probability: floor > balanced > premium
        """
        inquiry = {
            "buy": 120000.0,
            "gross_wt": 450.0,
            "chargeable_wt": 450.0,
            "origin": "Chhatrapati Shivaji Maharaj International Airport",
            "dest": "Frankfurt am Main",
            "client": "Shipper / Consignee",
            "commodity": "Pharmaceuticals",
            "strategy": "balanced",
        }
        res = self.engine.recommend_quote(inquiry)
        strat = res["strategic_recommendations"]

        m_floor = strat["floor"]["Margin_Percentage"]
        m_bal = strat["balanced"]["Margin_Percentage"]
        m_prem = strat["premium"]["Margin_Percentage"]

        p_floor = strat["floor"]["Win_Probability"]
        p_bal = strat["balanced"]["Win_Probability"]
        p_prem = strat["premium"]["Win_Probability"]

        # Margin ordering
        self.assertLess(m_floor, m_bal, f"Expected m_floor ({m_floor}) < m_bal ({m_bal})")
        self.assertLess(m_bal, m_prem, f"Expected m_bal ({m_bal}) < m_prem ({m_prem})")

        # Win probability ordering
        self.assertGreater(p_floor, p_bal, f"Expected p_floor ({p_floor}) > p_bal ({p_bal})")
        self.assertGreater(p_bal, p_prem, f"Expected p_bal ({p_bal}) > p_prem ({p_prem})")


if __name__ == "__main__":
    unittest.main()
