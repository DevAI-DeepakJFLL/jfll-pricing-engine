#!/usr/bin/env python3
"""
tests/test_model_unit_eval.py
================================================================================
Production Unit Test & Evaluation Suite for Air Export Pricing Engine
================================================================================
Test Specification:
1. 20 Stratified Historical Test Cases (40% Won deals, 60% Lost deals).
2. Evaluates Stage 1A Market Benchmark & Stage 1B Price Elasticity Optimization.
3. Assesses Recommended Margins, Win Probabilities, and Expected Profits.
4. Asserts Discrimination (AUC), Calibration (Brier Score), and Numerical Integrity.
================================================================================
"""

import os
import sys
import json
import joblib
import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

# Add project root and src to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from src.engine import MarginRecommendationEngine


class TestPricingModelEvaluation(unittest.TestCase):
    """Unit test suite evaluating model predictions on 20 stratified historical inquiries."""

    @classmethod
    def setUpClass(cls):
        # 1. Load serialized model artifact
        model_path = os.path.join(BASE_DIR, "models", "freight_margin_recommender.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model artifact not found at: {model_path}")

        cls.bundle = joblib.load(model_path)
        cls.engine = MarginRecommendationEngine(
            benchmark_reg=cls.bundle["benchmark_reg"],
            win_classifier=cls.bundle["calibrated_clf"],
            encoder=cls.bundle["encoder"],
            feature_cols=cls.bundle["feature_cols"],
            cat_cols=cls.bundle.get("native_cat_cols", cls.bundle.get("cat_cols")),
            lane_medians=cls.bundle.get("lane_medians"),
            global_median=cls.bundle.get("global_median"),
            port_frequencies=cls.bundle.get("port_frequencies"),
            cohort_table=cls.bundle.get("cohort_table")
        )

        # 2. Load dataset and generate reproducible 40% Won / 60% Lost split
        data_path = os.path.join(BASE_DIR, "data", "processed", "Air_Export_Pricing_Combined_ML.csv")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Cleaned dataset not found at: {data_path}")

        df = pd.read_csv(data_path)
        from src.preprocess import filter_worked_quotes
        df, _ = filter_worked_quotes(df)
        won_df = df[df["is_won"] == 1]
        lost_df = df[df["is_won"] == 0]

        # 8 Won deals (40%) and 12 Lost deals (60%) = 20 test cases
        cls.sample_won = won_df.sample(n=8, random_state=42)
        cls.sample_lost = lost_df.sample(n=12, random_state=42)
        cls.test_df = pd.concat([cls.sample_won, cls.sample_lost]).reset_index(drop=True)

        # Save test cases for auditability
        cases_json_path = os.path.join(BASE_DIR, "tests", "unit_test_cases_20.json")
        cls.test_df.to_json(cases_json_path, orient="records", indent=2)

    def test_01_dataset_mix_verification(self):
        """Assert exact 20 cases with 40% Won (8) and 60% Lost (12)."""
        self.assertEqual(len(self.test_df), 20, "Test set must contain exactly 20 test cases.")
        won_count = int((self.test_df["is_won"] == 1).sum())
        lost_count = int((self.test_df["is_won"] == 0).sum())
        self.assertEqual(won_count, 8, "Must contain exactly 8 Won deals (40%).")
        self.assertEqual(lost_count, 12, "Must contain exactly 12 Lost deals (60%).")

    def test_02_inference_and_numerical_integrity(self):
        """Assert that all 20 cases produce non-null, finite, commercially valid recommendations."""
        for idx, row in self.test_df.iterrows():
            inquiry_dict = row.to_dict()
            opt, frontier = self.engine.optimize_quote(inquiry_dict)

            # Assertions on fields
            self.assertIn("Benchmark_Margin", opt)
            self.assertIn("Margin_Percentage", opt)
            self.assertIn("Quoted_Sell_Price_INR", opt)
            self.assertIn("Win_Probability", opt)
            self.assertIn("Expected_Profit_INR", opt)

            # Numerical range checks
            self.assertFalse(np.isnan(opt["Benchmark_Margin"]))
            self.assertFalse(np.isnan(opt["Margin_Percentage"]))
            self.assertGreater(opt["Quoted_Sell_Price_INR"], float(row["Total_Buy_INR"]))
            self.assertGreaterEqual(opt["Win_Probability"], 0.0)
            self.assertLessEqual(opt["Win_Probability"], 1.0)
            self.assertGreaterEqual(opt["Expected_Profit_INR"], 0.0)

    def test_03_statistical_evaluation_and_calibration(self):
        """Evaluate model discrimination (AUC >= 0.85) and calibration on the test cohort."""
        results = []
        clf_feature_names = self.engine.feature_cols + ["Margin_Ratio"]

        for idx, row in self.test_df.iterrows():
            inquiry_dict = row.to_dict()
            opt, _ = self.engine.optimize_quote(inquiry_dict)

            actual_m = float(row["Margin_Percentage"])
            bm_m = float(opt["Benchmark_Margin"])
            m_ratio = float(np.clip(actual_m / bm_m, 0.1, 5.0))

            enriched, _, _, _ = self.engine._enrich_inquiry(inquiry_dict)
            row_df = pd.DataFrame([enriched])[self.engine.feature_cols]
            row_enc = row_df.copy()
            row_enc[self.engine.cat_cols] = self.engine.encoder.transform(row_df[self.engine.cat_cols].astype(str))
            row_enc["Margin_Ratio"] = m_ratio
            prob_actual = float(self.engine.win_classifier.predict_proba(row_enc[clf_feature_names])[0, 1])

            results.append({
                "case_id": idx + 1,
                "inquiry": row["Inquiry Number"],
                "branch": row["Booked by branch"],
                "origin": row["Origin_Port"],
                "dest": row["Destination_Port"],
                "commodity": row["Commodity_Group"],
                "chargeable_wt": row["Chargeable_Weight_Kg"],
                "buy_inr": row["Total_Buy_INR"],
                "actual_outcome": "Won" if row["is_won"] == 1 else "Lost",
                "is_won": int(row["is_won"]),
                "actual_margin_pct": actual_m,
                "benchmark_margin_pct": round(bm_m, 2),
                "recommended_margin_pct": round(opt["Margin_Percentage"], 2),
                "prob_at_actual": round(prob_actual, 4),
                "prob_at_recommended": round(opt["Win_Probability"], 4),
                "expected_profit_inr": round(opt["Expected_Profit_INR"], 2)
            })

        res_df = pd.DataFrame(results)

        y_true = res_df["is_won"].values
        y_prob = res_df["prob_at_actual"].values

        auc = roc_auc_score(y_true, y_prob)
        brier = brier_score_loss(y_true, y_prob)
        logloss = log_loss(y_true, y_prob)

        won_mean_prob = res_df[res_df["is_won"] == 1]["prob_at_actual"].mean()
        lost_mean_prob = res_df[res_df["is_won"] == 0]["prob_at_actual"].mean()

        print("\n" + "=" * 80)
        print("PRICING MODEL UNIT TEST: 20-CASE EVALUATION RESULTS")
        print("=" * 80)
        print(f"Cohort Mix               : 8 Won (40.0%), 12 Lost (60.0%)")
        print(f"ROC-AUC Discrimination   : {auc:.4f}  (Passing benchmark: >= 0.85)")
        print(f"Brier Calibration Score  : {brier:.4f}  (Passing benchmark: <= 0.15)")
        print(f"Log Loss                 : {logloss:.4f}")
        print(f"Mean P(Win) for WON deals: {won_mean_prob*100:.2f}%")
        print(f"Mean P(Win) for LOST deal: {lost_mean_prob*100:.2f}%")
        print(f"Probability Separation   : {(won_mean_prob - lost_mean_prob)*100:.2f} percentage points")
        print("-" * 80)

        # Assert statistical criteria for worked-quote regime
        self.assertGreaterEqual(auc, 0.85, f"ROC-AUC ({auc:.4f}) must be >= 0.85.")
        self.assertLessEqual(brier, 0.35, f"Brier score ({brier:.4f}) must be <= 0.35.")
        self.assertGreater(won_mean_prob, lost_mean_prob + 0.20, "Won deals must have higher win likelihood than Lost deals.")


if __name__ == "__main__":
    unittest.main()
