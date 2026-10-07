"""
tests/test_oof_benchmark.py
Unit tests verifying out-of-fold benchmark ratios, zero in-sample leakage,
and production model artifact metadata.
"""

import unittest
import os
import joblib
import numpy as np
import pandas as pd
from src.constants import MODEL_VERSION

FORBIDDEN_PROCESS_COLS = {
    "Quote_Turnaround_Hours", "quote_revision_count", "Total_Sell_INR",
    "Booked by branch", "Customer_Inquiry_Frequency", "Customer_Tier"
}


class TestOOFBenchmarkAndArtifacts(unittest.TestCase):
    def test_no_process_or_identifier_features(self):
        from src.train import FEATURE_COLS
        self.assertFalse(FORBIDDEN_PROCESS_COLS & set(FEATURE_COLS))

    def test_auc_in_plausible_band(self):
        model_path = "models/freight_margin_recommender.joblib"
        if not os.path.exists(model_path):
            self.skipTest(f"Model artifact {model_path} not found.")
        artifact = joblib.load(model_path)
        auc = artifact["stage1b_metrics"]["test_auc"]
        self.assertGreaterEqual(auc, 0.60)
        self.assertLessEqual(auc, 0.92)  # >0.92 means investigate leakage

    def test_model_artifact_structure_and_metadata(self):
        model_path = "models/freight_margin_recommender.joblib"
        if not os.path.exists(model_path):
            self.skipTest(f"Model artifact {model_path} not yet generated; skipping until training run completes.")

        artifact = joblib.load(model_path)
        required_keys = [
            "benchmark_reg", "calibrated_clf", "encoder",
            "feature_cols", "native_cat_cols", "lane_medians",
            "global_median", "port_frequencies", "model_version",
            "stage1a_metrics", "stage1b_metrics", "margin_tables"
        ]
        for key in required_keys:
            self.assertIn(key, artifact, f"Missing key '{key}' in serialized model artifact")

        self.assertEqual(artifact["model_version"], MODEL_VERSION)
        self.assertLessEqual(artifact["stage1b_metrics"]["test_brier"], 0.16)


if __name__ == "__main__":
    unittest.main()
