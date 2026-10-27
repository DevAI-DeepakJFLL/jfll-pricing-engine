#!/usr/bin/env python3
"""
scripts/train_win_model.py
Trains and saves the standalone calibrated monotonic WinModel artifact.
"""

import os
import sys
import pandas as pd

# Add repo root to path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from win_model import WinModel

DATA_PATH = os.path.join(REPO_ROOT, "data", "processed", "all_inquiries_win_rate_v3.csv")
OUTPUT_PATH = os.path.join(REPO_ROOT, "models", "win_model.joblib")


def main():
    if not os.path.exists(DATA_PATH):
        print(f"Error: dataset not found at {DATA_PATH}")
        sys.exit(1)

    print(f"Loading win inquiries from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} inquiries. Distribution:")
    print(df["is_won"].value_counts(normalize=True))

    print("Fitting WinModel with HistGradientBoostingClassifier (monotonic_cst=[-1]) & Isotonic calibration...")
    model = WinModel(random_state=42)
    model.fit(df)

    print(f"Training completed successfully.")
    print("Holdout Evaluation Metrics:")
    for k, v in model.metrics.items():
        print(f"  - {k}: {v}")

    model.save(OUTPUT_PATH)
    print(f"WinModel saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
