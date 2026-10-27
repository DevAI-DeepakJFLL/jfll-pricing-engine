"""
tests/test_win_model.py
Unit tests for the calibrated monotonic WinModel.
"""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest
from src.win_model import WinModel


@pytest.fixture
def sample_win_df():
    np.random.seed(42)
    n = 300
    months = ["2026-05"] * 100 + ["2026-06"] * 100 + ["2026-07"] * 100
    margins = np.random.uniform(2.0, 30.0, n)
    # Win probability strictly decreases with margin
    probs = 1.0 / (1.0 + np.exp((margins - 12.0) / 4.0))
    is_won = (np.random.rand(n) < probs).astype(int)

    df = pd.DataFrame({
        "Year_Month": months,
        "Margin_Percentage": margins,
        "is_won": is_won,
        "Chargeable_Weight_Kg": np.random.uniform(50.0, 500.0, n),
        "Total_Buy_INR": np.random.uniform(10000.0, 80000.0, n),
        "Unit_Buy_INR": np.random.uniform(100.0, 300.0, n),
        "Origin_Region": np.random.choice(["North India", "West India", "South India"], n),
        "Destination_Region": np.random.choice(["Europe", "North America", "Middle East"], n),
        "Commodity_Group": np.random.choice(["General Cargo", "Perishables", "Pharma"], n),
        "Company_Group": np.random.choice(["Direct Shipper", "Freight Forwarder", "Subagent"], n),
        "Weight_Tier": np.random.choice(["100-300kg", "300-500kg"], n),
        "cust_pit_won_count_180d": np.random.randint(0, 10, n),
        "cust_pit_median_margin_won": np.random.uniform(5.0, 15.0, n),
        "cust_pit_recency_days": np.random.uniform(5.0, 90.0, n),
        "lane_pit_median_margin_won_90d": np.random.uniform(6.0, 14.0, n),
        "Regional_Lane": ["North India -> Europe"] * n,
        "Incoterms": ["FOB"] * n,
        "Cargo_Density_Ratio": [1.0] * n,
    })
    return df


def test_win_model_fit_and_monotonicity(sample_win_df):
    model = WinModel(random_state=42)
    model.fit(sample_win_df)
    assert model.is_fitted
    assert "auc" in model.metrics

    feat_row = sample_win_df.iloc[[0]].copy()
    candidate_margins = [3.0, 6.0, 10.0, 15.0, 20.0, 30.0]
    probs = model.predict_margins(feat_row, candidate_margins)

    assert len(probs) == len(candidate_margins)
    # Monotonicity check: as margin increases, win probability must not increase
    for i in range(len(probs) - 1):
        assert probs[i] >= probs[i + 1] - 1e-6, f"Monotonicity violated: {probs[i]} < {probs[i+1]} at margin {candidate_margins[i]} vs {candidate_margins[i+1]}"


def test_win_model_save_and_load(sample_win_df):
    model = WinModel(random_state=42)
    model.fit(sample_win_df)

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        tmp_path = f.name

    try:
        model.save(tmp_path)
        loaded = WinModel().load(tmp_path)
        assert loaded.is_fitted
        assert loaded.metrics == model.metrics

        feat_row = sample_win_df.iloc[[0]].copy()
        orig_p = model.predict_margins(feat_row, [5.0, 15.0])
        loaded_p = loaded.predict_margins(feat_row, [5.0, 15.0])
        np.testing.assert_allclose(orig_p, loaded_p)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
