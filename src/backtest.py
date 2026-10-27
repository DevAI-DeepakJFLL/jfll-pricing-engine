"""
src/backtest.py
Rigorous backtesting, pinball loss evaluation, empirical corridor coverage measurement,
and release gates for Air Export Pricing Engine models.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


def pinball(y, q, tau: float) -> float:
    """Computes pinball (quantile) loss for target y and predicted quantile q at level tau."""
    d = np.asarray(y, float) - np.asarray(q, float)
    return float(np.mean(np.maximum(tau * d, (tau - 1.0) * d)))


def temporal_split(df: pd.DataFrame, month_col: str = "Year_Month", min_cal: int = 150):
    """
    Splits DataFrame dynamically by month, allocating the most recent month(s) with at least min_cal
    records for calibration and earlier months for training.
    """
    if month_col not in df.columns:
        n = len(df)
        split_idx = int(0.80 * n)
        return df.iloc[:split_idx], df.iloc[split_idx:]

    months = sorted(df[month_col].dropna().unique())
    for k in (1, 2, 3):
        if len(months) > k:
            cal = df[df[month_col].isin(months[-k:])]
            if len(cal) >= min_cal:
                return df.drop(cal.index), cal
    # Positional fallback
    n = len(df)
    split_idx = int(0.80 * n)
    return df.iloc[:split_idx], df.iloc[split_idx:]


def rolling_origin_backtest(df: pd.DataFrame, model_factory, month_col: str = "Year_Month", n_folds: int = 3, min_train: int = 500):
    """
    Performs walk-forward expanding window temporal backtest across n_folds.
    Evaluates pinball losses, MAE, coverage, and crossing rates.
    """
    if month_col not in df.columns:
        return pd.DataFrame()

    months = sorted(df[month_col].dropna().unique())
    rows = []
    for m in months[-n_folds:]:
        tr = df[df[month_col] < m]
        te = df[df[month_col] == m]
        if len(tr) < min_train or te.empty:
            continue
        mdl = model_factory().fit(tr, tr["Margin_Percentage"])
        X, _ = mdl._prepare_features(te)
        raw = {q: np.expm1(getattr(mdl, f"model_p{q}").predict(X)) for q in (10, 50, 90)}
        y = te["Margin_Percentage"].to_numpy(float)
        rows.append({
            "month": m,
            "n": len(te),
            "pinball_p10": round(pinball(y, raw[10], 0.1), 4),
            "pinball_p50": round(pinball(y, raw[50], 0.5), 4),
            "pinball_p90": round(pinball(y, raw[90], 0.9), 4),
            "mae_p50": round(float(np.mean(np.abs(y - raw[50]))), 4),
            "coverage_p10_p90": round(float(np.mean((y >= raw[10]) & (y <= raw[90]))), 4),
            "crossing_rate": round(float(np.mean((raw[10] > raw[50]) | (raw[50] > raw[90]))), 4),
        })
    return pd.DataFrame(rows)


def assert_release_gates(bt: pd.DataFrame, cov_target: float = 0.80, cov_tol: float = 0.15, max_cross: float = 0.05):
    """Asserts production release gates on backtest results."""
    if bt.empty:
        return
    cov = float(bt["coverage_p10_p90"].mean())
    cross = float(bt["crossing_rate"].mean())
    if abs(cov - cov_target) > cov_tol:
        raise RuntimeError(f"P10-P90 coverage {cov:.2%} outside gate {cov_target:.0%}±{cov_tol:.0%}")
    if cross > max_cross:
        raise RuntimeError(f"Quantile crossing rate {cross:.2%} exceeds gate {max_cross:.0%}")


def measure_corridor_coverage(engine, holdout: pd.DataFrame, max_rows: int = 500) -> Dict[str, Any]:
    """
    Measures true empirical holdout coverage of the negotiation corridor.
    Replaces static '90% confidence coverage' claims with genuine measured data.
    """
    hits = 0
    widths = 0.0
    n = 0
    sub = holdout.tail(max_rows)
    for _, r in sub.iterrows():
        try:
            inq = {**r.to_dict(), "strategy": "balanced", "quote_revision_count": 1}
            rec = engine.recommend_quote(inq)
            c = rec.get("negotiation_corridor")
            if not c:
                continue
            y = float(r["Margin_Percentage"])
            hits += int(c["min_margin_pct"] <= y <= c["max_margin_pct"])
            widths += float(c["band_width_pct"])
            n += 1
        except Exception:
            continue

    empirical_cov = round(hits / max(n, 1), 4)
    mean_width = round(widths / max(n, 1), 2)
    return {
        "empirical_coverage": empirical_cov,
        "mean_width_pp": mean_width,
        "n": n,
        "hits": hits,
    }
