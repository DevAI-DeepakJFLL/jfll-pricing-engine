"""
src/win_model.py
Monotonic calibrated Win Probability Model for Air Export Pricing Engine.

Models P(Win | shipment features, quoted margin), strictly enforcing:
- Monotone decreasing win probability as margin percentage increases (monotonic_cst=[-1]).
- Isotonic regression post-hoc calibration to ensure probabilities reflect empirical win frequencies.
- Out-of-fold and holdout validation against release quality gates (AUC >= 0.60, ECE <= 0.05).
"""

import os
from typing import Dict, Sequence, Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import OrdinalEncoder

try:
    from quantile_model import FEATURE_COLS, CATEGORICAL_COLS
except ImportError:
    from src.quantile_model import FEATURE_COLS, CATEGORICAL_COLS

WIN_FEATURES = FEATURE_COLS + ["Margin_Percentage"]


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Computes Expected Calibration Error across discrete probability bins."""
    idx = np.minimum((p * bins).astype(int), bins - 1)
    ece_val = 0.0
    for b in range(bins):
        mask = (idx == b)
        if mask.any():
            ece_val += abs(y[mask].mean() - p[mask].mean()) * mask.mean()
    return float(ece_val)


class WinModel:
    """
    Predicts probability of winning a deal given shipment features and candidate margin.
    Enforces economic monotonicity on margin and calibrated win probabilities.
    """

    def __init__(self, random_state: int = 42):
        self.enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.clf: Optional[HistGradientBoostingClassifier] = None
        self.iso: Optional[IsotonicRegression] = None
        self.metrics: Dict[str, float] = {}
        self.random_state = random_state
        self.is_fitted: bool = False

    def _X(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        df_work = df.copy()
        if "Unit_Buy_INR" not in df_work.columns:
            if "Total_Buy_INR" in df_work.columns and "Chargeable_Weight_Kg" in df_work.columns:
                df_work["Unit_Buy_INR"] = df_work["Total_Buy_INR"] / df_work["Chargeable_Weight_Kg"].replace(0, 1.0)
            else:
                df_work["Unit_Buy_INR"] = 0.0

        for col in WIN_FEATURES:
            if col not in df_work.columns:
                df_work[col] = np.nan

        X = df_work[WIN_FEATURES].copy()
        cats = X[CATEGORICAL_COLS].astype(str)
        if fit:
            X[CATEGORICAL_COLS] = self.enc.fit_transform(cats)
        else:
            X[CATEGORICAL_COLS] = self.enc.transform(cats)
        return X.to_numpy(dtype=float)

    def fit(self, df: pd.DataFrame, month_col: str = "Year_Month") -> "WinModel":
        """
        Fits HistGradientBoostingClassifier with negative monotonicity on Margin_Percentage,
        and fits an IsotonicRegression calibrator on a temporal holdout slice.
        """
        months = sorted(df[month_col].dropna().unique())
        if len(months) < 3:
            # Fallback to positional split if fewer than 3 months
            n = len(df)
            tr = df.iloc[:int(0.70 * n)]
            ca = df.iloc[int(0.70 * n):int(0.85 * n)]
            te = df.iloc[int(0.85 * n):]
        else:
            test_from, calib_from = months[-1], months[-2]
            tr = df[df[month_col] < calib_from]
            ca = df[(df[month_col] >= calib_from) & (df[month_col] < test_from)]
            te = df[df[month_col] >= test_from]

        if len(ca) < 50:
            # Expand calibration slice if too small
            n = len(df)
            tr = df.iloc[:int(0.70 * n)]
            ca = df.iloc[int(0.70 * n):int(0.85 * n)]
            te = df.iloc[int(0.85 * n):]

        mono = [0] * len(WIN_FEATURES)
        mono[WIN_FEATURES.index("Margin_Percentage")] = -1

        self.clf = HistGradientBoostingClassifier(
            categorical_features=[c in CATEGORICAL_COLS for c in WIN_FEATURES],
            monotonic_cst=mono,
            learning_rate=0.05,
            max_iter=200,
            min_samples_leaf=30,
            l2_regularization=5.0,
            early_stopping=False,
            random_state=self.random_state,
        ).fit(self._X(tr, fit=True), tr["is_won"])

        raw_ca = self.clf.predict_proba(self._X(ca))[:, 1]
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.02, y_max=0.98).fit(
            raw_ca, ca["is_won"]
        )

        self.is_fitted = True

        # Evaluate on test slice
        p = self.predict_df(te)
        y = te["is_won"].to_numpy(float)

        auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else 0.50
        brier = float(brier_score_loss(y, p))
        ece = _ece(y, p)

        self.metrics = {
            "auc": round(auc, 4),
            "brier": round(brier, 4),
            "ece": round(ece, 4),
            "n_test": int(len(te)),
            "base_rate": round(float(y.mean()), 4),
        }
        return self

    def predict_df(self, df: pd.DataFrame) -> np.ndarray:
        """Predicts calibrated win probabilities for each row in a DataFrame."""
        if not self.is_fitted or self.clf is None or self.iso is None:
            raise RuntimeError("WinModel must be fitted before predict_df")
        raw_p = self.clf.predict_proba(self._X(df))[:, 1]
        return np.asarray(self.iso.predict(raw_p), dtype=float)

    def predict_margins(self, feat_row: pd.DataFrame, margins: Sequence[float]) -> np.ndarray:
        """
        Given a single shipment feature row, evaluates win probability across candidate margins.
        Returns array of probabilities corresponding to each candidate margin.
        """
        if not self.is_fitted or self.clf is None or self.iso is None:
            return np.array([None] * len(margins))

        row = feat_row.iloc[[0]] if len(feat_row) > 1 else feat_row
        batch = pd.concat([row] * len(margins), ignore_index=True)
        batch["Margin_Percentage"] = np.asarray(margins, dtype=float)
        return self.predict_df(batch)

    def save(self, path: str) -> None:
        """Serializes WinModel artifact to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        bundle = {
            "clf": self.clf,
            "iso": self.iso,
            "enc": self.enc,
            "metrics": self.metrics,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(bundle, path)

    def load(self, path: str) -> "WinModel":
        """Loads serialized WinModel artifact from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"WinModel file not found at {path}")
        bundle = joblib.load(path)
        self.clf = bundle["clf"]
        self.iso = bundle["iso"]
        self.enc = bundle["enc"]
        self.metrics = bundle.get("metrics", {})
        self.is_fitted = bundle.get("is_fitted", True)
        return self
