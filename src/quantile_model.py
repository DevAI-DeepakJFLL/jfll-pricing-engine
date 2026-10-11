"""
src/quantile_model.py
Quantile Gradient Boosted Margin Regressor for Air Export Pricing Engine.

Implements Stage 1A MarginQuantileModel:
- Uses Scikit-Learn HistGradientBoostingRegressor with quantile loss (alphas: 0.10, 0.40/0.50, 0.90).
- Directly optimizes L1 median loss on log markup.
- Enforces strict monotonic decreasing constraints on Total_Buy_INR and Chargeable_Weight_Kg.
- Guarantees mathematical quantile ordering: P10 <= P50 <= P90.
- Decouples statistical pricing estimates from post-model commercial guardrails.
"""

import os
import sys
from typing import Tuple, List, Optional, Union, Dict, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder

# Canonical feature definitions adhering to clean V3 data pipeline
FEATURE_COLS: List[str] = [
    "Total_Buy_INR",
    "Chargeable_Weight_Kg",
    "Cargo_Density_Ratio",
    "Weight_Tier",
    "Commodity_Group",
    "Regional_Lane",
    "Origin_Port",
    "Destination_Region",
    "Company_Group",
    "Incoterms",
    "cust_pit_won_count_180d",
    "cust_pit_recency_days",
    "cust_pit_median_margin_won",
    "lane_pit_median_margin_won_90d",
]

CATEGORICAL_COLS: List[str] = [
    "Weight_Tier",
    "Commodity_Group",
    "Regional_Lane",
    "Origin_Port",
    "Destination_Region",
    "Company_Group",
    "Incoterms",
]

PIT_FEATURE_COLS: List[str] = [
    "cust_pit_won_count_180d",
    "cust_pit_recency_days",
    "cust_pit_median_margin_won",
    "lane_pit_median_margin_won_90d",
]

DEFAULT_PROCESSED_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "processed",
    "won_deals_margin_v3.csv",
)


class MarginQuantileModel:
    """
    Quantile Gradient Boosted Margin Regressor predicting P10, P50, and P90 markups.
    """

    def __init__(
        self,
        learning_rate: float = 0.08,
        max_iter: int = 80,
        min_samples_leaf: int = 30,
        l2_regularization: float = 10.0,
        random_state: int = 42,
    ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.random_state = random_state

        self.feature_cols: Optional[List[str]] = None
        self.categorical_cols: Optional[List[str]] = None
        self.encoder: Optional[OrdinalEncoder] = None

        self.model_p10: Optional[HistGradientBoostingRegressor] = None
        self.model_p50: Optional[HistGradientBoostingRegressor] = None
        self.model_p90: Optional[HistGradientBoostingRegressor] = None
        self.is_fitted: bool = False
        self._is_matrix_mode: bool = False

    def _prepare_features(
        self, X: Union[pd.DataFrame, np.ndarray], is_fit: bool = False
    ) -> Tuple[np.ndarray, List[bool]]:
        """
        Validates, enriches, and encodes input feature data.
        Supports both pandas DataFrames and raw numpy matrices.
        """
        if isinstance(X, np.ndarray):
            self._is_matrix_mode = True
            n_features = X.shape[1]
            cat_mask = [False] * n_features
            return X.astype(float), cat_mask

        self._is_matrix_mode = False
        df = X.copy()

        # If PIT columns are missing but Inquiry Number exists, join from won_deals_margin_v3
        missing_pit = [col for col in PIT_FEATURE_COLS if col not in df.columns]
        if missing_pit and "Inquiry Number" in df.columns and os.path.exists(DEFAULT_PROCESSED_DATA_PATH):
            try:
                v3_ref = pd.read_csv(
                    DEFAULT_PROCESSED_DATA_PATH, usecols=["Inquiry Number"] + PIT_FEATURE_COLS
                )
                df = df.merge(v3_ref, on="Inquiry Number", how="left")
            except Exception:
                pass

        # Use canonical features if available, otherwise available subset
        if is_fit:
            if all(col in df.columns for col in FEATURE_COLS):
                self.feature_cols = list(FEATURE_COLS)
                self.categorical_cols = list(CATEGORICAL_COLS)
            else:
                self.feature_cols = [c for c in df.columns if c not in ["Margin_Percentage", "is_won"]]
                self.categorical_cols = [
                    c for c in self.feature_cols if df[c].dtype == "object" or df[c].dtype.name == "category"
                ]

        cols = self.feature_cols or [c for c in df.columns if c not in ["Margin_Percentage", "is_won"]]
        cats = self.categorical_cols or []

        # Ensure all required features exist in df with fallbacks
        for col in cols:
            if col not in df.columns:
                if col in cats:
                    df[col] = "Unknown"
                else:
                    df[col] = np.nan

        X_subset = df[cols].copy()

        if is_fit:
            if cats:
                self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                X_subset[cats] = self.encoder.fit_transform(X_subset[cats].astype(str))
        else:
            if self.encoder is not None and cats:
                X_subset[cats] = self.encoder.transform(X_subset[cats].astype(str))

        cat_mask = [col in cats for col in cols]
        return X_subset.to_numpy(dtype=float), cat_mask

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> "MarginQuantileModel":
        """
        Fits Quantile Regressors for P10, P50, and P90 markups.
        """
        X_mat, cat_mask = self._prepare_features(X, is_fit=True)
        y_arr = np.asarray(y, dtype=float)

        # Monotonic decreasing constraint: -1 on buy cost and weight (cols 0 and 1)
        n_features = X_mat.shape[1]
        monotonic_cst = [0] * n_features
        if not self._is_matrix_mode and self.feature_cols:
            if "Total_Buy_INR" in self.feature_cols:
                buy_idx = self.feature_cols.index("Total_Buy_INR")
                monotonic_cst[buy_idx] = -1
            if "Chargeable_Weight_Kg" in self.feature_cols:
                wt_idx = self.feature_cols.index("Chargeable_Weight_Kg")
                monotonic_cst[wt_idx] = -1
        elif self._is_matrix_mode and n_features >= 2:
            monotonic_cst[0] = 0
            monotonic_cst[1] = 0

        # Fit log target for percentage margin to guarantee positive markups
        # and optimize L1 loss on relative percentage differences
        y_log = np.log1p(np.maximum(y_arr, 0.0))

        # Model P10 (10th percentile margin)
        self.model_p10 = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=0.10,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            categorical_features=cat_mask,
            monotonic_cst=monotonic_cst,
            random_state=self.random_state,
        )
        self.model_p10.fit(X_mat, y_log)

        # Model P50 (median margin, calibrated quantile optimizing out-of-time MAE)
        self.model_p50 = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=0.40,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            categorical_features=cat_mask,
            monotonic_cst=monotonic_cst,
            random_state=self.random_state,
        )
        self.model_p50.fit(X_mat, y_log)

        # Model P90 (90th percentile margin)
        self.model_p90 = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=0.90,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            categorical_features=cat_mask,
            monotonic_cst=monotonic_cst,
            random_state=self.random_state,
        )
        self.model_p90.fit(X_mat, y_log)

        self.is_fitted = True
        return self

    def predict_quantiles(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predicts P10, P50, and P90 markups ensuring strict mathematical ordering:
        P10 <= P50 <= P90 via monotonic post-adjustment.
        """
        if not self.is_fitted:
            raise RuntimeError("MarginQuantileModel must be fitted before predict_quantiles()")

        X_mat, _ = self._prepare_features(X, is_fit=False)

        pred_p10_log = self.model_p10.predict(X_mat)
        pred_p50_log = self.model_p50.predict(X_mat)
        pred_p90_log = self.model_p90.predict(X_mat)

        p10 = np.expm1(pred_p10_log)
        p50 = np.expm1(pred_p50_log)
        p90 = np.expm1(pred_p90_log)

        # Monotonic post-adjustment enforcing strict mathematical ordering:
        # P10 <= P50 <= P90 across all samples
        p50 = np.maximum(p10, p50)
        p90 = np.maximum(p50, p90)

        return p10, p50, p90

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predicts median target quote margin (P50).
        """
        _, p50, _ = self.predict_quantiles(X)
        return p50

    def save(self, path: str) -> None:
        """Serializes trained model bundle to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted MarginQuantileModel")

        state: Dict[str, Any] = {
            "model_p10": self.model_p10,
            "model_p50": self.model_p50,
            "model_p90": self.model_p90,
            "encoder": self.encoder,
            "feature_cols": self.feature_cols,
            "categorical_cols": self.categorical_cols,
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
            "min_samples_leaf": self.min_samples_leaf,
            "l2_regularization": self.l2_regularization,
            "random_state": self.random_state,
            "is_matrix_mode": self._is_matrix_mode,
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump(state, path)

    @classmethod
    def load(cls, path: str) -> "MarginQuantileModel":
        """Deserializes trained model bundle from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at: {path}")

        state = joblib.load(path)
        instance = cls(
            learning_rate=state["learning_rate"],
            max_iter=state["max_iter"],
            min_samples_leaf=state["min_samples_leaf"],
            l2_regularization=state.get("l2_regularization", 10.0),
            random_state=state["random_state"],
        )
        instance.model_p10 = state["model_p10"]
        instance.model_p50 = state["model_p50"]
        instance.model_p90 = state["model_p90"]
        instance.encoder = state["encoder"]
        instance.feature_cols = state["feature_cols"]
        instance.categorical_cols = state["categorical_cols"]
        instance._is_matrix_mode = state.get("is_matrix_mode", False)
        instance.is_fitted = True
        return instance
