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
from typing import Tuple, List, Optional, Union, Dict, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder

# Canonical feature definitions adhering to clean V3 data pipeline
FEATURE_COLS: List[str] = [
    "Total_Buy_INR",
    "Unit_Buy_INR",
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

        # Derive Unit_Buy_INR dynamically if omitted
        if "Unit_Buy_INR" not in df.columns:
            if "Total_Buy_INR" in df.columns and "Chargeable_Weight_Kg" in df.columns:
                df["Unit_Buy_INR"] = df["Total_Buy_INR"] / df["Chargeable_Weight_Kg"].clip(lower=0.1)
            elif "Airline_Freight_Rate_INR" in df.columns:
                df["Unit_Buy_INR"] = df["Airline_Freight_Rate_INR"]
            else:
                df["Unit_Buy_INR"] = 250.0

        # Use canonical features if available, otherwise available subset
        if is_fit:
            if all(col in df.columns for col in FEATURE_COLS):
                self.feature_cols = list(FEATURE_COLS)
                self.categorical_cols = list(CATEGORICAL_COLS)
            else:
                EXCLUDE_COLS = {
                    "Inquiry Number", "Quotation Number", "Inquiry Status", "Quote Status",
                    "is_won", "Margin_Percentage", "Margin_Amount_INR", "Total_Sell_INR",
                    "Sales Person", "Quoted Carrier", "Commodity Type", "customer_id",
                    "quote_date", "Year_Month", "Month_Name", "Booked by branch",
                    "Customer_Company"
                }
                self.feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
                self.categorical_cols = [
                    c for c in self.feature_cols
                    if c in CATEGORICAL_COLS or pd.api.types.is_string_dtype(df[c]) or df[c].dtype == "object" or df[c].dtype.name == "category"
                ]

        cols = self.feature_cols or list(FEATURE_COLS)
        cats = self.categorical_cols if self.categorical_cols is not None else list(CATEGORICAL_COLS)
        if is_fit:
            self.categorical_cols = list(cats)

        # Ensure all required features exist in df with robust fallbacks
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
        sample_weight: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> "MarginQuantileModel":
        """
        Fits Quantile Regressors for P10, P50, and P90 markups.
        Supports sample_weight for time-decay recency weighting.
        """
        X_mat, cat_mask = self._prepare_features(X, is_fit=True)
        y_arr = np.asarray(y, dtype=float)
        sw_arr = np.asarray(sample_weight, dtype=float) if sample_weight is not None else None

        # Monotonic decreasing constraints: -1 on buy cost, unit buy rate, and weight
        n_features = X_mat.shape[1]
        monotonic_cst = [0] * n_features
        if not self._is_matrix_mode and self.feature_cols:
            if "Total_Buy_INR" in self.feature_cols:
                buy_idx = self.feature_cols.index("Total_Buy_INR")
                monotonic_cst[buy_idx] = -1
            if "Unit_Buy_INR" in self.feature_cols:
                unit_idx = self.feature_cols.index("Unit_Buy_INR")
                monotonic_cst[unit_idx] = -1
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
        self.model_p10.fit(X_mat, y_log, sample_weight=sw_arr)

        # Model P50 (true median margin / L1 loss)
        self.model_p50 = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=0.50,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            categorical_features=cat_mask,
            monotonic_cst=monotonic_cst,
            random_state=self.random_state,
        )
        self.model_p50.fit(X_mat, y_log, sample_weight=sw_arr)

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
        self.model_p90.fit(X_mat, y_log, sample_weight=sw_arr)

        # Enforce exact monotonic constraints on all tree predictors
        self._enforce_tree_monotonicity(monotonic_cst, cat_mask)

        self.is_fitted = True
        return self

    def _enforce_tree_monotonicity(
        self, monotonic_cst: List[int], cat_mask: List[bool]
    ) -> None:
        """
        Enforces monotonic split constraints on the leaf values of fitted tree predictors.
        Scikit-learn line-search for non-smooth quantile loss can overwrite leaf values
        without checking monotonic bounds, causing small step violations. This method
        walks each tree and strictly clips leaf values to uphold the monotonic contract.
        """
        is_cat_arr = np.array(cat_mask, dtype=bool)
        if len(monotonic_cst) > len(is_cat_arr):
            remapped_cst = np.array(monotonic_cst)
        else:
            remapped_cst = np.concatenate([
                np.zeros(is_cat_arr.sum(), dtype=int),
                np.array(monotonic_cst)[~is_cat_arr],
            ])

        def enforce_node(nodes: np.ndarray, node_idx: int, lower: float, upper: float) -> Tuple[float, float]:
            node = nodes[node_idx]
            if node["is_leaf"]:
                val = float(node["value"])
                val = float(np.clip(val, lower, upper))
                nodes[node_idx]["value"] = val
                return val, val

            feat = int(node["feature_idx"])
            cst = remapped_cst[feat] if feat < len(remapped_cst) else 0
            left_idx = int(node["left"])
            right_idx = int(node["right"])

            l_min, l_max = enforce_node(nodes, left_idx, lower, upper)
            r_min, r_max = enforce_node(nodes, right_idx, lower, upper)

            if cst == -1 and r_max > l_min:  # NEG: left should be >= right (r_max <= l_min)
                mid = (l_min + r_max) / 2.0
                l_min, l_max = enforce_node(nodes, left_idx, mid, upper)
                r_min, r_max = enforce_node(nodes, right_idx, lower, mid)
            elif cst == 1 and l_max > r_min:  # POS: left <= right (l_max <= r_min)
                mid = (l_max + r_min) / 2.0
                l_min, l_max = enforce_node(nodes, left_idx, lower, mid)
                r_min, r_max = enforce_node(nodes, right_idx, mid, upper)

            return min(l_min, r_min), max(l_max, r_max)

        for model_obj in [self.model_p10, self.model_p50, self.model_p90]:
            if model_obj is not None and hasattr(model_obj, "_predictors"):
                for pred_tuple in model_obj._predictors:
                    enforce_node(pred_tuple[0].nodes, 0, float("-inf"), float("+inf"))

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

    def unseen_categorical_report(self, X: pd.DataFrame) -> Dict[str, str]:
        """
        Reports any out-of-vocabulary categorical inputs not seen during training.
        """
        out = {}
        if self.encoder is None or not self.categorical_cols:
            return out
        for col, cats in zip(self.categorical_cols, self.encoder.categories_):
            if col in X.columns:
                val = str(X[col].iloc[0])
                if val not in set(cats):
                    out[col] = val
        return out

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
