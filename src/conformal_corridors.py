"""
src/conformal_corridors.py
Calibrated Split-Conformal Prediction Corridors for Air Export Pricing Engine.

Implements distribution-free split-conformal quantile prediction bands with finite-sample
coverage guarantees (Fix 5 / Issue M5).

Key Capabilities:
1. calibrate_conformal_band: Exact finite-sample distribution-free conformal quantile.
2. ConformalCorridorCalibrator:
   - Evaluates nonconformity scores R_i = |y_i - y_hat_i| on time-ordered calibration data.
   - Computes segment-level quantiles with Empirical Bayes shrinkage toward global:
     q_segment = (n_seg / (n_seg + k)) * q_raw + (k / (n_seg + k)) * q_global.
   - Provides point corridor predictions: [pred_margin - half_width, pred_margin + half_width].
   - Evaluates empirical out-of-time coverage and per-slab metrics.
   - Serialization and deserialization via joblib.
"""

import os
from typing import Dict, Tuple, Optional, Union, Any
import joblib
import numpy as np
import pandas as pd


def calibrate_conformal_band(residuals: Union[np.ndarray, pd.Series, list], alpha: float = 0.10) -> float:
    """
    Calculates the finite-sample distribution-free conformal quantile.

    For sample size n and target miscoverage rate alpha:
        k = ceil((n + 1) * (1 - alpha))
        index = clip(k - 1, 0, n - 1)
    Returns the half-width threshold q. On exchangeable holdout samples,
    the empirical coverage P(|Y - Y_hat| <= q) is guaranteed to be >= 1 - alpha.

    Parameters:
    -----------
    residuals : Union[np.ndarray, pd.Series, list]
        Array-like nonconformity scores or residuals |y - y_hat|.
    alpha : float, default 0.10
        Miscoverage rate (e.g., alpha=0.10 guarantees 90% marginal coverage).

    Returns:
    --------
    float: Calibrated half-width in percentage points.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    arr = np.abs(np.asarray(residuals, dtype=float).ravel())
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0:
        return 0.0

    sorted_res = np.sort(arr)
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    idx = min(max(k - 1, 0), n - 1)
    return float(sorted_res[idx])


class ConformalCorridorCalibrator:
    """
    Calibrated split-conformal corridor calibrator for air export pricing.

    Guarantees distribution-free marginal coverage (>= 1 - alpha) across time-ordered
    holdout data with Empirical Bayes segment shrinkage across weight slabs and
    customer relationship types.
    """

    def __init__(self, alpha: float = 0.10, k_prior: float = 20.0):
        """
        Parameters:
        -----------
        alpha : float, default 0.10
            Miscoverage rate (default 0.10 gives 90% confidence corridor).
        k_prior : float, default 20.0
            Empirical Bayes pseudo-observation shrinkage strength toward the global band.
        """
        self.alpha: float = float(alpha)
        self.k_prior: float = float(k_prior)
        self.global_half_width: float = 0.0
        self.slab_half_widths: Dict[str, float] = {}
        self.customer_type_half_widths: Dict[str, float] = {}  # "known" vs "new"
        self.calibration_n: int = 0
        self.is_fitted: bool = False

    def fit(
        self,
        y_true: Union[np.ndarray, pd.Series, list],
        y_pred: Union[np.ndarray, pd.Series, list],
        weight_tiers: Optional[Union[np.ndarray, pd.Series, list]] = None,
        is_known_customer: Optional[Union[np.ndarray, pd.Series, list]] = None,
    ) -> "ConformalCorridorCalibrator":
        """
        Fits conformal quantile bands on time-ordered calibration data.

        Computes global nonconformity scores R_i = |y_i - y_hat_i|.
        Computes segment-level quantiles with Empirical Bayes shrinkage toward global:
            q_segment = (n_seg / (n_seg + k)) * q_raw + (k / (n_seg + k)) * q_global.

        Parameters:
        -----------
        y_true : array-like
            Realized ground-truth margins (percentage points).
        y_pred : array-like
            Model predicted margins (percentage points).
        weight_tiers : optional array-like
            Weight tier category for each calibration sample (e.g. '<45kg', '100-300kg').
        is_known_customer : optional array-like
            Boolean indicator whether the customer has prior won quote history.

        Returns:
        --------
        self : ConformalCorridorCalibrator
        """
        y_true_arr = np.asarray(y_true, dtype=float).ravel()
        y_pred_arr = np.asarray(y_pred, dtype=float).ravel()
        if len(y_true_arr) != len(y_pred_arr):
            raise ValueError(
                f"y_true ({len(y_true_arr)}) and y_pred ({len(y_pred_arr)}) must have equal length"
            )

        residuals = np.abs(y_true_arr - y_pred_arr)
        valid_mask = ~np.isnan(residuals)
        residuals = residuals[valid_mask]
        self.calibration_n = len(residuals)

        if self.calibration_n == 0:
            raise ValueError("Cannot calibrate on empty or all-NaN residuals")

        # Global nonconformity score quantile
        self.global_half_width = calibrate_conformal_band(residuals, self.alpha)

        # Weight tier segment shrinkage
        self.slab_half_widths = {}
        if weight_tiers is not None:
            tiers_arr = np.asarray(weight_tiers).ravel()[valid_mask]
            unique_tiers = pd.Series(tiers_arr).dropna().unique()
            for tier in unique_tiers:
                tier_str = str(tier)
                tier_mask = (tiers_arr == tier)
                n_seg = int(np.sum(tier_mask))
                if n_seg > 0:
                    seg_res = residuals[tier_mask]
                    q_raw = calibrate_conformal_band(seg_res, self.alpha)
                    weight = n_seg / (n_seg + self.k_prior)
                    q_shrunk = weight * q_raw + (1.0 - weight) * self.global_half_width
                    self.slab_half_widths[tier_str] = float(q_shrunk)
                else:
                    self.slab_half_widths[tier_str] = float(self.global_half_width)

        # Customer relationship segment shrinkage ("known" vs "new")
        self.customer_type_half_widths = {}
        if is_known_customer is not None:
            known_arr = np.asarray(is_known_customer).ravel()[valid_mask].astype(bool)
            for cust_key, flag in [("known", True), ("new", False)]:
                cust_mask = (known_arr == flag)
                n_seg = int(np.sum(cust_mask))
                if n_seg > 0:
                    seg_res = residuals[cust_mask]
                    q_raw = calibrate_conformal_band(seg_res, self.alpha)
                    weight = n_seg / (n_seg + self.k_prior)
                    q_shrunk = weight * q_raw + (1.0 - weight) * self.global_half_width
                    self.customer_type_half_widths[cust_key] = float(q_shrunk)
                else:
                    self.customer_type_half_widths[cust_key] = float(self.global_half_width)

        self.is_fitted = True
        return self

    def get_half_width(
        self,
        weight_tier: Optional[str] = None,
        is_known_customer: bool = True,
    ) -> float:
        """
        Returns calibrated half-width (in percentage points) for a quote context.
        Applies segment shrinkage if available, falling back smoothly to global.

        Parameters:
        -----------
        weight_tier : optional str
            Weight slab code (e.g. '100-300kg', '>1000kg').
        is_known_customer : bool, default True
            Whether customer is recognized with past relationship history.

        Returns:
        --------
        float: Half-width in margin percentage points.
        """
        if not self.is_fitted:
            raise RuntimeError("ConformalCorridorCalibrator must be fitted before get_half_width()")

        # 1. If weight tier is fitted in slab dictionary, use its shrunk quantile
        if weight_tier is not None and str(weight_tier) in self.slab_half_widths:
            hw = self.slab_half_widths[str(weight_tier)]
            # If customer is explicitly new and we have new customer calibration, expand band proportionally
            # (Prevents Simpson's paradox / small retail courier variance leakage into heavy cargo)
            if not is_known_customer and "new" in self.customer_type_half_widths:
                expansion = min(1.30, max(1.05, self.customer_type_half_widths["new"] / max(1.0, self.global_half_width)))
                hw = hw * expansion
            return float(hw)

        # 2. Fall back to customer relationship type if available
        cust_key = "known" if is_known_customer else "new"
        if cust_key in self.customer_type_half_widths:
            return float(self.customer_type_half_widths[cust_key])

        # 3. Fall back to global calibrated quantile
        return float(self.global_half_width)

    def predict_corridor(
        self,
        pred_margin: float,
        weight_tier: Optional[str] = None,
        is_known_customer: bool = True,
    ) -> Tuple[float, float, float]:
        """
        Returns (corridor_low, corridor_high, half_width) around the predicted margin.

        Parameters:
        -----------
        pred_margin : float
            Point prediction of target margin percentage (e.g. from P50 quantile regressor).
        weight_tier : optional str
            Weight tier for segment calibration.
        is_known_customer : bool, default True
            Customer relationship flag.

        Returns:
        --------
        Tuple[float, float, float]: (corridor_low, corridor_high, half_width).
        """
        hw = self.get_half_width(weight_tier=weight_tier, is_known_customer=is_known_customer)
        corridor_low = float(pred_margin - hw)
        corridor_high = float(pred_margin + hw)
        return (corridor_low, corridor_high, float(hw))

    def evaluate_coverage(
        self,
        y_true: Union[np.ndarray, pd.Series, list],
        y_pred: Union[np.ndarray, pd.Series, list],
        weight_tiers: Optional[Union[np.ndarray, pd.Series, list]] = None,
    ) -> Dict[str, Any]:
        """
        Computes overall coverage rate, mean half-width, and per-slab coverage breakdown.

        Parameters:
        -----------
        y_true : array-like
            True realized margin percentages.
        y_pred : array-like
            Predicted margin percentages.
        weight_tiers : optional array-like
            Weight tier labels for slab-specific breakdown.

        Returns:
        --------
        Dict[str, Any]: Dictionary containing overall coverage, mean width, and slab metrics.
        """
        if not self.is_fitted:
            raise RuntimeError("ConformalCorridorCalibrator must be fitted before evaluate_coverage()")

        y_true_arr = np.asarray(y_true, dtype=float).ravel()
        y_pred_arr = np.asarray(y_pred, dtype=float).ravel()
        n = len(y_true_arr)

        tiers_arr = None
        if weight_tiers is not None:
            tiers_arr = np.asarray(weight_tiers).ravel()

        half_widths = np.zeros(n, dtype=float)
        for i in range(n):
            tier = str(tiers_arr[i]) if tiers_arr is not None else None
            half_widths[i] = self.get_half_width(weight_tier=tier)

        abs_errs = np.abs(y_true_arr - y_pred_arr)
        is_covered = abs_errs <= (half_widths + 1e-9)

        overall_coverage = float(np.mean(is_covered)) if n > 0 else 0.0
        mean_half_width = float(np.mean(half_widths)) if n > 0 else 0.0

        per_slab: Dict[str, Dict[str, Any]] = {}
        if tiers_arr is not None:
            for tier in pd.Series(tiers_arr).dropna().unique():
                tier_str = str(tier)
                mask = (tiers_arr == tier)
                n_slab = int(np.sum(mask))
                if n_slab > 0:
                    cov_slab = float(np.mean(is_covered[mask]))
                    hw_slab = float(np.mean(half_widths[mask]))
                    per_slab[tier_str] = {
                        "coverage_rate": cov_slab,
                        "overall_coverage": cov_slab,
                        "mean_half_width": hw_slab,
                        "n": n_slab,
                    }

        return {
            "overall_coverage": overall_coverage,
            "coverage_rate": overall_coverage,
            "mean_half_width": mean_half_width,
            "n_samples": n,
            "per_slab_coverage": per_slab,
            "slab_breakdown": per_slab,
        }

    def save(self, filepath: str) -> None:
        """Serializes calibrator state to disk using joblib."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted ConformalCorridorCalibrator")
        state: Dict[str, Any] = {
            "alpha": self.alpha,
            "k_prior": self.k_prior,
            "global_half_width": self.global_half_width,
            "slab_half_widths": self.slab_half_widths,
            "customer_type_half_widths": self.customer_type_half_widths,
            "calibration_n": self.calibration_n,
            "is_fitted": self.is_fitted,
        }
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(state, filepath)

    @classmethod
    def load(cls, filepath: str) -> "ConformalCorridorCalibrator":
        """Loads serialized calibrator state from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Calibrator file not found at: {filepath}")
        state = joblib.load(filepath)
        instance = cls(alpha=state["alpha"], k_prior=state["k_prior"])
        instance.global_half_width = state["global_half_width"]
        instance.slab_half_widths = state["slab_half_widths"]
        instance.customer_type_half_widths = state["customer_type_half_widths"]
        instance.calibration_n = state["calibration_n"]
        instance.is_fitted = state["is_fitted"]
        return instance
