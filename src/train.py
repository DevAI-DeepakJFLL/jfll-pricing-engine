"""
================================================================================
Production Model Training Pipeline: Two-Stage Market Benchmark & Elasticity Engine
================================================================================
Architecture:
  Stage 1A (Market Benchmark Regressor):
      Predicts market-clearing benchmark margin on won deals using rate-per-kg,
      lane rate index, weight, density, commodity, and trade corridors.
      Trained with honest 5-fold cross-validation out-of-fold benchmark estimation.
  Stage 1B (Calibrated Elasticity Classifier):
      Predicts win probability as a function of Margin_Ratio relative to market benchmark.
      Evaluated across 5-fold CV, temporal holdouts, and discrete margin buckets.
  Serialization:
      Persists model artifacts, encoders, rate indexes, and audit metadata to
      models/freight_margin_recommender.joblib.
================================================================================
"""

import os
import sys
import argparse
import datetime
from typing import Optional, Dict, Any, List, Tuple
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    roc_curve,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Support both module and standalone execution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import (
    MIN_BENCHMARK_MARGIN,
    MAX_BENCHMARK_MARGIN,
    MIN_CANDIDATE_MARGIN,
    MAX_CANDIDATE_MARGIN,
    MODEL_VERSION
)

try:
    import margin_framework as mf
    from preprocess import filter_worked_quotes
except ImportError:
    from src import margin_framework as mf
    from src.preprocess import filter_worked_quotes

# Feature definitions adhering to commercial representation standards
FEATURE_COLS: List[str] = [
    "Buy_Rate_Per_Kg",
    "Lane_Rate_Index",
    "Log_Chargeable_Weight",
    "Chargeable_Weight_Kg",
    "Gross_Weight_Kg",
    "Cargo_Density_Ratio",
    "Weight_Tier",
    "Commodity_Group",
    "Origin_Region",
    "Destination_Region",
    "Regional_Lane",
    "Origin_Port",
    "Dest_Port_Freq",
    "Company_Group",
    "Incoterms",
    "Month_Sin",
    "Month_Cos",
    "Air_Cargo_Season",
    "Is_Month_End",
    "Quote_DayOfWeek"
]

# Native categorical features supported by HistGradientBoosting (cardinality <= 255)
NATIVE_CAT_COLS: List[str] = [
    "Weight_Tier",
    "Commodity_Group",
    "Origin_Region",
    "Destination_Region",
    "Regional_Lane",
    "Origin_Port",
    "Company_Group",
    "Incoterms",
    "Air_Cargo_Season"
]


def compute_rate_features(
    df: pd.DataFrame,
    lane_medians: Optional[Dict[str, float]] = None,
    global_median: Optional[float] = None,
    port_frequencies: Optional[Dict[str, int]] = None
) -> pd.DataFrame:
    """
    Transform raw total buy and port names into commercially normalized rates & indices:
      - Buy_Rate_Per_Kg = Total_Buy_INR / Chargeable_Weight_Kg (clipped to commercial bounds)
      - Lane_Rate_Index = Buy_Rate_Per_Kg / median_lane_rate
      - Log_Chargeable_Weight = np.log1p(Chargeable_Weight_Kg)
      - Dest_Port_Freq = Historical frequency of destination port
    """
    out = df.copy()

    gross = out["Gross_Weight_Kg"].fillna(1.0).astype(float) if "Gross_Weight_Kg" in out.columns else pd.Series(1.0, index=out.index)
    ch_wt = out["Chargeable_Weight_Kg"].fillna(gross).astype(float)
    ch_wt = np.maximum(gross, ch_wt)
    ch_wt = np.ceil(ch_wt * 2.0) / 2.0
    out["Chargeable_Weight_Kg"] = ch_wt
    out["Gross_Weight_Kg"] = gross

    if "Total_Buy_INR" in out.columns:
        buy_total = out["Total_Buy_INR"].astype(float)
        buy_rate = (buy_total / ch_wt.replace(0, 1.0)).clip(lower=10.0, upper=8000.0)
    else:
        buy_rate = pd.Series(300.0, index=out.index)

    out["Buy_Rate_Per_Kg"] = buy_rate
    out["Log_Chargeable_Weight"] = np.log1p(ch_wt)

    # Compute lane rate index
    if lane_medians is not None:
        g_med = global_median if global_median is not None else 350.0
        lane_col = out["Regional_Lane"] if "Regional_Lane" in out.columns else pd.Series("Unknown", index=out.index)
        expected_lane_rate = lane_col.map(lane_medians).fillna(g_med)
        out["Lane_Rate_Index"] = (buy_rate / expected_lane_rate).clip(lower=0.2, upper=5.0)
    else:
        out["Lane_Rate_Index"] = 1.0

    # Frequency encode high-cardinality destination port
    if port_frequencies is not None and "Destination_Port" in out.columns:
        out["Dest_Port_Freq"] = out["Destination_Port"].map(port_frequencies).fillna(1).astype(float)
    else:
        out["Dest_Port_Freq"] = 1.0

    return out


def compute_cohort_quantile_table(df_won: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute empirical won-deal margin quantiles (P25, P50, P75, P90) across
    hierarchical trade corridors, weight slabs, and commodities.
    """
    def _calc_q(series: pd.Series) -> Dict[str, float]:
        return {
            "p25": round(float(series.quantile(0.25)), 2),
            "p50": round(float(series.quantile(0.50)), 2),
            "p75": round(float(series.quantile(0.75)), 2),
            "p90": round(float(series.quantile(0.90)), 2),
            "count": int(len(series))
        }

    global_q = _calc_q(df_won["Margin_Percentage"])

    lane_q = {}
    for lane, grp in df_won.groupby("Regional_Lane"):
        if len(grp) >= 8:
            lane_q[str(lane)] = _calc_q(grp["Margin_Percentage"])

    lane_tier_q = {}
    for (lane, tier), grp in df_won.groupby(["Regional_Lane", "Weight_Tier"]):
        if len(grp) >= 6:
            lane_tier_q[f"{lane}|{tier}"] = _calc_q(grp["Margin_Percentage"])

    full_cohort_q = {}
    for (lane, tier, comm), grp in df_won.groupby(["Regional_Lane", "Weight_Tier", "Commodity_Group"]):
        if len(grp) >= 4:
            full_cohort_q[f"{lane}|{tier}|{comm}"] = _calc_q(grp["Margin_Percentage"])

    return {
        "global": global_q,
        "lane": lane_q,
        "lane_tier": lane_tier_q,
        "full_cohort": full_cohort_q
    }


def train_and_evaluate(
    cleaned_data_path: str = "data/processed/Air_Export_Pricing_Combined_ML.csv",
    won_benchmark_path: Optional[str] = None,
    output_model_path: str = "models/freight_margin_recommender.joblib",
    plot_output_path: str = "docs/evaluation_plots.png"
):
    print("=" * 70)
    print(f"AIR EXPORT PRICING: TWO-STAGE PRODUCTION MODEL TRAINING ({MODEL_VERSION})")
    print("=" * 70)

    if not os.path.exists(cleaned_data_path):
        raise FileNotFoundError(f"Cleaned dataset not found at: {cleaned_data_path}")

    df_all = pd.read_csv(cleaned_data_path)
    df_all, dropped = filter_worked_quotes(df_all)
    print(f"[*] Dropped {len(dropped):,} unworked lost quotes; kept {len(df_all):,}")

    won_mask = df_all["is_won"] == 1
    df_won = df_all[won_mask].copy()
    print(f"    Verified Won Deals : {len(df_won):,} ({won_mask.mean() * 100.0:.1f}%)")
    print(f"    Verified Lost Deals: {(~won_mask).sum():,} ({(~won_mask).mean() * 100.0:.1f}%)")

    # 1. Compute lane medians and port frequency maps strictly from training data
    print("[*] Computing lane rate index reference benchmarks...")
    raw_buy_rate_won = (df_won["Total_Buy_INR"] / df_won["Chargeable_Weight_Kg"]).clip(10.0, 8000.0)
    df_won["Buy_Rate_Per_Kg"] = raw_buy_rate_won
    lane_medians = df_won.groupby("Regional_Lane")["Buy_Rate_Per_Kg"].median().to_dict()
    global_median = float(df_won["Buy_Rate_Per_Kg"].median())
    port_frequencies = df_all["Destination_Port"].value_counts().to_dict()

    print("[*] Computing empirical cohort margin quantiles...")
    cohort_table = compute_cohort_quantile_table(df_won)

    # 2. Enrich full dataset with rate features
    df_all = compute_rate_features(
        df_all,
        lane_medians=lane_medians,
        global_median=global_median,
        port_frequencies=port_frequencies
    )
    df_won = df_all[won_mask].copy()

    # 3. Fit categorical encoder for native categorical features
    print(f"[*] Fitting categorical encoder for {len(NATIVE_CAT_COLS)} native discrete features...")
    known_categories = {}
    for col in NATIVE_CAT_COLS:
        cats = set(df_all[col].astype(str).unique())
        if col == "Commodity_Group":
            cats.update(["Dangerous Goods", "Valuables", "Live Animals", "Other"])
        known_categories[col] = sorted(list(cats))

    encoder = OrdinalEncoder(
        categories=[known_categories[c] for c in NATIVE_CAT_COLS],
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )
    encoder.fit(df_all[NATIVE_CAT_COLS].astype(str))

    cat_indices = [i for i, c in enumerate(FEATURE_COLS) if c in NATIVE_CAT_COLS]

    X_all = df_all[FEATURE_COLS].copy()
    X_all[NATIVE_CAT_COLS] = encoder.transform(X_all[NATIVE_CAT_COLS].astype(str))

    X_won = X_all[won_mask].copy()
    y_won = df_won["Margin_Percentage"].clip(MIN_BENCHMARK_MARGIN, MAX_BENCHMARK_MARGIN)

    # 4. Stage 1A: 5-Fold Cross-Validation for Out-of-Fold (OOF) Benchmark Margins (TR-2)
    print("\n[*] Evaluating Stage 1A Benchmark Regressor with 5-Fold Cross-Validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_regressor = HistGradientBoostingRegressor(
        categorical_features=cat_indices,
        max_iter=200,
        min_samples_leaf=15,
        l2_regularization=2.0,
        random_state=42
    )

    y_won_cv_pred = cross_val_predict(cv_regressor, X_won, y_won, cv=kf)
    cv_mae = mean_absolute_error(y_won, y_won_cv_pred)
    cv_rmse = np.sqrt(mean_squared_error(y_won, y_won_cv_pred))
    cv_r2 = r2_score(y_won, y_won_cv_pred)

    print(f"    Stage 1A 5-Fold CV MAE  : {cv_mae:.3f}% margin")
    print(f"    Stage 1A 5-Fold CV RMSE : {cv_rmse:.3f}% margin")
    print(f"    Stage 1A 5-Fold CV R²   : {cv_r2:.4f}")

    # Train production benchmark regressor on full won dataset
    benchmark_reg = HistGradientBoostingRegressor(
        categorical_features=cat_indices,
        max_iter=200,
        min_samples_leaf=15,
        l2_regularization=2.0,
        random_state=42
    )
    benchmark_reg.fit(X_won, y_won)

    # Compute honest OOF Margin_Ratio for won deals; predict for lost deals
    pred_lost = benchmark_reg.predict(X_all[~won_mask]).clip(MIN_BENCHMARK_MARGIN, MAX_BENCHMARK_MARGIN)

    df_all.loc[won_mask, "Margin_Ratio"] = (
        df_all.loc[won_mask, "Margin_Percentage"] / y_won_cv_pred.clip(MIN_BENCHMARK_MARGIN, MAX_BENCHMARK_MARGIN)
    ).clip(0.1, 5.0)

    df_all.loc[~won_mask, "Margin_Ratio"] = (
        df_all.loc[~won_mask, "Margin_Percentage"] / pred_lost
    ).clip(0.1, 5.0)

    mean_ratio_won = float(df_all.loc[won_mask, "Margin_Ratio"].mean())
    std_ratio_won = float(df_all.loc[won_mask, "Margin_Ratio"].std())
    mean_ratio_lost = float(df_all.loc[~won_mask, "Margin_Ratio"].mean())
    print(f"\n[*] §1.2 Honest OOF Margin_Ratio Distribution:")
    print(f"    - Won Deals OOF Margin_Ratio Mean : {mean_ratio_won:.3f} (Std: {std_ratio_won:.3f})")
    print(f"    - Lost Deals Margin_Ratio Mean    : {mean_ratio_lost:.3f}")

    # 5. Stage 1B: Calibrated Elasticity Classifier
    clf_features = FEATURE_COLS + ["Margin_Ratio"]
    clf_cat_indices = [i for i, c in enumerate(clf_features) if c in NATIVE_CAT_COLS]

    X_clf = df_all[clf_features].copy()
    X_clf[NATIVE_CAT_COLS] = encoder.transform(X_clf[NATIVE_CAT_COLS].astype(str))
    y_clf = df_all["is_won"]

    # Temporal split replaces random split (last ~20% of months)
    cut = "2026-08"
    if "Year_Month" in df_all.columns:
        tr_mask = df_all["Year_Month"] < cut
        X_train, X_test = X_clf[tr_mask], X_clf[~tr_mask]
        y_train, y_test = y_clf[tr_mask], y_clf[~tr_mask]
        print(f"\n[*] Training Stage 1B Calibrated Classifier with Temporal Split at {cut} (Train: {len(X_train):,}, Test: {len(X_test):,})...")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_clf, y_clf, test_size=0.20, random_state=42, stratify=y_clf
        )
        print(f"\n[*] Training Stage 1B Calibrated Classifier with Random Split (Train: {len(X_train):,}, Test: {len(X_test):,})...")

    def make_monotonic_clf(feature_cols, cat_idx):
        mono = [0] * len(feature_cols)
        if "Margin_Ratio" in feature_cols:
            mono[feature_cols.index("Margin_Ratio")] = -1
        return HistGradientBoostingClassifier(
            categorical_features=cat_idx,
            monotonic_cst=mono,
            max_iter=200,
            min_samples_leaf=15,
            learning_rate=0.04,
            random_state=42
        )

    base_clf = make_monotonic_clf(clf_features, clf_cat_indices)
    calibrated_clf = CalibratedClassifierCV(estimator=base_clf, method="sigmoid", cv=3)
    calibrated_clf.fit(X_train, y_train)

    test_probs = calibrated_clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, test_probs)
    brier = brier_score_loss(y_test, test_probs)

    print("\n" + "-" * 50)
    print("STAGE 1B TEST EVALUATION METRICS")
    print("-" * 50)
    print(f"ROC-AUC Score : {auc:.4f}")
    print(f"Brier Score   : {brier:.4f}")

    # Margin-bucket calibration verification (TR-5)
    print("\n[*] Margin Bucket Calibration Verification:")
    test_df = df_all.loc[y_test.index].copy()
    test_df["pred_prob"] = test_probs

    buckets = [(0, 5), (5, 10), (10, 15), (15, 25), (25, 100)]
    bucket_records = []
    for b_low, b_high in buckets:
        sub = test_df[(test_df["Margin_Percentage"] >= b_low) & (test_df["Margin_Percentage"] < b_high)]
        if len(sub) > 0:
            obs = sub["is_won"].mean()
            pred = sub["pred_prob"].mean()
            cal_err = abs(obs - pred)
            bucket_records.append({"bucket": f"{b_low}-{b_high}%", "n": len(sub), "obs": obs, "pred": pred, "err": cal_err})
            print(f"    - Bucket [{b_low:>2}% - {b_high:>3}%]: n={len(sub):<4}, Observed={obs*100.0:>5.1f}%, Predicted={pred*100.0:>5.1f}%, Error={cal_err*100.0:>4.1f}%")

    # Fit final production classifier on full dataset
    print("\n[*] Fitting final production classifier on complete dataset...")
    final_base_clf = make_monotonic_clf(clf_features, clf_cat_indices)
    final_calibrated_clf = CalibratedClassifierCV(estimator=final_base_clf, method="sigmoid", cv=3)
    final_calibrated_clf.fit(X_clf, y_clf)

    # 6. Evaluation Plots
    os.makedirs(os.path.dirname(plot_output_path), exist_ok=True)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: ROC Curve
    fpr, tpr, _ = roc_curve(y_test, test_probs)
    ax1.plot(fpr, tpr, color="#23c2f2", lw=2.5, label=f"Calibrated GBDT (AUC = {auc:.3f})")
    ax1.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax1.set_title("Stage 1B: ROC Curve (Win Classifier)")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Probability Calibration Curve
    prob_true, prob_pred = calibration_curve(y_test, test_probs, n_bins=8, strategy="uniform")
    ax2.plot(prob_pred, prob_true, marker="o", color="#a7cf45", lw=2, label="Calibrated Curve")
    ax2.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    ax2.set_title("Stage 1B: Probability Calibration")
    ax2.set_xlabel("Mean Predicted Probability")
    ax2.set_ylabel("Observed Win Rate")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    # Panel 3: Stage 1A CV Residuals
    residuals = y_won - y_won_cv_pred
    ax3.hist(residuals, bins=25, color="#f28e2b", edgecolor="black", alpha=0.7)
    ax3.axvline(0, color="black", linestyle="--", lw=1.5)
    ax3.set_title(f"Stage 1A: CV Residuals (MAE: {cv_mae:.2f}%)")
    ax3.set_xlabel("Error (Actual - Predicted Margin %)")
    ax3.set_ylabel("Frequency")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_output_path, dpi=200)
    plt.close()
    print(f"[✓] Evaluation diagnostic charts saved to: {plot_output_path}")

    # 7. Serialize complete production payload with full metadata
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    won_worked = df_all[df_all["is_won"] == 1].copy()
    export_payload = {
        "benchmark_reg": benchmark_reg,
        "calibrated_clf": final_calibrated_clf,
        "encoder": encoder,
        "feature_cols": FEATURE_COLS,
        "native_cat_cols": NATIVE_CAT_COLS,
        "lane_medians": lane_medians,
        "global_median": global_median,
        "port_frequencies": port_frequencies,
        "cohort_table": cohort_table,
        "margin_tables": mf.build_tables(won_worked),
        "training_window": [
            str(df_all["Year_Month"].min()) if "Year_Month" in df_all.columns else "2026-01",
            str(df_all["Year_Month"].max()) if "Year_Month" in df_all.columns else "2026-09"
        ],
        "n_unworked_lost_dropped": int(len(dropped)),
        "sklearn_version": sklearn.__version__,
        "model_version": MODEL_VERSION,
        "trained_at": datetime.datetime.now().isoformat(),
        "stage1a_metrics": {
            "cv_mae": float(cv_mae),
            "cv_rmse": float(cv_rmse),
            "cv_r2": float(cv_r2)
        },
        "stage1b_metrics": {
            "test_auc": float(auc),
            "test_brier": float(brier)
        },
        "bucket_calibration": bucket_records
    }
    joblib.dump(export_payload, output_model_path)
    print(f"[✓] Serialized production engine ({MODEL_VERSION}) to: {output_model_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Two-Stage Margin Recommender")
    parser.add_argument("--cleaned", type=str, default="data/processed/Air_Export_Pricing_Combined_ML.csv")
    parser.add_argument("--won", type=str, default=None)
    parser.add_argument("--out_model", type=str, default="models/freight_margin_recommender.joblib")
    parser.add_argument("--out_plots", type=str, default="docs/evaluation_plots.png")
    args = parser.parse_args()

    train_and_evaluate(args.cleaned, args.won, args.out_model, args.out_plots)
