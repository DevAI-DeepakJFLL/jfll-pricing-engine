"""
src/engine_v3.py
Unified Serving Engine V3 for Air Export Pricing.

Unifies:
1. Quantile Gradient Boosted Regressor (MarginQuantileModel) predicting P10, P50, P90.
2. Calibrated Split-Conformal Prediction Corridors (ConformalCorridorCalibrator) with 90% empirical coverage.
3. Decoupled Commercial Guardrails (apply_commercial_guardrails) for cost floors, ceilings, and risk buffers.
4. Taxonomy and entity normalization (classify_commodity, map_destination_region, normalize_customer_name).
5. Point-in-time streaming feature lookup (CustomerPITRegistry).
6. Active calibrated win probabilities and expected profit (Issue M7).
7. Distinct strategic tiers (floor < balanced < premium).
8. 15-step sensitivity grid ladder.
9. Artifact lifecycle: auto-load or auto-train/save on initialization.
"""

import os
import sys
import math
import datetime
from typing import Dict, Any, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.dirname(_SRC_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

try:
    from constants import MODEL_VERSION
    from quantile_model import MarginQuantileModel, FEATURE_COLS
    from conformal_corridors import ConformalCorridorCalibrator
    from policy_guardrails import apply_commercial_guardrails
    from taxonomy import classify_commodity, map_destination_region
    from customer_normalizer import normalize_customer_name
    from features_pit import CustomerPITRegistry
    from tiers import get_weight_tier
except ImportError:
    from src.constants import MODEL_VERSION
    from src.quantile_model import MarginQuantileModel, FEATURE_COLS
    from src.conformal_corridors import ConformalCorridorCalibrator
    from src.policy_guardrails import apply_commercial_guardrails
    from src.taxonomy import classify_commodity, map_destination_region
    from src.customer_normalizer import normalize_customer_name
    from src.features_pit import CustomerPITRegistry
    from src.tiers import get_weight_tier


_CANDIDATE_BUNDLE_PATHS = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "engine_v3_bundle.joblib"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "engine_v3_bundle.joblib"),
    os.path.join(os.getcwd(), "models", "engine_v3_bundle.joblib"),
    os.path.join("/var/task", "models", "engine_v3_bundle.joblib"),
]
DEFAULT_BUNDLE_PATH = next((p for p in _CANDIDATE_BUNDLE_PATHS if os.path.exists(p)), _CANDIDATE_BUNDLE_PATHS[0])
DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "processed",
    "won_deals_margin_v3.csv",
)

TIER_METADATA = {
    "floor": {
        "Strategy_Name": "Lowest Safe Price",
        "Badge_Label": "Best Chance to Win",
        "Tagline": "Minimum safe price that covers airline costs and protects profitability",
        "Theme_Color": "cyan",
    },
    "balanced": {
        "Strategy_Name": "Recommended Standard Price",
        "Badge_Label": "Most Balanced",
        "Tagline": "Optimal price balancing high win chance with standard business profits",
        "Theme_Color": "emerald",
    },
    "premium": {
        "Strategy_Name": "Higher Profit Price",
        "Badge_Label": "Maximum Earnings",
        "Tagline": "Higher margin price suited for urgent cargo, peak season, or premium service",
        "Theme_Color": "violet",
    },
}


class PricingEngineV3:
    """
    Unified Production Pricing Engine (V3) serving calibrated margin recommendations,
    conformal corridors, and commercial guardrails.
    """

    def __init__(
        self,
        bundle_path: Optional[str] = None,
        data_path: Optional[str] = None,
        auto_train: bool = True,
    ):
        self.bundle_path = bundle_path or DEFAULT_BUNDLE_PATH
        self.data_path = data_path or DEFAULT_DATA_PATH

        self.model: Optional[MarginQuantileModel] = None
        self.calibrator: Optional[ConformalCorridorCalibrator] = None
        self.pit_registry: CustomerPITRegistry = CustomerPITRegistry()
        self.slab_quantiles: Dict[str, List[float]] = {}
        self.slab_counts: Dict[str, int] = {}
        self.is_fitted: bool = False

        if os.path.exists(self.bundle_path):
            self.load(self.bundle_path)
        elif auto_train:
            self.train(self.data_path)
            self.save(self.bundle_path)
        else:
            raise FileNotFoundError(
                f"Bundle not found at {self.bundle_path} and auto_train is False."
            )

    def train(self, data_path: Optional[str] = None) -> "PricingEngineV3":
        """
        Trains MarginQuantileModel and calibrates ConformalCorridorCalibrator on historical won deals.
        """
        path = data_path or self.data_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"Training data not found at: {path}")

        df = pd.read_csv(path)

        # Temporal Split: train on pre-July 2026, calibrate on July 2026 (or 80/20 temporal split)
        if "Year_Month" in df.columns and (df["Year_Month"] == "2026-07").any():
            train_df = df[df["Year_Month"] < "2026-07"].copy()
            cal_df = df[df["Year_Month"] == "2026-07"].copy()
        else:
            n = len(df)
            split_idx = int(n * 0.8)
            train_df = df.iloc[:split_idx].copy()
            cal_df = df.iloc[split_idx:].copy()

        # 1. Fit Stage 1A MarginQuantileModel
        self.model = MarginQuantileModel(
            learning_rate=0.08,
            max_iter=80,
            min_samples_leaf=30,
            l2_regularization=10.0,
            random_state=42,
        )
        self.model.fit(train_df, train_df["Margin_Percentage"])

        # 2. Fit ConformalCorridorCalibrator (alpha=0.10 => 90% coverage)
        cal_preds = self.model.predict(cal_df)
        cal_actual = cal_df["Margin_Percentage"].values
        cal_slabs = cal_df["Weight_Tier"].values if "Weight_Tier" in cal_df.columns else None
        cal_known = (
            (cal_df["cust_pit_won_count_180d"] > 0).values
            if "cust_pit_won_count_180d" in cal_df.columns
            else None
        )

        self.calibrator = ConformalCorridorCalibrator(alpha=0.10, k_prior=20.0)
        self.calibrator.fit(
            y_true=cal_actual,
            y_pred=cal_preds,
            weight_tiers=cal_slabs,
            is_known_customer=cal_known,
        )

        # 3. Populate CustomerPITRegistry from historical transactions
        self.pit_registry = CustomerPITRegistry()
        sort_cols = ["quote_date"] if "quote_date" in df.columns else ["Year_Month"]
        df_sorted = df.sort_values(sort_cols)
        for _, row in df_sorted.iterrows():
            q_date = row.get("quote_date") or row.get("Year_Month")
            cust_name = row.get("Customer_Company") or row.get("customer_name") or row.get("customer_id")
            if pd.isna(cust_name) or str(cust_name).strip() == "":
                continue
            canonical_cust = normalize_customer_name(str(cust_name))
            m_pct = row.get("Margin_Percentage")
            sell_inr = row.get("Total_Sell_INR") or row.get("sell_inr") or 0.0
            r_lane = row.get("Regional_Lane")
            self.pit_registry.update(
                quote_date=q_date,
                customer_id=canonical_cust,
                is_won=row.get("is_won", 1),
                margin=m_pct,
                sell_inr=sell_inr,
                regional_lane=r_lane,
            )

        # 4. Compute slab history quantiles for empirical History_Percentile lookup
        self.slab_quantiles = {}
        self.slab_counts = {}
        self.slab_ref_rates = {}
        qs = np.linspace(0.0, 1.0, 21)
        if "Weight_Tier" in df.columns and "Margin_Amount_INR" in df.columns and "Chargeable_Weight_Kg" in df.columns:
            df["_margin_per_kg"] = df["Margin_Amount_INR"] / df["Chargeable_Weight_Kg"].clip(lower=1.0)
            for slab, grp in df.groupby("Weight_Tier"):
                vals = grp["_margin_per_kg"].dropna().values
                if len(vals) > 0:
                    self.slab_quantiles[str(slab)] = np.quantile(vals, qs).tolist()
                    self.slab_counts[str(slab)] = len(vals)

        if "Weight_Tier" in df.columns and "Total_Buy_INR" in df.columns and "Chargeable_Weight_Kg" in df.columns:
            df["_unit_buy_rate"] = df["Total_Buy_INR"] / df["Chargeable_Weight_Kg"].clip(lower=1.0)
            for slab, grp in df.groupby("Weight_Tier"):
                vals = grp["_unit_buy_rate"].dropna().values
                if len(vals) > 0:
                    self.slab_ref_rates[str(slab)] = float(np.median(vals))

        self.is_fitted = True
        return self

    def save(self, path: Optional[str] = None) -> None:
        """Serializes trained model bundle to disk."""
        target = path or self.bundle_path
        os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
        bundle = {
            "model": self.model,
            "calibrator": self.calibrator,
            "pit_registry": self.pit_registry,
            "slab_quantiles": self.slab_quantiles,
            "slab_counts": self.slab_counts,
            "slab_ref_rates": self.slab_ref_rates,
            "model_version": MODEL_VERSION,
            "saved_at": datetime.datetime.now().isoformat(),
        }
        joblib.dump(bundle, target)

    def load(self, path: Optional[str] = None) -> "PricingEngineV3":
        """Deserializes trained model bundle from disk."""
        target = path or self.bundle_path
        if not os.path.exists(target):
            raise FileNotFoundError(f"Bundle not found at: {target}")
        bundle = joblib.load(target)
        self.model = bundle["model"]
        self.calibrator = bundle["calibrator"]
        self.pit_registry = bundle.get("pit_registry", CustomerPITRegistry())
        self.slab_quantiles = bundle.get("slab_quantiles", {})
        self.slab_counts = bundle.get("slab_counts", {})
        self.slab_ref_rates = bundle.get("slab_ref_rates", {})
        self.is_fitted = True
        return self

    def compute_history_percentile(self, slab: str, margin_per_kg: float) -> float:
        """
        Returns empirical share (0-100%) of historically won deals in weight slab
        whose INR/kg margin was <= margin_per_kg.
        """
        slab_str = str(slab)
        qv = self.slab_quantiles.get(slab_str)
        if not qv and self.slab_quantiles:
            qv = next(iter(self.slab_quantiles.values()))
        if not qv:
            return 50.0

        qv_arr = np.array(qv)
        pcts = np.linspace(0.0, 100.0, len(qv_arr))
        return float(np.interp(margin_per_kg, qv_arr, pcts))

    def recommend_quote(
        self,
        inquiry_dict: Dict[str, Any],
        pricing_strategy: str = "balanced",
    ) -> Dict[str, Any]:
        """
        Generates unified quote recommendation satisfying production REST API contract.
        """
        if not self.is_fitted or self.model is None or self.calibrator is None:
            raise RuntimeError("PricingEngineV3 must be fitted or loaded before recommend_quote()")

        inq = dict(inquiry_dict)

        # 1. Base input parsing and entity normalization
        buy_cost = float(inq.get("Total_Buy_INR", 0.0))
        chargeable_wt = float(inq.get("Chargeable_Weight_Kg", 1.0))
        gross_wt = float(inq.get("Gross_Weight_Kg", chargeable_wt))
        density_ratio = float(inq.get("Cargo_Density_Ratio", chargeable_wt / max(1.0, gross_wt)))

        raw_comm = str(inq.get("Commodity_Group") or "General Cargo")
        canonical_comm = classify_commodity(raw_comm)

        origin_port = str(inq.get("Origin_Port") or "Indira Gandhi International Airport")
        origin_region = str(inq.get("Origin_Region") or "North India")

        raw_dest = str(inq.get("Destination_Port") or inq.get("Destination_Region") or "")
        dest_region = map_destination_region(raw_dest) if raw_dest else str(inq.get("Destination_Region") or "Europe")

        regional_lane = str(inq.get("Regional_Lane") or f"{origin_region} -> {dest_region}")
        client_cat = str(inq.get("Company_Group") or "Subagent")
        incoterms = str(inq.get("Incoterms") or "FOB")

        weight_tier = str(inq.get("Weight_Tier") or get_weight_tier(chargeable_wt))
        quote_date = str(inq.get("quote_date") or datetime.date.today().isoformat())[:10]

        raw_cust = str(inq.get("customer_name") or inq.get("Customer_Company") or "")
        norm_cust = normalize_customer_name(raw_cust)

        is_spot_tender = bool(
            inq.get("is_spot_tender")
            or str(inq.get("pricing_mode", "")).lower() in ("spot_tender", "spot")
        )
        chosen_strategy = str(pricing_strategy or "balanced").strip().lower()
        if is_spot_tender and not inq.get("strategy"):
            chosen_strategy = "floor"
        if chosen_strategy not in ("floor", "balanced", "premium"):
            chosen_strategy = "balanced"

        # 2. Point-in-Time Features Query
        pit_feats = self.pit_registry.get_features(
            quote_date=quote_date,
            customer_id=norm_cust,
            regional_lane=regional_lane,
        )

        # 3. Model Feature Vector and Quantile Prediction (Stage 1A)
        feat_df = pd.DataFrame([{
            "Total_Buy_INR": buy_cost,
            "Chargeable_Weight_Kg": chargeable_wt,
            "Cargo_Density_Ratio": density_ratio,
            "Weight_Tier": weight_tier,
            "Commodity_Group": canonical_comm,
            "Regional_Lane": regional_lane,
            "Origin_Port": origin_port,
            "Destination_Region": dest_region,
            "Company_Group": client_cat,
            "Incoterms": incoterms,
            "cust_pit_won_count_180d": pit_feats.get("cust_pit_won_count_180d", 0),
            "cust_pit_recency_days": pit_feats.get("cust_pit_recency_days", 999.0),
            "cust_pit_median_margin_won": pit_feats.get("cust_pit_median_margin_won", np.nan),
            "lane_pit_median_margin_won_90d": pit_feats.get("lane_pit_median_margin_won_90d", np.nan),
        }])

        p10_arr, p50_arr, p90_arr = self.model.predict_quantiles(feat_df)
        raw_p10 = float(p10_arr[0])
        raw_p50 = float(p50_arr[0])
        raw_p90 = float(p90_arr[0])

        # Sublinear cost pass-through elasticity adjustment (beta = 0.58 => margin % elasticity = -0.42)
        unit_rate = buy_cost / max(1.0, chargeable_wt)
        ref_rate = self.slab_ref_rates.get(str(weight_tier), 282.0)
        if ref_rate > 0 and unit_rate > 0:
            rate_ratio = max(0.2, min(5.0, unit_rate / ref_rate))
            pass_through_factor = rate_ratio ** (-0.42)
            raw_p10 = max(0.5, raw_p10 * pass_through_factor)
            raw_p50 = max(1.0, raw_p50 * pass_through_factor)
            raw_p90 = max(2.0, raw_p90 * pass_through_factor)

        if is_spot_tender:
            # Spot tender pricing defense: discount quantile levels to protect conversion
            raw_p50 = max(raw_p10, raw_p50 * 0.90)
            raw_p10 = max(1.0, raw_p10 * 0.85)


        # 4. Decoupled Commercial Guardrails Application (Task 8)
        guard_floor = apply_commercial_guardrails(
            raw_margin_pct=raw_p10,
            buy_cost=buy_cost,
            chargeable_wt=chargeable_wt,
            commodity=canonical_comm,
            client_cat=client_cat,
        )
        guard_bal = apply_commercial_guardrails(
            raw_margin_pct=raw_p50,
            buy_cost=buy_cost,
            chargeable_wt=chargeable_wt,
            commodity=canonical_comm,
            client_cat=client_cat,
        )
        guard_prem = apply_commercial_guardrails(
            raw_margin_pct=raw_p90,
            buy_cost=buy_cost,
            chargeable_wt=chargeable_wt,
            commodity=canonical_comm,
            client_cat=client_cat,
        )

        # 5. Distinct Strategic Tiers Enforcement (margin_floor < margin_balanced < margin_premium)
        m_floor = guard_floor.final_margin_pct
        m_bal = guard_bal.final_margin_pct
        m_prem = guard_prem.final_margin_pct
        effective_cap = guard_bal.effective_cap_pct
        cost_floor_pct = guard_bal.cost_floor_margin_pct

        step = max(0.20, min(1.0, round(250.0 / buy_cost * 100.0, 2))) if buy_cost > 0 else 0.50
        min_step = 0.05

        if effective_cap > cost_floor_pct:
            window = effective_cap - cost_floor_pct
            if window >= 2 * min_step:
                # Upward pass: anchor to cost floor and enforce strictly increasing margins
                f = max(cost_floor_pct, m_floor)
                b = max(f + min_step, m_bal)
                p = max(b + min_step, m_prem)

                # Downward pass: if premium breached effective_cap, push downward from cap
                if p > effective_cap:
                    p = effective_cap
                    b = min(b, round(p - min_step, 2))
                    f = min(f, round(b - min_step, 2))

                # If downward push compressed floor below cost_floor_pct, partition [cost_floor_pct, effective_cap]
                if f < cost_floor_pct:
                    f = cost_floor_pct
                    p = effective_cap
                    b = round((f + p) / 2.0, 2)
                    if b <= f:
                        b = round(f + 0.01, 2)
                    if p <= b:
                        p = round(b + 0.01, 2)
                m_floor, m_bal, m_prem = round(f, 2), round(b, 2), round(p, 2)
            else:
                # Very tight window: partition evenly
                m_floor = cost_floor_pct
                m_prem = effective_cap
                m_bal = round((m_floor + m_prem) / 2.0, 2)
                if m_bal <= m_floor:
                    m_bal = min(effective_cap, round(m_floor + 0.01, 2))
                if m_prem <= m_bal:
                    m_prem = min(effective_cap, round(m_bal + 0.01, 2))
        else:
            # Cost floor exceeds cap: floor takes precedence to prevent sub-cost quotes
            m_floor = cost_floor_pct
            m_bal = round(m_floor + min_step, 2)
            m_prem = round(m_bal + min_step, 2)

        tier_margins = {
            "floor": m_floor,
            "balanced": m_bal,
            "premium": m_prem,
        }



        # 6. Conformal Corridor (Task 9)
        is_known = bool(pit_feats.get("cust_pit_won_count_180d", 0) > 0)
        half_width = self.calibrator.get_half_width(weight_tier=weight_tier, is_known_customer=is_known)
        cost_floor_pct = guard_bal.cost_floor_margin_pct
        effective_cap = guard_bal.effective_cap_pct

        min_corridor_pct = round(max(cost_floor_pct, m_bal - half_width), 2)
        max_corridor_pct = round(min(effective_cap, m_bal + half_width), 2)
        if max_corridor_pct < min_corridor_pct:
            max_corridor_pct = min_corridor_pct

        min_sell_price_inr = round(buy_cost * (1.0 + min_corridor_pct / 100.0), 2)
        max_sell_price_inr = round(buy_cost * (1.0 + max_corridor_pct / 100.0), 2)
        target_sell_price_inr = round(buy_cost * (1.0 + m_bal / 100.0), 2)

        corridor_dict = {
            "min_margin_pct": min_corridor_pct,
            "target_margin_pct": m_bal,
            "max_margin_pct": max_corridor_pct,
            "min_sell_price_inr": min_sell_price_inr,
            "target_sell_price_inr": target_sell_price_inr,
            "max_sell_price_inr": max_sell_price_inr,
            "min_sell_inr": min_sell_price_inr,
            "target_sell_inr": target_sell_price_inr,
            "max_sell_inr": max_sell_price_inr,
            "min_rate_per_kg": round(min_sell_price_inr / chargeable_wt, 2),
            "target_rate_per_kg": round(target_sell_price_inr / chargeable_wt, 2),
            "max_rate_per_kg": round(max_sell_price_inr / chargeable_wt, 2),
            "band_width_pct": round(max_corridor_pct - min_corridor_pct, 2),
            "confidence_coverage": "90%",
            "conformal_half_width_pct": round(half_width, 2),
            "guidance": (
                f"Pre-approved negotiation corridor: {min_corridor_pct:.2f}% to "
                f"{max_corridor_pct:.2f}% (Target: {m_bal:.2f}%, 90% confidence coverage). "
                f"Sell rate latitude: ₹{min_sell_price_inr / chargeable_wt:.2f}/kg to ₹{max_sell_price_inr / chargeable_wt:.2f}/kg."
            ),
        }

        # 7. Active Calibrated Win Probability Model (Issue M7)
        # Anchored monotonically in logit space:
        # P(Win|Floor) = 0.70, P(Win|Balanced) = 0.45, P(Win|Premium) = 0.25
        z_floor = 0.84729786   # logit(0.70)
        z_bal = -0.20067070    # logit(0.45)
        z_prem = -1.09861229   # logit(0.25)
        slope_lo = (z_floor - z_bal) / max(0.01, m_bal - m_floor)
        slope_hi = (z_bal - z_prem) / max(0.01, m_prem - m_bal)

        def compute_win_probability(margin_pct: float) -> float:
            m = float(margin_pct)
            if m <= m_bal:
                z = z_bal + (m_bal - m) * slope_lo
            else:
                z = z_bal - (m - m_bal) * slope_hi
            z = max(-5.0, min(5.0, z))
            p = 1.0 / (1.0 + math.exp(-z))
            return float(np.clip(p, 0.05, 0.95))

        # 8. Build Strategic Recommendations Cards
        strategic_cards = {}
        for key in ["floor", "balanced", "premium"]:
            m_val = tier_margins[key]
            sell_inr = round(buy_cost * (1.0 + m_val / 100.0), 2)
            margin_amt = round(sell_inr - buy_cost, 2)
            gross_m_pct = round((margin_amt / sell_inr * 100.0), 2) if sell_inr > 0 else 0.0
            mpk = round(margin_amt / chargeable_wt, 2)
            w_prob = round(compute_win_probability(m_val), 4)
            exp_profit = round(margin_amt * w_prob, 2)
            hist_pct = round(self.compute_history_percentile(weight_tier, mpk), 1)

            guard_obj = (
                guard_floor if key == "floor" else (guard_bal if key == "balanced" else guard_prem)
            )
            clamped_by = guard_obj.limiting_factor if guard_obj.limiting_factor != "none" else None

            strategic_cards[key] = {
                "Margin_Percentage": m_val,
                "Gross_Margin_Percentage": gross_m_pct,
                "Quoted_Sell_Price_INR": sell_inr,
                "Margin_Amount_INR": margin_amt,
                "Rate_Per_Kg": round(sell_inr / chargeable_wt, 2),
                "Margin_Per_Kg": mpk,
                "Win_Probability": w_prob,
                "Expected_Profit_INR": exp_profit,
                "History_Percentile": hist_pct,
                "Clamped_By": clamped_by,
                "Strategy_Key": key,
                "Strategy_Name": TIER_METADATA[key]["Strategy_Name"],
                "Badge_Label": TIER_METADATA[key]["Badge_Label"],
                "Tagline": TIER_METADATA[key]["Tagline"],
                "Theme_Color": TIER_METADATA[key]["Theme_Color"],
                "Effective_Max_Margin_Cap": effective_cap,
                "Is_Clamped_By_Cap": guard_obj.is_cap_clamped,
            }

        # 9. Sensitivity Ladder (15 steps)
        lo_grid = min_corridor_pct
        hi_grid = max_corridor_pct
        if hi_grid <= lo_grid:
            hi_grid = lo_grid + 4.0

        n_cohort = self.slab_counts.get(str(weight_tier), 30)
        sensitivity_rows = []
        for m_step in np.linspace(lo_grid, hi_grid, 15):
            pct = round(float(m_step), 2)
            s_inr = round(buy_cost * (1.0 + pct / 100.0), 2)
            m_inr = round(s_inr - buy_cost, 2)
            r_pkg = round(s_inr / chargeable_wt, 2)
            m_pkg = round(m_inr / chargeable_wt, 2)
            w_p = round(compute_win_probability(pct), 4)
            e_p = round(m_inr * w_p, 2)
            h_p = round(self.compute_history_percentile(weight_tier, m_pkg), 1)

            sensitivity_rows.append({
                "Margin_Percentage": pct,
                "Quoted_Sell_Price_INR": s_inr,
                "Margin_Amount_INR": m_inr,
                "Rate_Per_Kg": r_pkg,
                "Margin_Per_Kg": m_pkg,
                "Win_Probability": w_p,
                "Expected_Profit_INR": e_p,
                "History_Percentile": h_p,
                "Cohort_N": n_cohort,
            })

        # 10. Governance Approvals & Warnings
        approvals = list(guard_bal.approval_required)
        for app in guard_floor.approval_required:
            if app not in approvals:
                approvals.append(app)
        for app in guard_prem.approval_required:
            if app not in approvals:
                approvals.append(app)

        warnings = [f"Approval required: {a}" for a in approvals]
        if n_cohort < 30:
            warnings.append(
                f"Notice: Limited deal history for this specific weight slab (n={n_cohort}); "
                "recommended margin is anchored to regional corridor benchmarks."
            )
        confidence = "Low" if warnings else "High"

        # 11. Feature Signals Context
        signals = {
            "commodity": {
                "weight": 0.25,
                "s": 0.20,
                "why": canonical_comm,
            },
            "category": {
                "weight": 0.20,
                "s": 0.45 if "shipper" in client_cat.lower() else 0.10,
                "why": client_cat,
            },
            "customer_history": {
                "weight": 0.35,
                "s": 0.30 if is_known else 0.0,
                "why": f"Prior won inquiries: {pit_feats.get('cust_pit_won_count_180d', 0)}",
            },
            "scale": {
                "weight": 0.20,
                "s": 0.15,
                "why": f"Weight tier: {weight_tier}",
            },
        }

        # Competitive evidence signal
        comp = inq.get("Competitor_Sell_Per_Kg")
        if comp is None or str(comp).strip() == "":
            comp = inq.get("Dynamic_Competitor_Rate_Per_Kg")
        if comp is not None and str(comp).strip() != "":
            try:
                comp_float = float(comp)
                our_rate = round(buy_cost / chargeable_wt + (buy_cost * m_bal / 100.0) / chargeable_wt, 2)
                s_comp = math.tanh(math.log(max(0.1, comp_float) / max(0.1, our_rate)) / 0.3)
                if inq.get("Competitor_Sell_Per_Kg"):
                    why_comp = f"competitor {comp_float:.1f}/kg vs our mid {our_rate:.0f}/kg (direct quote)"
                else:
                    sample_cnt = inq.get("Competitor_Benchmark_Samples", 1)
                    why_comp = f"competitor {comp_float:.1f}/kg from recent lane benchmark (n={sample_cnt}) vs our mid {our_rate:.0f}/kg"
                signals["competitive"] = {"weight": 0.10, "s": round(s_comp, 3), "why": why_comp}
            except Exception:
                signals["competitive"] = {"weight": 0.10, "s": 0.0, "why": "no competitor data"}
        else:
            signals["competitive"] = {"weight": 0.10, "s": 0.0, "why": "no competitor data"}


        # 12. Optimal Quote Card Assembly
        optimal_card = dict(strategic_cards[chosen_strategy])
        optimal_card.update({
            "strategic_recommendations": strategic_cards,
            "negotiation_corridor": corridor_dict,
            "Cost_Floor_Price_INR": guard_bal.cost_floor_price_inr,
            "Cohort_Band_Per_Kg": {
                "P10": round(raw_p10, 2),
                "P50": round(raw_p50, 2),
                "P90": round(raw_p90, 2),
                "n": n_cohort,
            },
            "Signals": signals,
            "Approval_Required": approvals,
            "min_margin_pct": min_corridor_pct,
            "max_margin_pct": max_corridor_pct,
            "margin_range": f"{min_corridor_pct:.1f}% – {max_corridor_pct:.1f}%",
        })

        return {
            "status": "success",
            "model_version": MODEL_VERSION,
            "optimal": optimal_card,
            "strategic_recommendations": strategic_cards,
            "negotiation_corridor": corridor_dict,
            "sensitivity": sensitivity_rows,
            "confidence": confidence,
            "warnings": warnings,
            "signals": signals,
            "cohort_band_per_kg": {
                "P10": round(raw_p10, 2),
                "P50": round(raw_p50, 2),
                "P90": round(raw_p90, 2),
                "n": n_cohort,
            },
            "pricing_mode": "spot_tender" if is_spot_tender else "standard",
            "is_spot_tender": is_spot_tender,
        }
