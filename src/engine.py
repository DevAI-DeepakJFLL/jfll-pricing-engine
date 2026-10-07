"""
================================================================================
Two-Stage Margin Recommendation & Optimization Engine (v2.3 Remediated)
================================================================================
Features:
1. Stage 1A: Market Benchmark Regression with Regional Trade Corridors & Rate Index.
2. Stage 1B: Calibrated Elasticity Classifier across empirical Margin Ratios.
3. Hybrid Framework with Cohort Margin Quantile Lookup & Hard Cost Floor:
     - Absolute Rs floors: Min AWB fee (Rs 1,500), Min rate/kg (Rs 3.0), Ops handling (Rs 850).
     - Commodity-specific risk buffers (DG +6%, Pharma +4%, Perishable +3.5%, etc.).
4. Active Strategy Thresholds (C-1, E-2):
     - 'floor'   : Must-Win / Volume (Win Prob >= 40%)
     - 'balanced': Recommended Balance (Win Prob >= 25%, Max Expected Profit in Corridor)
     - 'premium' : Capacity Constrained / Peak Yield (Win Prob >= 15%, High Margin Capture)
5. Deterministic Date Handling & Q4 Out-of-Window Alerting (E-7).
6. Strict Tier Differentiation Guarantee (No duplicate balanced/premium cards).
================================================================================
"""

import os
import sys
import datetime
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import pandas as pd

# Support both module and standalone execution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import (
    MIN_BENCHMARK_MARGIN,
    MAX_BENCHMARK_MARGIN,
    MIN_CANDIDATE_MARGIN,
    MAX_CANDIDATE_MARGIN,
    STRATEGY_THRESHOLDS,
    MODEL_VERSION,
    compute_cost_floor_price,
    get_commodity_max_margin,
    get_client_category_max_margin,
    get_category_commodity_max_margin,
    COMMODITY_MAX_MARGIN_CAPS,
    CLIENT_CATEGORY_MAX_MARGIN_CAPS
)
from preprocess import get_air_cargo_season, compute_cyclical_month, map_origin_region, map_dest_region
from tiers import get_weight_tier, map_account_tier


@dataclass
class RetentionGuardrailConfig:
    """Configuration for key account retention adjustment."""
    enabled: bool = True
    freq_threshold: int = 15
    margin_ratio_cap: float = 1.8
    penalty_factor: float = 0.85


@dataclass
class EngineConfig:
    """Optimization safeguard parameters."""
    min_viable_prob_floor: float = 0.05
    collapse_threshold: float = 0.015
    dynamic_prob_retention: float = 0.60
    strategy_blend_factor: float = 0.85


class MarginRecommendationEngine:
    """
    Two-Stage Hybrid Pricing Engine:
    Combines Market Benchmark Regression, Calibrated Price-Elasticity,
    Empirical Cohort Quantiles, and Hard Commercial Cost Floors.
    """

    def __init__(
        self,
        benchmark_reg,
        win_classifier,
        encoder,
        feature_cols: List[str],
        cat_cols: List[str],
        lane_medians: Optional[Dict[str, float]] = None,
        global_median: Optional[float] = None,
        port_frequencies: Optional[Dict[str, int]] = None,
        cohort_table: Optional[Dict[str, Any]] = None,
        engine_config: Optional[EngineConfig] = None,
        retention_config: Optional[RetentionGuardrailConfig] = None,
        strategy_thresholds: Optional[Dict[str, float]] = None
    ):
        self.benchmark_reg = benchmark_reg
        self.win_classifier = win_classifier
        self.encoder = encoder
        self.feature_cols = feature_cols
        self.cat_cols = cat_cols
        self.lane_medians = lane_medians or {}
        self.global_median = global_median or 350.0
        self.port_frequencies = port_frequencies or {}
        self.cohort_table = cohort_table or {}
        self.engine_config = engine_config or EngineConfig()
        self.retention_config = retention_config or RetentionGuardrailConfig()
        self.strategy_thresholds = strategy_thresholds or STRATEGY_THRESHOLDS

    def get_cohort_quantiles(
        self,
        regional_lane: str,
        weight_tier: str,
        commodity_group: str
    ) -> Dict[str, float]:
        """
        Hierarchical lookup for empirical won-deal margin quantiles:
          1. Exact cohort (Regional_Lane + Weight_Tier + Commodity_Group)
          2. Lane + Weight_Tier
          3. Regional_Lane
          4. Global baseline fallback
        """
        if not self.cohort_table:
            return {"p25": 3.5, "p50": 7.0, "p75": 14.5, "p90": 25.0}

        k3 = f"{regional_lane}|{weight_tier}|{commodity_group}"
        if k3 in self.cohort_table.get("full_cohort", {}):
            return self.cohort_table["full_cohort"][k3]

        k2 = f"{regional_lane}|{weight_tier}"
        if k2 in self.cohort_table.get("lane_tier", {}):
            return self.cohort_table["lane_tier"][k2]

        if regional_lane in self.cohort_table.get("lane", {}):
            return self.cohort_table["lane"][regional_lane]

        return self.cohort_table.get("global", {"p25": 3.5, "p50": 7.0, "p75": 14.5, "p90": 25.0})

    def _enrich_inquiry(self, inquiry_dict: dict) -> Tuple[dict, bool, str, Optional[str]]:
        """
        Enrich inquiry with deterministic temporal, geographic, and rate features.
        Detects out-of-window dates (Q4) and unseen categories.
        """
        enriched = dict(inquiry_dict)

        # 1. Deterministic Date Parsing (E-7)
        quote_date_str = enriched.get("quote_date")
        is_out_of_window = False
        warning = None
        confidence = "High"

        if quote_date_str:
            try:
                # Accept YYYY-MM-DD or ISO datetime
                clean_date_str = str(quote_date_str).split("T")[0].split(" ")[0]
                dt = datetime.datetime.strptime(clean_date_str, "%Y-%m-%d")
                month = dt.month
                day = dt.day
                day_of_week = dt.weekday()
            except (ValueError, TypeError):
                now = datetime.datetime.now()
                month, day, day_of_week = now.month, now.day, now.weekday()
        else:
            now = datetime.datetime.now()
            month = enriched.get("Quote_Month", now.month)
            day = enriched.get("Quote_Day", now.day)
            day_of_week = enriched.get("Quote_DayOfWeek", now.weekday())

        # Q4 out-of-window check (historical training data spans Jan-Sep, months 1-9)
        if month in [10, 11, 12]:
            is_out_of_window = True
            confidence = "Low"
            warning = "Quote date is in Q4 (Oct-Dec) global peak season, which has no historical training observations in 9-month dataset. Model confidence is degraded to Low."

        enriched["Quote_Month"] = int(month)
        enriched["Quote_DayOfWeek"] = int(day_of_week)
        enriched["Is_Month_End"] = 1 if day >= 25 else 0
        enriched["Air_Cargo_Season"] = get_air_cargo_season(month)

        sin_val, cos_val = compute_cyclical_month(month)
        enriched["Month_Sin"] = sin_val
        enriched["Month_Cos"] = cos_val

        # 2. Geographic Hierarchy Normalization
        orig = str(enriched.get("Origin_Port", "Indira Gandhi International Airport"))
        dest = str(enriched.get("Destination_Port", "Dubai"))
        orig_reg = enriched.get("Origin_Region") or map_origin_region(orig)
        dest_reg = enriched.get("Destination_Region") or map_dest_region(dest)
        enriched["Origin_Region"] = orig_reg
        enriched["Destination_Region"] = dest_reg
        enriched["Regional_Lane"] = f"{orig_reg} -> {dest_reg}"

        # 3. Physical Weights & Density
        gross = float(enriched.get("Gross_Weight_Kg", 1.0))
        ch_wt = float(enriched.get("Chargeable_Weight_Kg", gross))
        ch_wt = max(gross, ch_wt)
        ch_wt = np.ceil(ch_wt * 2.0) / 2.0
        enriched["Gross_Weight_Kg"] = gross
        enriched["Chargeable_Weight_Kg"] = ch_wt
        enriched["Cargo_Density_Ratio"] = round(min(1.0, max(0.1, gross / ch_wt)), 4)

        # 4. Rates & Log Transformations
        buy_total = float(enriched["Total_Buy_INR"])
        buy_rate = min(8000.0, max(10.0, buy_total / ch_wt))
        enriched["Buy_Rate_Per_Kg"] = buy_rate
        enriched["Log_Chargeable_Weight"] = np.log1p(ch_wt)

        expected_lane_rate = self.lane_medians.get(enriched["Regional_Lane"], self.global_median)
        enriched["Lane_Rate_Index"] = min(5.0, max(0.2, buy_rate / expected_lane_rate))

        enriched["Dest_Port_Freq"] = float(self.port_frequencies.get(dest, 1.0))

        # Standardize Tiers and Customer Company Group
        enriched["Weight_Tier"] = get_weight_tier(ch_wt)
        freq = float(enriched.get("Customer_Inquiry_Frequency", 10))
        enriched["Customer_Inquiry_Frequency"] = freq
        enriched["Customer_Tier"] = map_account_tier(freq)
        enriched.setdefault("Commodity_Group", "General Cargo")
        enriched.setdefault("Incoterms", "FOB")
        enriched.setdefault("Business_Vertical", "Air Export Forwarding")
        enriched.setdefault("Origin_Port", orig)
        enriched.setdefault("Destination_Port", dest)

        comp_grp = str(enriched.get("Company_Group", "Subagent"))
        if any(k in comp_grp.lower() for k in ["direct", "shipper", "consignee"]):
            enriched["Company_Group"] = "Shipper / Consignee"
        elif "overseas" in comp_grp.lower():
            enriched["Company_Group"] = "Overseas Agent"
        elif "nvocc" in comp_grp.lower() or "coloader" in comp_grp.lower():
            enriched["Company_Group"] = "NVOCC / Coloader"
        elif "custom" in comp_grp.lower() or "cha" in comp_grp.lower():
            enriched["Company_Group"] = "Custom Broker"
        elif "gsa" in comp_grp.lower():
            enriched["Company_Group"] = "GSA"
        elif "transport" in comp_grp.lower():
            enriched["Company_Group"] = "Transporter"
        else:
            enriched["Company_Group"] = "Subagent"

        # Check unseen categoricals (E-9)
        try:
            test_cats = [str(enriched.get(c, "Unknown")) for c in self.cat_cols]
            test_cats_df = pd.DataFrame([test_cats], columns=self.cat_cols)
            enc_test = self.encoder.transform(test_cats_df)
            if (enc_test == -1).any():
                confidence = "Low"
        except Exception:
            confidence = "Low"

        return enriched, is_out_of_window, confidence, warning

    def optimize_quote(
        self,
        inquiry_dict: dict,
        candidate_margins: Optional[np.ndarray] = None,
        pricing_strategy: str = "balanced",
        min_win_prob: Optional[float] = None,
        custom_strategy_thresholds: Optional[Dict[str, float]] = None
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Execute full Two-Stage hybrid optimization:
          1. Calculate Cost Floor Price based on IATA commercial guardrails.
          2. Retrieve empirical cohort margin quantiles.
          3. Predict Stage 1A market-clearing benchmark margin.
          4. Vectorize Stage 1B calibrated win probability simulation.
          5. Extract 3 distinct commercial postures respecting active STRATEGY_THRESHOLDS.
        """
        enriched, is_out_of_window, confidence, warning = self._enrich_inquiry(inquiry_dict)
        base_buy = float(enriched["Total_Buy_INR"])
        ch_wt = float(enriched["Chargeable_Weight_Kg"])
        commodity = str(enriched.get("Commodity_Group", "General Cargo"))
        category = str(enriched.get("Company_Group", enriched.get("Client_Category", "Subagent")))
        lane = str(enriched.get("Regional_Lane", "Other Origin -> Other Destination"))
        tier = str(enriched.get("Weight_Tier", "<45kg"))

        # ----------------------------------------------------------------------
        # 1. UPFRONT CHECK: Category & Commodity Maximum Margin Ceilings
        # ----------------------------------------------------------------------
        cap_info = get_category_commodity_max_margin(
            commodity_group=commodity,
            client_category=category
        )
        comm_max_cap = cap_info["commodity_max_margin"]
        cat_max_cap = cap_info["client_category_max_margin"]
        effective_max_margin = cap_info["effective_max_margin"]
        limiting_factor = cap_info["limiting_factor"]

        # 2. Hard Cost Floor Price Calculation (C-2, E-8)
        cost_floor_price = compute_cost_floor_price(
            airline_buy_inr=base_buy,
            chargeable_wt_kg=ch_wt,
            commodity_group=commodity,
            min_pct=MIN_CANDIDATE_MARGIN
        )
        min_cost_margin_pct = round(((cost_floor_price - base_buy) / base_buy) * 100.0, 2)

        # Operational safety constraint: Cost floor is non-negotiable.
        # If min_cost_margin_pct > effective_max_margin, cost floor takes absolute precedence.
        effective_upper_cap = max(min_cost_margin_pct, effective_max_margin)

        # 3. Empirical Cohort Quantiles Lookup
        cohort_q = self.get_cohort_quantiles(lane, tier, commodity)

        # 4. Stage 1A Benchmark Prediction
        row_df = pd.DataFrame([enriched])[self.feature_cols]
        row_enc = row_df.copy()
        row_enc[self.cat_cols] = self.encoder.transform(row_df[self.cat_cols].astype(str))

        benchmark_margin = float(self.benchmark_reg.predict(row_enc)[0])
        benchmark_margin = max(MIN_BENCHMARK_MARGIN, min(MAX_BENCHMARK_MARGIN, benchmark_margin))
        # Ensure benchmark is bounded within [cost floor, effective upper ceiling]
        benchmark_margin = max(min_cost_margin_pct, min(benchmark_margin, effective_upper_cap))

        # 5. Search Grid Bounded by Cost Floor and Category/Commodity Ceiling (E-8)
        if candidate_margins is None:
            min_m = min_cost_margin_pct
            cohort_p90 = cohort_q.get("p90", effective_upper_cap)
            max_m = min(effective_upper_cap, max(min_m + 4.0, round(cohort_p90 * 1.15, 1)))
            if max_m < min_m:
                max_m = min_m
            candidate_margins = np.arange(min_m, max_m + 0.5, 0.5)

        # Ensure candidate margins are strictly bounded between cost floor and effective upper ceiling
        candidate_margins = candidate_margins[
            (candidate_margins >= min_cost_margin_pct) & 
            (candidate_margins <= effective_upper_cap)
        ]
        if len(candidate_margins) == 0:
            candidate_margins = np.array([min_cost_margin_pct])

        # 6. Vectorized Stage 1B Calibrated Evaluation
        n_cand = len(candidate_margins)
        batch_df = pd.DataFrame([enriched] * n_cand)[self.feature_cols]
        batch_enc = batch_df.copy()
        batch_enc[self.cat_cols] = self.encoder.transform(batch_df[self.cat_cols].astype(str))

        margin_ratios = (candidate_margins / benchmark_margin).clip(0.1, 5.0)
        batch_enc["Margin_Ratio"] = margin_ratios

        clf_feature_names = self.feature_cols + ["Margin_Ratio"]
        raw_probs = self.win_classifier.predict_proba(batch_enc[clf_feature_names])[:, 1]

        # Use calibrated win probabilities directly (E-1)
        win_probs = raw_probs

        selling_prices = np.round(base_buy * (1.0 + candidate_margins / 100.0), 2)
        margin_amounts = np.round(selling_prices - base_buy, 2)
        expected_profits = np.round(margin_amounts * win_probs, 2)
        rates_per_kg = np.round(selling_prices / ch_wt, 2)

        # Account Retention / Frequency Penalty for repeat accounts (§4.1)
        freq = float(enriched.get("Customer_Inquiry_Frequency", 1))
        if self.retention_config.enabled and freq > self.retention_config.freq_threshold:
            cap = benchmark_margin * self.retention_config.margin_ratio_cap
            retention_weight = np.where(candidate_margins > cap, self.retention_config.penalty_factor, 1.0)
            expected_profits = np.round(expected_profits * retention_weight, 2)

        results_df = pd.DataFrame({
            "Margin_Percentage": candidate_margins,
            "Quoted_Sell_Price_INR": selling_prices,
            "Margin_Amount_INR": margin_amounts,
            "Rate_Per_Kg": rates_per_kg,
            "Win_Probability": win_probs,
            "Expected_Profit_INR": expected_profits,
            "Benchmark_Margin": benchmark_margin
        })

        # 7. Extract 3 Distinct Strategic Postures with Active Thresholds & Caps
        thresholds = custom_strategy_thresholds or self.strategy_thresholds
        strat_recs, raw_quotes = self._extract_strategic_tiers(
            benchmark_margin=benchmark_margin,
            results_df=results_df,
            cohort_q=cohort_q,
            cost_floor_price=cost_floor_price,
            min_cost_margin_pct=min_cost_margin_pct,
            ch_wt=ch_wt,
            thresholds=thresholds,
            base_buy=base_buy,
            effective_max_margin=effective_upper_cap,
            commodity_max_cap=comm_max_cap,
            category_max_cap=cat_max_cap,
            limiting_factor=limiting_factor
        )

        strat_lower = (pricing_strategy or "balanced").strip().lower()
        if strat_lower in ("floor", "volume"):
            chosen_raw = raw_quotes["floor"]
        elif strat_lower in ("premium", "skimmer"):
            chosen_raw = raw_quotes["premium"]
        else:
            chosen_raw = raw_quotes["balanced"]

        optimal_quote = dict(chosen_raw)
        optimal_quote["Benchmark_Margin"] = round(benchmark_margin, 2)
        optimal_quote["Cost_Floor_Price_INR"] = round(cost_floor_price, 2)
        optimal_quote["Cost_Floor_Margin_Pct"] = round(min_cost_margin_pct, 2)
        optimal_quote["Commodity_Max_Margin_Cap"] = round(comm_max_cap, 2)
        optimal_quote["Category_Max_Margin_Cap"] = round(cat_max_cap, 2)
        optimal_quote["Effective_Max_Margin_Cap"] = round(effective_max_margin, 2)
        optimal_quote["Cap_Limiting_Factor"] = limiting_factor
        optimal_quote["Is_Clamped_By_Cap"] = bool(optimal_quote["Margin_Percentage"] >= effective_max_margin - 0.05)
        optimal_quote["is_out_of_window"] = is_out_of_window
        optimal_quote["Is_Out_Of_Window"] = is_out_of_window
        optimal_quote["confidence"] = confidence
        optimal_quote["Confidence_Level"] = confidence
        if warning:
            optimal_quote["out_of_window_warning"] = warning
            optimal_quote["Seasonal_Warning"] = warning
        optimal_quote["cohort_quantiles"] = cohort_q
        optimal_quote["strategic_recommendations"] = strat_recs

        return optimal_quote, results_df


    def _extract_strategic_tiers(
        self,
        benchmark_margin: float,
        results_df: pd.DataFrame,
        cohort_q: Dict[str, float],
        cost_floor_price: float,
        min_cost_margin_pct: float,
        ch_wt: float,
        thresholds: Dict[str, float],
        base_buy: Optional[float] = None,
        effective_max_margin: Optional[float] = None,
        commodity_max_cap: Optional[float] = None,
        category_max_cap: Optional[float] = None,
        limiting_factor: Optional[str] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Extract strictly distinct commercial postures adhering to active STRATEGY_THRESHOLDS
        and Category / Commodity Maximum Margin Ceilings:
          - Floor: Must-Win / Volume (P(Win) >= 0.40)
          - Balanced: Recommended (P(Win) >= 0.25, Max Expected Profit)
          - Premium: Capacity Constrained (P(Win) >= 0.15, High Margin Capture)
        """
        th_floor = thresholds.get("floor", 0.40)
        th_balanced = thresholds.get("balanced", 0.25)
        th_premium = thresholds.get("premium", 0.15)

        if effective_max_margin is None:
            effective_max_margin = float(results_df["Margin_Percentage"].max()) if not results_df.empty else MAX_CANDIDATE_MARGIN
        if base_buy is None and not results_df.empty:
            m0 = results_df["Margin_Percentage"].iloc[0]
            s0 = results_df["Quoted_Sell_Price_INR"].iloc[0]
            base_buy = round(s0 / (1.0 + m0 / 100.0), 2) if m0 != -100 else 10000.0
        elif base_buy is None:
            base_buy = 10000.0

        # 1. Floor Option (Must-Win / Volume)
        floor_candidates = results_df[results_df["Win_Probability"] >= th_floor]
        if not floor_candidates.empty:
            # Anchor to cohort P25 above cost floor
            p25_target = max(min_cost_margin_pct, cohort_q.get("p25", min_cost_margin_pct))
            diffs = (floor_candidates["Margin_Percentage"] - p25_target).abs()
            best_f_idx = diffs.idxmin()
            raw_floor = results_df.loc[best_f_idx].to_dict()
        else:
            # Fallback: candidate with highest win probability
            best_f_idx = results_df.sort_values(by=["Win_Probability", "Expected_Profit_INR"], ascending=[False, False]).index[0]
            raw_floor = results_df.loc[best_f_idx].to_dict()

        floor_margin = float(raw_floor["Margin_Percentage"])

        # 2. Balanced Option (Recommended - Profit Maximizer in Corridor)
        # Corridor: strictly above floor margin, target win prob >= 0.25
        bal_candidates = results_df[
            (results_df["Margin_Percentage"] > floor_margin) &
            (results_df["Win_Probability"] >= th_balanced)
        ]
        if not bal_candidates.empty:
            best_b_idx = bal_candidates.sort_values(
                by=["Expected_Profit_INR", "Win_Probability"],
                ascending=[False, False]
            ).index[0]
            raw_balanced = results_df.loc[best_b_idx].to_dict()
        else:
            # Look for higher margin than floor
            higher_cands = results_df[results_df["Margin_Percentage"] > floor_margin]
            if not higher_cands.empty:
                best_b_idx = higher_cands.sort_values(by=["Expected_Profit_INR"], ascending=False).index[0]
                raw_balanced = higher_cands.loc[best_b_idx].to_dict()
            else:
                # If only 1 candidate exists in grid, synthesize +1.0%
                raw_balanced = dict(raw_floor)
                raw_balanced["Margin_Percentage"] = floor_margin + 1.0
                raw_balanced["Quoted_Sell_Price_INR"] = round(raw_floor["Quoted_Sell_Price_INR"] * 1.01, 2)
                raw_balanced["Margin_Amount_INR"] = round(raw_balanced["Quoted_Sell_Price_INR"] - base_buy, 2)
                raw_balanced["Win_Probability"] = max(0.10, raw_floor["Win_Probability"] - 0.05)
                raw_balanced["Rate_Per_Kg"] = round(raw_balanced["Quoted_Sell_Price_INR"] / max(1.0, ch_wt), 2)
                raw_balanced["Expected_Profit_INR"] = round(raw_balanced["Margin_Amount_INR"] * raw_balanced["Win_Probability"], 2)

        balanced_margin = float(raw_balanced["Margin_Percentage"])

        # 3. Premium Option (Capacity Constrained / Peak Yield)
        # Margin strictly above balanced margin, win prob >= 0.15
        prem_candidates = results_df[
            (results_df["Margin_Percentage"] > balanced_margin) &
            (results_df["Win_Probability"] >= th_premium)
        ]
        if not prem_candidates.empty:
            # Anchor near cohort P75 or P90
            p75_target = max(balanced_margin + 2.0, cohort_q.get("p75", balanced_margin + 2.0))
            diffs = (prem_candidates["Margin_Percentage"] - p75_target).abs()
            best_p_idx = diffs.idxmin()
            raw_premium = results_df.loc[best_p_idx].to_dict()
        else:
            # Look for highest candidate above balanced
            higher_prem = results_df[results_df["Margin_Percentage"] > balanced_margin]
            if not higher_prem.empty:
                best_p_idx = higher_prem.index[-1]
                raw_premium = higher_prem.loc[best_p_idx].to_dict()
            else:
                raw_premium = dict(raw_balanced)
                raw_premium["Margin_Percentage"] = balanced_margin + 2.0
                raw_premium["Quoted_Sell_Price_INR"] = round(raw_balanced["Quoted_Sell_Price_INR"] * 1.02, 2)
                raw_premium["Margin_Amount_INR"] = round(raw_premium["Quoted_Sell_Price_INR"] - base_buy, 2)
                raw_premium["Win_Probability"] = max(0.05, raw_balanced["Win_Probability"] - 0.08)
                raw_premium["Rate_Per_Kg"] = round(raw_premium["Quoted_Sell_Price_INR"] / max(1.0, ch_wt), 2)
                raw_premium["Expected_Profit_INR"] = round(raw_premium["Margin_Amount_INR"] * raw_premium["Win_Probability"], 2)

        # Strict Upper Cap Clamping (Respecting Category & Commodity Ceilings)
        if raw_premium["Margin_Percentage"] > effective_max_margin:
            raw_premium["Margin_Percentage"] = round(effective_max_margin, 2)
            raw_premium["Quoted_Sell_Price_INR"] = round(base_buy * (1.0 + raw_premium["Margin_Percentage"] / 100.0), 2)
            raw_premium["Margin_Amount_INR"] = round(raw_premium["Quoted_Sell_Price_INR"] - base_buy, 2)
            raw_premium["Rate_Per_Kg"] = round(raw_premium["Quoted_Sell_Price_INR"] / max(1.0, ch_wt), 2)
            raw_premium["Expected_Profit_INR"] = round(raw_premium["Margin_Amount_INR"] * raw_premium["Win_Probability"], 2)

        if raw_balanced["Margin_Percentage"] >= raw_premium["Margin_Percentage"]:
            raw_balanced["Margin_Percentage"] = round(max(raw_floor["Margin_Percentage"] + 0.5, raw_premium["Margin_Percentage"] - 1.5), 2)
            raw_balanced["Quoted_Sell_Price_INR"] = round(base_buy * (1.0 + raw_balanced["Margin_Percentage"] / 100.0), 2)
            raw_balanced["Margin_Amount_INR"] = round(raw_balanced["Quoted_Sell_Price_INR"] - base_buy, 2)
            raw_balanced["Rate_Per_Kg"] = round(raw_balanced["Quoted_Sell_Price_INR"] / max(1.0, ch_wt), 2)
            raw_balanced["Expected_Profit_INR"] = round(raw_balanced["Margin_Amount_INR"] * raw_balanced["Win_Probability"], 2)

        if raw_floor["Margin_Percentage"] >= raw_balanced["Margin_Percentage"]:
            raw_floor["Margin_Percentage"] = round(max(min_cost_margin_pct, raw_balanced["Margin_Percentage"] - 1.0), 2)
            raw_floor["Quoted_Sell_Price_INR"] = round(base_buy * (1.0 + raw_floor["Margin_Percentage"] / 100.0), 2)
            raw_floor["Margin_Amount_INR"] = round(raw_floor["Quoted_Sell_Price_INR"] - base_buy, 2)
            raw_floor["Rate_Per_Kg"] = round(raw_floor["Quoted_Sell_Price_INR"] / max(1.0, ch_wt), 2)
            raw_floor["Expected_Profit_INR"] = round(raw_floor["Margin_Amount_INR"] * raw_floor["Win_Probability"], 2)

        # Strict Distinctness Guarantee (Fix duplicate recommendation bug)
        if raw_balanced["Margin_Percentage"] <= raw_floor["Margin_Percentage"]:
            raw_balanced["Margin_Percentage"] = round(raw_floor["Margin_Percentage"] + 0.5, 2)
            raw_balanced["Quoted_Sell_Price_INR"] = round(base_buy * (1.0 + raw_balanced["Margin_Percentage"] / 100.0), 2)
            raw_balanced["Margin_Amount_INR"] = round(raw_balanced["Quoted_Sell_Price_INR"] - base_buy, 2)
            raw_balanced["Rate_Per_Kg"] = round(raw_balanced["Quoted_Sell_Price_INR"] / max(1.0, ch_wt), 2)

        if raw_premium["Margin_Percentage"] <= raw_balanced["Margin_Percentage"]:
            raw_premium["Margin_Percentage"] = round(raw_balanced["Margin_Percentage"] + 1.0, 2)
            raw_premium["Quoted_Sell_Price_INR"] = round(base_buy * (1.0 + raw_premium["Margin_Percentage"] / 100.0), 2)
            raw_premium["Margin_Amount_INR"] = round(raw_premium["Quoted_Sell_Price_INR"] - base_buy, 2)
            raw_premium["Rate_Per_Kg"] = round(raw_premium["Quoted_Sell_Price_INR"] / max(1.0, ch_wt), 2)

        def _format_card(raw, key, name, badge, tagline, color):
            sell = float(raw["Quoted_Sell_Price_INR"])
            margin_amt = float(raw["Margin_Amount_INR"])
            markup_pct = float(raw["Margin_Percentage"])
            gross_margin_pct = round((margin_amt / sell) * 100.0, 2) if sell > 0 else markup_pct
            rate_kg = float(raw.get("Rate_Per_Kg", round(sell / max(1.0, ch_wt), 2)))
            eff_cap = float(effective_max_margin) if effective_max_margin is not None else MAX_CANDIDATE_MARGIN
            is_clamped = bool(markup_pct >= eff_cap - 0.05)

            return {
                "Margin_Percentage": round(markup_pct, 2),            # Markup on Buy
                "Gross_Margin_Percentage": round(gross_margin_pct, 2), # Margin on Sell (A-7)
                "Quoted_Sell_Price_INR": round(sell, 2),
                "Margin_Amount_INR": round(margin_amt, 2),
                "Rate_Per_Kg": round(rate_kg, 2),
                "Win_Probability": round(float(raw["Win_Probability"]), 4),
                "Expected_Profit_INR": round(float(raw["Expected_Profit_INR"]), 2),
                "Benchmark_Margin": round(benchmark_margin, 2),
                "Cost_Floor_Price_INR": round(cost_floor_price, 2),
                "Cost_Floor_Margin_Pct": round(min_cost_margin_pct, 2),
                "Commodity_Max_Margin_Cap": round(commodity_max_cap, 2) if commodity_max_cap is not None else None,
                "Category_Max_Margin_Cap": round(category_max_cap, 2) if category_max_cap is not None else None,
                "Effective_Max_Margin_Cap": round(eff_cap, 2),
                "Cap_Limiting_Factor": limiting_factor,
                "Is_Clamped_By_Cap": is_clamped,
                "Strategy_Key": key,
                "Strategy_Name": name,
                "Badge_Label": badge,
                "Tagline": tagline,
                "Theme_Color": color
            }


        rec_floor = _format_card(raw_floor, "floor", "Competitive Floor", "Must-Win / Volume", "Highest Win Probability • Secure Booking", "cyan")
        rec_balanced = _format_card(raw_balanced, "balanced", "Best-Recommended Balance", "Recommended", "Max Expected Profit • Standard Target", "emerald")
        rec_premium = _format_card(raw_premium, "premium", "High-Margin Premium", "Capacity Constrained", "Peak Margin Capture • Selective Volume", "violet")

        strat_dict = {
            "floor": rec_floor,
            "balanced": rec_balanced,
            "premium": rec_premium
        }

        return strat_dict, {"floor": raw_floor, "balanced": raw_balanced, "premium": raw_premium}

    def get_strategic_recommendations(
        self,
        benchmark_margin_or_quote: Any,
        results_df: pd.DataFrame
    ) -> Any:
        """Backward compatibility wrapper for legacy callers."""
        bm = float(benchmark_margin_or_quote.get("Benchmark_Margin", 10.0)) if isinstance(benchmark_margin_or_quote, dict) else float(benchmark_margin_or_quote)
        ch_wt = 100.0
        strat_dict, raw_map = self._extract_strategic_tiers(
            benchmark_margin=bm,
            results_df=results_df,
            cohort_q={"p25": 3.5, "p50": 7.0, "p75": 14.5, "p90": 25.0},
            cost_floor_price=float(results_df["Quoted_Sell_Price_INR"].min()),
            min_cost_margin_pct=float(results_df["Margin_Percentage"].min()),
            ch_wt=ch_wt,
            thresholds=self.strategy_thresholds
        )
        if isinstance(benchmark_margin_or_quote, dict):
            return strat_dict
        return strat_dict, raw_map
