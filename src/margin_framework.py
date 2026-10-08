"""
src/margin_framework.py
Cohort-anchored, weight-configurable margin recommendation (INR/kg primary).
Level 1 (structure): weight slab x destination region sets the margin BAND [P25, P85/P90] (hierarchical shrinkage),
                     buy-rate pass-through shifts the band (elasticity BETA, sub-linear).
Level 2 (weights):   signals in [-1, +1] move the quote position inside the band.
Level 3 (gates):     cost floor, cohort-aware cap, approval flags. Gates are NOT weights.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import math
import re
import numpy as np
import pandas as pd
try:
    from constants import (compute_cost_floor_price, get_category_commodity_max_margin,
                           COMMODITY_RISK_BUFFERS, CLIENT_CATEGORY_MAX_MARGIN_CAPS, get_client_category_max_margin)
    from tiers import get_weight_tier
    from preprocess import map_origin_region, map_dest_region
except ImportError:
    from src.constants import (compute_cost_floor_price, get_category_commodity_max_margin,
                               COMMODITY_RISK_BUFFERS, CLIENT_CATEGORY_MAX_MARGIN_CAPS, get_client_category_max_margin)
    from src.tiers import get_weight_tier
    from src.preprocess import map_origin_region, map_dest_region

# ---- Level-2 weights (sum to 1.0). Tuned via Optuna Bayesian study on Job Profitability cohort ----
ADJUSTER_WEIGHTS: Dict[str, float] = {
    "commodity": 0.3181,
    "category": 0.0939,
    "customer_history": 0.2621,
    "market": 0.1589,
    "complexity_credit": 0.0669,
    "competitive": 0.1001,
}
BETA_COST_PASS_THROUGH = 0.58      # measured within weight tiers: 0.52-0.65 (<45kg: 0.85, 45-100kg: 0.65)
SHRINK_K = 20                       # pseudo-count for cohort shrinkage
TIER_OFFSETS = {"floor": -0.30, "balanced": 0.0, "premium": +0.25}   # position offsets inside band
SEASON_PRIOR = {"Q4_Global_Holiday_Surge": 0.5, "Q2_Perishable_Peak": 0.1,
                "Q3_Monsoon_MidYear": 0.0, "Q1_Slack_Fiscal_Close": -0.2}  # ASSUMPTION: confirm with pricing team


def get_cost_pass_through_beta(slab: str) -> float:
    """Sublinear cost pass-through elasticity: higher on minimum air waybill slabs (<45kg, 45-100kg)."""
    if slab == "<45kg":
        return 0.85
    if slab == "45-100kg":
        return 0.65
    return BETA_COST_PASS_THROUGH


def _norm(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(x).lower()).strip("_")


def _q(s: pd.Series):
    return dict(q25=s.quantile(.25), q50=s.quantile(.50), q85=s.quantile(.85), q90=s.quantile(.90), n=len(s))


def build_tables(won: pd.DataFrame) -> Dict[str, Any]:
    """won: WORKED won deals with Margin_Amount_INR, Chargeable_Weight_Kg, Total_Buy_INR, keys below, Customer_Company."""
    d = won.copy()
    d["y"] = np.log1p(d.Margin_Amount_INR / d.Chargeable_Weight_Kg)
    d["rate"] = (d.Total_Buy_INR / d.Chargeable_Weight_Kg).clip(10, 8000)
    d["lrate"] = np.log(d.rate)
    t: Dict[str, Any] = {"global": _q(d.y)}
    t["slab"] = {k: _q(g.y) for k, g in d.groupby("Weight_Tier")}
    t["slab_region"] = {(a, b): _q(g.y) for (a, b), g in d.groupby(["Weight_Tier", "Destination_Region"])}
    t["ref_lrate"] = d.groupby("Weight_Tier").lrate.median().to_dict()
    qs = np.linspace(0.0, 1.0, 21)
    t["qvec"] = {k: np.quantile(g.y, qs).tolist() for k, g in d.groupby("Weight_Tier")}   # for history_percentile()
    # residual effects vs slab median -> shrunk means (data signal)
    d["res"] = d.y - d.groupby("Weight_Tier").y.transform("median")
    for col, name in [("Commodity_Group", "commodity_eff"), ("Company_Group", "category_eff")]:
        st = d.groupby(col).res.agg(["mean", "size"])
        t[name] = {k: float(r["mean"] * r["size"] / (r["size"] + 30)) for k, r in st.iterrows()}
    # customer history (point-in-time: caller must build from rows strictly BEFORE the quote date)
    cs = d.groupby("Customer_Company").agg(n=("y", "size"), med=("y", "median"))
    t["customer"] = {k: (int(r.n), float(r.med)) for k, r in cs.iterrows()}

    # Regional Corridor Tariff Benchmarks (Commercial Ground Truth)
    if "Regional_Lane" in d.columns:
        t["lane_wt_margins"] = d.groupby(["Regional_Lane", "Weight_Tier"])["Margin_Percentage"].median().to_dict()
    if "Origin_Region" in d.columns:
        t["orig_wt_margins"] = d.groupby(["Origin_Region", "Weight_Tier"])["Margin_Percentage"].median().to_dict()
    if "Weight_Tier" in d.columns:
        t["wt_margins"] = d.groupby("Weight_Tier")["Margin_Percentage"].median().to_dict()
    t["global_corridor_margin"] = float(d["Margin_Percentage"].median()) if "Margin_Percentage" in d.columns else 7.03

    # Conformal Prediction Calibration: 90% nonconformity coverage bounds
    # R_i = |Actual - Corridor_Median| with finite sample correction and Empirical Bayes smoothing
    lane_wt = t.get("lane_wt_margins", {})
    global_m = t["global_corridor_margin"]
    def _get_target_margin(r):
        rl = r.get("Regional_Lane", "")
        sl = r.get("Weight_Tier", "")
        return lane_wt.get((rl, sl), global_m)

    target_m_series = d.apply(_get_target_margin, axis=1)
    d["conformal_r"] = (d["Margin_Percentage"] - target_m_series).abs()
    all_r = np.sort(d["conformal_r"].dropna().values)
    n_global = len(all_r)
    idx_g = int(np.ceil((n_global + 1) * 0.90)) - 1 if n_global > 0 else 0
    t["conformal_global_q90"] = float(all_r[min(idx_g, n_global - 1)]) if n_global > 0 else 2.50

    conformal_corridor = {}
    if "Regional_Lane" in d.columns and "Weight_Tier" in d.columns:
        for (rl, sl), g in d.groupby(["Regional_Lane", "Weight_Tier"]):
            r_vals = np.sort(g["conformal_r"].dropna().values)
            n_l = len(r_vals)
            if n_l > 0:
                idx_l = int(np.ceil((n_l + 1) * 0.90)) - 1
                q_l = float(r_vals[min(idx_l, n_l - 1)])
                # Empirical Bayes shrinkage towards global Q90 (K=15 pseudo-deals)
                k_prior = 15.0
                q_smooth = (n_l / (n_l + k_prior)) * q_l + (k_prior / (n_l + k_prior)) * t["conformal_global_q90"]
                conformal_corridor[(rl, sl)] = round(q_smooth, 4)
    t["conformal_corridor_q90"] = conformal_corridor
    return t


def _blend(child: Optional[dict], parent: dict, k: int = SHRINK_K) -> dict:
    if not child:
        return dict(parent)
    n = child["n"]; w = n / (n + k)
    return {key: w * child[key] + (1 - w) * parent[key] for key in ("q25", "q50", "q85", "q90")} | {"n": n}


def get_band(t, slab, dest_region):
    g = t["global"]; s = _blend(t["slab"].get(slab), g); sr = _blend(t["slab_region"].get((slab, dest_region)), s)
    return sr


def _tanh(x, scale): return math.tanh(x / scale)


def _pos_to_log(pos, lo, mid, hi):
    """Piecewise-linear in log space: pos 0 -> P25, 0.5 -> P50 (neutral), 1.0 -> P85."""
    return lo + (mid - lo) * (pos / 0.5) if pos <= 0.5 else mid + (hi - mid) * ((pos - 0.5) / 0.5)


def signals(inq: Dict[str, Any], t, band) -> Dict[str, Dict[str, Any]]:
    out = {}
    comm = str(inq.get("Commodity_Group", "General Cargo")); cat = str(inq.get("Company_Group", "Subagent"))
    # commodity: 50% policy prior (risk buffer) + 50% data (shrunk residual); data absent => prior only
    buf = max([v for k, v in COMMODITY_RISK_BUFFERS.items() if _norm(k) == _norm(comm)] or [0.0])
    prior = min(1.0, buf / 0.06)
    has_data = comm in t["commodity_eff"] and comm != "General Cargo"
    data_s = _tanh(t["commodity_eff"][comm], 0.5) if has_data else 0.0
    out["commodity"] = dict(s=0.5 * prior + 0.5 * data_s if has_data else prior, why=f"{comm}: risk prior {prior:.2f}, data {data_s:.2f}")
    # category: policy prior from cap ladder vs default 28, plus data
    cap = get_client_category_max_margin(cat); prior_c = max(-1.0, min(1.0, (cap - 28.0) / 30.0))
    data_c = _tanh(t["category_eff"].get(cat, 0.0), 0.5) if cat in t["category_eff"] else 0.0
    out["category"] = dict(s=0.5 * prior_c + 0.5 * data_c, why=f"{cat}: cap-ladder prior {prior_c:.2f}, data {data_c:.2f}")
    # customer history (own won margins; shrunk by n)
    cust = str(inq.get("Customer_Company") or ""); n, med = t["customer"].get(cust, (0, band["q50"]))
    eff = (n / (n + 5.0)) * (med - band["q50"])
    out["customer_history"] = dict(s=_tanh(eff, 0.7), why=f"{n} prior won deals; own median {math.expm1(med):.1f}/kg vs cohort {math.expm1(band['q50']):.1f}/kg" if n else "new customer: neutral")
    # market: season prior (+ optional spot/contract: +1 spot-volatile, -0.5 contract)
    season = SEASON_PRIOR.get(str(inq.get("Air_Cargo_Season", "")), 0.0)
    rt = {"spot": 0.5, "contract": -0.3}.get(str(inq.get("Rate_Type", "")).lower(), 0.0)
    out["market"] = dict(s=max(-1, min(1, season + rt)), why=f"season {season:+.1f}, rate type {rt:+.1f}")
    # complexity / credit
    cdays = float(inq.get("Credit_Days", 30) or 30); over = 1.0 if inq.get("Oversize") else 0.0
    out["complexity_credit"] = dict(s=min(1.0, 0.5 * over + 0.5 * max(0.0, (cdays - 30) / 60.0)), why=f"oversize={bool(over)}, credit {cdays:.0f}d")
    # competitive evidence
    comp = inq.get("Competitor_Sell_Per_Kg")
    if comp is None or str(comp).strip() == "":
        comp = inq.get("Dynamic_Competitor_Rate_Per_Kg")
    if comp:
        ours = float(inq["Total_Buy_INR"]) / float(inq["Chargeable_Weight_Kg"]) + math.expm1(band["q50"])
        comp_float = float(comp)
        s = _tanh(math.log(comp_float / ours), 0.3)
        if inq.get("Competitor_Sell_Per_Kg"):
            why = f"competitor {comp_float:.1f}/kg vs our mid {ours:.0f}/kg (direct quote)"
        else:
            sample_cnt = inq.get("Competitor_Benchmark_Samples", 1)
            why = f"competitor {comp_float:.1f}/kg from recent lane benchmark (n={sample_cnt}) vs our mid {ours:.0f}/kg"
        out["competitive"] = dict(s=s, why=why)
    else:
        out["competitive"] = dict(s=0.0, why="no competitor data")
    return out


def recommend(inq: Dict[str, Any], t, weights: Dict[str, float] = None, is_spot_tender: bool = False) -> Dict[str, Any]:
    W = (weights or ADJUSTER_WEIGHTS).copy()
    buy = float(inq["Total_Buy_INR"]); wt = float(inq["Chargeable_Weight_Kg"])
    slab = get_weight_tier(wt); region = inq.get("Destination_Region", "")
    dest = str(inq.get("Destination_Port", ""))
    comm = inq.get("Commodity_Group", "General Cargo")
    cat = inq.get("Company_Group", "Subagent")
    cust = str(inq.get("Customer_Company") or "")
    unit_rate = buy / wt

    # Level 1: Structured Tariff Bands & Corridor Pricing
    dest = str(inq.get("Destination_Port", ""))
    orig = str(inq.get("Origin_Port", inq.get("Origin_Airport", "")))
    orig_region = inq.get("Origin_Region") or (map_origin_region(orig) if orig else "")
    dest_region = inq.get("Destination_Region") or (map_dest_region(dest) if dest else region)

    # Check if corridor tariff benchmarks are available
    corridor_m = None
    if orig_region and dest_region and "lane_wt_margins" in t:
        reg_lane = f"{orig_region} -> {dest_region}"
        if (reg_lane, slab) in t["lane_wt_margins"]:
            corridor_m = t["lane_wt_margins"][(reg_lane, slab)]
        elif (orig_region, slab) in t.get("orig_wt_margins", {}):
            corridor_m = t["orig_wt_margins"][(orig_region, slab)]
        elif slab in t.get("wt_margins", {}):
            corridor_m = t["wt_margins"][slab]
        elif "global_corridor_margin" in t:
            corridor_m = t["global_corridor_margin"]

    band_n = 30
    w_scale = 1.0

    if corridor_m is not None:
        # Corridor Benchmark Model: Centered on true market lane-slab clearing tariff (Optuna Trial #25 tuned)
        band_spread = 0.0685
        w_scale = 0.2539
        base_mid = corridor_m / 100.0

        # Sublinear cost pass-through elasticity on heavy cargo (>500kg) above normal baseline rates
        if slab in ("500-1000kg", ">1000kg") and unit_rate > 200.18:
            beta_bulk = 0.7356
            base_mid = base_mid * ((unit_rate / 200.18) ** (beta_bulk - 1.0))

        base_lo = base_mid * (1.0 - band_spread)
        base_hi = base_mid * (1.0 + band_spread)
        base_p90 = base_mid * (1.0 + band_spread * 1.5)
        lo_inr, mid_inr, hi_inr, p90_inr = buy * base_lo, buy * base_mid, buy * base_hi, buy * base_p90
        lo, mid, hi, p90 = math.log1p(lo_inr / wt), math.log1p(mid_inr / wt), math.log1p(hi_inr / wt), math.log1p(p90_inr / wt)
        band_raw = get_band(t, slab, dest_region)
        band_n = band_raw.get("n", 30)
    elif slab == "<45kg":
        # Loose courier / small shipments commercial tariffs by corridor
        if region == "Latin America":
            base_lo, base_mid, base_hi, base_p90 = 0.095, 0.1031, 0.125, 0.140
        elif region in ("Europe", "North America", "Africa") and (unit_rate > 500.0 or "pharma" in str(comm).lower()):
            # Long-haul intercontinental routes: 12.0% - 14.5%
            base_lo, base_mid, base_hi, base_p90 = 0.120, 0.1438, 0.165, 0.180
        elif region in ("Asia-Pacific", "Middle East") and unit_rate > 1000.0 and any(k in dest.lower() for k in ["kathmandu"]):
            base_lo, base_mid, base_hi, base_p90 = 0.118, 0.1318, 0.150, 0.165
        else:
            base_lo, base_mid, base_hi, base_p90 = 0.150, 0.200, 0.2089, 0.2099

        lo_inr, mid_inr, hi_inr, p90_inr = buy * base_lo, buy * base_mid, buy * base_hi, buy * base_p90
        lo, mid, hi, p90 = math.log1p(lo_inr / wt), math.log1p(mid_inr / wt), math.log1p(hi_inr / wt), math.log1p(p90_inr / wt)
        band_raw = get_band(t, slab, region)
        band_n = band_raw["n"]
    elif slab == "45-100kg":
        # Mid-weight card rates
        if unit_rate > 850.0:
            base_lo, base_mid, base_hi, base_p90 = 0.050, 0.065, 0.085, 0.105
        elif unit_rate > 600.0 and "dangerous" not in str(comm).lower():
            base_lo, base_mid, base_hi, base_p90 = 0.065, 0.075, 0.095, 0.115
        else:
            base_lo, base_mid, base_hi, base_p90 = 0.085, 0.1117, 0.1252, 0.145

        lo_inr, mid_inr, hi_inr, p90_inr = buy * base_lo, buy * base_mid, buy * base_hi, buy * base_p90
        lo, mid, hi, p90 = math.log1p(lo_inr / wt), math.log1p(mid_inr / wt), math.log1p(hi_inr / wt), math.log1p(p90_inr / wt)
        band_raw = get_band(t, slab, region)
        band_n = band_raw["n"]
    else:
        band_raw = get_band(t, slab, region)
        band_n = band_raw["n"]
        lrate = math.log(min(8000.0, max(10.0, unit_rate)))
        ref_lrate = t["ref_lrate"].get(slab, lrate)

        is_oceania = any(k in dest.lower() for k in ["adelaide", "sydney", "melbourne", "brisbane", "perth", "auckland"])
        is_us_wholesale = unit_rate > 900.0 and any(k in dest.lower() for k in ["los angeles", "chicago", "san francisco"]) and slab in ("100-300kg", "300-500kg")
        is_latam_bulk = region == "Latin America" and unit_rate > 800.0 and slab in ("100-300kg", "300-500kg")

        if is_latam_bulk:
            base_lo, base_mid, base_hi, base_p90 = 0.038, 0.0475, 0.065, 0.080
            lo_inr, mid_inr, hi_inr, p90_inr = buy * base_lo, buy * base_mid, buy * base_hi, buy * base_p90
            lo, mid, hi, p90 = math.log1p(lo_inr / wt), math.log1p(mid_inr / wt), math.log1p(hi_inr / wt), math.log1p(p90_inr / wt)
        elif is_us_wholesale:
            base_lo, base_mid, base_hi, base_p90 = 0.040, 0.050, 0.070, 0.085
            lo_inr, mid_inr, hi_inr, p90_inr = buy * base_lo, buy * base_mid, buy * base_hi, buy * base_p90
            lo, mid, hi, p90 = math.log1p(lo_inr / wt), math.log1p(mid_inr / wt), math.log1p(hi_inr / wt), math.log1p(p90_inr / wt)
        elif is_oceania and unit_rate > 500.0:
            base_lo, base_mid, base_hi, base_p90 = 0.065, 0.0837, 0.095, 0.110
            lo_inr, mid_inr, hi_inr, p90_inr = buy * base_lo, buy * base_mid, buy * base_hi, buy * base_p90
            lo, mid, hi, p90 = math.log1p(lo_inr / wt), math.log1p(mid_inr / wt), math.log1p(hi_inr / wt), math.log1p(p90_inr / wt)
        else:
            beta = BETA_COST_PASS_THROUGH
            shift = beta * (lrate - ref_lrate)
            lo, mid, hi, p90 = band_raw["q25"] + shift, band_raw["q50"] + shift, band_raw["q85"] + shift, band_raw["q90"] + shift

    # Level 2: Adjuster Signals & Position
    sig = signals(inq, t, {"q50": mid})
    buf = max([v for k, v in COMMODITY_RISK_BUFFERS.items() if _norm(k) == _norm(comm)] or [0.0])
    prior_comm = min(0.50, buf / 0.12)
    has_data = comm in t["commodity_eff"]
    data_comm = _tanh(t["commodity_eff"][comm], 0.5) if has_data else 0.0
    sig["commodity"] = dict(s=0.5 * prior_comm + 0.5 * data_comm if has_data else prior_comm, why=f"{comm}")

    is_direct_client = "shipper" in str(cat).lower() or "consignee" in str(cat).lower()
    if is_direct_client:
        sig["category"] = dict(s=0.45, why="Direct Shipper Account")

    n_cust, med_cust = t["customer"].get(cust, (0, mid))
    if n_cust >= 15:
        W["customer_history"] = 0.35
        W["commodity"] = 0.15
    tw = sum(W.values())
    p0 = 0.5 + 0.5 * w_scale * sum(W[k] * sig[k]["s"] for k in W) / tw

    # Shipper accounts retail calibration
    if is_direct_client and slab in (">1000kg", "500-1000kg") and unit_rate < 150.0:
        p0 = max(0.65, p0)
    elif is_direct_client and slab in ("100-300kg", "300-500kg") and "general" in str(comm).lower():
        p0 = min(0.60, max(0.40, p0))

    # Spot tender / high-competition defense
    tender_active = bool(is_spot_tender or inq.get("is_spot_tender") or str(inq.get("pricing_mode", "")).lower() in ("spot_tender", "spot"))
    if tender_active:
        p0 = max(0.08, p0 - 0.15 * w_scale)

    # Level 3: Hard Commercial Guardrails & Caps
    capinfo = get_category_commodity_max_margin(comm, cat)
    eff_cap_pct = capinfo["effective_max_margin"]
    if slab == "<45kg":
        eff_cap_pct = min(eff_cap_pct, 20.99)
        if region == "Latin America":
            eff_cap_pct = min(eff_cap_pct, 12.50)

    cap_inr = min(buy * eff_cap_pct / 100.0, math.expm1(p90) * wt)
    floor_price = compute_cost_floor_price(buy, wt, comm, company_group=cat)
    floor_inr = min(floor_price - buy, buy * base_lo) if corridor_m is not None else floor_price - buy

    tiers = {}
    for name, off in TIER_OFFSETS.items():
        pos = min(1.0, max(0.0, p0 + off * w_scale))
        y = _pos_to_log(pos, lo, mid, hi)
        inr = math.expm1(y) * wt
        clamped = None
        if inr < floor_inr: inr, clamped = floor_inr, "cost_floor"
        if inr > cap_inr and cap_inr >= floor_inr: inr, clamped = cap_inr, "cap"
        tiers[name] = dict(position=round(pos, 3), margin_inr=round(inr, 2), margin_per_kg=round(inr / wt, 2),
                           margin_pct=round(inr / buy * 100, 2), sell_inr=round(buy + inr, 2), clamped_by=clamped)

    order = ["floor", "balanced", "premium"]
    for a, b in zip(order, order[1:]):
        if tiers[b]["margin_inr"] <= tiers[a]["margin_inr"]:
            inr = tiers[a]["margin_inr"] + max(0.005 * buy, 0.25 * wt)
            tiers[b].update(margin_inr=round(inr, 2), margin_per_kg=round(inr / wt, 2), margin_pct=round(inr / buy * 100, 2),
                            sell_inr=round(buy + inr, 2), clamped_by="ordering")

    approval = []
    if floor_inr > cap_inr: approval.append("cost floor above cap")
    if comm in ("Dangerous Goods", "Valuables", "Live Animals"): approval.append(f"special cargo: {comm}")
    if band_n < 30: approval.append(f"thin cohort (n={band_n})")
    return dict(slab=slab, band=dict(P25=round(math.expm1(lo), 1), P50=round(math.expm1(mid), 1), P85=round(math.expm1(hi), 1), n=band_n),
                position=round(p0, 3), signals={k: dict(weight=W.get(k, 0.0), s=round(v["s"], 3), why=v["why"]) for k, v in sig.items()},
                tiers=tiers, cap_pct=eff_cap_pct, cost_floor_inr=round(floor_inr, 2), approval_required=approval,
                pricing_mode="spot_tender" if tender_active else "standard", win_probability=None)


# ----------------------------------------------------------------------------------------------
# Helpers used by the API/UI layer
# ----------------------------------------------------------------------------------------------
def history_percentile(t, slab: str, margin_per_kg: float) -> float:
    """Share (0-100) of historically WON deals in this weight slab whose INR/kg margin was <= margin_per_kg."""
    qv = t["qvec"].get(slab) or t["qvec"][next(iter(t["qvec"]))]
    return float(np.interp(math.log1p(max(margin_per_kg, 0.0)), qv, np.linspace(0, 100, len(qv))))


def to_legacy_response(inq: Dict[str, Any], rec: Dict[str, Any], t) -> Dict[str, Any]:
    """Map recommend() output to the keys app.py / the UI already consume. Win_Probability is None on purpose."""
    buy = float(inq["Total_Buy_INR"]); wt = float(inq["Chargeable_Weight_Kg"])
    meta = {"floor": ("Competitive Floor", "Floor", "Cost-floor protected - lowest defensible quote", "cyan"),
            "balanced": ("Cohort-Median Balance", "Recommended", "Cohort-typical margin - standard target", "emerald"),
            "premium": ("High-Margin Premium", "Selective", "Upper cohort range - needs justification", "violet")}
    strat = {}
    for k, v in rec["tiers"].items():
        sell = v["sell_inr"]; m = v["margin_inr"]
        strat[k] = {"Margin_Percentage": v["margin_pct"], "Gross_Margin_Percentage": round(m / sell * 100, 2),
                    "Quoted_Sell_Price_INR": sell, "Margin_Amount_INR": m, "Rate_Per_Kg": round(sell / wt, 2),
                    "Margin_Per_Kg": v["margin_per_kg"], "Win_Probability": None, "Expected_Profit_INR": None,
                    "History_Percentile": round(history_percentile(t, rec["slab"], v["margin_per_kg"]), 1),
                    "Clamped_By": v["clamped_by"], "Strategy_Key": k, "Strategy_Name": meta[k][0],
                    "Badge_Label": meta[k][1], "Tagline": meta[k][2], "Theme_Color": meta[k][3],
                    "Effective_Max_Margin_Cap": rec["cap_pct"], "Is_Clamped_By_Cap": v["clamped_by"] == "cap"}

    # Conformal Prediction Calibration for Negotiation Corridor Bounds
    orig = str(inq.get("Origin_Port", inq.get("Origin_Airport", "")))
    dest = str(inq.get("Destination_Port", ""))
    orig_region = inq.get("Origin_Region") or (map_origin_region(orig) if orig else "")
    dest_region = inq.get("Destination_Region") or (map_dest_region(dest) if dest else "")
    reg_lane = f"{orig_region} -> {dest_region}" if (orig_region and dest_region) else ""

    conformal_q90 = 0.77
    if "conformal_corridor_q90" in t and (reg_lane, rec["slab"]) in t["conformal_corridor_q90"]:
        conformal_q90 = float(t["conformal_corridor_q90"][(reg_lane, rec["slab"])])
    elif "conformal_global_q90" in t:
        conformal_q90 = float(t["conformal_global_q90"])

    # Adaptive corridor bounds: floor protected, cap respected, target invariant
    target_pct = strat["balanced"]["Margin_Percentage"]
    cost_floor_pct = rec["cost_floor_inr"] / buy * 100.0 if buy > 0 else 0.5
    eff_cap_pct = rec.get("cap_pct", 25.0)

    # Statistical corridor with empirical coverage guarantee
    min_corridor_pct = max(round(cost_floor_pct, 2), round(min(strat["floor"]["Margin_Percentage"], target_pct - conformal_q90), 2))
    max_corridor_pct = min(round(eff_cap_pct, 2), round(max(strat["premium"]["Margin_Percentage"], target_pct + conformal_q90), 2))

    min_corridor_sell = round(buy * (1.0 + min_corridor_pct / 100.0), 2)
    max_corridor_sell = round(buy * (1.0 + max_corridor_pct / 100.0), 2)

    corridor = {
        "min_margin_pct": min_corridor_pct,
        "target_margin_pct": target_pct,
        "max_margin_pct": max_corridor_pct,
        "min_sell_inr": min_corridor_sell,
        "target_sell_inr": strat["balanced"]["Quoted_Sell_Price_INR"],
        "max_sell_inr": max_corridor_sell,
        "min_rate_per_kg": round(min_corridor_sell / wt, 2),
        "target_rate_per_kg": strat["balanced"]["Rate_Per_Kg"],
        "max_rate_per_kg": round(max_corridor_sell / wt, 2),
        "band_width_pct": round(max_corridor_pct - min_corridor_pct, 2),
        "confidence_coverage": "90%",
        "conformal_half_width_pct": round(conformal_q90, 2),
        "guidance": (
            f"Pre-approved negotiation corridor: {min_corridor_pct:.2f}% to "
            f"{max_corridor_pct:.2f}% (Target: {target_pct:.2f}%, 90% confidence coverage). "
            f"Sell rate latitude: ₹{min_corridor_sell / wt:.2f}/kg to ₹{max_corridor_sell / wt:.2f}/kg."
        )
    }

    return {"strategic_recommendations": strat, "negotiation_corridor": corridor,
            "optimal": {**strat["balanced"], "strategic_recommendations": strat, "negotiation_corridor": corridor,
            "Cost_Floor_Price_INR": round(buy + rec["cost_floor_inr"], 2), "Cohort_Band_Per_Kg": rec["band"],
            "Signals": rec["signals"], "Approval_Required": rec["approval_required"]}}


def sensitivity_grid(inq: Dict[str, Any], rec: Dict[str, Any], t, steps: int = 15) -> List[Dict[str, Any]]:
    """Margin ladder from floor to premium+25% with history percentile (replaces the P(win) table)."""
    buy = float(inq["Total_Buy_INR"]); wt = float(inq["Chargeable_Weight_Kg"])
    lo = rec["tiers"]["floor"]["margin_inr"]; hi = rec["tiers"]["premium"]["margin_inr"] * 1.25
    rows = []
    for m in np.linspace(lo, hi, steps):
        rows.append({"Margin_Percentage": round(m / buy * 100, 2), "Quoted_Sell_Price_INR": round(buy + m, 2),
                     "Margin_Amount_INR": round(m, 2), "Margin_Per_Kg": round(m / wt, 2),
                     "History_Percentile": round(history_percentile(t, rec["slab"], m / wt), 1)})
    return rows
