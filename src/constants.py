"""
src/constants.py
Single source of truth for pricing bounds, margin search ranges,
commercial guardrails, operational floors, and system constants.
"""

import re
from typing import Dict, Optional, Any

# Model Version Identifier (Single Source of Truth across codebase and audit DB)
MODEL_VERSION: str = "v2.3-remediated-20261004"

# Market Benchmark Margin bounds (%)
MIN_BENCHMARK_MARGIN: float = 1.0
MAX_BENCHMARK_MARGIN: float = 48.0

# Candidate Margin search bounds (%)
MIN_CANDIDATE_MARGIN: float = 1.0  # Raised from 0.5% to 1.0% to block near-cost bids (C-2, F-09)
MAX_CANDIDATE_MARGIN: float = 45.0

# Strategy Win-Rate Minimum Floors (Active commercial constraints)
STRATEGY_THRESHOLDS: Dict[str, float] = {
    "floor": 0.40,        # Volume / Must-Win posture: minimum 40% win rate target
    "volume": 0.40,       # Backwards compatibility alias
    "balanced": 0.25,     # Recommended commercial sweet spot: minimum 25% win rate target
    "premium": 0.15,      # Capacity constrained / peak yield: minimum 15% win rate target
    "skimmer": 0.15       # Backwards compatibility alias
}

# ------------------------------------------------------------------------------
# HARD COMMERCIAL PROFITABILITY FLOORS (Section 10.1 & C-2, F-09)
# ------------------------------------------------------------------------------
MIN_MARGIN_PER_AWB_INR: float = 1500.0   # Fixed documentation, screening, AWB issuance cost floor
MIN_MARGIN_PER_KG_INR: float = 3.0       # Minimum unit operational contribution margin per kg
OPS_HANDLING_FEE_AWB_INR: float = 850.0  # Baseline origin handling & terminal documentation fee

# ------------------------------------------------------------------------------
# COMMODITY RISK BUFFERS TABLE (P-7, F-10, Section 10.1)
# ------------------------------------------------------------------------------
COMMODITY_RISK_BUFFERS: Dict[str, float] = {
    "general_cargo": 0.0,
    "General Cargo": 0.0,
    "engineering_machinery": 0.015,     # +1.5% for tie-down/crating/shoring inspections
    "Engineering & Machinery": 0.015,
    "courier_express": 0.020,           # +2.0% for tight cutoff & late lodgement SLA
    "Courier": 0.020,
    "other_specialized": 0.025,         # +2.5% for special equipment or non-standard handling
    "other": 0.025,
    "Other": 0.025,
    "live_animals": 0.050,
    "Live Animals": 0.050,
    "perishable_cold_chain": 0.035,     # +3.5% for temperature tracking & transit time urgency
    "Perishable Foodstuff": 0.035,
    "pharmaceuticals": 0.040,           # +4.0% for GDP compliance, validated passive packaging
    "Pharmaceuticals": 0.040,
    "valuable_cargo": 0.050,            # +5.0% for vaulting, tarmac escort, high-liability cover
    "Valuables": 0.050,
    "dangerous_goods": 0.060,           # +6.0% for IATA DGR inspection, checklist, hazard segregation
    "Dangerous Goods": 0.060
}


def compute_cost_floor_price(
    airline_buy_inr: float,
    chargeable_wt_kg: float,
    commodity_group: str = "general_cargo",
    min_pct: float = MIN_CANDIDATE_MARGIN,
    company_group: str = "Subagent"
) -> float:
    """
    Computes the commercial cost floor price.
    Protects against negative or near-cost quotes while avoiding artificial margin
    inflation on bulk freight.
    """
    wt = max(1.0, float(chargeable_wt_kg))
    buy = float(airline_buy_inr)
    rate = buy / wt

    # Small shipments (<= 45kg) strictly enforce full AWB documentation & handling fee
    if wt <= 45.0:
        unit_min = MIN_MARGIN_PER_KG_INR
        awb_min = MIN_MARGIN_PER_AWB_INR
        ops_fee = OPS_HANDLING_FEE_AWB_INR
    else:
        # Mid and bulk freight scale unit and documentation minimums realistically with rate and buy cost
        unit_min = min(MIN_MARGIN_PER_KG_INR, max(0.50, 0.015 * rate))
        awb_min = min(MIN_MARGIN_PER_AWB_INR, max(350.0, 0.015 * buy))
        ops_fee = min(OPS_HANDLING_FEE_AWB_INR, max(200.0, 0.010 * buy))

    # Commodity risk buffer:
    key = re.sub(r"[^a-z0-9]+", "_", str(commodity_group).lower()).strip("_")
    risk_buffer = COMMODITY_RISK_BUFFERS.get(key, 0.0)
    if risk_buffer == 0.0:
        if "pharma" in key or "vaccine" in key:
            risk_buffer = COMMODITY_RISK_BUFFERS["pharmaceuticals"]
        elif "perish" in key or "food" in key:
            risk_buffer = COMMODITY_RISK_BUFFERS["perishable_cold_chain"]
        elif "danger" in key or "dgr" in key or "haz" in key:
            risk_buffer = COMMODITY_RISK_BUFFERS["dangerous_goods"]
        elif "valuable" in key or "vun" in key:
            risk_buffer = COMMODITY_RISK_BUFFERS["valuable_cargo"]
        elif "engineer" in key or "machine" in key or "auto" in key:
            risk_buffer = COMMODITY_RISK_BUFFERS["engineering_machinery"]
        elif "courier" in key or "express" in key:
            risk_buffer = COMMODITY_RISK_BUFFERS["courier_express"]
        elif "animal" in key or "avi" in key:
            risk_buffer = COMMODITY_RISK_BUFFERS["live_animals"]
        elif "other" in key:
            risk_buffer = COMMODITY_RISK_BUFFERS["other"]

    # Intermediary wholesale discount
    if any(k in str(company_group).lower() for k in ["gsa", "nvocc", "coloader"]):
        risk_buffer *= 0.50

    pct_floor_price = (buy * (1.0 + (min_pct / 100.0) + risk_buffer)) + ops_fee
    abs_floor_price = buy + awb_min + (unit_min * wt)

    return float(max(pct_floor_price, abs_floor_price))


def build_cap_table(won, min_n=30, pct=0.90, k=30):
    """Data-derived cap % per (commodity, category, weight slab), shrunk toward the slab P90."""
    won = won.assign(mp=won.Margin_Percentage)
    g_slab = won.groupby("Weight_Tier").mp.quantile(pct)
    rows = {}
    for (c, cat, slab), g in won.groupby(["Commodity_Group", "Company_Group", "Weight_Tier"]):
        parent = g_slab[slab]
        n = len(g)
        rows[(c, cat, slab)] = (n, float((n * g.mp.quantile(pct) + k * parent) / (n + k)))
    return rows


# ------------------------------------------------------------------------------
# CATEGORY & COMMODITY MAXIMUM MARGIN CAPS (Ceilings)
# ------------------------------------------------------------------------------
# Maximum acceptable market margins by commodity type. Quoting above these
# ceilings leads to near-100% loss of deal in competitive air freight markets.
COMMODITY_MAX_MARGIN_CAPS: Dict[str, float] = {
    "General Cargo": 30.0,
    "general_cargo": 30.0,
    "Garments / Textiles": 25.0,
    "garments_textiles": 25.0,
    "garments": 25.0,
    "textiles": 25.0,
    "Auto Parts": 22.0,
    "auto_parts": 22.0,
    "automotive": 22.0,
    "Perishable Foodstuff": 26.0,
    "perishable_foodstuff": 26.0,
    "perishables": 26.0,
    "perishable_cold_chain": 26.0,
    "Engineering & Machinery": 25.0,
    "engineering_machinery": 25.0,
    "engineering": 25.0,
    "machinery": 25.0,
    "Pharmaceuticals": 35.0,
    "pharmaceuticals": 35.0,
    "pharma": 35.0,
    "Courier": 38.0,
    "courier": 38.0,
    "courier_express": 38.0,
    "Dangerous Goods": 42.0,
    "dangerous_goods": 42.0,
    "hazmat": 42.0,
    "Valuables": 45.0,
    "valuable_cargo": 45.0,
    "valuables": 45.0,
    "Live Animals": 35.0,
    "live_animals": 35.0,
    "Other": 28.0,
    "other": 28.0,
    "default": 28.0
}

# Maximum acceptable market margins by client category / customer group.
# Intermediaries (NVOCCs, GSAs, Overseas Agents) have tighter margin tolerances.
CLIENT_CATEGORY_MAX_MARGIN_CAPS: Dict[str, float] = {
    "Shipper / Consignee": 35.0,
    "shipper_consignee": 35.0,
    "shipper": 35.0,
    "consignee": 35.0,
    "Subagent": 28.0,
    "subagent": 28.0,
    "Overseas Agent": 24.0,
    "overseas_agent": 24.0,
    "NVOCC / Coloader": 22.0,
    "nvocc_coloader": 22.0,
    "nvocc": 22.0,
    "coloader": 22.0,
    "Custom Broker": 25.0,
    "custom_broker": 25.0,
    "cha": 25.0,
    "Transporter": 22.0,
    "transporter": 22.0,
    "GSA": 20.0,
    "gsa": 20.0,
    "Shipping Line": 18.0,
    "shipping_line": 18.0,
    "default": 30.0
}


def get_commodity_max_margin(commodity_group: str) -> float:
    """
    Returns the maximum acceptable margin percentage cap for a commodity type.
    """
    if not commodity_group:
        return float(COMMODITY_MAX_MARGIN_CAPS["default"])

    if commodity_group in COMMODITY_MAX_MARGIN_CAPS:
        return float(COMMODITY_MAX_MARGIN_CAPS[commodity_group])

    key = str(commodity_group).strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")
    if key in COMMODITY_MAX_MARGIN_CAPS:
        return float(COMMODITY_MAX_MARGIN_CAPS[key])

    # Partial matching for common aliases
    if "garment" in key or "textile" in key or "fabric" in key or "apparel" in key:
        return float(COMMODITY_MAX_MARGIN_CAPS["garments_textiles"])
    if "auto" in key or "vehicle" in key:
        return float(COMMODITY_MAX_MARGIN_CAPS["auto_parts"])
    if "perish" in key or "food" in key or "fruit" in key or "fish" in key or "meat" in key:
        return float(COMMODITY_MAX_MARGIN_CAPS["perishable_foodstuff"])
    if "pharma" in key or "drug" in key or "vaccine" in key or "med" in key:
        return float(COMMODITY_MAX_MARGIN_CAPS["pharmaceuticals"])
    if "danger" in key or "dgr" in key or "haz" in key:
        return float(COMMODITY_MAX_MARGIN_CAPS["dangerous_goods"])
    if "valuable" in key or "gold" in key or "jewel" in key or "silver" in key:
        return float(COMMODITY_MAX_MARGIN_CAPS["valuable_cargo"])
    if "courier" in key or "express" in key:
        return float(COMMODITY_MAX_MARGIN_CAPS["courier"])
    if "engineer" in key or "machine" in key:
        return float(COMMODITY_MAX_MARGIN_CAPS["engineering_machinery"])
    if "animal" in key or "avi" in key:
        return float(COMMODITY_MAX_MARGIN_CAPS["live_animals"])

    return float(COMMODITY_MAX_MARGIN_CAPS.get("default", 28.0))


def get_client_category_max_margin(client_category: str) -> float:
    """
    Returns the maximum acceptable margin percentage cap for a client category.
    """
    if not client_category:
        return float(CLIENT_CATEGORY_MAX_MARGIN_CAPS["default"])

    if client_category in CLIENT_CATEGORY_MAX_MARGIN_CAPS:
        return float(CLIENT_CATEGORY_MAX_MARGIN_CAPS[client_category])

    key = str(client_category).strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")
    if key in CLIENT_CATEGORY_MAX_MARGIN_CAPS:
        return float(CLIENT_CATEGORY_MAX_MARGIN_CAPS[key])

    if any(k in key for k in ["direct", "shipper", "consignee"]):
        return float(CLIENT_CATEGORY_MAX_MARGIN_CAPS["shipper_consignee"])
    if "overseas" in key:
        return float(CLIENT_CATEGORY_MAX_MARGIN_CAPS["overseas_agent"])
    if "nvocc" in key or "coloader" in key:
        return float(CLIENT_CATEGORY_MAX_MARGIN_CAPS["nvocc_coloader"])
    if "custom" in key or "cha" in key:
        return float(CLIENT_CATEGORY_MAX_MARGIN_CAPS["custom_broker"])
    if "gsa" in key:
        return float(CLIENT_CATEGORY_MAX_MARGIN_CAPS["gsa"])
    if "transport" in key:
        return float(CLIENT_CATEGORY_MAX_MARGIN_CAPS["transporter"])
    if "shipping" in key:
        return float(CLIENT_CATEGORY_MAX_MARGIN_CAPS["shipping_line"])
    if "subagent" in key or "broker" in key:
        return float(CLIENT_CATEGORY_MAX_MARGIN_CAPS["subagent"])

    return float(CLIENT_CATEGORY_MAX_MARGIN_CAPS.get("default", 30.0))


def get_category_commodity_max_margin(
    commodity_group: Optional[str] = "General Cargo",
    client_category: Optional[str] = "Subagent"
) -> Dict[str, Any]:
    """
    Checks and calculates the upfront maximum margin ceiling for the specified
    category and commodity type.
    """
    comm_cap = get_commodity_max_margin(str(commodity_group or "General Cargo"))
    cat_cap = get_client_category_max_margin(str(client_category or "Subagent"))

    if comm_cap <= cat_cap:
        effective_cap = comm_cap
        limiting_factor = f"Commodity ({commodity_group}: {comm_cap:.1f}%)"
    else:
        effective_cap = cat_cap
        limiting_factor = f"Client Category ({client_category}: {cat_cap:.1f}%)"

    return {
        "commodity_group": commodity_group,
        "commodity_max_margin": comm_cap,
        "client_category": client_category,
        "client_category_max_margin": cat_cap,
        "effective_max_margin": effective_cap,
        "limiting_factor": limiting_factor
    }

