"""
src/policy_guardrails.py
Decoupled post-model commercial guardrails, operational floors, margin ceilings,
risk buffers, and governance approval rules. (Issue M1, Fix 4)
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

try:
    from constants import (
        MIN_CANDIDATE_MARGIN,
        MIN_MARGIN_PER_AWB_INR,
        MIN_MARGIN_PER_KG_INR,
        OPS_HANDLING_FEE_AWB_INR,
        compute_cost_floor_price,
        get_commodity_max_margin,
        get_client_category_max_margin,
        resolve_risk_buffer,
    )
    from taxonomy import classify_commodity
except ImportError:
    from src.constants import (
        MIN_CANDIDATE_MARGIN,
        MIN_MARGIN_PER_AWB_INR,
        MIN_MARGIN_PER_KG_INR,
        OPS_HANDLING_FEE_AWB_INR,
        compute_cost_floor_price,
        get_commodity_max_margin,
        get_client_category_max_margin,
        resolve_risk_buffer,
    )
    from src.taxonomy import classify_commodity


@dataclass(frozen=True)
class GuardrailResult:
    """
    Immutable result object representing post-model policy guardrails application.
    """
    raw_margin_pct: float
    final_margin_pct: float
    margin_amount_inr: float
    sell_price_inr: float
    rate_per_kg_inr: float
    cost_floor_margin_pct: float
    cost_floor_price_inr: float
    effective_cap_pct: float
    is_floor_clamped: bool
    is_cap_clamped: bool
    limiting_factor: str  # "none", "cost_floor_awb", "cost_floor_per_kg", "cost_floor_risk", "commodity_cap", "client_category_cap"
    approval_required: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        """Convert GuardrailResult to a standard dictionary."""
        return asdict(self)


# Canonical reference to unified risk buffer in constants.py
_resolve_risk_buffer = resolve_risk_buffer


def apply_commercial_guardrails(
    raw_margin_pct: float,
    buy_cost: float,
    chargeable_wt: float,
    commodity: str = "General Cargo",
    client_cat: str = "Subagent",
) -> GuardrailResult:
    """
    Applies deterministic commercial profitability floors, risk buffers, and margin
    ceilings to raw model predictions or strategy quotes.

    Args:
        raw_margin_pct: Raw predicted margin percentage.
        buy_cost: Base airline buy cost in INR.
        chargeable_wt: Chargeable weight in kg.
        commodity: Commodity group / cargo description.
        client_cat: Customer group / client category.

    Returns:
        GuardrailResult: Frozen dataclass with final margin, prices, clamping flags,
                         limiting factor, and required governance approvals.
    """
    raw_m = float(raw_margin_pct)
    buy = max(0.0, float(buy_cost))
    wt = max(1.0, float(chargeable_wt))
    comm_str = str(commodity or "General Cargo").strip()
    client_str = str(client_cat or "Subagent").strip()

    # 1. Compute Ceilings (Caps)
    comm_cap = float(get_commodity_max_margin(comm_str))
    cat_cap = float(get_client_category_max_margin(client_str))
    effective_cap = round(min(comm_cap, cat_cap), 2)

    # 2. Compute Cost Floor Price & Margin %
    cost_floor_price = round(
        compute_cost_floor_price(
            airline_buy_inr=buy,
            chargeable_wt_kg=wt,
            commodity_group=comm_str,
            min_pct=MIN_CANDIDATE_MARGIN,
            company_group=client_str,
        ),
        2,
    )
    if buy > 0.0:
        cost_floor_margin = round(((cost_floor_price - buy) / buy) * 100.0, 2)
    else:
        cost_floor_margin = 0.0

    # 3. Determine Cost Floor Limiting Factor Breakdown
    rate = buy / wt if wt > 0.0 else 0.0
    if wt <= 45.0:
        unit_min = MIN_MARGIN_PER_KG_INR
        awb_min = MIN_MARGIN_PER_AWB_INR
        ops_fee = OPS_HANDLING_FEE_AWB_INR
    else:
        unit_min = min(MIN_MARGIN_PER_KG_INR, max(0.50, 0.015 * rate))
        awb_min = min(MIN_MARGIN_PER_AWB_INR, max(350.0, 0.015 * buy))
        ops_fee = min(OPS_HANDLING_FEE_AWB_INR, max(200.0, 0.010 * buy))

    risk_buffer = _resolve_risk_buffer(comm_str, client_str)
    pct_floor_price = (buy * (1.0 + (MIN_CANDIDATE_MARGIN / 100.0) + risk_buffer)) + ops_fee
    abs_floor_price = buy + awb_min + (unit_min * wt)

    if risk_buffer > 0.0 and pct_floor_price >= abs_floor_price:
        floor_limiting_factor = "cost_floor_risk"
    elif (unit_min * wt) > awb_min:
        floor_limiting_factor = "cost_floor_per_kg"
    else:
        floor_limiting_factor = "cost_floor_awb"

    # 4. Approvals Verification
    approvals: List[str] = []

    # Special cargo approval triggers
    canonical_comm = classify_commodity(comm_str)
    if canonical_comm in ("Dangerous Goods", "Valuables", "Live Animals"):
        approvals.append(f"special cargo: {canonical_comm}")
    elif any(x in comm_str.lower() for x in ["danger", "dgr", "haz", "un "]):
        approvals.append("special cargo: Dangerous Goods")
    elif any(x in comm_str.lower() for x in ["valuable", "vun", "gold", "jewel"]):
        approvals.append("special cargo: Valuables")
    elif any(x in comm_str.lower() for x in ["animal", "avi", "livestock"]):
        approvals.append("special cargo: Live Animals")

    # Conflicting bounds: Cost floor exceeds maximum ceiling
    is_conflicting_bounds = bool(cost_floor_margin > effective_cap)
    if is_conflicting_bounds:
        approvals.append("cost floor above cap")

    # 5. Clamping Logic & Limiting Factor
    if is_conflicting_bounds:
        # Cost floor takes absolute precedence over cap to prevent financial loss
        final_margin_pct = cost_floor_margin
        is_floor_clamped = True
        is_cap_clamped = False
        limiting_factor = floor_limiting_factor
    elif raw_m < cost_floor_margin:
        final_margin_pct = cost_floor_margin
        is_floor_clamped = True
        is_cap_clamped = False
        limiting_factor = floor_limiting_factor
    elif raw_m > effective_cap:
        final_margin_pct = effective_cap
        is_floor_clamped = False
        is_cap_clamped = True
        if comm_cap <= cat_cap:
            limiting_factor = "commodity_cap"
        else:
            limiting_factor = "client_category_cap"
    else:
        final_margin_pct = round(raw_m, 2)
        is_floor_clamped = False
        is_cap_clamped = False
        limiting_factor = "none"

    # 6. Financial Metrics Calculation
    if is_floor_clamped:
        sell_price_inr = round(cost_floor_price, 2)
        margin_amount_inr = round(sell_price_inr - buy, 2)
    else:
        sell_price_inr = round(buy * (1.0 + final_margin_pct / 100.0), 2)
        margin_amount_inr = round(sell_price_inr - buy, 2)

    rate_per_kg_inr = round(sell_price_inr / wt, 2)

    return GuardrailResult(
        raw_margin_pct=round(raw_m, 2),
        final_margin_pct=round(final_margin_pct, 2),
        margin_amount_inr=margin_amount_inr,
        sell_price_inr=sell_price_inr,
        rate_per_kg_inr=rate_per_kg_inr,
        cost_floor_margin_pct=round(cost_floor_margin, 2),
        cost_floor_price_inr=cost_floor_price,
        effective_cap_pct=round(effective_cap, 2),
        is_floor_clamped=is_floor_clamped,
        is_cap_clamped=is_cap_clamped,
        limiting_factor=limiting_factor,
        approval_required=tuple(approvals),
    )
