"""
src/tiers.py
Single source of truth for Weight-Tier, Weight-Break Arbitrage, and Account-Tier binning logic.
"""

from typing import List, Optional, Dict, Any

try:
    from history import add_customer_history
except ImportError:
    from src.history import add_customer_history

WEIGHT_TIER_BINS: List[float] = [0.0, 45.0, 100.0, 300.0, 500.0, 1000.0, float("inf")]
WEIGHT_TIER_LABELS: List[str] = ["<45kg", "45-100kg", "100-300kg", "300-500kg", "500-1000kg", ">1000kg"]

IATA_WEIGHT_BREAKS: List[float] = [45.0, 100.0, 300.0, 500.0, 1000.0]


def get_weight_tier(chargeable_wt: float) -> str:
    """Classify chargeable weight into discrete weight tiers."""
    for boundary, label in zip(WEIGHT_TIER_BINS[1:], WEIGHT_TIER_LABELS):
        if chargeable_wt < boundary:
            return label
    return WEIGHT_TIER_LABELS[-1]


def map_account_tier(freq: int) -> str:
    """
    Classify customer inquiry frequency into customer relationship tiers.

    Units: Annualized inquiries per year (or historical total inquiries over active window).
    Thresholds:
      - Key Account: >= 20 inquiries/yr (Strategic enterprise / frequent forwarder)
      - Regular: 6 to 19 inquiries/yr (Active recurring customer)
      - Occasional: 2 to 5 inquiries/yr (Ad-hoc / seasonal customer)
      - Spot / One-Off: < 2 inquiries/yr (Single shipment inquiry)
    """
    try:
        val = int(freq or 0)
    except (TypeError, ValueError):
        val = 0

    if val >= 20:
        return "Key Account"
    if val >= 6:
        return "Regular"
    if val >= 2:
        return "Occasional"
    return "Spot / One-Off"


def get_next_weight_break(chargeable_wt: float) -> Optional[float]:
    """Return the next higher IATA standard weight break in kg, or None if >= 1000kg."""
    for b in IATA_WEIGHT_BREAKS:
        if b > chargeable_wt:
            return b
    return None


def check_weight_break_arbitrage(
    chargeable_wt: float,
    current_total_cost: float,
    next_slab_rate_per_kg: Optional[float] = None,
    slab_rates: Optional[Dict[float, float]] = None,
    default_break_discount_pct: float = 0.12,
) -> Dict[str, Any]:
    """
    Check if 'bumping' the shipment's chargeable weight to the next IATA weight break
    results in a lower total cost (Weight-Break Arbitrage / Pivot Rule).

    In air-export freight forwarding, if a 92 kg shipment at Rs 150/kg costs Rs 13,800,
    but the airline's +100 kg break rate is Rs 130/kg, billing as 100 kg costs Rs 13,000,
    saving Rs 800.

    Parameters:
      chargeable_wt: Actual chargeable weight in kg.
      current_total_cost: Total buy/carrier cost at current chargeable weight.
      next_slab_rate_per_kg: Carrier rate/kg at the next higher break (if known).
      slab_rates: Optional dictionary mapping break threshold (kg) -> rate/kg.
      default_break_discount_pct: Default airline break discount (12% typical IATA tariff break).

    Returns:
      Dict containing has_arbitrage, recommended_chargeable_wt, savings, advisory, etc.
    """
    next_break = get_next_weight_break(chargeable_wt)
    if next_break is None:
        return {
            "has_arbitrage": False,
            "original_chargeable_wt": float(chargeable_wt),
            "recommended_chargeable_wt": float(chargeable_wt),
            "original_cost": float(current_total_cost),
            "arbitrage_cost": float(current_total_cost),
            "savings": 0.0,
            "savings_pct": 0.0,
            "next_weight_break": None,
            "advisory": None,
        }

    target_rate: Optional[float] = None
    if next_slab_rate_per_kg is not None and float(next_slab_rate_per_kg) > 0:
        target_rate = float(next_slab_rate_per_kg)
    elif slab_rates and next_break in slab_rates:
        target_rate = float(slab_rates[next_break])
    elif chargeable_wt > 0 and current_total_cost > 0:
        # Near a break threshold (within 20% of break, e.g. >= 80kg for 100kg break, >= 420kg for 500kg break)
        if chargeable_wt >= 0.80 * next_break:
            cur_rate = current_total_cost / chargeable_wt
            target_rate = round(cur_rate * (1.0 - default_break_discount_pct), 2)

    if target_rate is not None and target_rate > 0:
        arbitrage_cost = round(next_break * target_rate, 2)
        if arbitrage_cost < current_total_cost:
            savings = round(current_total_cost - arbitrage_cost, 2)
            savings_pct = round((savings / current_total_cost) * 100.0, 2) if current_total_cost > 0 else 0.0
            advisory = (
                f"Weight-break arbitrage detected: Bumping declared weight from {chargeable_wt:.1f} kg to "
                f"{next_break:.1f} kg reduces carrier cost from ₹{current_total_cost:,.2f} to ₹{arbitrage_cost:,.2f}, "
                f"saving ₹{savings:,.2f} ({savings_pct}%)."
            )
            return {
                "has_arbitrage": True,
                "original_chargeable_wt": float(chargeable_wt),
                "recommended_chargeable_wt": float(next_break),
                "original_cost": float(current_total_cost),
                "arbitrage_cost": float(arbitrage_cost),
                "savings": savings,
                "savings_pct": savings_pct,
                "next_weight_break": float(next_break),
                "target_rate_per_kg": float(target_rate),
                "advisory": advisory,
            }

    return {
        "has_arbitrage": False,
        "original_chargeable_wt": float(chargeable_wt),
        "recommended_chargeable_wt": float(chargeable_wt),
        "original_cost": float(current_total_cost),
        "arbitrage_cost": float(current_total_cost),
        "savings": 0.0,
        "savings_pct": 0.0,
        "next_weight_break": float(next_break),
        "advisory": None,
    }
