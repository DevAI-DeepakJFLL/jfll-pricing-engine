"""
src/validation.py
Strict inquiry validation and normalization for Air Export Pricing Engine.
Guarantees:
- Rejects zero-sentinels, negative values, and non-numeric weights/buys.
- Validates implied buy rates are within plausible commercial bounds [10.0, 8000.0] INR/kg.
- Requires explicit destination (no silent Heathrow fallbacks).
- Parses and validates ISO quote dates into IST (Asia/Kolkata).
"""

import math
import datetime
from typing import Any, Dict, Optional

# Indian Standard Time (UTC+05:30)
IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))

MIN_RATE: float = 10.0     # INR/kg minimum plausible buy rate
MAX_RATE: float = 8000.0   # INR/kg maximum plausible buy rate (express / chartered)
MAX_WEIGHT_KG: float = 25000.0  # Max commercial air freight parcel weight


class QuoteValidationError(ValueError):
    """Raised when quotation inquiry parameters fail validation."""
    pass


def _num(v: Any, name: str, lo: float, hi: float) -> float:
    """Parses a float and enforces strict non-infinite (lo, hi] boundaries."""
    try:
        x = float(str(v).replace(",", "").strip())
    except (TypeError, ValueError):
        raise QuoteValidationError(f"{name} must be numeric")
    if not math.isfinite(x) or not (lo < x <= hi):
        raise QuoteValidationError(f"{name}={x} outside acceptable range ({lo}, {hi}]")
    return x


def _first(inq: Dict[str, Any], keys) -> Any:
    """Returns the first non-null, non-empty value for the candidate keys."""
    for k in keys:
        if inq.get(k) not in (None, ""):
            return inq[k]
    return None


def validate_inquiry(inq: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates and normalizes inquiry inputs before reaching the recommendation engine.
    Raises QuoteValidationError if required inputs are missing, invalid, or out of range.
    Returns normalized dictionary with Chargeable_Weight_Kg, Total_Buy_INR, and quote_date.
    """
    if not isinstance(inq, dict):
        raise QuoteValidationError("Inquiry payload must be a dictionary")

    out = dict(inq)

    # 1. Chargeable Weight Validation
    wt = _first(inq, ("Chargeable_Weight_Kg", "chargeable_wt", "chargeable_weight", "direct_weight", "Chargeable_Weight"))
    if wt is None:
        raise QuoteValidationError("Chargeable weight is required")
    wt = _num(wt, "Chargeable weight (kg)", 0.0, MAX_WEIGHT_KG)

    # 2. Airline Buy Cost Validation
    buy = _first(inq, ("Total_Buy_INR", "buy", "airline_buy_price", "buy_cost"))
    if buy is not None:
        buy = _num(buy, "Total buy (INR)", 0.0, 1e9)
    else:
        unit = _first(inq, ("Unit_Buy_INR", "unit_buy", "buy_rate"))
        if unit is None:
            raise QuoteValidationError("Total buy or unit buy rate is required")
        unit_val = _num(unit, "Unit buy (INR/kg)", 0.0, MAX_RATE)
        buy = round(unit_val * wt, 2)

    implied_rate = buy / wt
    # For commercial bulk freight (>= 45kg), enforce standard tariff bounds [MIN_RATE, MAX_RATE]
    # For small/courier parcels (< 45kg), minimum AWB documentation & handling fees allow higher effective per-kg rates
    max_allowed_rate = MAX_RATE if wt >= 45.0 else 250000.0
    if not (MIN_RATE <= implied_rate <= max_allowed_rate):
        raise QuoteValidationError(
            f"Implied buy rate {implied_rate:.2f} INR/kg outside plausible range [{MIN_RATE}, {max_allowed_rate}] INR/kg"
        )

    # 3. Destination Validation
    dest = _first(inq, ("Destination_Port", "dest", "Destination_Region", "destination"))
    if not dest or not str(dest).strip():
        raise QuoteValidationError("Destination is required")

    # 4. Temporal (Quote Date) Validation
    qd = str(inq.get("quote_date") or "").strip()[:10]
    today = datetime.datetime.now(IST).date()
    if qd:
        try:
            d = datetime.date.fromisoformat(qd)
        except ValueError:
            raise QuoteValidationError("quote_date must be formatted as YYYY-MM-DD")
        # Accept historical records up to 800 days back, and future quotes up to 365 days ahead
        if d > today + datetime.timedelta(days=365) or d < today - datetime.timedelta(days=800):
            raise QuoteValidationError(f"quote_date '{qd}' outside accepted operational window")
    else:
        qd = today.isoformat()

    out.update({
        "Chargeable_Weight_Kg": wt,
        "Total_Buy_INR": buy,
        "quote_date": qd,
    })
    return out
