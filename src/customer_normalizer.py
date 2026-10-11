"""
src/customer_normalizer.py
Customer Master Entity Normalization (Issue D6).

Resolves fragmented legal entity duplicates, corporate suffixes,
punctuation variants, and known account aliases across the Pan-India
air export pricing dataset (collapsing 1,073 raw names into <= 1,030 canonical entities).
"""

import re
from typing import Dict, Optional

# Known multi-alias mappings for accounts in data/processed/Air_Export_Pricing_Combined_ML.csv
CUSTOMER_ALIAS_MAP: Dict[str, str] = {
    # Spacing and concatenations
    "AD LOYAL LOGISTICS": "ADLOYAL LOGISTICS",
    "CAPITAL LOG INDIA": "CAPITALLOG INDIA",
    "FOUR VECTER LOGISTICS": "FOURVECTER LOGISTICS",
    "M S LOGISTICS": "MS LOGISTICS",
    "M S MS LOGISTICS": "MS LOGISTICS",
    "OCEAN SKY LOGISTICS": "OCEANSKY LOGISTICS",
    "OCEANSKY LOGISTICS PVI": "OCEANSKY LOGISTICS",
    "ONE CLICK LOGISTICS INDIA": "ONECLICK LOGISTICS INDIA",
    "ONE TEAM LOGISTICS INDIA": "ONETEAM LOGISTICS INDIA",
    "TRANS GLOBAL LOGISTICS": "TRANSGLOBAL LOGISTICS",
    "BLUE APPLE LOGISTICS SOLUTIONS": "BLUEAPPLE LOGISTIC SOLUTIONS",
    "BLUEAPPLE LOGISTICS SOLUTIONS": "BLUEAPPLE LOGISTIC SOLUTIONS",
    # Typographical and OCR errors
    "ALONSO LOGISTICS HNDIA": "ALONSO LOGISTICS INDIA",
    "BELORG SHIPPING": "BELROG SHIPPING",
    "JETFREGHT": "JETFREIGHT",
    "HINDUSTAN LOGISTICSPVT": "HINDUSTAN LOGISTICS",
    "VRRDDHI FREIGHT": "VRIDDHI FREIGHT",
    # Singular / Plural variations
    "ATHENA GLOBAL LOGISTIC": "ATHENA GLOBAL LOGISTICS",
    "FREIGHT WING LOGISTIC": "FREIGHT WING LOGISTICS",
    "GEETA SHIPPING AND CLEARING SERVICE": "GEETA SHIPPING AND CLEARING SERVICES",
    # Branch and division aliases
    "ALLANASONS PRIVATE LIMITED (DELHI)": "ALLANASONS",
    "ALLANASONS PRIVATE LIMITED": "ALLANASONS",
    "INEXT LOGISTICS & SUPPLY CHAIN (CANADA) PVT LTD.": "INEXT LOGISTICS AND SUPPLY CHAIN",
    "INEXT LOGISTICS & SUPPLY CHAIN PRIVATE LIMITED": "INEXT LOGISTICS AND SUPPLY CHAIN",
    "M&M MILITZER AND MUENCH (INDIA) PRIVATE LIMITED": "M AND M MILITZER AND MUENCH INDIA",
    "M&M MILITZER & MUENCH INDIA PVT. LTD.": "M AND M MILITZER AND MUENCH INDIA",
    "MILITZER & MUENCH": "M AND M MILITZER AND MUENCH INDIA",
}

# Corporate legal structure suffixes to strip from the end of customer names
CORPORATE_SUFFIXES = [
    r"PRIVATE\s+LIMITED",
    r"PVT\s+LTD",
    r"PVT\s+LIMITED",
    r"PRIVATE\s+LTD",
    r"P\s+LTD",
    r"P\s+LIMITED",
    r"LIMITED",
    r"LTD",
    r"INCORPORATED",
    r"INC",
    r"LLP",
    r"LLC",
    r"FZE",
    r"FZCO",
    r"CORPORATION",
    r"CORP",
    r"OPC",
    r"B\s+V",
    r"BV",
    r"PVT",
    r"CO",
    r"COMPANY",
]


def normalize_customer_name(raw_name: Optional[str]) -> str:
    """
    Normalizes a customer company name by:
    1. Trimming whitespace and converting to uppercase.
    2. Checking known alias map.
    3. Stripping noisy parenthetical branch qualifiers (e.g. DELHI, CANADA).
    4. Normalizing ampersands ('&' -> 'AND') and removing non-alphanumeric punctuation.
    5. Stripping trailing corporate legal suffixes (e.g. PVT LTD, LIMITED, INC, LLP).
    6. Resolving cleaned names against the alias dictionary.

    Args:
        raw_name: The raw string customer name from TMS / CRM quotes.

    Returns:
        Canonical entity name in uppercase, or empty string if input is falsy.
    """
    if not raw_name or not isinstance(raw_name, str):
        return ""

    raw_clean = raw_name.strip()
    if not raw_clean:
        return ""

    # Check exact match against alias map before cleaning
    if raw_clean in CUSTOMER_ALIAS_MAP:
        return CUSTOMER_ALIAS_MAP[raw_clean]

    s = raw_clean.upper()
    if s in CUSTOMER_ALIAS_MAP:
        return CUSTOMER_ALIAS_MAP[s]

    # Strip branch qualifiers in parentheses e.g. (DELHI), (CANADA), (LONDON), (EUROPE), (HUF)
    s = re.sub(r"\((DELHI|CANADA|LONDON|EUROPE|INDIA|I|PROP|HUF)\b.*?\)", "", s)

    # Normalize ampersands
    s = re.sub(r"&", " AND ", s)

    # Remove non-alphanumeric punctuation and noise
    s = re.sub(r"[^A-Z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    if s in CUSTOMER_ALIAS_MAP:
        return CUSTOMER_ALIAS_MAP[s]

    # Iteratively strip corporate legal suffixes from end of string
    changed = True
    while changed:
        changed = False
        for suf in CORPORATE_SUFFIXES:
            pattern = r"\b" + suf + r"$"
            match = re.search(pattern, s)
            if match:
                s = re.sub(pattern, "", s).strip()
                changed = True
                break

    # Final lookup in alias dictionary after suffix removal
    if s in CUSTOMER_ALIAS_MAP:
        return CUSTOMER_ALIAS_MAP[s]

    return s


def get_canonical_customer_id(raw_name: Optional[str]) -> str:
    """
    Returns a deterministic entity identifier slug for a customer.

    Format: 'CUST_<SLUG>' (e.g., 'CUST_ALLANASONS').
    If the name is empty or unresolved, returns 'CUST_UNKNOWN'.

    Args:
        raw_name: The raw string customer name.

    Returns:
        Deterministic customer entity ID string.
    """
    normalized = normalize_customer_name(raw_name)
    if not normalized:
        return "CUST_UNKNOWN"

    slug = re.sub(r"[^A-Z0-9]+", "_", normalized).strip("_")
    return f"CUST_{slug}" if slug else "CUST_UNKNOWN"
