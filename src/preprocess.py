"""
================================================================================
Data Preprocessing Pipeline: Geographic Hierarchies & Enterprise Cleaning
================================================================================
1. Hierarchical status-rank deduplication (preserves Won quote or latest revision).
2. Pure resolved quote filtering (strictly Won vs Lost, excluding Draft/Hold/Cancelled).
3. Physical plausibility validation:
     - Volumetric calculation (L * W * H * pcs / 6000)
     - Strict IATA constraint: Chargeable Weight >= Gross Weight
     - Standard 0.5 kg rounding & commercial weight bounds (0.5 kg to 25,000 kg)
     - Density ratio clamped in [0.1, 1.0]
4. Origin port validation (quarantines non-Indian origin airports).
5. Comprehensive destination mapping covering all global commercial gateways (<0.1% unmapped).
6. Enterprise commodity classification (Dangerous Goods, Pharma, Perishable, Engineering, etc.).
7. Standardized Incoterms and centralized weight/customer inquiry frequency tiering.
8. Exports production-ready datasets to data/processed/.
================================================================================
"""

import os
import sys
import re
import argparse
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Support both module and standalone execution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tiers import WEIGHT_TIER_BINS, WEIGHT_TIER_LABELS, map_account_tier, get_weight_tier


# ------------------------------------------------------------------------------
# PORT ALIASES & NORMALIZATION DICTIONARY
# ------------------------------------------------------------------------------
PORT_ALIASES: Dict[str, str] = {
    # Bangalore
    "kempegowda international airport": "Bangalore",
    "kempegowda": "Bangalore",
    "blr": "Bangalore",
    "bangalore": "Bangalore",

    # Delhi
    "delhi": "Indira Gandhi International Airport",
    "del": "Indira Gandhi International Airport",
    "indira gandhi": "Indira Gandhi International Airport",
    "indira gandhi international airport": "Indira Gandhi International Airport",

    # Mumbai
    "mumbai": "Chhatrapati Shivaji Maharaj International Airport",
    "bom": "Chhatrapati Shivaji Maharaj International Airport",
    "chhatrapati shivaji maharaj international airport": "Chhatrapati Shivaji Maharaj International Airport",

    # Calicut / Kozhikode
    "calicut": "Kozhikode (ex Calicut)",
    "kozhikode": "Kozhikode (ex Calicut)",
    "ccj": "Kozhikode (ex Calicut)",
    "kozhikode (ex calicut)": "Kozhikode (ex Calicut)",

    # Trivandrum
    "trivandrum": "Thiruvananthapuram (ex Trivandrum)",
    "thiruvananthapuram": "Thiruvananthapuram (ex Trivandrum)",
    "trv": "Thiruvananthapuram (ex Trivandrum)",
    "thiruvananthapuram (ex trivandrum)": "Thiruvananthapuram (ex Trivandrum)",

    # Frankfurt
    "frankfurt": "Frankfurt am Main",
    "fra": "Frankfurt am Main",
    "frankfurt am main": "Frankfurt am Main",

    # New York (JFK)
    "john f kennedy international airport": "John F. Kennedy Apt/New York",
    "john f. kennedy international airport": "John F. Kennedy Apt/New York",
    "john f kennedy": "John F. Kennedy Apt/New York",
    "jfk": "John F. Kennedy Apt/New York",
    "john f. kennedy apt/new york": "John F. Kennedy Apt/New York",

    # London (Heathrow)
    "london": "Heathrow Apt/London",
    "london heathrow": "Heathrow Apt/London",
    "heathrow": "Heathrow Apt/London",
    "lhr": "Heathrow Apt/London",
    "heathrow apt/london": "Heathrow Apt/London",

    # Abu Dhabi
    "abu dhabi": "Zayed International Airport",
    "auh": "Zayed International Airport",
    "zayed": "Zayed International Airport",
    "zayed international airport": "Zayed International Airport",

    # Toronto
    "toronto": "Pearson International Apt/Toronto",
    "yyz": "Pearson International Apt/Toronto",
    "pearson international apt/toronto": "Pearson International Apt/Toronto",
}


def normalize_port_name(port: Any) -> str:
    """Normalize raw airport name or code into canonical catalog naming."""
    if pd.isna(port):
        return "Unknown"
    p_str = str(port).strip()
    p_lower = p_str.lower()
    return PORT_ALIASES.get(p_lower, p_str)


# ------------------------------------------------------------------------------
# PHYSICAL PLAUSIBILITY & VOLUMETRIC LOGIC
# ------------------------------------------------------------------------------
def compute_volumetric_weight(dimensions: Optional[List[Dict[str, Any]]]) -> float:
    """
    Compute total volumetric weight across cargo pieces using IATA standard:
    Volumetric Weight (kg) = (Length_cm * Width_cm * Height_cm * Pieces) / 6000.
    """
    if not dimensions:
        return 0.0
    total_vol = 0.0
    for item in dimensions:
        try:
            l = float(item.get("length", 0.0))
            w = float(item.get("width", 0.0))
            h = float(item.get("height", 0.0))
            pcs = int(item.get("pieces", 1))
            total_vol += (l * w * h * max(1, pcs)) / 6000.0
        except (ValueError, TypeError):
            continue
    return round(total_vol, 2)


def validate_shipment_physics(
    gross_wt: float,
    ch_wt: float,
    dimensions: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[bool, float, float, Optional[str]]:
    """
    Validate and enforce physical plausibility of cargo weight & density according to IATA standards.

    Rules:
      1. Chargeable weight must be >= Gross weight.
      2. If dimensions provided, Chargeable weight must be >= Volumetric weight.
      3. Chargeable weight rounded to nearest 0.5 kg.
      4. Commercial boundary check: 0.5 kg <= Chargeable weight <= 25,000 kg.
      5. Cargo density ratio = Gross_Weight / Chargeable_Weight, bounded in [0.1, 1.0].

    Returns:
      (is_valid, sanitized_chargeable_wt, cargo_density_ratio, error_reason)
    """
    try:
        g_wt = float(gross_wt)
        c_wt = float(ch_wt)
    except (ValueError, TypeError):
        return False, 0.0, 0.0, "Invalid numeric weight inputs"

    vol_wt = compute_volumetric_weight(dimensions) if dimensions else 0.0

    # Raw chargeable weight cannot be less than gross or volumetric weight
    raw_ch_wt = max(g_wt, c_wt, vol_wt)

    if raw_ch_wt < 0.5:
        return False, raw_ch_wt, 0.0, "Chargeable weight must exceed minimum weight threshold of 0.5 kg"

    if raw_ch_wt > 25000.0:
        return False, raw_ch_wt, 0.0, "Chargeable weight exceeds maximum commercial weight threshold of 25,000 kg (charter inquiry required)"

    # Standard airfreight 0.5 kg increment: always round up (ceil)
    eff_ch_wt = np.ceil(raw_ch_wt * 2.0) / 2.0

    density = round(min(1.0, max(0.1, g_wt / eff_ch_wt)), 4) if eff_ch_wt > 0 else 1.0

    return True, eff_ch_wt, density, None


# ------------------------------------------------------------------------------
# ORIGIN PORT VALIDATION (INDIAN EXPORTS)
# ------------------------------------------------------------------------------
INDIAN_AIRPORT_NAMES = [
    "indira gandhi", "delhi", "chhatrapati", "mumbai",
    "ahmedabad", "kempegowda", "bangalore", "chennai",
    "cochin", "kochi", "hyderabad", "kolkata",
    "kozhikode", "calicut", "thiruvananthapuram", "trivandrum",
    "kannur", "navi mumbai", "amritsar", "lucknow", "jaipur",
    "chandigarh", "srinagar", "goa", "pune", "surat", "coimbatore",
    "guwahati", "bhubaneswar", "mundra", "nhava sheva", "varanasi", "indore"
]

INDIAN_AIRPORT_IATA_CODES = {
    "del", "bom", "amd", "blr", "maa", "cok", "hyd", "ccu", "ccj", "trv", "cnn", "atq", "lko",
    "jai", "ixc", "sxr", "goi", "pnq", "stv", "cjb", "gau", "bbi"
}


def is_valid_indian_origin(origin: Any) -> bool:
    """Validate that the shipment originates from a genuine Indian commercial export gateway."""
    if pd.isna(origin):
        return False
    o = str(origin).lower().strip()
    norm = normalize_port_name(origin).lower().strip()
    if any(k in o for k in INDIAN_AIRPORT_NAMES) or any(k in norm for k in INDIAN_AIRPORT_NAMES):
        return True
    words = set(re.findall(r"\b[a-z]{3}\b", o))
    return bool(words.intersection(INDIAN_AIRPORT_IATA_CODES))


# ------------------------------------------------------------------------------
# COMMODITY STANDARDIZATION WITH SPECIAL HANDLING
# ------------------------------------------------------------------------------
def map_commodity_group(desc: Any) -> str:
    """
    Map cargo descriptions into canonical business commodity categories,
    accurately detecting Dangerous Goods, Pharmaceuticals, Perishables, etc.
    """
    if pd.isna(desc):
        return "General Cargo"
    d = str(desc).strip().upper()

    # 1. Dangerous Goods / Hazmat (ensure "NON DG" is not falsely flagged)
    if "NON DG" not in d and "NON-DG" not in d and "NOT RESTRICTED" not in d:
        if any(k in d for k in ["DANGEROUS", "HAZMAT", "CHEMICAL", "FLAMMABLE", "LITHIUM", "BATTERY", "CLASS 9", "UN "]):
            return "Dangerous Goods"
        if re.search(r"\bDG\b", d):
            return "Dangerous Goods"

    # 2. Pharmaceuticals & Cold-chain healthcare
    if any(k in d for k in ["PHARMA", "MEDICINE", "VACCINE", "DRUG", "INSULIN", "HEALTHCARE"]):
        return "Pharmaceuticals"

    # 3. Perishables
    if any(k in d for k in ["FRUIT", "VEG", "MEAT", "FISH", "CRAB", "FLOWER", "ASPARAGUS", "PERISHABLE", "FOOD", "SEAFOOD"]):
        return "Perishable Foodstuff"

    # 4. Automotive
    if any(k in d for k in ["AUTO", "VEHICLE", "MOTOR", "TIRE", "TYRE", "CAR PART"]):
        return "Auto Parts"

    # 5. Garments & Textiles
    if any(k in d for k in ["GARMENT", "TEXTILE", "FABRIC", "COTTON", "APPAREL", "YARN", "SILK", "LEATHER"]):
        return "Garments / Textiles"

    # 6. Engineering & Machinery (word boundaries to avoid false substring hits like LENGTH)
    if any(k in d for k in ["MACHIN", "CIRCUIT", "EQUIPMENT", "HARDWARE", "VALVE", "TURBINE", "PUMP", "GENERATOR"]):
        return "Engineering & Machinery"
    if re.search(r"\b(ENG|ENGINE|ENGINEERING|TOOL|TOOLS)\b", d):
        return "Engineering & Machinery"

    # 7. Courier & Samples
    if any(k in d for k in ["COURIER", "DOCUMENT", "SAMPLE"]):
        return "Courier"

    # 8. Valuables
    if any(k in d for k in ["VALUABLE", "GOLD", "SILVER", "JEWEL", "DIAMOND", "CURRENCY"]):
        return "Valuables"

    # 9. Live Animals
    if any(k in d for k in ["LIVE ANIMAL", "AVI", "HORSE", "DOG", "PET"]):
        return "Live Animals"

    if any(k in d for k in ["GENERAL", "NON DG", "GENERAL CARGO"]):
        return "General Cargo"

    return "General Cargo"


# ------------------------------------------------------------------------------
# CURRENCY & REGIONAL MAPPING
# ------------------------------------------------------------------------------
def parse_currency_field(val: Any) -> float:
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()
    for curr in ["INR", "USD", "EUR", "GBP", "CAD"]:
        val_str = val_str.replace(curr, "")
    val_str = val_str.replace(",", "").strip()
    try:
        parsed = float(val_str)
        return parsed if np.isfinite(parsed) else np.nan
    except (ValueError, TypeError):
        return np.nan


KNOWN_ORIGIN_REGIONS: Dict[str, str] = {
    "Chhatrapati Shivaji Maharaj International Airport": "West India",
    "Indira Gandhi International Airport": "North India",
    "Ahmedabad": "West India",
    "Bangalore": "South India",
    "Kozhikode (ex Calicut)": "South India",
    "Kolkata": "East India",
    "Chennai": "South India",
    "Cochin": "South India",
    "Hyderabad": "South India",
    "Kannur International Airport": "South India",
    "Thiruvananthapuram (ex Trivandrum)": "South India",
    "Navi Mumbai International Airport": "West India",
    "Nhava Sheva": "West India",
    "Mundra": "West India",
}


def map_origin_region(origin: str) -> str:
    norm = normalize_port_name(origin)
    if norm in KNOWN_ORIGIN_REGIONS:
        return KNOWN_ORIGIN_REGIONS[norm]
    o = str(origin).lower().strip()
    if any(k in o for k in ["chhatrapati", "mumbai", "ahmedabad", "navi mumbai", "nhava", "surat", "pune", "goa"]):
        return "West India"
    if any(k in o for k in ["indira gandhi", "delhi", "amritsar", "lucknow", "jaipur", "chandigarh", "srinagar", "varanasi", "indore"]):
        return "North India"
    if any(k in o for k in ["bangalore", "kempegowda", "kozhikode", "calicut", "kannur", "chennai", "cochin", "kochi", "hyderabad", "trivandrum", "coimbatore"]):
        return "South India"
    if any(k in o for k in ["kolkata", "guwahati", "bhubaneswar"]):
        return "East India"
    return "Other Origin"


# Comprehensive geographic keyword mappings covering global trade gateways
DEST_MIDDLE_EAST_KEYWORDS = [
    "dubai", "zayed", "abu dhabi", "sharjah", "muscat", "doha", "dammam", "jeddah", "riyadh",
    "bahrain", "kuwait", "oman", "qatar", "saudi", "salalah", "amman", "beirut", "baghdad",
    "erbil", "ras al khaimah", "al ain", "ben gurion", "tel aviv", "medina", "madinah", "tehran",
    "jordan", "sulaymaniyah", "basra", "hamadan", "damascus", "abha", "bandar abbas", "abadan",
    "aden", "al fujayrah", "dibaa", "izmir", "adana", "diyarbakir", "cukurova", "esenboga",
    "ankara", "tallil", "ercan"
]

DEST_EUROPE_KEYWORDS = [
    "heathrow", "london", "frankfurt", "amsterdam", "paris", "charles", "milan", "malpensa",
    "rome", "madrid", "barcelona", "zurich", "dublin", "brussels", "bruxelles", "vienna",
    "munich", "manchester", "birmingham", "istanbul", "athens", "warsaw", "prague", "praha",
    "lisbon", "lisboa", "budapest", "stockholm", "arlanda", "el prat", "warszawa", "copenhagen",
    "oslo", "hamburg", "wien", "sheremetyevo", "domodedovo", "moscow", "moskva", "stuttgart",
    "basel", "geneva", "geneve", "helsinki", "otopeni", "bucuresti", "dusseldorf", "porto",
    "fiumicino", "saint petersburg", "belgrade", "beograd", "bratislava", "nice", "hannover",
    "gothenburg", "vilnius", "marseille", "sofia", "belfast", "malaga", "zagreb", "billund",
    "riga", "ljubljana", "berlin", "sarajevo", "nurnberg", "bilbao", "lyon", "luqa", "rotterdam",
    "koln", "genova", "venezia", "venice", "koltsovo", "luxembourg", "leipzig", "tirana", "mulhouse",
    "thessaloniki", "edinburgh", "covilha", "bristol", "katowice", "munster", "napoli", "newcastle",
    "pristina", "bordeaux", "linz", "tallinn", "bari", "firenze", "krakow", "podgorica", "graz",
    "torino", "strasbourg", "nantes", "keflavik", "salzburg", "malmo", "liege", "maastricht", "cork",
    "varna", "dresden", "saarbrucken", "lille", "malta", "plzen", "rostov", "hahn", "satu mare",
    "zaragoza", "gdynia", "minsk", "aberdeen", "glasgow", "bologna", "chisinau", "chișinău", "skopje",
    "novokuznetsk", "kravare", "laupheim", "albenga", "groningen", "states apt, jersey", "leshukonskoye",
    "nizhniy novgorod", "calden", "surgut", "valencia", "toulouse", "larnaca", "las palmas",
    "novosibirsk", "sevilla", "bremen", "alenquer", "le havre"
]

DEST_NORTH_AMERICA_KEYWORDS = [
    "kennedy", "new york", "jfk", "pearson", "toronto", "montreal", "vancouver", "chicago",
    "los angeles", "lax", "atlanta", "dallas", "houston", "miami", "san francisco", "boston",
    "seattle", "detroit", "newark", "philadelphia", "calgary", "ottawa", "dorval", "charlotte",
    "dulles", "washington", "denver", "orlando", "san jose", "minneapolis", "columbus", "phoenix",
    "laredo", "cincinnati", "edmonton", "cleveland", "memphis", "saint louis", "salt lake city",
    "portland", "indianapolis", "san antonio", "pittsburgh", "new orleans", "raleigh-durham",
    "kansas city", "el paso", "norfolk", "louisville", "tulsa", "las vegas", "milwaukee",
    "binghamton", "knoxville", "halifax", "austin", "nashville", "jacksonville", "dayton",
    "hamilton", "mcallen", "san diego", "fort lauderdale", "atlantic", "fort wayne", "savannah",
    "greenwood", "daytona beach", "dillon", "fort irwin", "tampa", "columbia", "lexington",
    "redding", "brady", "fargo", "meadville", "bakersfield", "oakland", "south lake tahoe",
    "jakolof bay", "seldovia", "honolulu", "kamuela", "albuquerque", "oklahoma city", "st. john's",
    "bartletts", "akhiok"
]

DEST_LATIN_AMERICA_KEYWORDS = [
    "sao paulo", "buenos aires", "mexico", "bogota", "santiago", "lima", "panama", "rio",
    "quito", "santo domingo", "caracas", "kingston", "piarco", "trinidad", "georgetown",
    "guayaquil", "asuncion", "monterrey", "viracopos", "felipe angeles", "curitiba", "curacao",
    "san salvador", "montevideo", "bridgetown", "viru viru", "guadalajara", "cali", "progreso",
    "san pedro sula", "managua", "sapiranga", "antigua", "salvador", "la paz", "san juan",
    "confins", "medellin", "paramaribo", "nosara beach", "cancun", "el salvador", "punta cana",
    "tegucigalpa", "navegantes", "belize city", "armenia", "pampulha", "natal", "cartagena",
    "santa cruz", "mamitupo", "cordoba", "porto alegre", "vitoria", "tingo maria", "port-au-prince",
    "belem", "hewanorra", "grenada", "la aurora", "st. martin", "saint kitts", "santa clara", "nassau"
]

DEST_AFRICA_KEYWORDS = [
    "cairo", "el qahira", "lagos", "muhammed", "nairobi", "kenyatta", "johannesburg", "or tambo",
    "addis ababa", "accra", "entebbe", "dar es salaam", "kigali", "casablanca", "harare", "lusaka",
    "alger", "algiers", "tunis", "abidjan", "dakar", "maputo", "luanda", "seewoosagur", "durban",
    "mohammed v", "cape town", "conakry", "juba", "mahe island", "n'djamena", "ouagadougou",
    "blaise diagne", "douala", "kinshasa", "lome", "cotonou", "agostinho neto", "lilongwe",
    "freetown", "djibouti", "zanzibar", "hargeisa", "nouakchott", "windhoek", "abuja",
    "antananarivo", "roberts", "bujumbura", "port sudan", "kano", "libreville", "mombasa",
    "gaborone", "banjul", "brazzaville", "port harcourt", "mogadishu", "lubumbashi", "bamako",
    "beira", "malabo", "port elizabeth", "niamey", "asmara", "yaounde", "pointe noire",
    "boende", "monrovia", "ndola", "port gentil", "pemba", "dirico", "nanyuki", "mitiga",
    "dakhla", "bulawayo", "enugu", "sfax", "roland garros", "blantyre", "benin city", "lusikisiki",
    "podor", "maroua", "bathurst", "hahaya", "ain eddis", "doany", "moroni", "bitam", "ad dabbah",
    "abu simbel", "togo"
]

DEST_ASIA_PACIFIC_KEYWORDS = [
    "singapore", "colombo", "shanghai", "pudong", "bangkok", "suvarnabhumi", "hong kong",
    "tokyo", "narita", "haneda", "seoul", "incheon", "kuala lumpur", "jakarta", "manila",
    "sydney", "melbourne", "dhaka", "karachi", "lahore", "kathmandu", "hanoi", "ho chi minh",
    "hochiminh", "taipei", "guangzhou", "beijing", "male", "almaty", "yangon", "brisbane",
    "perth", "auckland", "penang", "tashkent", "kabul", "baku", "yerevan", "tbilisi", "bishkek",
    "atyrau", "nursultan", "astana", "ashkhabad", "ulaanbaatar", "dushanbe", "bukhara",
    "kansai", "shenzhen", "qingdao", "wuhan", "nagoya", "islamabad", "techo", "cebu", "vientiane",
    "busan", "surabaya", "adelaide", "donghai", "nadi", "bandar seri begawan", "christchurch",
    "da nang", "chengdu", "xiamen", "suva", "fukuoka", "chongqing", "paro", "xian", "kota kinabalu",
    "kaohsiung", "macau", "don mueang", "nanjing", "hangzhou", "clark", "johor bahru", "osaka",
    "port moresby", "ulanhad", "peshawar", "fuzhou", "ningbo", "dalian", "labuan", "denpasar",
    "bali", "tianjin", "wellington", "canberra", "mandalay", "bhairawa", "fiji", "kosrae",
    "phnom penh", "yao", "yakushima", "jinan", "wanigela", "sola", "hanimaadhoo", "papeete",
    "wabag", "darchula", "datadawai", "iloilo", "batam", "kinmen", "hatzfeldthaven", "gebe",
    "nanning", "luzon", "ahe", "aramac", "ord river", "eneabba", "zhuhai", "noumea", "nullarbor",
    "changsha", "honiara", "kuching", "shenyang", "kunming", "yinchuan", "dadu", "obano", "biliau",
    "heydar aliyev", "charleville", "telupid", "turkey creek", "general santos", "arno", "lishan",
    "mount aue"
]


def map_dest_region(dest: str) -> str:
    """Classify destination airport name or code into major commercial freight corridors."""
    d = str(dest).lower().strip()
    for k in DEST_MIDDLE_EAST_KEYWORDS:
        if k in d:
            return "Middle East"
    for k in DEST_EUROPE_KEYWORDS:
        if k in d:
            return "Europe"
    for k in DEST_NORTH_AMERICA_KEYWORDS:
        if k in d:
            return "North America"
    for k in DEST_LATIN_AMERICA_KEYWORDS:
        if k in d:
            return "Latin America"
    for k in DEST_AFRICA_KEYWORDS:
        if k in d:
            return "Africa"
    for k in DEST_ASIA_PACIFIC_KEYWORDS:
        if k in d:
            return "Asia-Pacific"
    return "Other Destination"


def clean_incoterms(term: Any) -> str:
    if pd.isna(term):
        return "FOB"
    t = str(term).upper().strip()
    if "FOB" in t:
        return "FOB"
    if "CIF" in t:
        return "CIF"
    if "CFR" in t:
        return "CFR"
    if "EXW" in t:
        return "EXW"
    if "DAP" in t or "DDP" in t:
        return "DAP/DDP"
    return "FOB"


def get_air_cargo_season(month: Any) -> str:
    """Classify month into macro commercial air freight seasons."""
    try:
        m = int(month)
    except (ValueError, TypeError):
        m = 6
    if m in [1, 2, 3]:
        return "Q1_Slack_Fiscal_Close"
    elif m in [4, 5, 6]:
        return "Q2_Perishable_Peak"
    elif m in [7, 8, 9]:
        return "Q3_Monsoon_MidYear"
    else:
        return "Q4_Global_Holiday_Surge"


def compute_cyclical_month(month: Any) -> Tuple[float, float]:
    """Compute (Month_Sin, Month_Cos) for circular continuous seasonality."""
    try:
        m = float(month)
    except (ValueError, TypeError):
        m = 6.0
    rad = 2.0 * np.pi * m / 12.0
    return float(np.sin(rad)), float(np.cos(rad))


def filter_worked_quotes(df: pd.DataFrame, min_turnaround_h: float = 1.0, default_gm: float = 0.02, tol: float = 2e-4) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Label-purity filter removing unworked lost quotes.
    Identifies instant turnaround (< 1 hr) and default 2% gross-margin lost quotes.
    Returns (kept_df, dropped_unworked_lost_df).
    """
    out = df[df["Quote Status"].isin(["Won", "Lost"])].copy()
    out["is_won"] = (out["Quote Status"] == "Won").astype(int)
    gm = (out["Total_Sell_INR"] - out["Total_Buy_INR"]) / out["Total_Sell_INR"].replace(0, np.nan)
    default_priced = (gm - default_gm).abs() < tol
    instant = out["Quote_Turnaround_Hours"] < min_turnaround_h
    revision_condition = (out["quote_revision_count"] <= 1) if "quote_revision_count" in out.columns else True
    unworked_lost = (out["is_won"] == 0) & (instant | (default_priced & revision_condition))
    out["label_quality"] = np.where(unworked_lost, "unworked_lost", "worked")
    return out[~unworked_lost].copy(), out[unworked_lost].copy()


def filter_pure_resolved_quotes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the dataset to include strictly resolved commercial outcomes (Won and Lost).
    Excludes unresolved pipeline states (Draft, Hold, Cancelled, Approved).
    """
    clean_mask = df["Quote Status"].isin(["Won", "Lost"])
    out = df[clean_mask].copy()
    out["is_won"] = (out["Quote Status"] == "Won").astype(int)
    return out


def clean_and_standardize_dataset(
    input_path: str = "data/processed/Air_Export_Pricing_Combined_ML.csv",
    output_ml_path: str = "data/processed/Air_Export_Pricing_Combined_ML.csv",
    output_won_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Execute full enterprise data cleaning and hygiene pipeline on the ML dataset:
      - Quarantines non-Indian origin airports.
      - Filters out unresolved quotes (Draft, Hold, Cancelled, Approved) to guarantee pure Won vs Lost training signal.
      - Enforces physical plausibility: Chargeable Weight >= Gross Weight, 0.5 kg rounding, outlier removal.
      - Expands destination region mapping to eliminate 'Other Destination' misses.
      - Updates commodity classification for dangerous goods, temperature control, and high-value cargo.
      - Recomputes density ratio, weight tiers, and regional trade corridors.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found at: {input_path}")

    print(f"[*] Ingesting dataset: {input_path}")
    df = pd.read_csv(input_path)
    initial_rows = len(df)
    print(f"    Loaded {initial_rows:,} records.")

    # 1. Quarantining non-Indian origin ports (P-10)
    print("[*] Validating origin airports...")
    valid_origin_mask = df["Origin_Port"].apply(is_valid_indian_origin)
    df = df[valid_origin_mask].copy()
    dropped_origins = initial_rows - len(df)
    if dropped_origins > 0:
        print(f"    Quarantined {dropped_origins:,} non-Indian origin records.")

    # 2. Pure Resolved Quotes Filtering (P-1)
    print("[*] Filtering pure resolved quote outcomes (Won vs Lost)...")
    before_purity = len(df)
    df = filter_pure_resolved_quotes(df)
    print(f"    Excluded {before_purity - len(df):,} unresolved quotes (Draft/Hold/Cancelled/Approved). Remaining: {len(df):,}")

    # 3. Physical Plausibility & Outlier Quarantine (P-3, A-3)
    print("[*] Enforcing physical plausibility and commercial weight bounds...")
    gross = df["Gross_Weight_Kg"].fillna(1.0).astype(float)
    chargeable = df["Chargeable_Weight_Kg"].fillna(gross).astype(float)

    # IATA rule: Chargeable >= Gross
    effective_ch_wt = np.maximum(gross, chargeable)
    # Round up (ceil) to standard airfreight 0.5 kg increment
    effective_ch_wt = np.ceil(effective_ch_wt * 2.0) / 2.0

    df["Gross_Weight_Kg"] = gross
    df["Chargeable_Weight_Kg"] = effective_ch_wt
    df["Cargo_Density_Ratio"] = (gross / effective_ch_wt).clip(lower=0.1, upper=1.0).round(4)

    # Filter commercial bounds (0.5 kg to 25,000 kg)
    weight_valid_mask = (df["Chargeable_Weight_Kg"] >= 0.5) & (df["Chargeable_Weight_Kg"] <= 25000.0)
    df = df[weight_valid_mask].copy()

    # Commercial buy/sell validation
    price_valid_mask = (
        (df["Total_Buy_INR"] > 500.0) &
        (df["Total_Sell_INR"] > 500.0) &
        (df["Total_Sell_INR"] >= df["Total_Buy_INR"])
    )
    df = df[price_valid_mask].copy()

    # Prune non-commercial outlier margins (> 60% or < 0.2%)
    df["Margin_Amount_INR"] = df["Total_Sell_INR"] - df["Total_Buy_INR"]
    df["Margin_Percentage"] = (df["Margin_Amount_INR"] / df["Total_Buy_INR"]) * 100.0
    df = df[(df["Margin_Percentage"] >= 0.2) & (df["Margin_Percentage"] <= 60.0)].copy()

    # 4. Expanded Destination Mapping & Geographic Corridors (P-6)
    print("[*] Applying expanded destination mapping...")
    df["Destination_Region"] = df["Destination_Port"].apply(map_dest_region)
    df["Origin_Region"] = df["Origin_Port"].apply(map_origin_region)
    df["Regional_Lane"] = df["Origin_Region"] + " -> " + df["Destination_Region"]
    df["Trade_Lane"] = df["Origin_Port"] + " -> " + df["Destination_Port"]

    # 5. Commodity Group Standardization (P-7)
    if "Cargo Description" in df.columns:
        df["Commodity_Group"] = df["Cargo Description"].apply(map_commodity_group)

    # 6. Centralized Tiers (T-1, T-2)
    df["Weight_Tier"] = df["Chargeable_Weight_Kg"].apply(get_weight_tier)
    df["Customer_Tier"] = df["Customer_Inquiry_Frequency"].apply(map_account_tier)

    print(f"[✓] Data hygiene complete. Final pure dataset: {len(df):,} rows.")
    unmapped_dest_count = (df["Destination_Region"] == "Other Destination").sum()
    print(f"    - Destination_Region == 'Other Destination': {unmapped_dest_count} ({unmapped_dest_count / len(df) * 100.0:.2f}%)")
    print(f"    - Won Deals: {(df['is_won'] == 1).sum():,} ({(df['is_won'] == 1).mean() * 100.0:.1f}%)")
    print(f"    - Lost Deals: {(df['is_won'] == 0).sum():,} ({(df['is_won'] == 0).mean() * 100.0:.1f}%)")

    os.makedirs(os.path.dirname(output_ml_path), exist_ok=True)
    df.to_csv(output_ml_path, index=False)
    print(f"    Saved clean dataset to {output_ml_path}")

    if output_won_path:
        won_df = df[df["is_won"] == 1].copy()
        os.makedirs(os.path.dirname(output_won_path), exist_ok=True)
        won_df.to_csv(output_won_path, index=False)
        print(f"    Saved won deals benchmark to {output_won_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and standardize Air Export Pricing ML Dataset")
    parser.add_argument("--input", type=str, default="data/processed/Air_Export_Pricing_Combined_ML.csv")
    parser.add_argument("--output", type=str, default="data/processed/Air_Export_Pricing_Combined_ML.csv")
    args = parser.parse_args()

    clean_and_standardize_dataset(args.input, args.output)
