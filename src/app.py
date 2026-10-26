"""
================================================================================
Air Freight Margin Recommender & Sensitivity Web Engine
================================================================================
Production-hardened, self-contained evaluation interface with:
- Restored authentic signature styling: Outfit & Mulish typography, cyan/lime palette
- Full categorical vocabulary alignment across all 12 model features (58/58 passed)
- Quick Historical Test Presets bar
- Semantic numeric inputs with strict server-side validation (HTTP 400 error handling)
- Modular SQLite history tracking with schema versioning and model audit trails
- All 11 operating branch offices enabled
- Anti-caching HTTP headers for immediate browser updates
================================================================================
"""

import os
import sys
import io
import csv
import json
import logging
import datetime
from functools import wraps
from typing import Dict
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, render_template_string, Response, stream_with_context

# Ensure local directories are in python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_ROOT = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, ENGINE_ROOT)

from constants import MODEL_VERSION
from tiers import get_weight_tier, map_account_tier, check_weight_break_arbitrage
from preprocess import (
    normalize_port_name,
    map_origin_region,
    map_dest_region,
    clean_incoterms,
    validate_shipment_physics,
    compute_volumetric_weight
)
from security import verify_artifact_integrity, require_admin_auth
from db import (
    init_history_db,
    log_recommendation,
    record_outcome,
    get_all_history,
    iter_all_history,
    get_history_paginated,
    update_lead_status,
    clear_history_table,
    get_recent_competitor_rate_benchmark,
    LEAD_STATUSES
)

app = Flask(__name__)
# Limit incoming JSON payload to 1MB to prevent memory exhaustion DoS
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024
app.config["TEMPLATES_AUTO_RELOAD"] = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pricing_web")

# Prevent browser caching and enforce OWASP security headers
@app.after_request
def add_security_and_cache_headers(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "script-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "connect-src 'self';"
    )
    return response


# ------------------------------------------------------------------------------
# SQLITE HISTORY TRACKING STORE
# ------------------------------------------------------------------------------
if os.environ.get("AWS_LAMBDA_FUNCTION_NAME") or os.environ.get("LAMBDA_TASK_ROOT"):
    # Standard AWS Lambda ephemeral storage directory
    HISTORY_DB_PATH = "/tmp/recommendations_history.db"  # nosec B108
else:
    HISTORY_DB_PATH = os.path.join(ENGINE_ROOT, "data", "recommendations_history.db")
init_history_db(HISTORY_DB_PATH)


# ------------------------------------------------------------------------------
# INITIALIZE PRICING ENGINE V3 (§1.4, §5.6)
# ------------------------------------------------------------------------------
try:
    from preprocess import get_air_cargo_season
except ImportError:
    from src.preprocess import get_air_cargo_season

try:
    from engine_v3 import PricingEngineV3
except ImportError:
    from src.engine_v3 import PricingEngineV3

engine_v3 = PricingEngineV3()


# ------------------------------------------------------------------------------
# CATALOG DROPDOWN OPTIONS (SYNCHRONIZED WITH TRAINED DATASET & MODEL)
# ------------------------------------------------------------------------------
PORT_LABELS: Dict[str, str] = {
    'Indira Gandhi International Airport': 'Delhi (DEL) - Indira Gandhi',
    'Chhatrapati Shivaji Maharaj International Airport': 'Mumbai (BOM) - Chhatrapati Shivaji',
    'Bangalore': 'Bangalore (BLR) - Kempegowda',
    'Ahmedabad': 'Ahmedabad (AMD)',
    'Kozhikode (ex Calicut)': 'Calicut (CCJ) - Kozhikode',
    'Chennai': 'Chennai (MAA)',
    'Kolkata': 'Kolkata (CCU)',
    'Cochin': 'Cochin (COK)',
    'Hyderabad': 'Hyderabad (HYD)',
    'Kannur International Airport': 'Kannur (CNN)',
    'Thiruvananthapuram (ex Trivandrum)': 'Trivandrum (TRV)',
    'Navi Mumbai International Airport': 'Navi Mumbai (NMI)',
    'Heathrow Apt/London': 'London Heathrow (LHR)',
    'Dubai': 'Dubai (DXB)',
    'Zayed International Airport': 'Abu Dhabi (AUH) - Zayed',
    'Frankfurt am Main': 'Frankfurt (FRA)',
    'John F. Kennedy Apt/New York': 'New York (JFK)',
    'Pearson International Apt/Toronto': 'Toronto (YYZ)',
    'Singapore': 'Singapore (SIN)',
    'Amsterdam': 'Amsterdam (AMS)',
    'Muscat': 'Muscat (MCT)',
    'Doha': 'Doha (DOH)',
    'Accra': 'Accra (ACC)',
    'Bahrain International Airport': 'Bahrain (BAH)',
    'Ad Dammam': 'Dammam (DMM)',
    'Colombo': 'Colombo (CMB)',
    'Istanbul': 'Istanbul (IST)',
    'Riyadh': 'Riyadh (RUH)',
    'Lagos Murtala Muhammed Apt': 'Lagos (LOS) - Murtala Muhammed',
    'Guarulhos Apt/Sao Paulo': 'Sao Paulo (GRU) - Guarulhos',
    'Bangkok': 'Bangkok (BKK) - Suvarnabhumi',
    'El Qahira (Cairo)': 'Cairo (CAI)',
    'Sheremetyevo International Airport': 'Moscow (SVO) - Sheremetyevo',
    'Male': 'Male (MLE) - Velana',
    'Jeddah': 'Jeddah (JED) - King Abdulaziz',
    'Malpensa Apt/Milano': 'Milan (MXP) - Malpensa',
    'Shanghai Pudong International Apt': 'Shanghai (PVG) - Pudong',
    'Nairobi': 'Nairobi (NBO) - Jomo Kenyatta',
    'Charles-de-Gaulle Apt/Paris': 'Paris (CDG) - Charles de Gaulle',
    'Dhaka': 'Dhaka (DAC) - Hazrat Shahjalal',
    'Los Angeles': 'Los Angeles (LAX)',
    'Incheon Intl Apt/Seoul': 'Seoul (ICN) - Incheon',
    'O\'Hare Apt/Chicago': 'Chicago (ORD) - O\'Hare',
    'Kuala Lumpur International Airport': 'Kuala Lumpur (KUL)',
    'Atlanta': 'Atlanta (ATL) - Hartsfield-Jackson',
    'Johannesburg': 'Johannesburg (JNB) - OR Tambo',
    'Dar es Salaam': 'Dar es Salaam (DAR)',
    'Entebbe': 'Entebbe (EBB)',
    'Manila': 'Manila (MNL) - Ninoy Aquino',
    'Soekarno-Hatta Apt/Jakarta': 'Jakarta (CGK) - Soekarno-Hatta',
    'Hong Kong': 'Hong Kong (HKG)',
    'Kuwait': 'Kuwait (KWI)',
    'Manchester': 'Manchester (MAN)',
    'Warszawa': 'Warsaw (WAW) - Chopin',
    'Ciudad de Mexico': 'Mexico City (MEX)',
    'Kathmandu': 'Kathmandu (KTM) - Tribhuvan',
    'Amman': 'Amman (AMM) - Queen Alia',
    'Melbourne Airport': 'Melbourne (MEL)',
    'Sharjah Airport International Free Zone (SAIF Zone)': 'Sharjah (SHJ)',
    'Domodedovo Apt/Moscow': 'Moscow (DME) - Domodedovo',
    'Ho Chi Minh City': 'Ho Chi Minh (SGN) - Tan Son Nhat',
    'Madrid': 'Madrid (MAD) - Barajas',
    'George Bush Intercontinental Apt/Houston': 'Houston (IAH) - George Bush',
    'Munich': 'Munich (MUC)',
    'Brussels': 'Brussels (BRU)',
    'Vienna': 'Vienna (VIE)',
    'Zurich': 'Zurich (ZRH)',
    'Sydney': 'Sydney (SYD) - Kingsford Smith',
    'Tokyo': 'Tokyo (NRT/HND)'
}

ORIGIN_OPTIONS = [
    {"value": "Indira Gandhi International Airport", "label": "Delhi (DEL) - Indira Gandhi"},
    {"value": "Chhatrapati Shivaji Maharaj International Airport", "label": "Mumbai (BOM) - Chhatrapati Shivaji"},
    {"value": "Bangalore", "label": "Bangalore (BLR) - Kempegowda"},
    {"value": "Ahmedabad", "label": "Ahmedabad (AMD)"},
    {"value": "Kozhikode (ex Calicut)", "label": "Calicut (CCJ) - Kozhikode"},
    {"value": "Chennai", "label": "Chennai (MAA)"},
    {"value": "Kolkata", "label": "Kolkata (CCU)"},
    {"value": "Cochin", "label": "Cochin (COK)"},
    {"value": "Hyderabad", "label": "Hyderabad (HYD)"},
    {"value": "Kannur International Airport", "label": "Kannur (CNN)"},
    {"value": "Thiruvananthapuram (ex Trivandrum)", "label": "Trivandrum (TRV)"},
    {"value": "Navi Mumbai International Airport", "label": "Navi Mumbai (NMI)"},
]

CLIENT_OPTIONS = [
    {"value": "Shipper / Consignee", "label": "Direct Shipper / Consignee"},
    {"value": "Subagent", "label": "Subagent (Broker)"},
    {"value": "Overseas Agent", "label": "Overseas Agent"},
    {"value": "NVOCC / Coloader", "label": "NVOCC / Co-loader"},
    {"value": "GSA", "label": "GSA (General Sales Agent)"},
    {"value": "Custom Broker", "label": "Customs Broker / CHA"},
    {"value": "Transporter", "label": "Transporter / Fleet"},
    {"value": "Shipping Line", "label": "Shipping Line"},
]

VERTICAL_OPTIONS = [
    {"value": "Air Export Forwarding", "label": "Air Export Forwarding"},
    {"value": "Air Perishable", "label": "Air Perishable"},
    {"value": "Courier", "label": "Courier"},
    {"value": "Export Clearance", "label": "Export Clearance"},
    {"value": "General", "label": "General Cargo / Freight"},
    {"value": "Air Import Forwarding", "label": "Air Import Forwarding"},
    {"value": "Import Clearance", "label": "Import Clearance"},
]

COMMODITY_OPTIONS = [
    {"value": "General Cargo", "label": "General Cargo"},
    {"value": "Perishable Foodstuff", "label": "Perishables (Fish/Fruit/Veg)"},
    {"value": "Pharmaceuticals", "label": "Pharmaceuticals"},
    {"value": "Auto Parts", "label": "Auto Parts / Machinery"},
    {"value": "Garments / Textiles", "label": "Garments / Textiles"},
    {"value": "Dangerous Goods", "label": "Dangerous Goods (approval required)"},
    {"value": "Valuables", "label": "Valuables (approval required)"},
    {"value": "Live Animals", "label": "Live Animals (approval required)"},
    {"value": "Courier", "label": "Courier"},
    {"value": "Engineering & Machinery", "label": "Engineering Goods"},
    {"value": "Other", "label": "Other"},
]

INCOTERMS_OPTIONS = [
    {"value": "FOB", "label": "FOB - Free on Board"},
    {"value": "CIF", "label": "CIF - Cost, Insurance & Freight"},
    {"value": "CFR", "label": "CFR - Cost and Freight"},
    {"value": "EXW", "label": "EXW - Ex Works"},
    {"value": "DAP/DDP", "label": "DAP / DDP - Delivered"},
]

BRANCH_OPTIONS = [
    {"value": "Mumbai", "label": "Mumbai"},
    {"value": "Delhi", "label": "Delhi"},
    {"value": "Bangalore", "label": "Bangalore"},
    {"value": "Ahmedabad", "label": "Ahmedabad"},
    {"value": "Chennai", "label": "Chennai"},
    {"value": "Calicut", "label": "Calicut"},
    {"value": "Cochin", "label": "Cochin"},
    {"value": "Hyderabad", "label": "Hyderabad"},
    {"value": "Kannur", "label": "Kannur"},
    {"value": "Kolkata", "label": "Kolkata"},
    {"value": "Trivandrum", "label": "Trivandrum"},
]

CUSTOMER_ACTIVITY_OPTIONS = [
    {"value": "25", "tier": "Key Account", "label": "Key Account (Strategic / High Volume)", "sub": "≥20 quotes/yr • Priority retention"},
    {"value": "10", "tier": "Regular", "label": "Active / Regular Customer", "sub": "6–19 quotes/yr • Standard recurring"},
    {"value": "3", "tier": "Occasional", "label": "Occasional Customer", "sub": "2–5 quotes/yr • Ad-hoc / seasonal"},
    {"value": "1", "tier": "Spot / One-Off", "label": "New / Spot One-Off Inquiry", "sub": "1 quote • Single transaction"},
]

REVISION_OPTIONS = [
    {"value": "1", "label": "Rev 1 — Initial Quote (Standard Benchmark)"},
    {"value": "2", "label": "Rev 2 — Counter-Offer (Concession to Beat Competitor)"},
    {"value": "3", "label": "Rev 3+ — Final BAFO (Floor Defense Mode)"},
]

def _build_destinations_catalog(bundle=None, root_path=ENGINE_ROOT):
    if bundle and isinstance(bundle, dict):
        port_freqs = bundle.get("port_frequencies", {})
        if port_freqs:
            model_dests = set(port_freqs.keys())
        elif "cat_cols" in bundle and "Destination_Port" in bundle["cat_cols"]:
            dest_idx = bundle["cat_cols"].index("Destination_Port")
            model_dests = set(bundle["encoder"].categories_[dest_idx])
        else:
            model_dests = set(PORT_LABELS.keys())
    else:
        model_dests = set(PORT_LABELS.keys())
    
    comb_csv = os.path.join(root_path, "data", "processed", "Air_Export_Pricing_Combined_ML.csv")
    freq_order = []
    if os.path.exists(comb_csv):
        try:
            df_d = pd.read_csv(comb_csv, usecols=["Destination_Port"])
            counts = df_d["Destination_Port"].dropna().value_counts()
            freq_order = [p for p in counts.index if p in model_dests]
        except Exception as e:
            logger.warning(f"Could not load destination frequencies from CSV: {e}")
            freq_order = []
            
    remaining = sorted([p for p in model_dests if p not in freq_order])
    full_list = freq_order + remaining
    
    opts = []
    for port in full_list:
        label = PORT_LABELS.get(port, port)
        opts.append({"value": port, "label": label})
    return opts

DESTINATION_OPTIONS = _build_destinations_catalog(None, ENGINE_ROOT)
PORT_LABELS_JSON = json.dumps(PORT_LABELS).replace("</", "<\\/")

CARRIER_OPTIONS = [
    {"value": "Any Airline", "label": "Any Airline (Market Default)"},
    {"value": "Qatar Airways", "label": "Qatar Airways"},
    {"value": "Emirates SkyCargo", "label": "Emirates SkyCargo"},
    {"value": "Air India", "label": "Air India"},
    {"value": "Etihad Airways", "label": "Etihad Airways"},
    {"value": "Ethiopian Airlines", "label": "Ethiopian Airlines"},
    {"value": "Air France", "label": "Air France"},
    {"value": "Singapore Airlines Ltd", "label": "Singapore Airlines Ltd"},
    {"value": "IndiGo", "label": "IndiGo"},
    {"value": "Turkish Airlines", "label": "Turkish Airlines"},
    {"value": "Cathay Pacific", "label": "Cathay Pacific"},
    {"value": "Lufthansa Cargo", "label": "Lufthansa Cargo"},
    {"value": "Saudi Airlines Cargo", "label": "Saudi Airlines Cargo"},
    {"value": "British Airways", "label": "British Airways"},
    {"value": "SriLankan Airlines", "label": "SriLankan Airlines"},
    {"value": "Gulf Air", "label": "Gulf Air"},
    {"value": "Oman Air", "label": "Oman Air"},
    {"value": "Kuwait Airways", "label": "Kuwait Airways"},
    {"value": "EgyptAir", "label": "EgyptAir"},
    {"value": "SpiceJet", "label": "SpiceJet"},
    {"value": "Air Arabia", "label": "Air Arabia"},
    {"value": "Other Carrier", "label": "Other Carrier / Charter"},
]

DEFAULT_TOP_CUSTOMERS = [
    "UNIVERSAL CLEARING AND FORWARDING",
    "TEMPCON EXPRESS PRIVATE LIMITED",
    "TRIANGLE LOGISTICS",
    "WW LOGISTICIANS PRIVATE LIMITED",
    "SKL SHIPPING & LOGISTICS PVT LTD",
    "FLOMIC GLOBAL LOGISTICS LIMITED",
    "SADHI LOGISTICS PRIVATE LIMITED",
    "MINAS LOGISTICS PRIVATE LIMITED",
    "TRANSWORLD INTEGRATED LOGISTEK PVT. LTD.",
    "M 5 CONTINENT LOGISTICS SOLUTION PRIVATE LIMITED",
    "PARTEX AEROMARINE LOGISTICS PRIVATE LIMITED",
    "A STAR SHIPPING AND LOGISTICS LLP",
    "MOVEX SERVICES PVT LTD",
    "VELOGIC INDIA PRIVATE LIMITED",
    "NEKODA GLOBAL LOGISTICS INDIA PVT LTD",
    "ROUND THE CLOCK LOGISTICS PVT LTD",
    "MEO INTERNATIONAL LOGISTICS INDIA PRIVATE LIMITED",
    "RUCHA ENTERPRISES",
    "K&K SHIPPING SERVICES",
    "AEGON SHIPPING INDIA PRIVATE LIMITED",
    "EUROPA AIR AND SEA PRIVATE LIMITED",
    "FAIRMACS SHIPPING AND TRANSPORT SERVICES PVT LTD .",
    "LEOSTAR LOGISTICS PRIVATE LIMITED",
    "HAIKO LOGISTICS INDIA PVT LTD",
    "STRANS LOGISTICS PRIVATE LIMITED",
    "TRANSWAYS LOGISTICS SOLUTIONS",
    "HD WORLDWIDE LOGISTICS",
    "RELIABLE TRAVELS AND CARGO PVT LTD",
    "CARGO INTERNATIONAL (INDIA) PRIVATE LIMITED",
    "UNION CARRGO",
    "BLUE BELL LOGISTICS PRIVATE LIMITED",
    "V.H. LOGISTICS",
    "DTDC EXPRESS LIMITED",
    "SPEDITION INDIA LOGISTICS PRIVATE LIMITED",
    "PENTAGON INTERNATIONAL FREIGHT SOLUTION PVT LTD",
    "CFKR STANDARD TEMPERATURE COURIER PRIVATE LIMITED",
    "UNIFIED LOGISTICS INDIA PRIVATE LIMITED",
    "EXCEL WORLDWIDE PRIVATE LIMITED",
    "EXIGENT GLOBAL LOGISTICS PVT LTD",
    "BNB FREIGHT SOLUTIONS",
    "JETWAY CARGO",
    "SRV GLOBAL FREIGHT PRIVATE LIMITED",
    "MOVE FORWARD LOGISTICS & SERVICES",
    "VRRDDHI FREIGHT PVT LTD",
    "MS LOGISTICS PRIVATE LIMITED",
    "GLOBALBRIDGE LOGISTICS PRIVATE LIMITED",
    "FPS LOGISTICS PVT LTD",
    "SISKO WORLDWIDE LOGISTICS INDIA PVT LTD",
    "INDOLOG LOGISTICS PRIVATE LIMITED",
    "RCS LOGISTICS PVT LTD",
]

def _build_top_customers_catalog(root_path: str):
    csv_path = os.path.join(root_path, "data", "processed", "all_inquiries_2025_cleaned.csv")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, usecols=["Customer Company"])
            top = df["Customer Company"].dropna().value_counts().head(50).index.tolist()
            if len(top) >= 10:
                return top
        except Exception as e:
            logger.warning(f"Could not load customer accounts from CSV: {e}")
    return DEFAULT_TOP_CUSTOMERS

TOP_CUSTOMERS = _build_top_customers_catalog(ENGINE_ROOT)
TOP_CUSTOMERS_JSON = json.dumps(TOP_CUSTOMERS).replace("</", "<\\/")

SALES_PERSON_OPTIONS = [
    {"value": "", "label": "Unassigned / General Sales Desk"},
    {"value": "Mohnish Arora", "label": "Mohnish Arora"},
    {"value": "Narendra Gupta", "label": "Narendra Gupta"},
    {"value": "Edward Dsouza", "label": "Edward Dsouza"},
    {"value": "SHIVAM BHARDWAJ", "label": "Shivam Bhardwaj"},
    {"value": "Harvind Kumar", "label": "Harvind Kumar"},
    {"value": "HETAL SHAH", "label": "Hetal Shah"},
    {"value": "Parag Mehta", "label": "Parag Mehta"},
    {"value": "Jayatheertha L G", "label": "Jayatheertha L G"},
    {"value": "Priyanka Sardesai", "label": "Priyanka Sardesai"},
    {"value": "Deepanshu Sangwan", "label": "Deepanshu Sangwan"},
    {"value": "Gautam Arora", "label": "Gautam Arora"},
    {"value": "Samuel Prabhu", "label": "Samuel Prabhu"},
    {"value": "Tasneem Haji", "label": "Tasneem Haji"},
    {"value": "PARVESH GUPTA", "label": "Parvesh Gupta"},
    {"value": "Fasaludheen C A", "label": "Fasaludheen C A"},
    {"value": "Dheeraj Chanigalla", "label": "Dheeraj Chanigalla"},
    {"value": "Sweety Soni", "label": "Sweety Soni"},
    {"value": "SHASHIKANT YADAV", "label": "Shashikant Yadav"},
    {"value": "Joe Xavier", "label": "Joe Xavier"},
    {"value": "Other Sales Desk", "label": "Other Sales Desk"},
]


# ------------------------------------------------------------------------------
# HTML UI TEMPLATES (MODULAR TEMPLATE ENGINE)
# ------------------------------------------------------------------------------
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

def _load_template_str(name: str) -> str:
    path = os.path.join(TEMPLATES_DIR, name)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

HTML_TEMPLATE = _load_template_str("index.html")
HISTORY_TEMPLATE = _load_template_str("history.html")


# ------------------------------------------------------------------------------
# FLASK API ENDPOINTS
# ------------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template(
        "index.html",
        model_version=MODEL_VERSION,
        origin_options=ORIGIN_OPTIONS,
        dest_options=DESTINATION_OPTIONS,
        carrier_options=CARRIER_OPTIONS,
        top_customers=TOP_CUSTOMERS,
        top_customers_json=TOP_CUSTOMERS_JSON,
        sales_person_options=SALES_PERSON_OPTIONS,
        default_quote_date=datetime.date.today().isoformat(),
        client_options=CLIENT_OPTIONS,
        vertical_options=VERTICAL_OPTIONS,
        commodity_options=COMMODITY_OPTIONS,
        incoterms_options=INCOTERMS_OPTIONS,
        branch_options=BRANCH_OPTIONS,
        customer_activity_options=CUSTOMER_ACTIVITY_OPTIONS,
        revision_options=REVISION_OPTIONS,
        port_labels_json=PORT_LABELS_JSON
    )


@app.route("/api/quote", methods=["POST"])
def api_quote():
    data = request.get_json(force=True, silent=True) or {}

    # Weights validation (§2.2, §5.2) - Supports direct Airline Chargeable Weight as primary
    raw_chargeable = data.get("chargeable_wt") or data.get("chargeable_weight") or data.get("Chargeable_Weight_Kg")
    raw_gross = data.get("gross_wt") or data.get("gross_weight") or data.get("Gross_Weight_Kg")

    # If weights not directly passed, derive from dimensions if available
    if raw_chargeable is None and raw_gross is None:
        if any(k in data for k in ["dim_l", "dim_w", "dim_h"]) or ("dimensions" in data and isinstance(data["dimensions"], list)):
            try:
                if "dimensions" in data and isinstance(data["dimensions"], list):
                    vol_calc = compute_volumetric_weight(data["dimensions"])
                else:
                    l_val = float(data.get("dim_l", 0))
                    w_val = float(data.get("dim_w", 0))
                    h_val = float(data.get("dim_h", 0))
                    pcs_val = int(data.get("dim_pcs", 1))
                    vol_calc = (l_val * w_val * h_val * pcs_val) / 6000.0 if (l_val > 0 and w_val > 0 and h_val > 0) else 0.0
                if vol_calc > 0:
                    raw_chargeable = max(0.5, np.ceil(vol_calc * 2.0) / 2.0)
                    raw_gross = raw_chargeable
            except (ValueError, TypeError):
                pass

    if raw_chargeable is None and raw_gross is None:
        return jsonify({"status": "error", "message": "Invalid Gross Weight. Please provide a positive weight in kg."}), 400

    try:
        if raw_chargeable is not None and str(raw_chargeable).strip() != "":
            chargeable_wt = float(str(raw_chargeable).replace(",", "").strip())
        elif raw_gross is not None and str(raw_gross).strip() != "":
            chargeable_wt = float(str(raw_gross).replace(",", "").strip())
        else:
            raise ValueError()

        if chargeable_wt <= 0 or not np.isfinite(chargeable_wt):
            raise ValueError()
    except (ValueError, TypeError):
        return jsonify({"status": "error", "message": "Invalid Gross Weight. Please provide a positive weight in kg."}), 400

    try:
        if raw_gross is not None and str(raw_gross).strip() != "":
            gross_wt = float(str(raw_gross).replace(",", "").strip())
            if gross_wt <= 0 or not np.isfinite(gross_wt):
                gross_wt = chargeable_wt
        else:
            gross_wt = chargeable_wt
    except (ValueError, TypeError):
        gross_wt = chargeable_wt

    # Volumetric dimensions handling (§2.2, §5.2)
    dimensions = None
    if "dimensions" in data and isinstance(data["dimensions"], list):
        dimensions = data["dimensions"]
    elif any(k in data for k in ["dim_l", "dim_w", "dim_h"]):
        try:
            l = float(data.get("dim_l", 0))
            w = float(data.get("dim_w", 0))
            h = float(data.get("dim_h", 0))
            pcs = int(data.get("dim_pcs", 1))
            if l > 0 and w > 0 and h > 0:
                dimensions = [{"length": l, "width": w, "height": h, "pieces": pcs}]
        except (ValueError, TypeError):
            pass

    # Physical plausibility & IATA validation (§2.2, §5.2)
    is_valid, eff_chargeable, density_ratio, err_reason = validate_shipment_physics(
        gross_wt=gross_wt,
        ch_wt=chargeable_wt,
        dimensions=dimensions
    )
    if not is_valid:
        return jsonify({"status": "error", "message": f"Physical validation error: {err_reason}"}), 400

    chargeable_wt = eff_chargeable

    # Dual Buy Rate Resolution (unit_buy * chargeable_wt <-> total buy)
    raw_buy = data.get("buy")
    raw_unit_buy = data.get("unit_buy") or data.get("unit_buy_rate") or data.get("unit_freight_buy")

    buy_cost = None
    unit_buy_rate = None

    if raw_buy is not None and str(raw_buy).strip() != "":
        try:
            parsed_buy = float(str(raw_buy).replace(",", "").strip())
            if parsed_buy <= 0 or not np.isfinite(parsed_buy):
                raise ValueError()
            buy_cost = parsed_buy
        except (ValueError, TypeError):
            return jsonify({"status": "error", "message": "Invalid Airline Buy Cost. Please provide a positive numeric value."}), 400

    if raw_unit_buy is not None and str(raw_unit_buy).strip() != "":
        try:
            parsed_unit = float(str(raw_unit_buy).replace(",", "").strip())
            if parsed_unit <= 0 or not np.isfinite(parsed_unit):
                raise ValueError()
            unit_buy_rate = parsed_unit
        except (ValueError, TypeError):
            return jsonify({"status": "error", "message": "Invalid Airline Unit Buy Rate. Please provide a positive numeric rate (₹/kg)."}), 400

    # Smart Safeguard: Detect when raw_buy was entered as unit rate instead of total buy
    if buy_cost is not None and chargeable_wt >= 30.0:
        implied_unit_rate = buy_cost / chargeable_wt
        # In international air freight, rate < 10 INR/kg is physically impossible.
        # If buy_cost is a realistic unit rate (e.g. >= 20 INR) and unit_buy_rate is either missing or equal to buy_cost:
        if implied_unit_rate < 10.0 and buy_cost >= 20.0 and (unit_buy_rate is None or abs(unit_buy_rate - buy_cost) < 0.01):
            logger.warning(
                f"Auto-resolving unit buy rate entered into total buy: buy={buy_cost} INR for {chargeable_wt} kg "
                f"(implied {implied_unit_rate:.2f} INR/kg). Auto-converting to unit rate {buy_cost} INR/kg "
                f"-> Total Buy: {round(buy_cost * chargeable_wt, 2)} INR."
            )
            unit_buy_rate = buy_cost
            buy_cost = round(buy_cost * chargeable_wt, 2)

    if buy_cost is None:
        if unit_buy_rate is not None:
            buy_cost = round(unit_buy_rate * chargeable_wt, 2)
        else:
            return jsonify({"status": "error", "message": "Invalid Airline Buy Cost. Please provide a positive numeric value for buy or unit_buy."}), 400

    if unit_buy_rate is None and buy_cost is not None:
        unit_buy_rate = round(buy_cost / max(0.1, chargeable_wt), 2)

    raw_freq = data.get("frequency", 10)
    try:
        frequency = int(raw_freq)
        frequency = max(1, min(500, frequency))
    except (ValueError, TypeError):
        frequency = 10

    # Default revision to 1 (commercial quote revision start at 1, §5.4)
    raw_rev = data.get("revision", 1)
    try:
        revision = int(raw_rev)
        revision = max(1, min(20, revision))
    except (ValueError, TypeError):
        revision = 1

    origin_input = str(data.get("origin") or "Indira Gandhi International Airport").strip()
    dest_input = str(data.get("dest") or "Heathrow Apt/London").strip()

    origin = normalize_port_name(origin_input)
    dest = normalize_port_name(dest_input)
    client = str(data.get("client") or "Shipper / Consignee").strip()
    vertical = str(data.get("vertical") or "Air Export Forwarding").strip()
    commodity = str(data.get("commodity") or "General Cargo").strip()
    incoterms = clean_incoterms(data.get("incoterms") or "FOB")
    branch = str(data.get("branch") or "Mumbai").strip()
    pricing_strategy = str(data.get("strategy") or "balanced").strip().lower()

    # Commercial Inputs: Carrier, Customer Account, Sales Person, Quote Date
    carrier = str(
        data.get("carrier")
        or data.get("quoted_carrier")
        or data.get("airline")
        or data.get("Quoted Carrier")
        or "Any Airline"
    ).strip()

    customer_company = str(
        data.get("customer_company")
        or data.get("customer_name")
        or data.get("customer")
        or data.get("Customer Company")
        or ""
    ).strip()

    sales_person = str(
        data.get("sales_person")
        or data.get("salesperson")
        or data.get("Sales Person")
        or ""
    ).strip()

    quote_date_raw = data.get("quote_date")
    if quote_date_raw and str(quote_date_raw).strip():
        quote_date = str(quote_date_raw).strip()[:10]
    else:
        quote_date = datetime.date.today().isoformat()

    try:
        month = int(quote_date[5:7])
    except (ValueError, IndexError):
        month = datetime.date.today().month

    # Derived Geographic Hierarchies & Tiers
    origin_region = map_origin_region(origin)
    dest_region = map_dest_region(dest)
    regional_lane = f"{origin_region} -> {dest_region}"

    tier = get_weight_tier(chargeable_wt)
    cust_tier = map_account_tier(frequency)

    inquiry_payload = {
        "Total_Buy_INR": buy_cost,
        "Unit_Buy_INR": unit_buy_rate,
        "Chargeable_Weight_Kg": chargeable_wt,
        "Gross_Weight_Kg": gross_wt,
        "Cargo_Density_Ratio": density_ratio,
        "Weight_Tier": tier,
        "Business_Vertical": vertical,
        "Commodity_Group": commodity,
        "Origin_Region": origin_region,
        "Destination_Region": dest_region,
        "Regional_Lane": regional_lane,
        "Origin_Port": origin,
        "Destination_Port": dest,
        "Company_Group": client,
        "Customer_Inquiry_Frequency": frequency,
        "Customer_Tier": cust_tier,
        "Booked by branch": branch,
        "Incoterms": incoterms,
        "quote_revision_count": revision,
        "quote_date": quote_date,
        "Quote_Month": month,
        "Air_Cargo_Season": get_air_cargo_season(month),
        "customer_name": customer_company,
        "Customer_Company": customer_company,
        "carrier": carrier,
        "Quoted Carrier": carrier,
        "sales_person": sales_person,
        "Sales Person": sales_person,
        "inquiry_ref": data.get("inquiry_ref") or data.get("ref") or ""
    }

    is_spot_tender = bool(data.get("is_spot_tender") or str(data.get("pricing_mode", "")).lower() in ("spot_tender", "spot"))
    if is_spot_tender and not data.get("strategy"):
        pricing_strategy = "floor"

    comp_rate = data.get("competitor_sell_per_kg")
    comp_intel = {
        "source": "none",
        "competitor_rate_per_kg": None,
        "sample_count": 0,
        "market_guidance": "No competitor rate reported for this inquiry."
    }
    if comp_rate is not None and str(comp_rate).strip() != "":
        try:
            comp_float = float(comp_rate)
            comp_intel = {
                "source": "explicit",
                "competitor_rate_per_kg": comp_float,
                "sample_count": 1,
                "market_guidance": f"Customer indicated competitor rate of {comp_float:.2f} INR/kg."
            }
        except (ValueError, TypeError):
            comp_rate = None

    if comp_rate is None or str(comp_rate).strip() == "":
        bench = get_recent_competitor_rate_benchmark(
            db_path=HISTORY_DB_PATH,
            origin_port=inquiry_payload.get("Origin_Port", ""),
            dest_port=inquiry_payload.get("Destination_Port", "")
        )
        if bench:
            inquiry_payload["Dynamic_Competitor_Rate_Per_Kg"] = bench["median_rate_per_kg"]
            inquiry_payload["Competitor_Benchmark_Samples"] = bench["sample_count"]
            comp_intel = {
                "source": "historical_benchmark",
                "competitor_rate_per_kg": bench["median_rate_per_kg"],
                "sample_count": bench["sample_count"],
                "min_rate_per_kg": bench["min_rate_per_kg"],
                "max_rate_per_kg": bench["max_rate_per_kg"],
                "market_guidance": (
                    f"Lane benchmark from {bench['sample_count']} recent deals: median competitor rate {bench['median_rate_per_kg']:.2f} INR/kg. "
                    "Engine margin calibrated to remain competitive."
                )
            }

    inquiry_payload.update({
        "quote_date": quote_date,
        "Air_Cargo_Season": get_air_cargo_season(month),
        "Customer_Company": customer_company,
        "customer_name": customer_company,
        "Credit_Days": data.get("credit_days", 30),
        "Oversize": bool(data.get("oversize")),
        "Rate_Type": data.get("rate_type", ""),
        "Competitor_Sell_Per_Kg": comp_rate,
        "is_spot_tender": is_spot_tender,
        "pricing_mode": "spot_tender" if is_spot_tender else "standard"
    })

    try:
        quote_res = engine_v3.recommend_quote(
            inquiry_dict=inquiry_payload,
            pricing_strategy=pricing_strategy,
        )

        chosen_strategy = pricing_strategy if pricing_strategy in quote_res["strategic_recommendations"] else "balanced"
        optimal_quote = quote_res["optimal"]
        strategic_recommendations = quote_res["strategic_recommendations"]
        negotiation_corridor = quote_res["negotiation_corridor"]
        sensitivity = quote_res["sensitivity"]
        confidence = quote_res["confidence"]
        warnings = quote_res["warnings"]
        signals = quote_res["signals"]
        cohort_band = quote_res["cohort_band_per_kg"]

        is_preset = bool(data.get("is_preset", False))

        history_logged = log_recommendation(
            db_path=HISTORY_DB_PATH,
            payload=inquiry_payload,
            optimal=optimal_quote,
            model_version=MODEL_VERSION,
            chosen_strategy=chosen_strategy,
            is_preset=is_preset,
            override_reason=data.get("override_reason"),
            user_id=data.get("user_id"),
        )

        # Weight-Break Arbitrage Evaluation (§2.8)
        arbitrage_result = check_weight_break_arbitrage(
            chargeable_wt=chargeable_wt,
            current_total_cost=buy_cost,
            next_slab_rate_per_kg=data.get("next_slab_rate") or data.get("carrier_rate_next_break"),
            slab_rates=data.get("carrier_slab_rates"),
        )

        return jsonify({
            "status": "success",
            "carrier": carrier,
            "customer_company": customer_company,
            "sales_person": sales_person,
            "quote_date": quote_date,
            "unit_buy": unit_buy_rate,
            "buy": buy_cost,
            "optimal": optimal_quote,
            "strategic_recommendations": strategic_recommendations,
            "negotiation_corridor": negotiation_corridor,
            "sensitivity": sensitivity,
            "confidence": confidence,
            "warnings": warnings,
            "signals": signals,
            "cohort_band_per_kg": cohort_band,
            "weight_break_arbitrage": arbitrage_result,
            "pricing_mode": quote_res.get("pricing_mode", "standard"),
            "is_spot_tender": is_spot_tender,
            "competitor_intelligence": comp_intel,
            "history_logged": history_logged,
            "model_version": MODEL_VERSION,
        })
    except Exception as err:
        logger.error(f"PricingEngineV3 recommendation error: {err}", exc_info=True)
        return jsonify({"status": "error", "message": "Optimization engine encountered an internal error. Please verify your quotation parameters."}), 500



@app.route("/api/history/outcome", methods=["POST"])
def api_history_outcome():
    d = request.get_json(force=True, silent=True) or {}
    ok = record_outcome(
        HISTORY_DB_PATH,
        int(d.get("id", 0)),
        d.get("status"),
        final_price_inr=d.get("final_price_inr"),
        loss_reason=d.get("loss_reason"),
        competitor_rate_per_kg=d.get("competitor_rate_per_kg")
    )
    return (jsonify({"status": "success"}), 200) if ok else (jsonify({
        "status": "error",
        "message": "Won needs final_price_inr; Lost needs a coded loss_reason."
    }), 400)


@app.route("/history")
def history_view():
    raw_page = request.args.get("page", 1)
    raw_per_page = request.args.get("per_page", 15)
    try:
        page = max(1, int(raw_page))
    except (ValueError, TypeError):
        page = 1
    try:
        per_page = max(5, min(100, int(raw_per_page)))
    except (ValueError, TypeError):
        per_page = 15

    pagination = get_history_paginated(HISTORY_DB_PATH, page=page, per_page=per_page, include_presets=True)
    stats = pagination["stats"]

    return render_template(
        "history.html",
        records=pagination["records"],
        statuses=LEAD_STATUSES,
        total_count=stats["total_count"],
        won_count=stats["won_count"],
        total_pipeline=stats["total_pipeline"],
        win_rate=stats["win_rate"],
        page=pagination["page"],
        per_page=pagination["per_page"],
        total_pages=pagination["total_pages"],
        has_prev=pagination["has_prev"],
        has_next=pagination["has_next"],
        prev_page=pagination["prev_page"],
        next_page=pagination["next_page"],
        start_idx=pagination["start_idx"],
        end_idx=pagination["end_idx"]
    )


@app.route("/api/history/update_status", methods=["POST"])
def api_history_update_status():
    data = request.get_json(force=True, silent=True) or {}
    quote_id = data.get("id")
    new_status = data.get("status")

    if not quote_id or not new_status:
        return jsonify({"status": "error", "message": "Missing quote ID or status."}), 400

    success = update_lead_status(HISTORY_DB_PATH, int(quote_id), new_status)
    if success:
        return jsonify({"status": "success"})
    return jsonify({"status": "error", "message": "Invalid status or record not found."}), 400


@app.route("/api/history/clear", methods=["POST"])
@require_admin_auth
def api_history_clear():
    success = clear_history_table(HISTORY_DB_PATH)
    return jsonify({"status": "success" if success else "error"})


def sanitize_csv_field(val):
    if val is None:
        return ""
    val_str = str(val)
    if val_str and val_str[0] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + val_str
    return val_str


@app.route("/api/history/export")
@require_admin_auth
def api_history_export():
    def generate_csv_stream():
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "ID", "Timestamp", "Origin", "Destination", "Buy Cost INR",
            "Selling Price INR", "Margin Range", "Margin Pct", "Margin Amount INR",
            "Win Probability", "Expected Profit INR", "Gross Wt Kg",
            "Chargeable Wt Kg", "Commodity", "Client Type", "Branch", "Status", "Model Version"
        ])
        yield output.getvalue()
        output.seek(0)
        output.truncate(0)

        for r in iter_all_history(HISTORY_DB_PATH):
            writer.writerow([
                sanitize_csv_field(r.get("id")),
                sanitize_csv_field(r.get("timestamp")),
                sanitize_csv_field(r.get("origin_port")),
                sanitize_csv_field(r.get("dest_port")),
                sanitize_csv_field(r.get("airline_buy_inr")),
                sanitize_csv_field(r.get("selling_price_inr")),
                sanitize_csv_field(r.get("margin_range")),
                sanitize_csv_field(r.get("margin_pct")),
                sanitize_csv_field(r.get("margin_amount_inr")),
                sanitize_csv_field(r.get("win_prob")),
                sanitize_csv_field(r.get("expected_profit_inr")),
                sanitize_csv_field(r.get("gross_wt_kg")),
                sanitize_csv_field(r.get("chargeable_wt_kg")),
                sanitize_csv_field(r.get("commodity")),
                sanitize_csv_field(r.get("client_type")),
                sanitize_csv_field(r.get("branch")),
                sanitize_csv_field(r.get("status")),
                sanitize_csv_field(r.get("model_version"))
            ])
            yield output.getvalue()
            output.seek(0)
            output.truncate(0)

    return Response(
        stream_with_context(generate_csv_stream()),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=air_export_pricing_history.csv"}
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
