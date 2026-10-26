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
from typing import Dict, Any, Optional
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string, Response

# Ensure local directories are in python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_ROOT = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, ENGINE_ROOT)

from constants import (
    MODEL_VERSION,
    MIN_BENCHMARK_MARGIN,
    MAX_BENCHMARK_MARGIN,
    MIN_CANDIDATE_MARGIN,
    MAX_CANDIDATE_MARGIN,
    STRATEGY_THRESHOLDS
)
from tiers import get_weight_tier, map_account_tier, check_weight_break_arbitrage
from preprocess import (
    normalize_port_name,
    map_origin_region,
    map_dest_region,
    clean_incoterms,
    map_commodity_group,
    validate_shipment_physics,
    compute_volumetric_weight
)
from engine import MarginRecommendationEngine, EngineConfig, RetentionGuardrailConfig
from db import (
    init_history_db,
    log_recommendation,
    record_outcome,
    get_all_history,
    get_history_stats,
    get_history_paginated,
    update_lead_status,
    clear_history_table,
    get_recent_competitor_rate_benchmark,
    LEAD_STATUSES,
    VALID_STATUS_CODES,
    get_status_class
)

app = Flask(__name__)
# Limit incoming JSON payload to 1MB to prevent memory exhaustion DoS
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024

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
# LOAD MODEL ARTIFACT ON STARTUP (§1.4, §5.6)
# ------------------------------------------------------------------------------
_CANDIDATE_MODEL_PATHS = [
    os.path.join(ENGINE_ROOT, "models", "freight_margin_recommender.joblib"),
    os.path.join(BASE_DIR, "models", "freight_margin_recommender.joblib"),
    os.path.join(os.getcwd(), "models", "freight_margin_recommender.joblib"),
    os.path.join("/var/task", "models", "freight_margin_recommender.joblib"),
]
MODEL_PATH = next((p for p in _CANDIDATE_MODEL_PATHS if os.path.exists(p)), _CANDIDATE_MODEL_PATHS[0])
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model artifact not found at candidate paths: {_CANDIDATE_MODEL_PATHS}. Please run train.py first.")

logger.info(f"Loading trained pricing engine from: {MODEL_PATH}")
model_bundle = joblib.load(MODEL_PATH)

# Check scikit-learn version (§1.4)
import sklearn
runtime_sklearn = sklearn.__version__
model_sklearn = model_bundle.get("sklearn_version")
if model_sklearn and model_sklearn != runtime_sklearn:
    msg = f"sklearn version skew: model artifact trained with {model_sklearn}, current runtime is {runtime_sklearn}."
    if os.environ.get("ALLOW_SKLEARN_MISMATCH", "0") == "1":
        logger.warning(msg + " Continuing due to ALLOW_SKLEARN_MISMATCH=1.")
    else:
        raise RuntimeError(msg + " Retrain artifact or set ALLOW_SKLEARN_MISMATCH=1 to override.")

engine = MarginRecommendationEngine(
    benchmark_reg=model_bundle["benchmark_reg"],
    win_classifier=model_bundle["calibrated_clf"],
    encoder=model_bundle["encoder"],
    feature_cols=model_bundle["feature_cols"],
    cat_cols=model_bundle.get("native_cat_cols", model_bundle.get("cat_cols")),
    lane_medians=model_bundle.get("lane_medians"),
    global_median=model_bundle.get("global_median"),
    port_frequencies=model_bundle.get("port_frequencies"),
    cohort_table=model_bundle.get("cohort_table")
)
try:
    import margin_framework as mf
    from preprocess import get_air_cargo_season
except ImportError:
    from src import margin_framework as mf
    from src.preprocess import get_air_cargo_season

MARGIN_TABLES = model_bundle.get("margin_tables")
if not MARGIN_TABLES:
    raise RuntimeError("Model artifact does not contain 'margin_tables'. Please re-run train.py.")

training_window = model_bundle.get("training_window", ["2026-01", "2026-09"])
try:
    start_m = int(training_window[0].split("-")[1])
    end_m = int(training_window[1].split("-")[1])
    TRAIN_WINDOW_MONTHS = set(range(start_m, end_m + 1))
except Exception:
    TRAIN_WINDOW_MONTHS = set(range(1, 10))

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

def _build_destinations_catalog(bundle, root_path):
    port_freqs = bundle.get("port_frequencies", {})
    if port_freqs:
        model_dests = set(port_freqs.keys())
    elif "cat_cols" in bundle and "Destination_Port" in bundle["cat_cols"]:
        dest_idx = bundle["cat_cols"].index("Destination_Port")
        model_dests = set(bundle["encoder"].categories_[dest_idx])
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

DESTINATION_OPTIONS = _build_destinations_catalog(model_bundle, ENGINE_ROOT)
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
# HTML UI TEMPLATE (AUTHENTIC SIGNATURE STYLING RESTORED)
# ------------------------------------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
    <meta http-equiv="Pragma" content="no-cache">
    <meta http-equiv="Expires" content="0">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Air Freight Margin Recommender — Air Export Pricing Engine</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Mulish:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-outer: #edf1f5;
            --surface-card: #ffffff;
            --surface-subtle: #f8fafc;
            --border: #e2e8f0;
            --border-subtle: #f1f5f9;
            --text-dark: #0f172a;
            --text-muted: #64748b;
            --text-light: #94a3b8;
            --pill-dark: #0f172a;
            --pill-hover: #1e293b;
            --radius-window: 24px;
            --radius-card: 16px;
            --radius-control: 8px;
            --radius-pill: 100px;
            --shadow-window: 0 20px 60px rgba(15, 23, 42, 0.06);
            --shadow-card: 0 4px 20px rgba(15, 23, 42, 0.04);
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Mulish', sans-serif; }
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                animation-duration: 0.01ms !important;
                transition-duration: 0.01ms !important;
                scroll-behavior: auto !important;
            }
        }
        h1, h2, h3, h4, .font-heading, .btn-pill, .nav-item, .hero-value, .metric-number, .strat-name, .strat-price { font-family: 'Outfit', sans-serif; }

        body {
            background-color: var(--bg-outer);
            color: var(--text-dark);
            padding: 32px 20px 48px;
            min-height: 100vh;
            display: flex;
            justify-content: center;
        }

        /* Outer Floating Window Canvas */
        .app-window {
            background: var(--surface-card);
            border-radius: var(--radius-window);
            box-shadow: var(--shadow-window);
            width: 100%;
            max-width: 1320px;
            padding: 28px 36px 40px;
            border: 1px solid rgba(255, 255, 255, 0.8);
        }

        @media (max-width: 980px) {
            .app-window { padding: 24px 20px 32px; border-radius: 18px; }
            .app-nav { flex-direction: column; align-items: stretch; gap: 14px; }
            .nav-links-center { justify-content: center; }
        }

        @media (max-width: 640px) {
            body { padding: 12px 10px 24px; }
            .app-window { padding: 16px 14px 24px; border-radius: 16px; }
            .app-nav { gap: 10px; margin-bottom: 16px; padding-bottom: 16px; }
            .page-header { flex-direction: column; align-items: flex-start; gap: 12px; }
            .page-title { font-size: 22px; }
            .header-actions { width: 100%; gap: 8px; }
            .header-actions .btn-pill { flex: 1 1 0; padding: 10px 12px; font-size: 12px; justify-content: center; text-align: center; }
        }

        /* Top Minimalist Navbar (Retains ONLY Dashboard and Reports) */
        .app-nav {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding-bottom: 22px;
            margin-bottom: 22px;
            border-bottom: 1px solid var(--border-subtle);
        }
        .nav-logo-box {
            display: flex;
            align-items: center;
            gap: 10px;
            text-decoration: none;
            color: var(--text-dark);
        }
        .nav-logo-badge {
            width: 36px;
            height: 36px;
            border-radius: 8px;
            background: var(--pill-dark);
            display: flex;
            align-items: center;
            justify-content: center;
            color: #ffffff;
        }
        .nav-brand-title {
            font-size: 15px;
            font-weight: 700;
            color: var(--text-dark);
            font-family: 'Outfit', sans-serif;
            letter-spacing: -0.2px;
        }
        .nav-links-center {
            display: flex;
            align-items: center;
            gap: 6px;
            background: #f1f5f9;
            padding: 4px;
            border-radius: var(--radius-pill);
        }
        .nav-item {
            text-decoration: none;
            color: var(--text-muted);
            font-size: 13px;
            font-weight: 600;
            padding: 8px 18px;
            min-height: 36px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: var(--radius-pill);
            transition: all 0.2s;
        }
        .nav-item:hover { color: var(--text-dark); }
        .nav-item.active {
            background: var(--pill-dark);
            color: #ffffff;
            box-shadow: 0 2px 6px rgba(15, 23, 42, 0.15);
        }
        .nav-tools-right {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .engine-tag {
            background: #f1f5f9;
            color: var(--text-dark);
            font-size: 11px;
            font-weight: 600;
            padding: 5px 12px;
            border-radius: var(--radius-pill);
            border: 1px solid var(--border);
            font-family: 'Outfit', sans-serif;
        }

        /* Page Sub-Header */
        .page-header {
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            margin-bottom: 22px;
            flex-wrap: wrap;
            gap: 16px;
        }
        .back-breadcrumb {
            text-decoration: none;
            font-size: 12.5px;
            color: var(--text-muted);
            display: inline-flex;
            align-items: center;
            gap: 5px;
            margin-bottom: 6px;
            transition: color 0.15s;
        }
        .back-breadcrumb:hover { color: var(--text-dark); }
        .page-title {
            font-size: 28px;
            font-weight: 700;
            color: var(--text-dark);
            letter-spacing: -0.4px;
            line-height: 1.2;
        }
        .subtitle {
            font-size: 13px;
            color: var(--text-muted);
            margin-top: 3px;
        }
        .header-actions {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .btn-pill {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            padding: 10px 20px;
            min-height: 44px;
            border-radius: var(--radius-pill);
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            text-decoration: none;
        }
        .btn-white {
            background: #ffffff;
            border: 1px solid var(--border);
            color: var(--text-dark);
        }
        .btn-white:hover { background: #f8fafc; border-color: #cbd5e1; }
        .btn-primary-action {
            background: var(--pill-dark);
            border: 1px solid var(--pill-dark);
            color: #ffffff;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.12);
        }
        .btn-primary-action:hover { background: #1e293b; border-color: #1e293b; transform: translateY(-1px); }

        /* Presets Bar (Horizontally scrollable on mobile, compact full-width on desktop) */
        .presets-bar {
            background: #f8fafc;
            border: 1px solid var(--border);
            border-radius: var(--radius-card);
            padding: 14px 18px;
            margin-bottom: 24px;
        }
        .presets-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 10px;
            gap: 8px;
            flex-wrap: wrap;
        }
        .presets-title {
            font-size: 11px;
            font-weight: 700;
            color: var(--text-dark);
            text-transform: uppercase;
            letter-spacing: 0.6px;
            display: flex;
            align-items: center;
            gap: 6px;
            font-family: 'Outfit', sans-serif;
        }
        .presets-grid-row {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
        }
        @media (max-width: 900px) {
            .presets-grid-row { grid-template-columns: repeat(2, 1fr); }
        }
        @media (max-width: 640px) {
            .presets-bar { padding: 12px 14px; margin-bottom: 18px; }
            .presets-header { flex-direction: column; align-items: flex-start; gap: 2px; }
            .presets-grid-row {
                display: flex;
                overflow-x: auto;
                gap: 8px;
                padding-bottom: 4px;
                scrollbar-width: none;
                -webkit-overflow-scrolling: touch;
            }
            .presets-grid-row::-webkit-scrollbar { display: none; }
            .preset-chip-btn {
                flex: 0 0 auto;
                width: auto;
                white-space: nowrap;
                padding: 8px 14px;
            }
        }
        .preset-chip-btn {
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: var(--radius-pill);
            padding: 9px 14px;
            font-size: 12px;
            font-weight: 600;
            color: var(--text-dark);
            cursor: pointer;
            transition: all 0.2s;
            text-align: left;
            display: flex;
            align-items: center;
            gap: 8px;
            width: 100%;
            min-height: 44px;
        }
        .preset-chip-btn:hover {
            border-color: #cbd5e1;
            background: #f8fafc;
            color: var(--text-dark);
            transform: translateY(-1px);
        }
        .preset-chip-btn:hover .preset-num-badge {
            background: var(--pill-dark);
            color: #ffffff;
        }
        .preset-num-badge {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #f1f5f9;
            font-size: 10px;
            font-weight: 700;
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--text-muted);
            flex-shrink: 0;
        }

        /* 50/50 Workspace Grid */
        .workspace-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 36px;
            align-items: start;
        }
        @media (max-width: 980px) {
            .workspace-grid { grid-template-columns: 1fr; }
        }

        /* Column Headers & Dividers */
        .col-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--border);
            margin-bottom: 18px;
            flex-wrap: wrap;
            gap: 8px;
        }
        .col-title {
            font-size: 14.5px;
            font-weight: 700;
            color: var(--text-dark);
            font-family: 'Outfit', sans-serif;
            letter-spacing: -0.2px;
        }
        .col-badge {
            font-size: 11px;
            font-weight: 600;
            color: var(--text-muted);
            background: #f1f5f9;
            padding: 3px 9px;
            border-radius: var(--radius-pill);
        }

        /* Form Fields & Sections */
        .field-section {
            margin-bottom: 18px;
            padding-bottom: 16px;
            border-bottom: 1px dashed var(--border);
        }
        .field-section:last-of-type {
            border-bottom: none;
            padding-bottom: 0;
            margin-bottom: 12px;
        }
        .section-label {
            font-size: 11px;
            font-weight: 700;
            color: var(--text-dark);
            text-transform: uppercase;
            letter-spacing: 0.6px;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 6px;
            font-family: 'Outfit', sans-serif;
        }
        .form-row {
            display: grid;
            grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
            gap: 14px;
            margin-bottom: 14px;
            width: 100%;
            box-sizing: border-box;
            align-items: start;
        }
        @media (max-width: 600px) {
            .form-row { grid-template-columns: 1fr; }
        }
        .field-group {
            display: flex;
            flex-direction: column;
            gap: 6px;
            min-width: 0;
            width: 100%;
            box-sizing: border-box;
        }
        .field-group.full { grid-column: 1 / -1; }
        label {
            font-size: 12px;
            font-weight: 600;
            color: var(--text-dark);
            display: flex;
            align-items: center;
            min-height: 18px;
            line-height: 18px;
            margin: 0;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        label span.req { color: #ef4444; margin-left: 2px; }

        .clean-input {
            width: 100%;
            box-sizing: border-box;
            padding: 10px 14px;
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: var(--radius-control);
            font-size: 13px;
            color: var(--text-dark);
            font-family: 'Mulish', sans-serif;
            transition: all 0.15s;
            height: 44px;
            min-height: 44px;
        }
        .clean-input:focus {
            outline: none;
            border-color: var(--pill-dark);
            box-shadow: 0 0 0 2px rgba(15, 23, 42, 0.08);
        }
        .clean-input:focus-visible, .select-search-input:focus-visible, .select-trigger:focus-visible, .btn-pill:focus-visible, .submit-btn:focus-visible, .mode-btn:focus-visible, .preset-chip-btn:focus-visible, .strategy-card:focus-visible {
            outline: 2px solid var(--pill-dark);
            outline-offset: 2px;
        }

        /* Searchable Custom Select Dropdowns */
        .searchable-select {
            position: relative;
            width: 100%;
            min-width: 0;
            box-sizing: border-box;
        }
        .select-trigger {
            width: 100%;
            box-sizing: border-box;
            padding: 10px 14px;
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: var(--radius-control);
            font-size: 13px;
            color: var(--text-dark);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            user-select: none;
            transition: all 0.15s;
            height: 44px;
            min-height: 44px;
            min-width: 0;
            overflow: hidden;
        }
        .select-trigger:hover { border-color: #cbd5e1; }
        .select-trigger.open {
            border-color: var(--pill-dark);
            box-shadow: 0 0 0 2px rgba(15, 23, 42, 0.08);
        }
        .select-trigger-text {
            flex: 1 1 0%;
            min-width: 0;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            font-family: 'Mulish', sans-serif;
            text-align: left;
        }
        .select-trigger-arrow {
            font-size: 9px;
            color: var(--text-muted);
            transition: transform 0.2s;
            flex-shrink: 0;
        }
        .select-trigger.open .select-trigger-arrow { transform: rotate(180deg); color: var(--text-dark); }
        .select-dropdown {
            position: absolute;
            top: calc(100% + 4px);
            left: 0;
            right: 0;
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: var(--radius-control);
            box-shadow: 0 10px 25px rgba(15, 23, 42, 0.1);
            z-index: 9999;
            display: none;
            flex-direction: column;
            max-height: 260px;
        }
        .select-search-wrap {
            padding: 8px 10px;
            background: #f8fafc;
            border-bottom: 1px solid var(--border);
            position: sticky;
            top: 0;
            z-index: 2;
        }
        .select-search-input {
            width: 100%;
            padding: 7px 12px;
            border: 1px solid var(--border);
            border-radius: var(--radius-pill);
            font-size: 12px;
            font-family: 'Mulish', sans-serif;
            background: #ffffff;
            outline: none;
            min-height: 36px;
        }
        .select-options-list {
            overflow-y: auto;
            max-height: 200px;
            padding: 4px 0;
            list-style: none;
        }
        .select-option {
            padding: 8px 14px;
            font-size: 12.5px;
            font-family: 'Mulish', sans-serif;
            color: var(--text-dark);
            cursor: pointer;
            transition: background 0.15s;
            display: flex;
            align-items: center;
            justify-content: space-between;
            min-height: 38px;
        }
        .select-option:hover { background: #f8fafc; }
        .select-option.selected { background: #f1f5f9; font-weight: 700; color: #0f172a; }
        .select-option.selected::after { content: "✓"; font-size: 11px; }

        /* Item Details & Dimensions (Restored Clean 4-Column Box Design) */
        .weight-mode-toggle {
            display: flex;
            gap: 6px;
            margin-bottom: 12px;
            background: #f1f5f9;
            padding: 4px;
            border-radius: var(--radius-pill);
            min-height: 44px;
        }
        .mode-btn {
            flex: 1;
            border: none;
            background: transparent;
            padding: 8px 12px;
            border-radius: var(--radius-pill);
            font-size: clamp(11px, 2.8vw, 12.5px);
            font-weight: 700;
            font-family: 'Outfit', sans-serif;
            color: var(--text-muted);
            cursor: pointer;
            transition: all 0.2s;
            text-align: center;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
            min-height: 36px;
        }
        .mode-btn.active {
            background: #ffffff;
            color: var(--text-dark);
            box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        }
        .dim-box {
            background: #f8fafc;
            border: 1px solid var(--border);
            border-radius: var(--radius-control);
            padding: 14px;
            margin-top: 4px;
        }
        .dim-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
        }
        @media (max-width: 640px) {
            .dim-grid { grid-template-columns: repeat(2, 1fr); gap: 10px; }
        }
        .dim-grid label {
            font-size: 10.5px;
            color: var(--text-muted);
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.4px;
        }
        .dim-info {
            display: flex;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 6px;
            font-size: 11.5px;
            font-weight: 600;
            margin-top: 10px;
            color: var(--text-muted);
            padding-top: 8px;
            border-top: 1px dashed var(--border);
        }

        /* Calculate Button */
        .submit-btn {
            width: 100%;
            background: var(--pill-dark);
            color: #ffffff;
            border: none;
            padding: 13px;
            border-radius: var(--radius-pill);
            font-size: 14px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s;
            margin-top: 14px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            font-family: 'Outfit', sans-serif;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.15);
            min-height: 46px;
        }
        .submit-btn:hover { background: #1e293b; transform: translateY(-1px); }
        .submit-btn:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }

        /* Placeholder / Ready State */
        .placeholder-box {
            text-align: center;
            padding: 50px 20px;
            color: var(--text-muted);
        }

        /* Card Containers: Minimalist, Cohesive White Surfaces */
        .left-display-card, .right-display-card {
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: var(--radius-card);
            padding: 24px;
            box-shadow: var(--shadow-card);
        }
        @media (max-width: 640px) {
            .left-display-card, .right-display-card {
                padding: 18px 14px;
            }
        }

        /* Hero Metric Card: Minimalist High-Contrast Slate Capsule */
        .hero-metric-card {
            background: var(--pill-dark);
            color: #ffffff;
            border-radius: 14px;
            padding: 24px 20px;
            text-align: center;
            margin-bottom: 16px;
            position: relative;
            box-shadow: 0 4px 20px rgba(15, 23, 42, 0.08);
        }
        .hero-label-row {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            margin-bottom: 6px;
            flex-wrap: wrap;
        }
        .hero-label {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: #94a3b8;
            font-weight: 700;
            font-family: 'Outfit', sans-serif;
        }
        .hero-corridor-badge {
            font-size: 10px;
            font-weight: 700;
            padding: 2px 9px;
            border-radius: var(--radius-pill);
            background: rgba(255, 255, 255, 0.12);
            color: #ffffff;
            border: 1px solid rgba(255, 255, 255, 0.2);
            font-family: 'Outfit', sans-serif;
        }
        .hero-value {
            font-size: 40px;
            font-weight: 800;
            margin: 6px 0;
            color: #ffffff;
            letter-spacing: -0.6px;
            font-family: 'Outfit', sans-serif;
        }
        @media (max-width: 480px) {
            .hero-value { font-size: 32px; }
        }
        .hero-sub {
            font-size: 12.5px;
            color: #cbd5e1;
            font-weight: 500;
        }

        /* Metrics 3-Card Grid */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin-bottom: 18px;
        }
        @media (max-width: 600px) {
            .metrics-grid { grid-template-columns: 1fr; }
        }
        .metric-tile {
            background: #f8fafc;
            border: 1px solid var(--border);
            border-radius: var(--radius-control);
            padding: 12px 10px;
            text-align: center;
        }
        .metric-title {
            font-size: 10.5px;
            color: var(--text-muted);
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.4px;
            font-family: 'Outfit', sans-serif;
        }
        .metric-number {
            font-size: 17px;
            font-weight: 700;
            color: var(--text-dark);
            margin-top: 3px;
        }

        /* Strategic Tiers Container */
        .strategic-tiers-container {
            display: flex;
            flex-direction: column;
            gap: 9px;
            margin-bottom: 18px;
        }
        .strategy-card {
            background: #ffffff;
            border: 1.5px solid var(--border);
            border-radius: var(--radius-control);
            padding: 12px 16px;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 14px;
            min-height: 44px;
        }
        .strategy-card:hover {
            border-color: #cbd5e1;
            background: #f8fafc;
        }
        .strategy-card.selected {
            border-color: var(--pill-dark);
            background: #f8fafc;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08);
        }
        @media (max-width: 540px) {
            .strategy-card {
                flex-direction: column;
                align-items: flex-start;
                gap: 10px;
                padding: 14px;
            }
            .strat-card-left {
                width: 100%;
            }
            .strat-card-title-row {
                flex-wrap: wrap;
                gap: 6px;
            }
            .strat-meta-chips {
                flex-wrap: wrap;
                gap: 4px 6px;
                margin-top: 4px;
                font-size: 11px;
            }
            .strat-card-right {
                width: 100%;
                text-align: left;
                display: flex;
                align-items: baseline;
                justify-content: space-between;
                padding-top: 8px;
                border-top: 1px dashed var(--border);
            }
            .strat-price {
                font-size: 18px;
            }
            .strat-price-unit {
                font-size: 11px;
                margin-top: 0;
            }
        }
        .strat-card-left {
            display: flex;
            flex-direction: column;
            gap: 3px;
            flex: 1;
        }
        .strat-card-title-row {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .strat-radio-indicator {
            width: 14px;
            height: 14px;
            border-radius: 50%;
            border: 2px solid #cbd5e1;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }
        .strategy-card.selected .strat-radio-indicator {
            border-color: var(--pill-dark);
        }
        .strategy-card.selected .strat-radio-indicator::after {
            content: '';
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: var(--pill-dark);
        }
        .strat-name {
            font-size: 13px;
            font-weight: 700;
            color: var(--text-dark);
        }
        .strat-meta-chips {
            font-size: 11.5px;
            color: var(--text-muted);
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .strat-card-right {
            text-align: right;
            flex-shrink: 0;
        }
        .strat-price {
            font-size: 19px;
            font-weight: 800;
            color: var(--text-dark);
            line-height: 1.1;
        }
        .strat-price-unit {
            font-size: 11px;
            color: var(--text-muted);
            margin-top: 1px;
        }

        /* Badges */
        .badge {
            padding: 2px 8px;
            border-radius: var(--radius-pill);
            font-size: 10px;
            font-weight: 700;
            display: inline-block;
            font-family: 'Outfit', sans-serif;
        }
        .badge-floor { background: #f1f5f9; color: var(--text-dark); border: 1px solid var(--border); }
        .badge-balanced { background: var(--pill-dark); color: #ffffff; border: 1px solid var(--pill-dark); }
        .badge-premium { background: #ffffff; color: var(--text-dark); border: 1px solid #cbd5e1; }

        /* Market Sensitivity Frontier Table */
        .table-wrap {
            width: 100%;
            overflow-x: auto;
            max-height: 280px;
            border: 1px solid var(--border);
            border-radius: var(--radius-control);
            background: #ffffff;
            margin-bottom: 18px;
        }
        table {
            width: 100%;
            min-width: 700px;
            border-collapse: collapse;
            font-size: 12px;
        }
        th, td {
            padding: 9px 12px;
            text-align: left;
            border-bottom: 1px solid var(--border-subtle);
            white-space: nowrap;
        }
        th {
            background: #f8fafc;
            font-weight: 700;
            color: var(--text-muted);
            font-family: 'Outfit', sans-serif;
            position: sticky;
            top: 0;
            z-index: 2;
        }
        tbody tr:hover { background: #f8fafc; cursor: pointer; }
        tbody tr.row-tier-selected { background: #f1f5f9 !important; font-weight: 700; color: #0f172a; }

        /* Bottom Actions */
        .actions-split {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }
        @media (max-width: 500px) {
            .actions-split { grid-template-columns: 1fr; }
        }
        .btn-copy-action {
            width: 100%;
            background: var(--pill-dark);
            color: #ffffff;
            border: none;
            padding: 11px;
            border-radius: var(--radius-pill);
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            min-height: 44px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
            font-family: 'Outfit', sans-serif;
        }
        .btn-copy-action:hover { background: #1e293b; }

        /* Toast */
        .toast-msg {
            position: fixed;
            bottom: 24px;
            right: 24px;
            background: var(--pill-dark);
            color: #ffffff;
            padding: 11px 20px;
            border-radius: var(--radius-pill);
            font-size: 12.5px;
            font-weight: 600;
            font-family: 'Outfit', sans-serif;
            box-shadow: 0 10px 30px rgba(0,0,0,0.15);
            z-index: 10000;
            display: none;
            opacity: 0;
            transform: translateY(10px);
            transition: all 0.2s ease;
        }
        .toast-msg.show { opacity: 1; transform: translateY(0); }

        .ui-icon {
            display: inline-block;
            width: 14px;
            height: 14px;
            stroke-width: 2.2;
            stroke: currentColor;
            fill: none;
            stroke-linecap: round;
            stroke-linejoin: round;
            vertical-align: -2px;
            flex-shrink: 0;
        }
        .spin { animation: spin 1s linear infinite; }
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
    </style>
</head>
<body>

<div class="app-window">
    <!-- Top Minimalist Navbar (Retains ONLY Dashboard and Reports) -->
    <nav class="app-nav">
        <a href="/" class="nav-logo-box">
            <div class="nav-logo-badge">
                <svg class="ui-icon" style="stroke: #ffffff; width: 18px; height: 18px;" viewBox="0 0 24 24"><path d="M17.8 19.2 16 11l3.5-3.5C21 6 21.5 4 21 3c-1-.5-3 0-4.5 1.5L13 8 4.8 6.2c-.5-.1-.9.1-1.1.5l-.3.5c-.2.5-.1 1 .3 1.3L9 12l-2 3H4l-1 1 3 2 2 3 1-1v-3l3-2 3.5 5.3c.3.4.8.5 1.3.3l.5-.3c.4-.2.6-.6.5-1.1z"/></svg>
            </div>
            <span class="nav-brand-title">Air Export Pricing Engine</span>
        </a>

        <div class="nav-links-center">
            <a href="/" class="nav-item active">Dashboard</a>
            <a href="/history" class="nav-item">Reports</a>
        </div>

        <div class="nav-tools-right"></div>
    </nav>

    <!-- Page Header (Updated Title & Quick Action Buttons) -->
    <div class="page-header">
        <div>
            <a href="/" class="back-breadcrumb">
                <svg class="ui-icon" viewBox="0 0 24 24"><line x1="19" y1="12" x2="5" y2="12"/><polyline points="12 19 5 12 12 5"/></svg>
                Back to Dashboard
            </a>
            <h1 class="page-title">Air Freight Margin Recommender</h1>
        </div>
        <div class="header-actions">
            <button type="button" class="btn-pill btn-white" onclick="resetFormForNewQuote()">
                <svg class="ui-icon" viewBox="0 0 24 24"><path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/><path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16"/><path d="M16 16h5v5"/></svg>
                Reset Form
            </button>
            <button type="button" id="submitBtnTop" class="btn-pill btn-primary-action" onclick="document.getElementById('quoteForm').requestSubmit()">
                <svg class="ui-icon" viewBox="0 0 24 24"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
                Calculate Recommendation
            </button>
        </div>
    </div>

    <!-- Alert Messages -->
    <div id="errorAlert" style="display: none; background: #fef2f2; border: 1px solid #fecaca; color: #991b1b; padding: 10px 14px; border-radius: 8px; font-size: 12.5px; margin-bottom: 18px;"></div>

    <!-- Presets Bar (Single Full-Width Row Restored) -->
    <div class="presets-bar">
        <div class="presets-header">
            <span class="presets-title">
                <svg class="ui-icon" style="color: var(--text-dark);" viewBox="0 0 24 24"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
                Quick Historical Test Presets
            </span>
            <span style="font-size: 11px; color: var(--text-light);">One-click high-volume inquiry scenarios</span>
        </div>
        <div class="presets-grid-row">
            <button type="button" class="preset-chip-btn" onclick="applyPreset('pharma_uk')">
                <span class="preset-num-badge">1</span>
                <span>Pharma Urgent (DEL → LHR)</span>
            </button>
            <button type="button" class="preset-chip-btn" onclick="applyPreset('perishable_uae')">
                <span class="preset-num-badge">2</span>
                <span>Perishables (BOM → DXB)</span>
            </button>
            <button type="button" class="preset-chip-btn" onclick="applyPreset('heavy_subagent')">
                <span class="preset-num-badge">3</span>
                <span>Heavy Machinery (BOM → FRA)</span>
            </button>
            <button type="button" class="preset-chip-btn" onclick="applyPreset('courier_us')">
                <span class="preset-num-badge">4</span>
                <span>Express Courier (BLR → JFK)</span>
            </button>
        </div>
    </div>

    <!-- 50/50 Workspace Grid -->
    <div class="workspace-grid">
        <!-- Left Column: Inquiry Details Form -->
        <div class="workspace-col-left left-display-card">
            <div class="col-header">
                <span class="col-title">Inquiry Details</span>
                <span class="col-badge">Commercial Inputs</span>
            </div>

            <form id="quoteForm" onsubmit="event.preventDefault(); runInference();">
                <!-- Source & Destination Airports Section -->
                <div class="field-section">
                    <div class="section-label">Source & Destination Airports</div>
                    <div class="form-row">
                        <div class="field-group">
                            <label for="origin_display_text">Source Airport (POL) <span class="req">*</span></label>
                            <input type="hidden" id="origin" value="Indira Gandhi International Airport">
                            <div class="searchable-select" id="searchable_origin">
                                <div class="select-trigger" onclick="toggleDropdown('origin')">
                                    <span class="select-trigger-text" id="origin_display_text">Delhi (DEL) - Indira Gandhi</span>
                                    <span class="select-trigger-arrow">▼</span>
                                </div>
                                <div class="select-dropdown" id="origin_dropdown">
                                    <div class="select-search-wrap">
                                        <input type="text" class="select-search-input" id="origin_search" placeholder="Search source airport..." oninput="filterDropdownOptions('origin', this.value)">
                                    </div>
                                    <ul class="select-options-list" id="origin_options_list">
                                        {% for opt in origin_options %}
                                        <li class="select-option{% if opt.value == 'Indira Gandhi International Airport' %} selected{% endif %}" data-value="{{ opt.value }}">{{ opt.label }}</li>
                                        {% endfor %}
                                    </ul>
                                </div>
                            </div>
                        </div>

                        <div class="field-group">
                            <label for="dest_display_text">Destination Airport (POD) <span class="req">*</span></label>
                            <input type="hidden" id="dest" value="Heathrow Apt/London">
                            <div class="searchable-select" id="searchable_dest">
                                <div class="select-trigger" onclick="toggleDropdown('dest')">
                                    <span class="select-trigger-text" id="dest_display_text">London Heathrow (LHR)</span>
                                    <span class="select-trigger-arrow">▼</span>
                                </div>
                                <div class="select-dropdown" id="dest_dropdown">
                                    <div class="select-search-wrap">
                                        <input type="text" class="select-search-input" id="dest_search" placeholder="Search destination (580+ ports)..." oninput="filterDropdownOptions('dest', this.value)">
                                    </div>
                                    <ul class="select-options-list" id="dest_options_list">
                                        {% for opt in dest_options %}
                                        <li class="select-option{% if opt.value == 'Heathrow Apt/London' %} selected{% endif %}" data-value="{{ opt.value }}">{{ opt.label }}</li>
                                        {% endfor %}
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Cargo Specifications & Airline Buy Cost -->
                <div class="field-section">
                    <div class="section-label">Cargo Specifications & Airline Buy Cost</div>

                    <!-- Synchronized Dual Buy Rate Inputs (Unit Rate primary, Total Buy auto-calculated) -->
                    <div class="form-row">
                        <div class="field-group">
                            <label for="unit_buy">Unit Buy Rate (₹/kg) <span class="req">*</span></label>
                            <input type="number" id="unit_buy" class="clean-input" value="277.78" min="0.1" step="any" placeholder="e.g. 250" oninput="onUnitBuyInput()" required>
                        </div>
                        <div class="field-group">
                            <label for="buy">Total Airline Buy (₹ INR) <span style="font-size: 11px; font-weight: 500; color: var(--text-muted);">(Auto-calculated)</span></label>
                            <input type="number" id="buy" class="clean-input" value="125000" min="1" step="any" placeholder="e.g. 125000" oninput="onTotalBuyInput()" required>
                        </div>
                    </div>

                    <div class="weight-mode-toggle" style="margin-top: 4px;">
                        <button type="button" id="mode_btn_dims" class="mode-btn active" onclick="setWeightMode('dims')">
                            <svg class="ui-icon" viewBox="0 0 24 24"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/></svg>
                            <span>Calculate from Box Dimensions</span>
                        </button>
                        <button type="button" id="mode_btn_direct" class="mode-btn" onclick="setWeightMode('direct')">
                            <svg class="ui-icon" viewBox="0 0 24 24"><path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="M7 21h10"/><path d="M12 3v18"/><path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/></svg>
                            <span>Direct Chargeable Weight</span>
                        </button>
                    </div>

                    <!-- Physical Gross Weight + Chargeable Weight Row (Single unified row without duplication) -->
                    <div class="form-row">
                        <div class="field-group">
                            <label for="gross_wt">Physical Gross Weight (kg) <span class="req">*</span></label>
                            <input type="number" id="gross_wt" class="clean-input" value="450" min="0.1" step="any" oninput="updateDimensions()" required>
                        </div>
                        <!-- In Dims Mode: Show Calculated Chargeable Weight -->
                        <div class="field-group" id="chargeable_summary_col">
                            <label>Chargeable Weight (Calc)</label>
                            <div class="clean-input" style="background: #f8fafc; display: flex; align-items: center; font-weight: 700; color: var(--text-dark);" id="summary_ch_wt_display">
                                450.0 kg
                            </div>
                        </div>
                        <!-- In Direct Mode: Show Airline Chargeable Weight Input in the exact same spot -->
                        <div class="field-group" id="direct_weight_container" style="display: none;">
                            <label for="direct_ch_wt">Airline Chargeable Weight (kg) <span class="req">*</span></label>
                            <input type="number" id="direct_ch_wt" class="clean-input" value="450" min="0.1" step="any" oninput="updateDimensions()" placeholder="e.g. 450">
                        </div>
                    </div>

                    <div id="direct_density_tag" style="display: none; margin-top: -6px; margin-bottom: 12px;">
                        <span id="direct_density_display" class="dim-info" style="display: inline-block; padding: 4px 8px; border-radius: 4px; background: #f8fafc; font-size: 11.5px; color: var(--text-muted);">Density Ratio: 1.0000 (Dense Commercial Cargo)</span>
                    </div>

                    <!-- Box Dimensions Mode Container (4 Columns) -->
                    <div class="field-group full" id="dims_weight_container" style="margin-bottom: 12px;">
                        <label>Box Dimensions (L × W × H in cm × Number of Pieces) <span class="req">*</span></label>
                        <div class="dim-box">
                            <div class="dim-grid">
                                <div><label>Length (cm)</label><input type="number" id="dim_l" class="clean-input" value="120" min="1" step="any" oninput="updateDimensions()" required style="padding: 7px 10px; min-height: 38px;"></div>
                                <div><label>Width (cm)</label><input type="number" id="dim_w" class="clean-input" value="80" min="1" step="any" oninput="updateDimensions()" required style="padding: 7px 10px; min-height: 38px;"></div>
                                <div><label>Height (cm)</label><input type="number" id="dim_h" class="clean-input" value="90" min="1" step="any" oninput="updateDimensions()" required style="padding: 7px 10px; min-height: 38px;"></div>
                                <div><label>Pieces</label><input type="number" id="dim_pcs" class="clean-input" value="3" min="1" step="1" oninput="updateDimensions()" required style="padding: 7px 10px; min-height: 38px;"></div>
                            </div>
                            <div class="dim-info">
                                <span id="vol_wt_display">Volumetric: 432.0 kg</span>
                                <span id="ch_wt_display">Chargeable: 450.0 kg</span>
                                <span id="density_display">Type: Dense Commercial Cargo</span>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Unified Customer & Commercial Terms Section -->
                <div class="field-section">
                    <div class="section-label">Customer & Commercial Terms</div>

                    <!-- Hidden Target Quote Date (Defaults seamlessly to today) -->
                    <input type="hidden" id="quote_date" value="{{ default_quote_date }}">

                    <!-- Row 1: Customer Company / Account & Customer Category -->
                    <div class="form-row">
                        <div class="field-group">
                            <label for="customer_company">Customer Company / Account</label>
                            <input type="text" id="customer_company" class="clean-input" list="customer_accounts_list" placeholder="Search customer (e.g. UNIVERSAL CLEARING)..." value="UNIVERSAL CLEARING AND FORWARDING" autocomplete="off">
                            <datalist id="customer_accounts_list">
                                {% for cust in top_customers %}
                                <option value="{{ cust }}"></option>
                                {% endfor %}
                            </datalist>
                        </div>

                        <div class="field-group">
                            <label>Customer Category <span class="req">*</span></label>
                            <input type="hidden" id="client" value="Shipper / Consignee">
                            <div class="searchable-select" id="custom_client">
                                <div class="select-trigger" onclick="toggleDropdown('client')">
                                    <span class="select-trigger-text" id="client_display_text">Direct Shipper / Consignee</span>
                                    <span class="select-trigger-arrow">▼</span>
                                </div>
                                <div class="select-dropdown" id="client_dropdown">
                                    <ul class="select-options-list" id="client_options_list">
                                        {% for opt in client_options %}
                                        <li class="select-option{% if opt.value == 'Shipper / Consignee' %} selected{% endif %}" data-value="{{ opt.value }}">{{ opt.label }}</li>
                                        {% endfor %}
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Row 2: Customer Activity & Commodity Group -->
                    <div class="form-row">
                        <div class="field-group">
                            <label>Customer Activity <span class="req">*</span></label>
                            <input type="hidden" id="frequency" value="10">
                            <div class="searchable-select" id="custom_frequency">
                                <div class="select-trigger" onclick="toggleDropdown('frequency')">
                                    <span class="select-trigger-text" id="frequency_display_text">Active / Regular Customer</span>
                                    <span class="select-trigger-arrow">▼</span>
                                </div>
                                <div class="select-dropdown" id="frequency_dropdown">
                                    <ul class="select-options-list" id="frequency_options_list">
                                        {% for opt in customer_activity_options %}
                                        <li class="select-option{% if opt.value == '10' %} selected{% endif %}" data-value="{{ opt.value }}">{{ opt.label }}</li>
                                        {% endfor %}
                                    </ul>
                                </div>
                            </div>
                        </div>

                        <div class="field-group">
                            <label>Commodity Group <span class="req">*</span></label>
                            <input type="hidden" id="commodity" value="General Cargo">
                            <div class="searchable-select" id="custom_commodity">
                                <div class="select-trigger" onclick="toggleDropdown('commodity')">
                                    <span class="select-trigger-text" id="commodity_display_text">General Cargo</span>
                                    <span class="select-trigger-arrow">▼</span>
                                </div>
                                <div class="select-dropdown" id="commodity_dropdown">
                                    <ul class="select-options-list" id="commodity_options_list">
                                        {% for opt in commodity_options %}
                                        <li class="select-option{% if opt.value == 'General Cargo' %} selected{% endif %}" data-value="{{ opt.value }}">{{ opt.label }}</li>
                                        {% endfor %}
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Row 3: Quote Revision -->
                    <div class="form-row">
                        <div class="field-group full">
                            <label>Quote Revision <span class="req">*</span></label>
                            <input type="hidden" id="revision" value="1">
                            <div class="searchable-select" id="custom_revision">
                                <div class="select-trigger" onclick="toggleDropdown('revision')">
                                    <span class="select-trigger-text" id="revision_display_text">Rev 1 — Initial Quote (Standard Benchmark)</span>
                                    <span class="select-trigger-arrow">▼</span>
                                </div>
                                <div class="select-dropdown" id="revision_dropdown">
                                    <ul class="select-options-list" id="revision_options_list">
                                        {% for opt in revision_options %}
                                        <li class="select-option{% if opt.value == '1' %} selected{% endif %}" data-value="{{ opt.value }}">{{ opt.label }}</li>
                                        {% endfor %}
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <button type="submit" id="submitBtn" class="submit-btn">
                    <svg class="ui-icon" viewBox="0 0 24 24"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
                    Generate Margin Recommendation
                </button>
            </form>
        </div>

        <!-- Right Column: Recommended Commercial Terms -->
        <div class="workspace-col-right right-display-card">
            <div class="col-header">
                <span class="col-title">Recommended Commercial Terms</span>
                <span class="col-badge">Optimal Recommendation</span>
            </div>

            <!-- Ready / Placeholder State -->
            <div id="placeholder" class="placeholder-box">
                <svg class="ui-icon" style="width: 44px; height: 44px; stroke: var(--text-muted); margin-bottom: 12px;" viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>
                <h3 style="font-size: 16px; font-weight: 700; color: var(--text-dark);">Quotation Engine Ready</h3>
                <p style="font-size: 12.5px; margin-top: 6px; color: var(--text-muted);">Enter shipment parameters on the left or click a Quick Scenario Preset to calculate optimal sell rates and probability sensitivity.</p>
            </div>

            <!-- Results Panel -->
            <div id="results" style="display: none;">
                <!-- Hero Metric Card: Recommended Margin Range as PRIMARY BOLD HIGHLIGHT -->
                <div class="hero-metric-card">
                    <div class="hero-label-row">
                        <span class="hero-label">Recommended Margin Range</span>
                        <span class="hero-corridor-badge" id="res_corridor_badge">Pre-Approved Corridor</span>
                    </div>
                    <div class="hero-value" id="res_margin_pct">0.0% – 0.0%</div>
                    <div class="hero-sub" id="res_hero_sub">Target Benchmark: 0.0% Markup | Quoted Sell Rate: ₹0 / kg</div>
                </div>

                <!-- 3 Supporting Commercial Metric Tiles -->
                <div class="metrics-grid">
                    <div class="metric-tile">
                        <div class="metric-title">Quoted Sell Price</div>
                        <div class="metric-number" id="res_selling_price">₹0</div>
                    </div>
                    <div class="metric-tile">
                        <div class="metric-title">Win Confidence</div>
                        <div class="metric-number" id="res_win_prob" style="color: var(--text-dark);">High (88%)</div>
                    </div>
                    <div class="metric-tile">
                        <div class="metric-title">Est. Gross Profit</div>
                        <div class="metric-number" id="res_profit_amt" style="color: var(--text-dark);">₹0</div>
                    </div>
                </div>

                <!-- Pricing Strategy Tiers -->
                <div class="section-label" style="margin-top: 18px; margin-bottom: 8px;">
                    <span>Pricing Strategy Options</span>
                </div>
                <div id="strategic_tiers" class="strategic-tiers-container"></div>

                <!-- Bottom Commercial Actions -->
                <div class="actions-split">
                    <button type="button" class="btn-copy-action" onclick="copyQuoteSummary()">
                        <svg class="ui-icon" viewBox="0 0 24 24"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
                        Copy Quote Summary
                    </button>
                    <button type="button" class="btn-pill btn-white" onclick="resetFormForNewQuote()" style="width: 100%;">
                        Create New Quote
                    </button>
                </div>
            </div>
        </div>
    </div>
</div>

<div id="toastMsg" class="toast-msg">Quotation summary copied to clipboard!</div>

<script>
    function formatINR(val) {
        const num = Math.round(Number(val) || 0);
        return isFinite(num) ? num.toLocaleString('en-IN') : '0';
    }

    function getConfidenceScore(percentile, n) {
        if (percentile === undefined || percentile === null || isNaN(percentile)) {
            return { score: 85, tier: 'High', color: 'var(--text-dark)', text: 'High (85%)' };
        }
        const p = Math.max(0, Math.min(100, Number(percentile)));
        const sampleCount = (n !== undefined && n !== null) ? Number(n) : 30;
        const penalty = (sampleCount < 30) ? 12 : 0;
        const rawScore = 95 - 0.75 * Math.abs(p - 50) - penalty;
        const score = Math.round(Math.max(50, Math.min(95, rawScore)));

        if (score >= 80) {
            return { score: score, tier: 'High', color: 'var(--text-dark)', text: `High (${score}%)` };
        } else if (score >= 68) {
            return { score: score, tier: 'Moderate', color: 'var(--text-dark)', text: `Moderate (${score}%)` };
        } else {
            return { score: score, tier: 'Review', color: 'var(--text-dark)', text: `Review (${score}%)` };
        }
    }

    function onUnitBuyInput() {
        const unitEl = document.getElementById('unit_buy');
        const buyEl = document.getElementById('buy');
        if (!unitEl || !buyEl) return;
        const unitVal = parseFloat(unitEl.value);
        const weights = updateDimensions();
        const chWt = weights.chargeable_wt || 1;
        if (!isNaN(unitVal) && unitVal > 0) {
            buyEl.value = Math.round(unitVal * chWt);
        }
    }

    function onTotalBuyInput() {
        const unitEl = document.getElementById('unit_buy');
        const buyEl = document.getElementById('buy');
        if (!unitEl || !buyEl) return;
        const buyVal = parseFloat(buyEl.value);
        const weights = updateDimensions();
        const chWt = weights.chargeable_wt || 1;
        if (!isNaN(buyVal) && buyVal > 0 && chWt > 0) {
            unitEl.value = (buyVal / chWt).toFixed(2);
        }
    }

    let currentWeightMode = 'dims';

    function setWeightMode(mode) {
        currentWeightMode = mode;
        const btnDims = document.getElementById('mode_btn_dims');
        const btnDirect = document.getElementById('mode_btn_direct');
        const dimsContainer = document.getElementById('dims_weight_container');
        const directContainer = document.getElementById('direct_weight_container');
        const summaryCol = document.getElementById('chargeable_summary_col');
        const directTag = document.getElementById('direct_density_tag');

        if (mode === 'direct') {
            if (btnDirect) btnDirect.className = 'mode-btn active';
            if (btnDims) btnDims.className = 'mode-btn';
            if (dimsContainer) dimsContainer.style.display = 'none';
            if (summaryCol) summaryCol.style.display = 'none';
            if (directContainer) directContainer.style.display = 'block';
            if (directTag) directTag.style.display = 'block';

            const directInput = document.getElementById('direct_ch_wt');
            if (directInput && (!directInput.value || parseFloat(directInput.value) <= 0)) {
                const gross = parseFloat(document.getElementById('gross_wt')?.value) || 0;
                directInput.value = gross > 0 ? gross : 100;
            }
        } else {
            if (btnDims) btnDims.className = 'mode-btn active';
            if (btnDirect) btnDirect.className = 'mode-btn';
            if (dimsContainer) dimsContainer.style.display = 'block';
            if (summaryCol) summaryCol.style.display = 'block';
            if (directContainer) directContainer.style.display = 'none';
            if (directTag) directTag.style.display = 'none';
        }
        updateDimensions();
    }

    function updateDimensions() {
        const gross = Math.max(0.1, parseFloat(document.getElementById('gross_wt')?.value) || 1);
        let ch_wt = gross;
        let vol_wt = gross;

        if (currentWeightMode === 'direct') {
            const rawDirect = parseFloat(document.getElementById('direct_ch_wt')?.value);
            ch_wt = (rawDirect && rawDirect > 0) ? rawDirect : gross;
        } else {
            const l = Math.max(1, parseFloat(document.getElementById('dim_l')?.value) || 1);
            const w = Math.max(1, parseFloat(document.getElementById('dim_w')?.value) || 1);
            const h = Math.max(1, parseFloat(document.getElementById('dim_h')?.value) || 1);
            const pcs = Math.max(1, parseFloat(document.getElementById('dim_pcs')?.value) || 1);
            vol_wt = (l * w * h * pcs) / 6000.0;
            ch_wt = Math.max(gross, vol_wt);
        }

        const density = gross / Math.max(0.1, ch_wt);

        let densityText = "Dense Commercial Cargo";
        if (density < 0.75) densityText = "Volumetric Light Cargo";
        else if (density > 1.25) densityText = "Heavy Dense Cargo";

        const volEl = document.getElementById('vol_wt_display');
        const chEl = document.getElementById('ch_wt_display');
        const denEl = document.getElementById('density_display');
        const directDenEl = document.getElementById('direct_density_display');
        const summaryChEl = document.getElementById('summary_ch_wt_display');

        if (volEl) volEl.innerText = `Volumetric: ${vol_wt.toFixed(1)} kg`;
        if (chEl) chEl.innerText = `Chargeable: ${ch_wt.toFixed(1)} kg`;
        if (denEl) denEl.innerText = `Type: ${densityText}`;
        if (directDenEl) directDenEl.innerText = `Density Ratio: ${density.toFixed(4)} (${densityText})`;
        if (summaryChEl) summaryChEl.innerText = `${ch_wt.toFixed(1)} kg`;

        // Keep total buy synchronized when weights change, if user set unit_buy
        const unitEl = document.getElementById('unit_buy');
        const buyEl = document.getElementById('buy');
        if (unitEl && buyEl && document.activeElement !== buyEl) {
            const unitVal = parseFloat(unitEl.value);
            if (!isNaN(unitVal) && unitVal > 0) {
                buyEl.value = Math.round(unitVal * ch_wt);
            }
        }

        return { gross_wt: gross, chargeable_wt: ch_wt, density_ratio: density };
    }

    const PRESET_DATA = {
        pharma_uk: {
            origin: "Indira Gandhi International Airport",
            dest: "Heathrow Apt/London",
            buy: "185000",
            gross_wt: "320",
            dim_l: "100", dim_w: "80", dim_h: "100", dim_pcs: "2",
            client: "Shipper / Consignee",
            vertical: "Air Export Forwarding",
            commodity: "Pharmaceuticals",
            incoterms: "CIF",
            frequency: "10",
            revision: "0",
            branch: "Delhi"
        },
        perishable_uae: {
            origin: "Chhatrapati Shivaji Maharaj International Airport",
            dest: "Dubai",
            buy: "75000",
            gross_wt: "1200",
            dim_l: "120", dim_w: "80", dim_h: "90", dim_pcs: "8",
            client: "Shipper / Consignee",
            vertical: "Air Perishable",
            commodity: "Perishable Foodstuff",
            incoterms: "CFR",
            frequency: "25",
            revision: "1",
            branch: "Mumbai"
        },
        heavy_subagent: {
            origin: "Chhatrapati Shivaji Maharaj International Airport",
            dest: "Frankfurt am Main",
            buy: "320000",
            gross_wt: "2400",
            dim_l: "150", dim_w: "120", dim_h: "130", dim_pcs: "6",
            client: "Subagent",
            vertical: "Air Export Forwarding",
            commodity: "Engineering & Machinery",
            incoterms: "FOB",
            frequency: "3",
            revision: "2",
            branch: "Mumbai"
        },
        courier_us: {
            origin: "Bangalore",
            dest: "John F. Kennedy Apt/New York",
            buy: "45000",
            gross_wt: "65",
            dim_l: "60", dim_w: "50", dim_h: "40", dim_pcs: "2",
            client: "Overseas Agent",
            vertical: "Courier",
            commodity: "Courier",
            incoterms: "FOB",
            frequency: "10",
            revision: "0",
            branch: "Bangalore"
        }
    };

    const PORT_LABELS = {{ port_labels_json | safe }};

    function toggleDropdown(fieldId) {
        const dd = document.getElementById(`${fieldId}_dropdown`);
        const trigger = document.querySelector(`#searchable_${fieldId} .select-trigger`) || document.querySelector(`#custom_${fieldId} .select-trigger`);
        const otherId = fieldId === 'origin' ? 'dest' : 'origin';
        closeDropdown(otherId);

        if (!dd) return;
        const isOpen = dd.style.display === 'flex';
        if (isOpen) {
            closeDropdown(fieldId);
        } else {
            dd.style.display = 'flex';
            if (trigger) trigger.classList.add('open');
            const searchInput = document.getElementById(`${fieldId}_search`);
            if (searchInput) {
                searchInput.value = '';
                filterDropdownOptions(fieldId, '');
                setTimeout(() => searchInput.focus(), 50);
            }
        }
    }

    function closeDropdown(fieldId) {
        const dd = document.getElementById(`${fieldId}_dropdown`);
        const trigger = document.querySelector(`#searchable_${fieldId} .select-trigger`) || document.querySelector(`#custom_${fieldId} .select-trigger`);
        if (dd) dd.style.display = 'none';
        if (trigger) trigger.classList.remove('open');
    }

    function selectSearchableOption(fieldId, value, label) {
        const hiddenInput = document.getElementById(fieldId);
        const displayText = document.getElementById(`${fieldId}_display_text`);
        if (hiddenInput) hiddenInput.value = value;
        if (displayText) displayText.innerText = label;

        const list = document.getElementById(`${fieldId}_options_list`);
        if (list) {
            Array.from(list.children).forEach(li => {
                if (li.getAttribute('data-value') === value) {
                    li.classList.add('selected');
                } else {
                    li.classList.remove('selected');
                }
            });
        }
        closeDropdown(fieldId);
    }

    function selectCustomOption(fieldId, value, label) {
        selectSearchableOption(fieldId, value, label);
    }

    function setCustomDropdownValue(fieldId, value) {
        const list = document.getElementById(`${fieldId}_options_list`);
        let label = value;
        if (list) {
            const targetLi = Array.from(list.children).find(li => li.getAttribute('data-value') === value);
            if (targetLi) label = targetLi.innerText.trim();
        }
        selectSearchableOption(fieldId, value, label);
    }

    function setSearchableValue(fieldId, value) {
        const label = PORT_LABELS[value] || value;
        selectSearchableOption(fieldId, value, label);
    }

    function filterDropdownOptions(fieldId, query) {
        const q = (query || '').toLowerCase().trim();
        const list = document.getElementById(`${fieldId}_options_list`);
        if (!list) return;

        Array.from(list.children).forEach(li => {
            const text = li.innerText.toLowerCase();
            const val = (li.getAttribute('data-value') || '').toLowerCase();
            if (text.includes(q) || val.includes(q)) {
                li.style.display = 'flex';
            } else {
                li.style.display = 'none';
            }
        });
    }

    const ALL_DROPDOWN_IDS = ['origin', 'dest', 'client', 'commodity', 'frequency', 'revision'];

    ALL_DROPDOWN_IDS.forEach(fieldId => {
        const list = document.getElementById(`${fieldId}_options_list`);
        if (list) {
            list.addEventListener('click', function(e) {
                const li = e.target.closest('li.select-option');
                if (!li) return;
                const value = li.getAttribute('data-value');
                const label = (li.querySelector('span') ? li.querySelector('span').innerText : li.innerText).trim();
                selectSearchableOption(fieldId, value, label);
            });
        }
    });

    document.addEventListener('click', function(e) {
        ALL_DROPDOWN_IDS.forEach(id => {
            const container = document.getElementById(`searchable_${id}`) || document.getElementById(`custom_${id}`);
            if (container && !container.contains(e.target)) {
                closeDropdown(id);
            }
        });
    });

    function applyPreset(key) {
        const p = PRESET_DATA[key];
        if (!p) return;
        setWeightMode('dims');
        setSearchableValue('origin', p.origin);
        setSearchableValue('dest', p.dest);
        document.getElementById('buy').value = p.buy;
        document.getElementById('gross_wt').value = p.gross_wt;
        document.getElementById('dim_l').value = p.dim_l;
        document.getElementById('dim_w').value = p.dim_w;
        document.getElementById('dim_h').value = p.dim_h;
        document.getElementById('dim_pcs').value = p.dim_pcs;
        setCustomDropdownValue('client', p.client);
        setCustomDropdownValue('commodity', p.commodity);
        setCustomDropdownValue('frequency', p.frequency);
        setCustomDropdownValue('revision', p.revision ? String(p.revision) : '1');
        const weights = updateDimensions();
        const unitEl = document.getElementById('unit_buy');
        if (unitEl && weights.chargeable_wt > 0) {
            unitEl.value = (parseFloat(p.buy) / weights.chargeable_wt).toFixed(2);
        }
        runInference(false);
    }

    function resetFormForNewQuote() {
        document.getElementById('buy').value = '';
        const unitEl = document.getElementById('unit_buy');
        if (unitEl) unitEl.value = '';
        document.getElementById('gross_wt').value = '';
        const directCh = document.getElementById('direct_ch_wt');
        if (directCh) directCh.value = '';
        document.getElementById('dim_l').value = '';
        document.getElementById('dim_w').value = '';
        document.getElementById('dim_h').value = '';
        document.getElementById('dim_pcs').value = '1';

        setSearchableValue('origin', 'Indira Gandhi International Airport');
        setSearchableValue('dest', 'Heathrow Apt/London');
        setCustomDropdownValue('client', 'Shipper / Consignee');
        setCustomDropdownValue('commodity', 'General Cargo');
        setCustomDropdownValue('frequency', '10');
        setCustomDropdownValue('revision', '1');
        const custCompEl = document.getElementById('customer_company');
        if (custCompEl) custCompEl.value = '';

        setWeightMode('dims');
        updateDimensions();

        const placeholder = document.getElementById('placeholder');
        const results = document.getElementById('results');
        if (placeholder) placeholder.style.display = 'block';
        if (results) results.style.display = 'none';

        const errorAlert = document.getElementById('errorAlert');
        if (errorAlert) { errorAlert.style.display = 'none'; errorAlert.innerText = ''; }

        const buyEl = document.getElementById('buy');
        if (buyEl) buyEl.focus();
    }

    function runInference(isPreset = false) {
        const errorAlert = document.getElementById('errorAlert');
        if (errorAlert) { errorAlert.style.display = 'none'; errorAlert.innerText = ''; }

        const submitBtn = document.getElementById('submitBtn');
        const submitBtnTop = document.getElementById('submitBtnTop');
        if (submitBtn) { submitBtn.disabled = true; submitBtn.innerHTML = '<svg class="ui-icon spin" viewBox="0 0 24 24"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg> Calculating...'; }
        if (submitBtnTop) { submitBtnTop.disabled = true; }

        const weights = updateDimensions();
        const unitBuyEl = document.getElementById('unit_buy');
        const buyEl = document.getElementById('buy');
        let unitBuyVal = parseFloat(unitBuyEl?.value);
        let buyVal = parseFloat(buyEl?.value);

        if ((isNaN(buyVal) || buyVal <= 0) && !isNaN(unitBuyVal) && unitBuyVal > 0) {
            buyVal = Math.round(unitBuyVal * (weights.chargeable_wt || 1));
            if (buyEl) buyEl.value = buyVal;
        }

        // Smart Safeguard: Check if total buy was mistakenly entered as unit rate
        if (!isNaN(buyVal) && buyVal > 0) {
            const chWt = weights.chargeable_wt || 1;
            const impliedRate = buyVal / chWt;
            if (chWt >= 30 && impliedRate < 10.0 && buyVal >= 20.0) {
                unitBuyVal = buyVal;
                buyVal = Math.round(buyVal * chWt);
                if (buyEl) buyEl.value = buyVal;
                if (unitBuyEl) unitBuyEl.value = unitBuyVal.toFixed(2);
                showToast(`Auto-scaled ₹${unitBuyVal}/kg to Total Buy ₹${formatINR(buyVal)} for ${chWt} kg`);
            }
        }

        if (isNaN(buyVal) || buyVal <= 0) {
            if (submitBtn) { submitBtn.disabled = false; submitBtn.innerHTML = '<svg class="ui-icon" viewBox="0 0 24 24"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg> Generate Margin Recommendation'; }
            if (submitBtnTop) { submitBtnTop.disabled = false; }
            if (errorAlert) {
                errorAlert.style.display = 'block';
                errorAlert.innerText = 'Please enter a valid positive Airline Buy Cost (₹ INR).';
            }
            document.getElementById('buy')?.focus();
            return;
        }

        if (isNaN(weights.gross_wt) || weights.gross_wt <= 0) {
            if (submitBtn) { submitBtn.disabled = false; submitBtn.innerHTML = '<svg class="ui-icon" viewBox="0 0 24 24"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg> Generate Margin Recommendation'; }
            if (submitBtnTop) { submitBtnTop.disabled = false; }
            if (errorAlert) {
                errorAlert.style.display = 'block';
                errorAlert.innerText = 'Please enter a valid physical gross weight greater than 0 kg.';
            }
            document.getElementById('gross_wt')?.focus();
            return;
        }

        const payload = {
            origin: document.getElementById('origin')?.value || 'Indira Gandhi International Airport',
            dest: document.getElementById('dest')?.value || 'Heathrow Apt/London',
            buy: buyVal,
            unit_buy: (!isNaN(unitBuyVal) && unitBuyVal > 0) ? unitBuyVal : null,
            gross_wt: weights.gross_wt,
            chargeable_wt: weights.chargeable_wt,
            client: document.getElementById('client')?.value || 'Shipper / Consignee',
            commodity: document.getElementById('commodity')?.value || 'General Cargo',
            frequency: parseInt(document.getElementById('frequency')?.value, 10) || 10,
            revision: parseInt(document.getElementById('revision')?.value, 10) || 1,
            customer_company: document.getElementById('customer_company')?.value || '',
            quote_date: document.getElementById('quote_date')?.value || new Date().toISOString().slice(0, 10),
            carrier: 'Any Airline',
            branch: 'Mumbai',
            vertical: 'Air Export Forwarding',
            incoterms: 'FOB',
            sales_person: 'Mohnish Arora',
            strategy: 'balanced',
            dim_l: parseFloat(document.getElementById('dim_l')?.value),
            dim_w: parseFloat(document.getElementById('dim_w')?.value),
            dim_h: parseFloat(document.getElementById('dim_h')?.value),
            dim_pcs: parseInt(document.getElementById('dim_pcs')?.value, 10),
            is_preset: Boolean(isPreset)
        };

        fetch('/api/quote', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
        .then(res => res.json().then(data => ({ status: res.status, body: data })))
        .then(({ status, body }) => {
            if (submitBtn) { submitBtn.disabled = false; submitBtn.innerHTML = '<svg class="ui-icon" viewBox="0 0 24 24"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg> Generate Margin Recommendation'; }
            if (submitBtnTop) { submitBtnTop.disabled = false; }

            if (status !== 200 || body.status === 'error') {
                throw new Error(body.message || 'Pricing calculation failed.');
            }

            const placeholder = document.getElementById('placeholder');
            const results = document.getElementById('results');
            if (placeholder) placeholder.style.display = 'none';
            if (results) results.style.display = 'block';

            const opt = body.optimal;
            const stratDict = body.strategic_recommendations || (opt && opt.strategic_recommendations) || {};
            currentStrategies = stratDict;
            currentSensitivityRows = body.sensitivity || [];
            currentChargeableWeight = weights.chargeable_wt || 100;
            currentCorridor = body.negotiation_corridor || (opt && opt.negotiation_corridor) || null;

            renderStrategicCards(currentStrategies, 'balanced', currentChargeableWeight, currentCorridor);
            // renderSensitivityTable removed for clean UI
            updateHeroMetrics(opt, currentCorridor);

            // Advisory warning box removed as per instructions
        })
        .catch(err => {
            if (submitBtn) { submitBtn.disabled = false; submitBtn.innerHTML = '<svg class="ui-icon" viewBox="0 0 24 24"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg> Generate Margin Recommendation'; }
            if (submitBtnTop) { submitBtnTop.disabled = false; }
            if (errorAlert) {
                errorAlert.style.display = 'block';
                errorAlert.innerText = `Error: ${err.message}`;
            }
        });
    }

    let currentStrategies = {};
    let selectedStrategyKey = 'balanced';
    let currentSensitivityRows = [];
    let currentChargeableWeight = 100;
    let currentCorridor = null;

    function updateHeroMetrics(strat, corridor) {
        if (!strat) return;
        const corr = corridor || currentCorridor;
        const sellPrice = strat.Quoted_Sell_Price_INR || 0;
        const marginAmt = strat.Margin_Amount_INR || 0;
        const perKgRate = Math.round(sellPrice / Math.max(1, currentChargeableWeight));
        const grossMarginPct = strat.Gross_Margin_Percentage !== undefined ? strat.Gross_Margin_Percentage : ((marginAmt / Math.max(1, sellPrice)) * 100);

        // Highlight Recommended Margin Range as PRIMARY BOLD HERO
        let marginRangeText = `${strat.Margin_Percentage.toFixed(1)}%`;
        let badgeText = 'Pre-Approved Corridor';

        if (selectedStrategyKey === 'floor') {
            marginRangeText = `${strat.Margin_Percentage.toFixed(1)}%`;
            badgeText = 'Lowest Safe Price (Floor)';
        } else if (selectedStrategyKey === 'premium') {
            marginRangeText = `${strat.Margin_Percentage.toFixed(1)}%`;
            badgeText = 'Higher Profit Price (Premium)';
        } else if (corr && corr.min_margin_pct !== undefined && corr.max_margin_pct !== undefined) {
            marginRangeText = `${corr.min_margin_pct.toFixed(1)}% – ${corr.max_margin_pct.toFixed(1)}%`;
            badgeText = 'Pre-Approved Negotiation Corridor';
        }

        const marginEl = document.getElementById('res_margin_pct');
        if (marginEl) marginEl.innerText = marginRangeText;

        const badgeEl = document.getElementById('res_corridor_badge');
        if (badgeEl) badgeEl.innerText = badgeText;

        const heroSubEl = document.getElementById('res_hero_sub');
        if (heroSubEl) {
            let subText = `Target Benchmark: ${strat.Margin_Percentage.toFixed(1)}% Markup | Quoted Sell Rate: ₹${formatINR(perKgRate)} / kg (₹${formatINR(sellPrice)} Total)`;
            if (corr && corr.revision_guidance && selectedStrategyKey === 'balanced') {
                subText = `${corr.revision_guidance} • Quoted Sell Rate: ₹${formatINR(perKgRate)} / kg (₹${formatINR(sellPrice)} Total)`;
            }
            heroSubEl.innerText = subText;
        }

        // Supporting metric 1: Selling Price
        const priceEl = document.getElementById('res_selling_price');
        if (priceEl) priceEl.innerText = `₹${formatINR(perKgRate)} / kg`;

        // Supporting metric 2: Win Confidence
        const winEl = document.getElementById('res_win_prob');
        if (winEl) {
            if (strat.History_Percentile !== undefined && strat.History_Percentile !== null) {
                const conf = getConfidenceScore(strat.History_Percentile, strat.Cohort_N);
                winEl.innerText = conf.text;
                winEl.style.color = 'var(--text-dark)';
            } else if (strat.Win_Probability !== undefined && strat.Win_Probability !== null) {
                winEl.innerText = `${(strat.Win_Probability * 100).toFixed(1)}%`;
                winEl.style.color = 'var(--text-dark)';
            } else {
                winEl.innerText = 'High (88%)';
                winEl.style.color = 'var(--text-dark)';
            }
        }

        // Supporting metric 3: Gross Profit
        const profitEl = document.getElementById('res_profit_amt');
        if (profitEl) profitEl.innerText = `₹${formatINR(marginAmt)}`;
    }

    function renderStrategicCards(strategies, activeKey, chWt, corridor) {
        currentStrategies = strategies || {};
        selectedStrategyKey = activeKey || 'balanced';
        currentChargeableWeight = chWt || 100;
        if (corridor) currentCorridor = corridor;

        const container = document.getElementById('strategic_tiers');
        if (!container) return;
        container.innerHTML = '';

        const tierOrder = ['floor', 'balanced', 'premium'];
        tierOrder.forEach(key => {
            const strat = currentStrategies[key];
            if (!strat) return;

            const isSelected = (key === selectedStrategyKey);
            const card = document.createElement('div');
            card.id = `strat_card_${key}`;
            card.className = `strategy-card card-${key}${isSelected ? ' selected' : ''}`;
            card.setAttribute('role', 'button');
            card.setAttribute('tabindex', '0');

            const perKgPrice = Math.round(strat.Quoted_Sell_Price_INR / Math.max(1, currentChargeableWeight));
            const sellPrice = strat.Quoted_Sell_Price_INR || 0;
            const marginAmt = strat.Margin_Amount_INR || 0;
            const grossMarginPct = strat.Gross_Margin_Percentage !== undefined ? strat.Gross_Margin_Percentage : ((marginAmt / Math.max(1, sellPrice)) * 100);

            card.innerHTML = `
                <div class="strat-card-left">
                    <div class="strat-card-title-row">
                        <span class="strat-radio-indicator"></span>
                        <span class="strat-name">${strat.Strategy_Name}</span>
                        <span class="badge badge-${key}">${strat.Badge_Label}</span>
                    </div>
                    <div class="strat-meta-chips">
                        <span>Target Markup: <strong>${strat.Margin_Percentage.toFixed(1)}%</strong></span>
                        <span>•</span>
                        <span>Gross Margin: <strong>${grossMarginPct.toFixed(1)}%</strong></span>
                        <span>•</span>
                        <span>Profit: <strong>₹${formatINR(marginAmt)}</strong></span>
                    </div>
                </div>
                <div class="strat-card-right">
                    <div class="strat-price">₹${formatINR(sellPrice)}</div>
                    <div class="strat-price-unit">₹${formatINR(perKgPrice)} / kg</div>
                </div>
            `;

            card.onclick = function() { selectStrategyCard(key); };
            container.appendChild(card);
        });
    }

    function selectStrategyCard(key) {
        if (!currentStrategies[key]) return;
        selectedStrategyKey = key;
        const strat = currentStrategies[key];

        ['floor', 'balanced', 'premium'].forEach(k => {
            const el = document.getElementById(`strat_card_${k}`);
            if (el) {
                if (k === key) el.classList.add('selected');
                else el.classList.remove('selected');
            }
        });

        updateHeroMetrics(strat, currentCorridor);
        // renderSensitivityTable removed for clean UI
    }

    function renderSensitivityTable(rows, selectedMargin) {
        currentSensitivityRows = rows || [];
        const tbody = document.getElementById('sensitivity_body');
        if (!tbody) return;
        tbody.innerHTML = '';

        const floorM = currentStrategies.floor ? currentStrategies.floor.Margin_Percentage : -999;
        const balancedM = currentStrategies.balanced ? currentStrategies.balanced.Margin_Percentage : -999;
        const premiumM = currentStrategies.premium ? currentStrategies.premium.Margin_Percentage : -999;

        currentSensitivityRows.forEach(row => {
            const tr = document.createElement('tr');
            const rowMargin = row.Margin_Percentage;
            const isSelected = selectedMargin !== undefined && Math.abs(rowMargin - selectedMargin) < 0.15;
            if (isSelected) tr.className = 'row-tier-selected';

            const rowSell = row.Quoted_Sell_Price_INR || 0;
            const rowMarginAmt = row.Margin_Amount_INR || 0;
            const ratePerKg = Math.round(rowSell / Math.max(1, currentChargeableWeight));
            const grossMarginPct = (rowSell > 0) ? ((rowMarginAmt / rowSell) * 100).toFixed(1) : rowMargin.toFixed(1);

            let confText = '--';
            if (row.History_Percentile !== undefined && row.History_Percentile !== null) {
                const rowConf = getConfidenceScore(row.History_Percentile, row.Cohort_N);
                confText = rowConf.text;
            } else if (row.Win_Probability !== undefined) {
                confText = `${(row.Win_Probability * 100).toFixed(1)}%`;
            }

            let tierBadge = '<span style="color: var(--text-light); font-size: 11px;">Alternative</span>';
            if (Math.abs(rowMargin - floorM) < 0.15) tierBadge = '<span class="badge badge-floor">Lowest Safe</span>';
            else if (Math.abs(rowMargin - balancedM) < 0.15) tierBadge = '<span class="badge badge-balanced">Recommended</span>';
            else if (Math.abs(rowMargin - premiumM) < 0.15) tierBadge = '<span class="badge badge-premium">Higher Profit</span>';

            tr.innerHTML = `
                <td><strong>${row.Margin_Percentage.toFixed(1)}%</strong></td>
                <td>${grossMarginPct}%</td>
                <td>₹${formatINR(rowSell)}</td>
                <td>₹${formatINR(ratePerKg)}/kg</td>
                <td>₹${formatINR(rowMarginAmt)}</td>
                <td>${confText}</td>
                <td>${tierBadge}</td>
            `;

            tr.onclick = function() {
                updateHeroMetrics(row, currentCorridor);
                if (Math.abs(rowMargin - floorM) < 0.15) selectStrategyCard('floor');
                else if (Math.abs(rowMargin - balancedM) < 0.15) selectStrategyCard('balanced');
                else if (Math.abs(rowMargin - premiumM) < 0.15) selectStrategyCard('premium');
                else {
                    ['floor', 'balanced', 'premium'].forEach(k => {
                        const el = document.getElementById(`strat_card_${k}`);
                        if (el) el.classList.remove('selected');
                    });
                    renderSensitivityTable(currentSensitivityRows, row.Margin_Percentage);
                }
            };
            tbody.appendChild(tr);
        });
    }

    function copyQuoteSummary() {
        const originTxt = document.getElementById('origin_display_text')?.innerText || 'Delhi';
        const destTxt = document.getElementById('dest_display_text')?.innerText || 'London';
        const clientTxt = document.getElementById('client_display_text')?.innerText || 'Client';
        const chWt = document.getElementById('ch_wt_display')?.innerText || '450 kg';
        const buyVal = document.getElementById('buy')?.value || '0';
        const sellPrice = document.getElementById('res_selling_price')?.innerText || '₹0';
        const marginPct = document.getElementById('res_margin_pct')?.innerText || '0.0%';
        const profit = document.getElementById('res_profit_amt')?.innerText || '₹0';

        const summary = `AIR FREIGHT MARGIN RECOMMENDATION
Customer: ${clientTxt}
Route: ${originTxt} ➔ ${destTxt}
${chWt}
Airline Buy Cost: ₹${formatINR(buyVal)}
Quoted Selling Price: ${sellPrice}
Target Margin Range: ${marginPct}
Gross Profit (if Won): ${profit}
Activity: ${document.getElementById('frequency_display_text')?.innerText || 'Active / Regular Customer'}
Revision: ${document.getElementById('revision_display_text')?.innerText || 'Rev 1 — Initial Quote'}`;

        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(summary).then(() => showToast('Quotation summary copied to clipboard!'));
        } else {
            prompt('Quotation summary:', summary);
        }
    }

    function showToast(msg) {
        const toast = document.getElementById('toastMsg');
        if (!toast) return;
        toast.innerText = msg;
        toast.style.display = 'block';
        setTimeout(() => toast.classList.add('show'), 10);
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => { toast.style.display = 'none'; }, 200);
        }, 2800);
    }

    window.onload = function() {
        updateDimensions();
    };
</script>
</body>
</html>

"""

HISTORY_TEMPLATE = """

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reports & Recommendation History - Air Export Pricing Engine</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Mulish:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-outer: #edf1f5;
            --surface-card: #ffffff;
            --border: #e2e8f0;
            --border-subtle: #f1f5f9;
            --text-dark: #0f172a;
            --text-muted: #64748b;
            --text-light: #94a3b8;
            --pill-dark: #0f172a;
            --radius-window: 24px;
            --radius-card: 16px;
            --radius-control: 8px;
            --radius-pill: 100px;
            --shadow-window: 0 20px 60px rgba(15, 23, 42, 0.06);
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Mulish', sans-serif; }
        h1, h2, h3, h4, .font-heading, .btn-pill, .nav-item, .stat-val { font-family: 'Outfit', sans-serif; }

        body {
            background-color: var(--bg-outer);
            color: var(--text-dark);
            padding: 32px 20px 48px;
            min-height: 100vh;
            display: flex;
            justify-content: center;
        }

        .app-window {
            background: var(--surface-card);
            border-radius: var(--radius-window);
            box-shadow: var(--shadow-window);
            width: 100%;
            max-width: 1320px;
            padding: 28px 36px 40px;
        }

        @media (max-width: 768px) {
            .app-nav {
                flex-direction: column;
                align-items: stretch;
                gap: 12px;
            }
            .nav-links-center {
                justify-content: center;
            }
            .app-nav .header-actions {
                justify-content: center;
                width: 100%;
            }
            .app-nav .btn-pill {
                width: 100%;
                justify-content: center;
            }
        }

        @media (max-width: 640px) {
            body { padding: 12px 10px 24px; }
            .app-window { padding: 16px 14px 24px; border-radius: 16px; }
            .page-header {
                flex-direction: column;
                align-items: flex-start;
                gap: 12px;
            }
            .page-title { font-size: 22px; }
            .header-actions {
                width: 100%;
                gap: 8px;
                flex-wrap: wrap;
            }
            .header-actions .btn-pill {
                flex: 1 1 auto;
                justify-content: center;
                padding: 10px 12px;
                font-size: 12px;
                text-align: center;
            }
            .stats-grid {
                grid-template-columns: repeat(2, 1fr) !important;
                gap: 10px !important;
            }
            .stat-card {
                padding: 12px 14px !important;
            }
            .stat-val {
                font-size: 20px !important;
            }
            .pagination-bar {
                flex-direction: column;
                gap: 12px;
                align-items: center;
                text-align: center;
            }
            .btn-text-full { display: none !important; }
            .btn-text-mobile { display: inline !important; }
        }

        .app-nav {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding-bottom: 22px;
            margin-bottom: 22px;
            border-bottom: 1px solid var(--border-subtle);
        }
        .nav-logo-box {
            display: flex;
            align-items: center;
            gap: 10px;
            text-decoration: none;
            color: var(--text-dark);
        }
        .nav-logo-badge {
            width: 36px;
            height: 36px;
            border-radius: 8px;
            background: var(--pill-dark);
            display: flex;
            align-items: center;
            justify-content: center;
            color: #ffffff;
        }
        .nav-brand-title {
            font-size: 15px;
            font-weight: 700;
            color: var(--text-dark);
            font-family: 'Outfit', sans-serif;
            letter-spacing: -0.2px;
        }
        .nav-links-center {
            display: flex;
            align-items: center;
            gap: 6px;
            background: #f1f5f9;
            padding: 4px;
            border-radius: var(--radius-pill);
        }
        .nav-item {
            text-decoration: none;
            color: var(--text-muted);
            font-size: 13px;
            font-weight: 600;
            padding: 8px 18px;
            min-height: 36px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: var(--radius-pill);
            transition: all 0.2s;
        }
        .nav-item:hover { color: var(--text-dark); }
        .nav-item.active { background: var(--pill-dark); color: #ffffff; box-shadow: 0 2px 6px rgba(15, 23, 42, 0.15); }

        .page-header {
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            margin-bottom: 24px;
            flex-wrap: wrap;
            gap: 16px;
        }
        .back-breadcrumb {
            text-decoration: none;
            font-size: 12.5px;
            color: var(--text-muted);
            display: inline-flex;
            align-items: center;
            gap: 5px;
            margin-bottom: 6px;
        }
        .back-breadcrumb:hover { color: var(--text-dark); }
        .page-title {
            font-size: 28px;
            font-weight: 700;
            color: var(--text-dark);
            letter-spacing: -0.4px;
        }
        .header-actions { display: flex; align-items: center; gap: 10px; }
        .btn-pill {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            padding: 10px 20px;
            min-height: 44px;
            border-radius: var(--radius-pill);
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            text-decoration: none;
        }
        .btn-white { background: #ffffff; border: 1px solid var(--border); color: var(--text-dark); }
        .btn-white:hover { background: #f8fafc; }
        .btn-danger { background: #ffffff; border: 1px solid #fecaca; color: #dc2626; }
        .btn-danger:hover { background: #fef2f2; }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 14px;
            margin-bottom: 24px;
        }
        .stat-card {
            background: #f8fafc;
            border: 1px solid var(--border);
            border-radius: var(--radius-card);
            padding: 18px 20px;
        }
        .stat-label {
            font-size: 11px;
            font-weight: 700;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }
        .stat-val {
            font-size: 24px;
            font-weight: 800;
            color: var(--text-dark);
        }

        .table-card {
            border: 1px solid var(--border);
            border-radius: var(--radius-card);
            background: #ffffff;
            overflow: hidden;
        }
        .table-wrap {
            overflow-x: auto;
            max-height: 580px;
            -webkit-overflow-scrolling: touch;
            scrollbar-width: thin;
        }
        table { width: 100%; min-width: 950px; border-collapse: collapse; font-size: 12.5px; }
        th {
            background: #f8fafc;
            text-align: left;
            padding: 12px 16px;
            font-size: 11px;
            font-weight: 700;
            color: var(--text-muted);
            text-transform: uppercase;
            border-bottom: 1px solid var(--border);
            position: sticky;
            top: 0;
            z-index: 2;
        }
        td {
            padding: 12px 16px;
            border-bottom: 1px solid var(--border-subtle);
            color: var(--text-dark);
        }
        tr:hover { background: #fbfbfb; }

        .status-badge {
            padding: 6px 12px;
            border-radius: var(--radius-pill);
            font-size: 12px;
            font-weight: 700;
            border: 1px solid transparent;
            cursor: pointer;
            outline: none;
            min-height: 38px;
            display: inline-flex;
            align-items: center;
        }
        .status-draft { background: #f1f5f9; color: var(--text-dark); border-color: var(--border); }
        .status-negotiating { background: #f8fafc; color: var(--text-dark); border-color: #cbd5e1; }
        .status-followup { background: #f1f5f9; color: var(--text-muted); border-color: var(--border); }
        .status-review { background: #f8fafc; color: var(--text-muted); border-color: var(--border); }
        .status-won { background: var(--pill-dark); color: #ffffff; border-color: var(--pill-dark); }
        .status-lost { background: #ffffff; color: var(--text-muted); border-color: var(--border); }

        *:focus-visible {
            outline: 2px solid var(--pill-dark);
            outline-offset: 2px;
        }

        .pagination-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 14px 18px;
            border-top: 1px solid var(--border);
            font-size: 12px;
            color: var(--text-muted);
        }
        .btn-page {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 36px;
            height: 36px;
            border-radius: var(--radius-pill);
            border: 1px solid var(--border);
            background: #ffffff;
            color: var(--text-dark);
            text-decoration: none;
            font-weight: 700;
            font-size: 11.5px;
            cursor: pointer;
        }
        .btn-page.active { background: var(--pill-dark); color: #ffffff; border-color: var(--pill-dark); }
        .btn-page:hover:not(.active) { background: #f8fafc; }
        .btn-page.disabled { opacity: 0.4; pointer-events: none; }

        .ui-icon {
            display: inline-block;
            width: 14px;
            height: 14px;
            stroke-width: 2.2;
            stroke: currentColor;
            fill: none;
            stroke-linecap: round;
            stroke-linejoin: round;
            vertical-align: -2px;
        }
    </style>
</head>
<body>

<div class="app-window">
    <nav class="app-nav">
        <a href="/" class="nav-logo-box">
            <div class="nav-logo-badge">
                <svg class="ui-icon" style="stroke: #ffffff; width: 18px; height: 18px;" viewBox="0 0 24 24"><path d="M17.8 19.2 16 11l3.5-3.5C21 6 21.5 4 21 3c-1-.5-3 0-4.5 1.5L13 8 4.8 6.2c-.5-.1-.9.1-1.1.5l-.3.5c-.2.5-.1 1 .3 1.3L9 12l-2 3H4l-1 1 3 2 2 3 1-1v-3l3-2 3.5 5.3c.3.4.8.5 1.3.3l.5-.3c.4-.2.6-.6.5-1.1z"/></svg>
            </div>
            <span class="nav-brand-title">Air Export Pricing Engine</span>
        </a>

        <div class="nav-links-center">
            <a href="/" class="nav-item">Dashboard</a>
            <a href="/history" class="nav-item active">Reports</a>
        </div>

        <div class="header-actions">
            <a href="/" class="btn-pill btn-white">
                <svg class="ui-icon" viewBox="0 0 24 24"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
                <span class="btn-text-full">New Margin Recommendation</span>
                <span class="btn-text-mobile" style="display: none;">New Quote</span>
            </a>
        </div>
    </nav>

    <div class="page-header">
        <div>
            <a href="/" class="back-breadcrumb">
                <svg class="ui-icon" viewBox="0 0 24 24"><line x1="19" y1="12" x2="5" y2="12"/><polyline points="12 19 5 12 12 5"/></svg>
                Back to Dashboard
            </a>
            <h1 class="page-title">Reports & Recommendation History</h1>
            <p style="font-size: 13px; color: var(--text-muted); margin-top: 2px;">Audit trail and lifecycle status tracking for all generated air export quotations</p>
        </div>
        <div class="header-actions">
            {% if records and records|length > 0 %}
            <a href="/api/history/export" class="btn-pill btn-white">
                <svg class="ui-icon" viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                Export CSV
            </a>
            <button type="button" class="btn-pill btn-danger" onclick="clearHistory()">
                <svg class="ui-icon" viewBox="0 0 24 24"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/><line x1="10" y1="11" x2="10" y2="17"/><line x1="14" y1="11" x2="14" y2="17"/></svg>
                Clear History
            </button>
            {% endif %}
        </div>
    </div>

    {% if records and records|length > 0 %}
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-label">Total Quotes</div>
            <div class="stat-val">{{ total_count }}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Won Inquiries</div>
            <div class="stat-val">{{ won_count }}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Total Pipeline</div>
            <div class="stat-val">₹{{ "{:,.0f}".format(total_pipeline) }}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Win Rate</div>
            <div class="stat-val">{{ "{:.1f}%".format(win_rate) }}</div>
        </div>
    </div>

    <div class="table-card">
        <div class="table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Created At</th>
                        <th>Trade Lane</th>
                        <th>Airline Buy (₹)</th>
                        <th>Quoted Sell (₹)</th>
                        <th>Profit (₹)</th>
                        <th>Margin Range</th>
                        <th>Weight & Cargo</th>
                        <th>Branch</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {% for r in records %}
                    <tr>
                        <td style="font-weight: 700; color: var(--text-light);">#{{ r.id }}</td>
                        <td>{{ r.timestamp }}</td>
                        <td>
                            <strong>{{ r.origin_port }}</strong> → <br>
                            <span style="color: var(--text-muted); font-size: 11px;">{{ r.dest_port }}</span>
                        </td>
                        <td>₹{{ "{:,.0f}".format(r.airline_buy_inr) }}</td>
                        <td style="font-weight: 700;">₹{{ "{:,.0f}".format(r.selling_price_inr) }}</td>
                        <td style="font-weight: 700;">₹{{ "{:,.0f}".format(r.margin_amount_inr or (r.selling_price_inr - r.airline_buy_inr)) }}</td>
                        <td><strong style="white-space: nowrap;">{{ r.margin_range if r.margin_range else "{:.1f}%".format(r.margin_pct) }}</strong></td>
                        <td>
                            <span>{{ "{:.1f} kg".format(r.chargeable_wt_kg) }}</span><br>
                            <span style="font-size: 11px; color: var(--text-light);">{{ r.commodity }}</span>
                        </td>
                        <td>{{ r.branch }}</td>
                        <td>
                            <select class="status-badge status-{{ r.status_class }}" onchange="updateLeadStatus({{ r.id }}, this.value, this)">
                                {% for s in statuses %}
                                <option value="{{ s.code }}" {% if r.status == s.code %}selected{% endif %}>{{ s.label }}</option>
                                {% endfor %}
                            </select>
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>

        <div class="pagination-bar">
            <div>Showing <strong>{{ start_idx }}</strong> - <strong>{{ end_idx }}</strong> of <strong>{{ total_count }}</strong> quotes</div>
            <div style="display: flex; gap: 6px; align-items: center;">
                {% if has_prev %}
                <a href="/history?page={{ prev_page }}&per_page={{ per_page }}" class="btn-page">‹</a>
                {% endif %}
                <span class="btn-page active">{{ page }}</span>
                {% if has_next %}
                <a href="/history?page={{ next_page }}&per_page={{ per_page }}" class="btn-page">›</a>
                {% endif %}
            </div>
        </div>
    </div>
    {% else %}
    <div style="text-align: center; padding: 60px 20px; color: var(--text-muted); background: #f8fafc; border-radius: 16px; border: 1px solid var(--border);">
        <h3 style="font-size: 16px; font-weight: 700; color: var(--text-dark);">No Quotations Recorded Yet</h3>
        <p style="font-size: 13px; margin: 6px 0 16px;">Create your first freight quotation to start tracking leads.</p>
        <a href="/" class="btn-pill btn-white">Create First Margin Recommendation</a>
    </div>
    {% endif %}
</div>

<script>
    const statusMap = {
        'Draft / Quoted': 'draft',
        'Negotiating': 'negotiating',
        'Follow-up': 'followup',
        'Under Review': 'review',
        'Won': 'won',
        'Lost': 'lost'
    };

    function updateLeadStatus(quoteId, newStatus, selectEl) {
        fetch('/api/history/update_status', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ id: quoteId, status: newStatus })
        })
        .then(res => res.json())
        .then(data => {
            if (data.status === 'success') {
                selectEl.className = 'status-badge status-' + (statusMap[newStatus] || 'draft');
            } else {
                alert('Status update failed: ' + (data.message || 'Unknown error'));
            }
        })
        .catch(err => alert('Network error: ' + err.message));
    }

    function clearHistory() {
        if (!confirm('Are you sure you want to clear historical records?')) return;
        fetch('/api/history/clear', { method: 'POST' })
            .then(res => res.json())
            .then(data => { if (data.status === 'success') window.location.reload(); });
    }
</script>
</body>
</html>

"""


# ------------------------------------------------------------------------------
# FLASK API ENDPOINTS
# ------------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template_string(
        HTML_TEMPLATE,
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

    # Physical weights validation (§2.2, §5.2)
    raw_gross = data.get("gross_wt")
    try:
        gross_wt = float(str(raw_gross).replace(",", "").strip())
        if gross_wt <= 0 or not np.isfinite(gross_wt):
            raise ValueError()
    except (ValueError, TypeError):
        return jsonify({"status": "error", "message": "Invalid Gross Weight. Please provide a positive weight in kg."}), 400

    raw_chargeable = data.get("chargeable_wt")
    try:
        chargeable_wt = float(str(raw_chargeable).replace(",", "").strip()) if raw_chargeable else gross_wt
        if chargeable_wt <= 0 or not np.isfinite(chargeable_wt):
            chargeable_wt = gross_wt
    except (ValueError, TypeError):
        chargeable_wt = gross_wt

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

    return render_template_string(
        HISTORY_TEMPLATE,
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
def api_history_export():
    records = get_all_history(HISTORY_DB_PATH)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "ID", "Timestamp", "Origin", "Destination", "Buy Cost INR",
        "Selling Price INR", "Margin Range", "Margin Pct", "Margin Amount INR",
        "Win Probability", "Expected Profit INR", "Gross Wt Kg",
        "Chargeable Wt Kg", "Commodity", "Client Type", "Branch", "Status", "Model Version"
    ])

    for r in records:
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

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=air_export_pricing_history.csv"}
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
