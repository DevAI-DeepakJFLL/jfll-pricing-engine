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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pricing_web")

# Prevent browser caching
@app.after_request
def add_cache_headers(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# ------------------------------------------------------------------------------
# SQLITE HISTORY TRACKING STORE
# ------------------------------------------------------------------------------
HISTORY_DB_PATH = os.path.join(ENGINE_ROOT, "data", "recommendations_history.db")
init_history_db(HISTORY_DB_PATH)


# ------------------------------------------------------------------------------
# LOAD MODEL ARTIFACT ON STARTUP (§1.4, §5.6)
# ------------------------------------------------------------------------------
MODEL_PATH = os.path.join(ENGINE_ROOT, "models", "freight_margin_recommender.joblib")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, "models", "freight_margin_recommender.joblib")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model artifact not found at canonical path: {MODEL_PATH}. Please run train.py first.")

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

import datetime
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
PORT_LABELS_JSON = json.dumps(PORT_LABELS)


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
    <title>Air Freight Margin Recommender</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@500;600;700;800&family=Mulish:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #0284c7;
            --primary-dark: #0369a1;
            --primary-light: #e0f2fe;
            --secondary: #16a34a;
            --secondary-dark: #15803d;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --border: #e2e8f0;
            --text-dark: #0f172a;
            --text-muted: #64748b;
            --radius-card: 16px;
            --radius-control: 10px;
            --radius-pill: 9999px;
            --shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.06);
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Mulish', sans-serif; }
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
                scroll-behavior: auto !important;
            }
        }
        h1, h2, h3, h4, .card-header, .hero-value, .submit-btn, .preset-btn, .nav-history-btn { font-family: 'Outfit', sans-serif; }
        body { background-color: var(--bg); color: var(--text-dark); padding: 32px 20px; line-height: 1.5; }
        .container { max-width: 1200px; margin: 0 auto; }

        /* Header */
        header { margin-bottom: 24px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; }
        h1 { font-size: 26px; font-weight: 800; color: var(--text-dark); }
        p.subtitle { color: var(--text-muted); font-size: 13.5px; margin-top: 3px; }

        /* Presets Bar */
        .preset-bar { background: #f0f9ff; border: 1px solid #bae6fd; border-radius: var(--radius-card); padding: 16px 20px; margin-bottom: 24px; }
        .preset-title { font-size: 11.5px; font-weight: 700; color: var(--primary-dark); display: block; margin-bottom: 10px; letter-spacing: 0.5px; text-transform: uppercase; }
        .preset-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; }
        .preset-btn { background: white; border: 1px solid #cbd5e1; border-radius: var(--radius-pill); padding: 10px 16px; min-height: 44px; display: inline-flex; align-items: center; font-size: 12px; font-weight: 600; cursor: pointer; transition: all 0.2s; text-align: left; width: 100%; color: var(--text-dark); }
        .preset-btn:hover { background: var(--primary); color: white; border-color: var(--primary); }

        /* Layout & Grid Defenses */
        .layout { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 24px; align-items: start; }
        @media (max-width: 900px) { .layout { grid-template-columns: minmax(0, 1fr); } }

        /* SVG UI Icons */
        .ui-icon {
            display: inline-block;
            width: 15px;
            height: 15px;
            stroke-width: 2.2;
            stroke: currentColor;
            fill: none;
            stroke-linecap: round;
            stroke-linejoin: round;
            vertical-align: -2px;
            flex-shrink: 0;
        }
        .ui-icon-lg {
            display: inline-block;
            width: 42px;
            height: 42px;
            stroke-width: 1.8;
            stroke: var(--primary);
            fill: none;
            stroke-linecap: round;
            stroke-linejoin: round;
            margin-bottom: 8px;
        }
        .spin {
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }

        /* Cards */
        .card { min-width: 0; max-width: 100%; box-sizing: border-box; background: var(--card-bg); border: 1px solid var(--border); border-radius: var(--radius-card); padding: 24px; box-shadow: var(--shadow); margin-bottom: 24px; }
        .card.card-input { border-left: 4px solid var(--primary); }
        .card.card-output { border-left: 4px solid var(--secondary); }
        .card-header { font-size: 16px; font-weight: 700; margin-bottom: 18px; display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid var(--border); padding-bottom: 12px; }

        /* Form Inputs */
        .field-section { margin-bottom: 18px; }
        .section-label { font-size: 11px; font-weight: 700; color: var(--primary-dark); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 10px; display: flex; align-items: center; gap: 8px; }
        .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
        .form-group { display: flex; flex-direction: column; gap: 5px; }
        .form-group.full { grid-column: 1 / -1; }
        label { font-size: 12px; font-weight: 600; color: var(--text-dark); }
        input { width: 100%; padding: 10px 12px; border: 1px solid var(--border); border-radius: var(--radius-control); font-size: 13px; background: #ffffff; color: var(--text-dark); font-family: 'Mulish', sans-serif; transition: all 0.2s ease; }
        input:focus { outline: none; border-color: var(--primary); box-shadow: 0 0 0 3px rgba(2, 132, 199, 0.2); }

        select {
            width: 100%;
            padding: 10px 36px 10px 12px;
            border: 1px solid var(--border);
            border-radius: var(--radius-control);
            font-size: 13px;
            font-family: 'Mulish', sans-serif;
            background-color: #ffffff;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6' viewBox='0 0 10 6'%3E%3Cpath fill='%2364748b' d='M0 0l5 6 5-6z'/%3E%3C/svg%3E");
            background-repeat: no-repeat;
            background-position: right 14px center;
            background-size: 10px 6px;
            color: var(--text-dark);
            cursor: pointer;
            outline: none;
            -webkit-appearance: none;
            -moz-appearance: none;
            appearance: none;
            transition: all 0.2s ease;
        }
        select:hover {
            border-color: var(--primary);
        }
        select:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(2, 132, 199, 0.2);
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6' viewBox='0 0 10 6'%3E%3Cpath fill='%230369a1' d='M0 0l5 6 5-6z'/%3E%3C/svg%3E");
        }
        select option {
            padding: 10px 12px;
            font-size: 13px;
            color: var(--text-dark);
            background: #ffffff;
        }

        /* Searchable Custom Dropdown (Strict Brand Alignment: Outfit/Mulish, Sky/Emerald) */
        .searchable-select { position: relative; width: 100%; }
        .select-trigger {
            width: 100%;
            padding: 10px 12px;
            border: 1px solid var(--border);
            border-radius: var(--radius-control);
            font-size: 13px;
            background: #ffffff;
            color: var(--text-dark);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            user-select: none;
            transition: all 0.2s;
            min-height: 42px;
        }
        .select-trigger:hover { border-color: var(--primary); }
        .select-trigger.open {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(2, 132, 199, 0.2);
            border-bottom-left-radius: 4px;
            border-bottom-right-radius: 4px;
        }
        .select-trigger-text {
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            font-family: 'Mulish', sans-serif;
        }
        .select-trigger-arrow {
            font-size: 10px;
            color: var(--text-muted);
            transition: transform 0.2s;
            flex-shrink: 0;
        }
        .select-trigger.open .select-trigger-arrow {
            transform: rotate(180deg);
            color: var(--primary);
        }
        .select-dropdown {
            position: absolute;
            top: 100%;
            left: 0;
            right: 0;
            margin-top: 4px;
            background: #ffffff;
            border: 1.5px solid var(--primary);
            border-radius: var(--radius-control);
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.15);
            z-index: 9999;
            display: none;
            flex-direction: column;
            max-height: 280px;
            animation: dropdownFadeIn 0.15s ease-out;
        }
        @keyframes dropdownFadeIn {
            from { opacity: 0; transform: translateY(-4px); }
            to { opacity: 1; transform: translateY(0); }
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
            padding: 8px 12px;
            border: 1px solid var(--border);
            border-radius: var(--radius-pill);
            font-size: 12px;
            font-family: 'Mulish', sans-serif;
            background: #ffffff;
            outline: none;
            min-height: 36px;
        }
        .select-search-input:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 2px rgba(2, 132, 199, 0.2);
        }
        .select-options-list {
            overflow-y: auto;
            max-height: 220px;
            padding: 4px 0;
            list-style: none;
            margin: 0;
        }
        .select-option {
            padding: 10px 12px;
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
        .select-option:hover {
            background: #f0f9ff;
            color: var(--primary-dark);
        }
        .select-option.selected {
            background: #e0f2fe;
            color: var(--primary-dark);
            font-weight: 700;
        }
        .select-option.selected::after {
            content: "✓";
            color: var(--primary-dark);
            font-weight: 800;
            font-size: 11px;
        }
        .select-no-results {
            padding: 14px 12px;
            text-align: center;
            font-size: 12px;
            color: var(--text-muted);
            font-family: 'Mulish', sans-serif;
        }

        /* Dimensions Subform */
        .weight-mode-toggle { display: flex; gap: 8px; margin-bottom: 12px; background: #f1f5f9; padding: 4px; border-radius: var(--radius-pill); min-height: 44px; }
        .mode-btn { flex: 1; border: none; background: transparent; padding: 10px 14px; border-radius: var(--radius-pill); font-size: 12px; font-weight: 700; font-family: 'Outfit', sans-serif; color: var(--text-muted); cursor: pointer; transition: all 0.2s; text-align: center; min-height: 36px; display: inline-flex; align-items: center; justify-content: center; gap: 8px; }
        .mode-btn.active { background: #ffffff; color: var(--text-dark); box-shadow: 0 2px 6px rgba(0,0,0,0.08); }
        .mode-btn:hover:not(.active) { color: var(--text-dark); }
        .dim-box { background: #f8fafc; border: 1px solid var(--border); border-radius: var(--radius-control); padding: 12px; margin-top: 4px; }
        .dim-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }
        .dim-grid label { font-size: 10px; color: var(--text-muted); }
        .dim-info { display: flex; justify-content: space-between; flex-wrap: wrap; gap: 6px; font-size: 11px; font-weight: 700; margin-top: 8px; color: var(--primary-dark); }


        /* Submit Button */
        .submit-btn { width: 100%; background: var(--primary); color: white; border: none; padding: 14px; border-radius: var(--radius-pill); font-size: 15px; font-weight: 700; cursor: pointer; transition: background 0.2s; margin-top: 14px; box-shadow: 0 4px 14px rgba(2, 132, 199, 0.35); min-height: 48px; }
        .submit-btn:hover { background: var(--primary-dark); }
        .submit-btn:disabled { opacity: 0.6; cursor: not-allowed; }

        /* Alert Box */
        .alert-error { background: #fef2f2; border: 1px solid #fecaca; color: #991b1b; padding: 12px 16px; border-radius: 12px; font-size: 13px; margin-bottom: 16px; display: none; }
        .alert-notice { background: #fffbeb; border: 1px solid #fde68a; color: #92400e; padding: 12px 16px; border-radius: 12px; font-size: 13px; margin-bottom: 16px; display: none; }

        /* Results Display */
        .results-panel { display: none; }
        .hero-metric { background: linear-gradient(135deg, #0284c7, #0369a1); color: white; border-radius: var(--radius-card); padding: 22px; text-align: center; margin-bottom: 18px; box-shadow: 0 6px 20px rgba(2, 132, 199, 0.25); }
        .hero-label { font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.5px; opacity: 0.92; }
        .hero-value { font-size: 36px; font-weight: 800; margin: 4px 0; }
        .hero-sub { font-size: 13.5px; opacity: 0.96; }

        .metrics-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 18px; }
        @media (max-width: 800px) { .metrics-grid { grid-template-columns: repeat(2, 1fr); } }
        @media (max-width: 480px) { .metrics-grid { grid-template-columns: 1fr; } }
        .metric-card { background: #f8fafc; border: 1px solid var(--border); border-radius: var(--radius-control); padding: 12px; text-align: center; }
        .metric-title { font-size: 10.5px; color: var(--text-muted); font-weight: 700; text-transform: uppercase; }
        .metric-number { font-size: 18px; font-weight: 700; color: var(--text-dark); margin-top: 3px; font-family: 'Outfit', sans-serif; }

        /* Strategic Recommendation Options (Spacious Horizontal Layout) */
        .strategic-tiers-container {
            display: flex;
            flex-direction: column;
            gap: 10px;
            margin-bottom: 22px;
        }
        .strategy-card {
            background: #ffffff;
            border: 1.5px solid var(--border);
            border-radius: var(--radius-control);
            padding: 12px 16px;
            cursor: pointer;
            transition: all 0.2s ease;
            position: relative;
            outline: none;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            min-height: 60px;
            text-align: left;
        }
        .strategy-card:hover {
            border-color: #94a3b8;
            background: #f8fafc;
        }
        .strategy-card:focus-visible {
            box-shadow: 0 0 0 3px rgba(2, 132, 199, 0.35);
        }
        .strategy-card.selected {
            border-width: 2px;
        }
        .strategy-card.card-floor.selected {
            border-color: #0284c7;
            background: #f0f9ff;
            box-shadow: 0 2px 8px rgba(2, 132, 199, 0.12);
        }
        .strategy-card.card-balanced.selected {
            border-color: #16a34a;
            background: #f0fdf4;
            box-shadow: 0 2px 8px rgba(22, 163, 74, 0.12);
        }
        .strategy-card.card-premium.selected {
            border-color: #7c3aed;
            background: #faf5ff;
            box-shadow: 0 2px 8px rgba(124, 58, 237, 0.12);
        }
        .strat-card-left {
            display: flex;
            flex-direction: column;
            gap: 4px;
            flex: 1;
            min-width: 0;
        }
        .strat-card-title-row {
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
        }
        .strat-radio-indicator {
            width: 16px;
            height: 16px;
            border-radius: 50%;
            border: 2px solid #cbd5e1;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
            transition: all 0.15s ease;
        }
        .strategy-card.selected .strat-radio-indicator {
            border-color: currentColor;
        }
        .strategy-card.selected .strat-radio-indicator::after {
            content: '';
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: currentColor;
        }
        .strategy-card.card-floor.selected .strat-radio-indicator { color: #0284c7; }
        .strategy-card.card-balanced.selected .strat-radio-indicator { color: #16a34a; }
        .strategy-card.card-premium.selected .strat-radio-indicator { color: #7c3aed; }

        .strat-name {
            font-size: 13.5px;
            font-weight: 700;
            color: var(--text-dark);
            font-family: 'Outfit', sans-serif;
            letter-spacing: -0.2px;
        }
        .strat-meta-chips {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 12px;
            color: var(--text-muted);
            font-family: 'Mulish', sans-serif;
            flex-wrap: wrap;
        }
        .strat-meta-val {
            font-weight: 700;
            color: var(--text-dark);
        }
        .strat-card-right {
            text-align: right;
            flex-shrink: 0;
        }
        .strat-price {
            font-size: 20px;
            font-weight: 800;
            font-family: 'Outfit', sans-serif;
            color: var(--text-dark);
            line-height: 1.15;
        }
        .strat-price-unit {
            font-size: 11px;
            font-weight: 600;
            color: var(--text-muted);
            margin-top: 2px;
        }

        /* Badges for Strategic Tiers */
        .badge { padding: 3px 9px; border-radius: var(--radius-pill); font-size: 10px; font-weight: 700; display: inline-block; }
        .badge-optimal { background: #e7f8cb; color: #436e0d; }
        .badge-floor { background: #e0f2fe; color: #0369a1; }
        .badge-balanced { background: #dcfce7; color: #15803d; }
        .badge-premium { background: #ede9fe; color: #6d28d9; }

        /* Sensitivity Table with User-Friendly Horizontal & Vertical Scrolling */
        .table-scroll-hint {
            font-size: 11px;
            color: var(--primary-dark);
            background: #f0f9ff;
            border: 1px solid #bae6fd;
            padding: 4px 10px;
            border-radius: var(--radius-pill);
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        .table-wrap {
            width: 100%;
            max-width: 100%;
            overflow-x: auto;
            overflow-y: auto;
            max-height: 360px;
            border-radius: var(--radius-control);
            border: 1.5px solid var(--border);
            background: #ffffff;
            box-sizing: border-box;
            -webkit-overflow-scrolling: touch;
            scrollbar-width: thin;
            scrollbar-color: #0284c7 #e2e8f0;
        }
        .table-wrap::-webkit-scrollbar {
            height: 10px;
            width: 8px;
        }
        .table-wrap::-webkit-scrollbar-track {
            background: #e2e8f0;
            border-radius: 6px;
        }
        .table-wrap::-webkit-scrollbar-thumb {
            background: #0284c7;
            border-radius: 6px;
            border: 2px solid #e2e8f0;
        }
        .table-wrap::-webkit-scrollbar-thumb:hover {
            background: #0369a1;
        }
        table {
            width: 100%;
            min-width: 860px;
            border-collapse: separate;
            border-spacing: 0;
            font-size: 12px;
        }
        th, td {
            padding: 10px 14px;
            text-align: left;
            border-bottom: 1px solid var(--border);
            white-space: nowrap;
        }
        th {
            background: #f8fafc;
            font-weight: 700;
            color: var(--text-muted);
            position: sticky;
            top: 0;
            z-index: 2;
            box-shadow: 0 1px 0 var(--border);
        }
        tbody tr {
            cursor: pointer;
            transition: background 0.15s ease;
        }
        tbody tr:hover {
            background: #f8fafc;
        }
        .optimal-row { background: #f2fbf4; font-weight: 700; color: #15803d; }
        .row-tier-floor { background: #f0f9ff; }
        .row-tier-balanced { background: #f2fbf4; font-weight: 700; }
        .row-tier-premium { background: #faf5ff; }
        .row-tier-selected { outline: 2px solid var(--primary); background: #e0f2fe !important; }

        /* Placeholder */
        .placeholder-state { text-align: center; padding: 48px 20px; color: var(--text-muted); }
        .placeholder-icon { font-size: 38px; margin-bottom: 10px; opacity: 0.7; }

        /* Navigation & Action Buttons */
        .nav-history-btn { text-decoration: none; background: #ffffff; color: var(--text-dark); border: 1.5px solid var(--border); padding: 7px 16px; border-radius: var(--radius-pill); font-size: 13px; font-weight: 700; display: inline-flex; align-items: center; gap: 7px; box-shadow: 0 1px 4px rgba(0,0,0,0.05); transition: all 0.2s; }
        .nav-history-btn:hover { background: var(--primary); color: white; border-color: var(--primary); }
        .btn-generate-new { background: #ffffff; color: var(--primary-dark); border: 1.5px solid var(--primary); padding: 5px 13px; border-radius: var(--radius-pill); font-size: 12px; font-weight: 700; cursor: pointer; transition: all 0.2s; display: inline-flex; align-items: center; gap: 5px; }
        .btn-generate-new:hover { background: var(--primary); color: white; }
        .btn-generate-new-bottom { width: 100%; background: #f1f5f9; color: #334155; border: 1px solid var(--border); padding: 12px; border-radius: var(--radius-pill); font-size: 14px; font-weight: 700; cursor: pointer; transition: all 0.2s; margin-top: 14px; font-family: 'Outfit', sans-serif; }
        .btn-generate-new-bottom:hover { background: #e2e8f0; color: #0f172a; }

        /* Sleek Contextual Tooltips */
        .tip-target { position: relative; display: inline-flex; align-items: center; gap: 4px; }
        .info-dot {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 14px;
            height: 14px;
            border-radius: 50%;
            background: #e2e8f0;
            color: #64748b;
            font-size: 9.5px;
            font-weight: 700;
            font-family: 'Outfit', sans-serif;
            cursor: pointer;
            line-height: 1;
            transition: all 0.2s;
            flex-shrink: 0;
            user-select: none;
        }
        .tip-target:hover .info-dot {
            background: var(--primary);
            color: white;
        }
        .tip-bubble {
            visibility: hidden;
            opacity: 0;
            position: absolute;
            bottom: 130%;
            left: 50%;
            transform: translateX(-50%);
            background: #0f172a;
            color: #f8fafc;
            padding: 8px 11px;
            border-radius: 8px;
            font-size: 11px;
            font-weight: 500;
            font-family: 'Mulish', sans-serif;
            white-space: normal;
            width: 210px;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.2);
            z-index: 1000;
            pointer-events: none;
            transition: opacity 0.2s, visibility 0.2s;
            line-height: 1.35;
            text-transform: none;
            letter-spacing: normal;
        }
        .tip-bubble::after {
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            margin-left: -5px;
            border-width: 5px;
            border-style: solid;
            border-color: #0f172a transparent transparent transparent;
        }
        .tip-target:hover .tip-bubble {
            visibility: visible;
            opacity: 1;
        }
        th .tip-bubble {
            bottom: auto;
            top: 130%;
        }
        th .tip-bubble::after {
            top: auto;
            bottom: 100%;
            border-color: transparent transparent #0f172a transparent;
        }
    </style>
</head>
<body>
<div class="container">
    <header>
        <div>
            <h1>Air Freight Margin Recommender</h1>
            <p class="subtitle">Two-Stage Market Benchmark & Calibrated Price Elasticity Engine</p>
        </div>
        <div style="display: flex; gap: 12px; align-items: center;">
            <a href="/history" class="nav-history-btn">
                <svg class="ui-icon" viewBox="0 0 24 24"><path d="M9 5H7a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2h-2"/><rect x="9" y="3" width="6" height="4" rx="2"/><path d="M9 12h6"/><path d="M9 16h4"/></svg>
                Recommendation History
            </a>
            <span class="badge badge-optimal" style="font-size: 12px; padding: 6px 15px;">Prototype Engine {{ model_version }}</span>
        </div>
    </header>

    <!-- Active Commercial Guardrails Banner (§5.7) -->
    <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: var(--radius-control); padding: 9px 16px; margin-bottom: 20px; font-size: 12px; color: #166534; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
        <div style="display: flex; align-items: center; gap: 8px; font-weight: 700;">
            <span style="display:inline-block; width:9px; height:9px; border-radius:50%; background:#22c55e;"></span>
            <span>Active Commercial Guardrails:</span>
        </div>
        <div style="display: flex; gap: 14px; font-weight: 600; flex-wrap: wrap;">
            <span>🛡️ IATA Min Weight Floor (0.5kg ceil)</span>
            <span>⚖️ Weight-Break Arbitrage Shield</span>
            <span>💰 Cost Floor (₹1,500/AWB min)</span>
            <span>📅 Seasonal Window Protection</span>
        </div>
    </div>

    <!-- Error Alert Box -->
    <div id="errorAlert" class="alert-error"></div>
    <div id="noticeAlert" class="alert-notice"></div>

    <!-- Quick Scenario Presets -->
    <div class="preset-bar">
        <span class="preset-title">Quick Historical Test Presets</span>
        <div class="preset-grid">
            <button type="button" class="preset-btn" onclick="applyPreset('pharma_uk')">1. Pharma Urgent (DEL → LHR)</button>
            <button type="button" class="preset-btn" onclick="applyPreset('perishable_uae')">2. Perishables (BOM → DXB)</button>
            <button type="button" class="preset-btn" onclick="applyPreset('heavy_subagent')">3. Heavy Machinery (BOM → FRA)</button>
            <button type="button" class="preset-btn" onclick="applyPreset('courier_us')">4. Express Courier (BLR → JFK)</button>
        </div>
    </div>

    <div class="layout">
        <!-- Input Form -->
        <div class="card card-input">
            <div class="card-header">
                <span>Shipment & Commercial Parameters</span>
            </div>

            <form id="quoteForm" onsubmit="event.preventDefault(); runInference();">
                <div class="field-section">
                    <div class="section-label">Origin & Destination Route</div>
                    <div class="form-grid">
                        <div class="form-group">
                            <label>Origin Airport (POL)</label>
                            <input type="hidden" id="origin" value="Indira Gandhi International Airport">
                            <div class="searchable-select" id="searchable_origin">
                                <div class="select-trigger" onclick="toggleDropdown('origin')">
                                    <span class="select-trigger-text" id="origin_display_text">Delhi (DEL) - Indira Gandhi</span>
                                    <span class="select-trigger-arrow">▼</span>
                                </div>
                                <div class="select-dropdown" id="origin_dropdown">
                                    <div class="select-search-wrap">
                                        <input type="text" class="select-search-input" id="origin_search" placeholder="Search origin airport..." oninput="filterDropdownOptions('origin', this.value)">
                                    </div>
                                    <ul class="select-options-list" id="origin_options_list">
                                        {% for opt in origin_options %}
                                        <li class="select-option{% if opt.value == 'Indira Gandhi International Airport' %} selected{% endif %}" data-value="{{ opt.value }}">{{ opt.label }}</li>
                                        {% endfor %}
                                    </ul>
                                </div>
                            </div>
                        </div>

                        <div class="form-group">
                            <label>Destination Airport (POD)</label>
                            <input type="hidden" id="dest" value="Heathrow Apt/London">
                            <div class="searchable-select" id="searchable_dest">
                                <div class="select-trigger" onclick="toggleDropdown('dest')">
                                    <span class="select-trigger-text" id="dest_display_text">London Heathrow (LHR)</span>
                                    <span class="select-trigger-arrow">▼</span>
                                </div>
                                <div class="select-dropdown" id="dest_dropdown">
                                    <div class="select-search-wrap">
                                        <input type="text" class="select-search-input" id="dest_search" placeholder="Search destination airport (580+ ports)..." oninput="filterDropdownOptions('dest', this.value)">
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

                <div class="field-section">
                    <div class="section-label">Airline Cost & Cargo Dimensions</div>
                    
                    <div class="weight-mode-toggle">
                        <button type="button" id="mode_btn_dims" class="mode-btn active" onclick="setWeightMode('dims')">
                            <svg class="ui-icon" viewBox="0 0 24 24"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/></svg>
                            <span>Calculate from Box Dimensions</span>
                        </button>
                        <button type="button" id="mode_btn_direct" class="mode-btn" onclick="setWeightMode('direct')">
                            <svg class="ui-icon" viewBox="0 0 24 24"><path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="M7 21h10"/><path d="M12 3v18"/><path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/></svg>
                            <span>Direct Chargeable Weight</span>
                        </button>
                    </div>

                    <div class="form-grid">
                        <div class="form-group">
                            <label for="buy">Airline Buy Cost (₹ INR)</label>
                            <input type="number" id="buy" value="125000" min="100" step="any" placeholder="e.g. 125000" required>
                        </div>

                        <div class="form-group">
                            <label for="gross_wt">Physical Gross Weight (kg)</label>
                            <input type="number" id="gross_wt" value="450" min="0.1" step="any" oninput="updateDimensions()" required>
                        </div>

                        <!-- Direct Chargeable Weight Mode Container -->
                        <div class="form-group full" id="direct_weight_container" style="display: none;">
                            <label for="direct_ch_wt">Airline Chargeable Weight (kg)</label>
                            <input type="number" id="direct_ch_wt" value="450" min="0.1" step="any" oninput="updateDimensions()" placeholder="e.g. 371.5">
                            <div class="dim-info" style="margin-top: 6px;">
                                <span id="direct_density_display">Density Ratio: 1.0000 (Dense Commercial Cargo)</span>
                            </div>
                        </div>

                        <!-- Box Dimensions Mode Container -->
                        <div class="form-group full" id="dims_weight_container">
                            <label>Box Dimensions (L × W × H in cm × Number of Pieces)</label>
                            <div class="dim-box">
                                <div class="dim-grid">
                                    <div><label>Length (cm)</label><input type="number" id="dim_l" value="120" min="1" step="any" oninput="updateDimensions()" required></div>
                                    <div><label>Width (cm)</label><input type="number" id="dim_w" value="80" min="1" step="any" oninput="updateDimensions()" required></div>
                                    <div><label>Height (cm)</label><input type="number" id="dim_h" value="90" min="1" step="any" oninput="updateDimensions()" required></div>
                                    <div><label>Pieces</label><input type="number" id="dim_pcs" value="3" min="1" step="1" oninput="updateDimensions()" required></div>
                                </div>
                                <div class="dim-info">
                                    <span id="vol_wt_display">Volumetric: 432.0 kg</span>
                                    <span id="ch_wt_display">Chargeable: 450.0 kg</span>
                                    <span id="density_display">Type: Dense Commercial Cargo</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="field-section">
                    <div class="section-label">Commercial Profile & Terms</div>
                    <div class="form-grid">
                        <div class="form-group">
                            <label>Client Category</label>
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

                        <div class="form-group">
                            <label>Business Vertical</label>
                            <input type="hidden" id="vertical" value="Air Export Forwarding">
                            <div class="searchable-select" id="custom_vertical">
                                <div class="select-trigger" onclick="toggleDropdown('vertical')">
                                    <span class="select-trigger-text" id="vertical_display_text">Air Export Forwarding</span>
                                    <span class="select-trigger-arrow">▼</span>
                                </div>
                                <div class="select-dropdown" id="vertical_dropdown">
                                    <ul class="select-options-list" id="vertical_options_list">
                                        {% for opt in vertical_options %}
                                        <li class="select-option{% if opt.value == 'Air Export Forwarding' %} selected{% endif %}" data-value="{{ opt.value }}">{{ opt.label }}</li>
                                        {% endfor %}
                                    </ul>
                                </div>
                            </div>
                        </div>

                        <div class="form-group">
                            <label>Commodity Group</label>
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

                        <div class="form-group">
                            <label>Incoterms</label>
                            <input type="hidden" id="incoterms" value="FOB">
                            <div class="searchable-select" id="custom_incoterms">
                                <div class="select-trigger" onclick="toggleDropdown('incoterms')">
                                    <span class="select-trigger-text" id="incoterms_display_text">FOB - Free on Board</span>
                                    <span class="select-trigger-arrow">▼</span>
                                </div>
                                <div class="select-dropdown" id="incoterms_dropdown">
                                    <ul class="select-options-list" id="incoterms_options_list">
                                        {% for opt in incoterms_options %}
                                        <li class="select-option{% if opt.value == 'FOB' %} selected{% endif %}" data-value="{{ opt.value }}">{{ opt.label }}</li>
                                        {% endfor %}
                                    </ul>
                                </div>
                            </div>
                        </div>

                        <div class="form-group">
                            <label for="frequency">Account Inquiries / Yr</label>
                            <input type="number" id="frequency" value="10" min="1" max="500" step="1">
                        </div>

                        <div class="form-group">
                            <label>Booking Branch Office</label>
                            <input type="hidden" id="branch" value="Mumbai">
                            <div class="searchable-select" id="custom_branch">
                                <div class="select-trigger" onclick="toggleDropdown('branch')">
                                    <span class="select-trigger-text" id="branch_display_text">Mumbai</span>
                                    <span class="select-trigger-arrow">▼</span>
                                </div>
                                <div class="select-dropdown" id="branch_dropdown">
                                    <ul class="select-options-list" id="branch_options_list">
                                        {% for opt in branch_options %}
                                        <li class="select-option{% if opt.value == 'Mumbai' %} selected{% endif %}" data-value="{{ opt.value }}">{{ opt.label }}</li>
                                        {% endfor %}
                                    </ul>
                                </div>
                            </div>
                        </div>

                        <div class="form-group full">
                            <label for="revision">Quote Revision Count</label>
                            <input type="number" id="revision" value="1" min="1" max="20" step="1">
                        </div>
                    </div>
                </div>


                <button type="submit" id="submitBtn" class="submit-btn">
                    <svg class="ui-icon" style="width: 17px; height: 17px;" viewBox="0 0 24 24"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
                    Generate Margin Recommendation
                </button>
            </form>
        </div>

        <!-- Output Display Card -->
        <div class="card card-output">
            <div class="card-header">
                <span>Recommended Commercial Terms</span>
                <button type="button" class="btn-generate-new" onclick="resetFormForNewQuote()">
                    <svg class="ui-icon" viewBox="0 0 24 24"><path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/><path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16"/><path d="M16 16h5v5"/></svg>
                    Generate New
                </button>
            </div>

            <!-- Initial Placeholder -->
            <div id="placeholder" class="placeholder-state">
                <div class="placeholder-icon">
                    <svg class="ui-icon-lg" viewBox="0 0 24 24"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>
                </div>
                <h3 style="font-size: 16px; font-weight: 700;">Ready for Pricing Inquiry</h3>
                <p style="font-size: 12.5px; margin-top: 5px;">Enter route details, airline cost, and cargo specifications on the left, then click Generate Margin Recommendation.</p>
            </div>

            <!-- Results Panel -->
            <div id="results" class="results-panel">
                <div class="hero-metric">
                    <div class="hero-label">Recommended Quoted Selling Price</div>
                    <div class="hero-value" id="res_selling_price">₹0</div>
                    <div class="hero-sub" id="res_hero_sub">Target Margin: 0.0% (₹0 Gross Profit if Won)</div>
                </div>

                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-title">
                            <span class="tip-target">
                                Recommended Margin %
                                <span class="info-dot">i</span>
                                <span class="tip-bubble">Optimal markup percentage calculated for this shipment corridor and profile.</span>
                            </span>
                        </div>
                        <div class="metric-number" id="res_margin_pct" style="color: var(--primary-dark);">0.0%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">
                            <span class="tip-target">
                                Historical Percentile
                                <span class="info-dot">i</span>
                                <span class="tip-bubble">Empirical share of historically won deals in this weight slab at or below this margin.</span>
                            </span>
                        </div>
                        <div class="metric-number" id="res_win_prob" style="color: #16a34a;">P50</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">
                            <span class="tip-target">
                                Gross Profit (if Won)
                                <span class="info-dot">i</span>
                                <span class="tip-bubble">Actual gross margin in Rupees pocketed if the deal is won (Sell Rate minus Airline Buy Cost).</span>
                            </span>
                        </div>
                        <div class="metric-number" id="res_profit_amt" style="color: #16a34a;">₹0</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">
                            <span class="tip-target">
                                Max Margin Cap
                                <span class="info-dot">i</span>
                                <span class="tip-bubble">Governing commercial maximum margin ceiling for this cargo category & commodity type.</span>
                            </span>
                        </div>
                        <div class="metric-number" id="res_max_cap" style="color: #6366f1;">0.0%</div>
                    </div>
                </div>


                <div class="section-label" style="margin-top: 18px; margin-bottom: 10px;">
                    <span class="tip-target">
                        Top Strategic Recommendations
                        <span class="info-dot">i</span>
                        <span class="tip-bubble">Choose a commercial pricing tier. Quoted terms and table focus update instantly.</span>
                    </span>
                </div>
                <div id="strategic_tiers" class="strategic-tiers-container"></div>

                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 18px; margin-bottom: 8px;">
                    <div class="section-label" style="margin-bottom: 0;">Market Sensitivity Frontier</div>
                    <span class="table-scroll-hint">
                        <svg class="ui-icon" style="color: var(--primary);" viewBox="0 0 24 24"><path d="m18 8 4 4-4 4"/><path d="M2 12h20"/><path d="m6 8-4 4 4 4"/></svg>
                        Scroll horizontally to view full frontier
                    </span>
                </div>
                <div class="table-wrap">
                    <table>
                        <thead>
                            <tr>
                                <th>
                                    <span class="tip-target">
                                        Markup %
                                        <span class="info-dot">i</span>
                                        <span class="tip-bubble">Markup percentage applied over airline buy cost: (Sell - Buy) / Buy.</span>
                                    </span>
                                </th>
                                <th>
                                    <span class="tip-target">
                                        Gross Margin %
                                        <span class="info-dot">i</span>
                                        <span class="tip-bubble">Gross profit percentage on quoted sell price: (Sell - Buy) / Sell.</span>
                                    </span>
                                </th>
                                <th>
                                    <span class="tip-target">
                                        Selling Price
                                        <span class="info-dot">i</span>
                                        <span class="tip-bubble">Total rate quoted to customer (Airline Buy + Margin).</span>
                                    </span>
                                </th>
                                <th>
                                    <span class="tip-target">
                                        Rate / Kg
                                        <span class="info-dot">i</span>
                                        <span class="tip-bubble">Effective quoted selling price per chargeable kilogram.</span>
                                    </span>
                                </th>
                                <th>
                                    <span class="tip-target">
                                        Gross Profit (if Won)
                                        <span class="info-dot">i</span>
                                        <span class="tip-bubble">Total gross margin in Rupees pocketed if booking is won.</span>
                                    </span>
                                </th>
                                <th>
                                    <span class="tip-target">
                                        Profit / Kg
                                        <span class="info-dot">i</span>
                                        <span class="tip-bubble">Gross margin pocketed per chargeable kilogram.</span>
                                    </span>
                                </th>
                                <th>
                                    <span class="tip-target">
                                        Won-Deal Percentile
                                        <span class="info-dot">i</span>
                                        <span class="tip-bubble">Empirical share of historically won deals in this weight slab at or below this margin.</span>
                                    </span>
                                </th>
                                <th>
                                    <span class="tip-target">
                                        Margin / Kg
                                        <span class="info-dot">i</span>
                                        <span class="tip-bubble">Empirical gross margin per kilogram.</span>
                                    </span>
                                </th>
                                <th>Recommendation</th>
                            </tr>
                        </thead>
                        <tbody id="sensitivity_body"></tbody>
                    </table>
                </div>

                <button type="button" class="btn-generate-new-bottom" onclick="resetFormForNewQuote()">
                    <svg class="ui-icon" viewBox="0 0 24 24"><path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/><path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16"/><path d="M16 16h5v5"/></svg>
                    Clear Fields & Create New Quote
                </button>
            </div>
        </div>
    </div>
</div>

<script>
    function formatINR(val) {
        const num = Math.round(Number(val) || 0);
        return isFinite(num) ? num.toLocaleString('en-IN') : '0';
    }


    let currentWeightMode = 'dims'; // 'dims' or 'direct'

    function setWeightMode(mode) {
        currentWeightMode = mode;
        const btnDims = document.getElementById('mode_btn_dims');
        const btnDirect = document.getElementById('mode_btn_direct');
        const dimsContainer = document.getElementById('dims_weight_container');
        const directContainer = document.getElementById('direct_weight_container');

        if (mode === 'direct') {
            if (btnDirect) btnDirect.className = 'mode-btn active';
            if (btnDims) btnDims.className = 'mode-btn';
            if (dimsContainer) dimsContainer.style.display = 'none';
            if (directContainer) directContainer.style.display = 'block';

            // Sync current chargeable weight to direct input if empty
            const directInput = document.getElementById('direct_ch_wt');
            if (directInput && (!directInput.value || parseFloat(directInput.value) <= 0)) {
                const gross = parseFloat(document.getElementById('gross_wt')?.value) || 0;
                directInput.value = gross > 0 ? gross : 100;
            }
        } else {
            if (btnDims) btnDims.className = 'mode-btn active';
            if (btnDirect) btnDirect.className = 'mode-btn';
            if (dimsContainer) dimsContainer.style.display = 'block';
            if (directContainer) directContainer.style.display = 'none';
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

        if (volEl) volEl.innerText = `Volumetric: ${vol_wt.toFixed(1)} kg`;
        if (chEl) chEl.innerText = `Chargeable: ${ch_wt.toFixed(1)} kg`;
        if (denEl) denEl.innerText = `Type: ${densityText}`;
        if (directDenEl) directDenEl.innerText = `Density Ratio: ${density.toFixed(4)} (${densityText})`;

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
            frequency: "18",
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
            frequency: "5",
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
            frequency: "8",
            revision: "0",
            branch: "Bangalore"
        }
    };

    // Searchable Select Logic
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

        // Update selected class
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
        let matchCount = 0;

        Array.from(list.children).forEach(li => {
            if (li.classList.contains('select-no-results')) return;
            const text = li.innerText.toLowerCase();
            const val = (li.getAttribute('data-value') || '').toLowerCase();
            if (text.includes(q) || val.includes(q)) {
                li.style.display = 'flex';
                matchCount++;
            } else {
                li.style.display = 'none';
            }
        });

        let noRes = list.querySelector('.select-no-results');
        if (matchCount === 0) {
            if (!noRes) {
                noRes = document.createElement('li');
                noRes.className = 'select-no-results';
                noRes.innerText = 'No matching options found';
                list.appendChild(noRes);
            }
            noRes.style.display = 'block';
        } else if (noRes) {
            noRes.style.display = 'none';
        }
    }

    const ALL_DROPDOWN_IDS = ['origin', 'dest', 'client', 'vertical', 'commodity', 'incoterms', 'branch'];

    // Event delegation for all select dropdowns
    ALL_DROPDOWN_IDS.forEach(fieldId => {
        const list = document.getElementById(`${fieldId}_options_list`);
        if (list) {
            list.addEventListener('click', function(e) {
                const li = e.target.closest('li.select-option');
                if (!li || li.classList.contains('select-no-results')) return;
                const value = li.getAttribute('data-value');
                const label = li.innerText.trim();
                selectSearchableOption(fieldId, value, label);
            });
        }
    });

    // Close any dropdown when clicked outside
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
        setCustomDropdownValue('vertical', p.vertical);
        setCustomDropdownValue('commodity', p.commodity);
        setCustomDropdownValue('incoterms', p.incoterms);
        document.getElementById('frequency').value = p.frequency;
        document.getElementById('revision').value = p.revision || '1';
        setCustomDropdownValue('branch', p.branch);
        updateDimensions();
        runInference(true);
    }

    function resetFormForNewQuote() {
        document.getElementById('buy').value = '';
        document.getElementById('gross_wt').value = '';
        const directCh = document.getElementById('direct_ch_wt');
        if (directCh) directCh.value = '';
        document.getElementById('dim_l').value = '';
        document.getElementById('dim_w').value = '';
        document.getElementById('dim_h').value = '';
        document.getElementById('dim_pcs').value = '1';
        document.getElementById('frequency').value = '10';
        document.getElementById('revision').value = '1';

        setSearchableValue('origin', 'Indira Gandhi International Airport');
        setSearchableValue('dest', 'Heathrow Apt/London');
        setCustomDropdownValue('client', 'Shipper / Consignee');
        setCustomDropdownValue('vertical', 'Air Export Forwarding');
        setCustomDropdownValue('commodity', 'General Cargo');
        setCustomDropdownValue('incoterms', 'FOB');
        setCustomDropdownValue('branch', 'Mumbai');

        updateDimensions();

        const results = document.getElementById('results');
        const placeholder = document.getElementById('placeholder');
        if (results) results.style.display = 'none';
        if (placeholder) placeholder.style.display = 'block';
        const stratTiers = document.getElementById('strategic_tiers');
        if (stratTiers) stratTiers.innerHTML = '';
        currentStrategies = {};
        currentSensitivityRows = [];

        const errorAlert = document.getElementById('errorAlert');
        const noticeAlert = document.getElementById('noticeAlert');
        if (errorAlert) { errorAlert.style.display = 'none'; errorAlert.innerText = ''; }
        if (noticeAlert) { noticeAlert.style.display = 'none'; noticeAlert.innerText = ''; }

        const buyEl = document.getElementById('buy');
        if (buyEl) {
            buyEl.focus();
            buyEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    }

    function runInference(isPreset = false) {
        const errorAlert = document.getElementById('errorAlert');
        const noticeAlert = document.getElementById('noticeAlert');
        if (errorAlert) { errorAlert.style.display = 'none'; errorAlert.innerText = ''; }
        if (noticeAlert) { noticeAlert.style.display = 'none'; noticeAlert.innerText = ''; }

        const submitBtn = document.getElementById('submitBtn');
        if (submitBtn) { submitBtn.disabled = true; submitBtn.innerHTML = '<svg class="ui-icon spin" viewBox="0 0 24 24"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg> Calculating Optimal Quote...'; }

        const weights = updateDimensions();
        const payload = {
            origin: document.getElementById('origin')?.value,
            dest: document.getElementById('dest')?.value,
            buy: parseFloat(document.getElementById('buy')?.value),
            gross_wt: weights.gross_wt,
            chargeable_wt: weights.chargeable_wt,
            client: document.getElementById('client')?.value,
            vertical: document.getElementById('vertical')?.value,
            commodity: document.getElementById('commodity')?.value,
            incoterms: document.getElementById('incoterms')?.value,
            frequency: parseInt(document.getElementById('frequency')?.value, 10),
            revision: parseInt(document.getElementById('revision')?.value, 10) || 1,
            branch: document.getElementById('branch')?.value,
            strategy: 'balanced',
            dim_l: parseFloat(document.getElementById('dim_l')?.value),
            dim_w: parseFloat(document.getElementById('dim_w')?.value),
            dim_h: parseFloat(document.getElementById('dim_h')?.value),
            dim_pcs: parseInt(document.getElementById('dim_pcs')?.value, 10),
            quote_date: new Date().toISOString().slice(0, 10),
            is_preset: Boolean(isPreset)
        };

        fetch('/api/quote', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
        .then(res => res.json().then(data => ({ status: res.status, body: data })))
        .then(({ status, body }) => {
            if (submitBtn) { submitBtn.disabled = false; submitBtn.innerHTML = '<svg class="ui-icon" style="width: 17px; height: 17px;" viewBox="0 0 24 24"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg> Generate Margin Recommendation'; }

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

            renderStrategicCards(currentStrategies, 'balanced', currentChargeableWeight);
            renderSensitivityTable(currentSensitivityRows, opt.Margin_Percentage);
            updateHeroMetrics(opt);

            if (body.warnings && body.warnings.length > 0 && noticeAlert) {
                noticeAlert.style.display = 'block';
                noticeAlert.innerText = body.warnings.join('  |  ');
            } else if (body.history_logged === false && noticeAlert) {
                noticeAlert.style.display = 'block';
                noticeAlert.innerText = 'Notice: Quote generated successfully, but audit history logging was offline.';
            }
        })
        .catch(err => {
            if (submitBtn) { submitBtn.disabled = false; submitBtn.innerHTML = '<svg class="ui-icon" style="width: 17px; height: 17px;" viewBox="0 0 24 24"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg> Generate Margin Recommendation'; }
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

    function updateHeroMetrics(strat) {
        if (!strat) return;
        const grossMarginPct = strat.Gross_Margin_Percentage !== undefined ? strat.Gross_Margin_Percentage : ((strat.Margin_Amount_INR / Math.max(1, strat.Quoted_Sell_Price_INR)) * 100);
        document.getElementById('res_selling_price').innerText = `₹${formatINR(strat.Quoted_Sell_Price_INR)}`;
        document.getElementById('res_hero_sub').innerText = `Markup on Buy: ${strat.Margin_Percentage.toFixed(1)}% | Gross Margin on Sell: ${grossMarginPct.toFixed(1)}% (₹${formatINR(strat.Margin_Amount_INR)} Gross Profit if Won)`;
        document.getElementById('res_margin_pct').innerText = `${strat.Margin_Percentage.toFixed(1)}% (GM: ${grossMarginPct.toFixed(1)}%)`;
        if (strat.History_Percentile !== undefined && strat.History_Percentile !== null) {
            document.getElementById('res_win_prob').innerText = `P${strat.History_Percentile.toFixed(0)}`;
        } else if (strat.Win_Probability !== undefined && strat.Win_Probability !== null) {
            document.getElementById('res_win_prob').innerText = `${(strat.Win_Probability * 100).toFixed(1)}%`;
        } else {
            document.getElementById('res_win_prob').innerText = '--';
        }
        document.getElementById('res_profit_amt').innerText = `₹${formatINR(strat.Margin_Amount_INR)}`;
        const expEl = document.getElementById('res_exp_val');
        if (expEl) {
            if (strat.Expected_Profit_INR !== undefined && strat.Expected_Profit_INR !== null) {
                expEl.innerText = `₹${formatINR(strat.Expected_Profit_INR)}`;
            } else if (strat.Margin_Per_Kg !== undefined) {
                expEl.innerText = `₹${strat.Margin_Per_Kg}/kg`;
            }
        }
        const capEl = document.getElementById('res_max_cap');
        if (capEl && strat.Effective_Max_Margin_Cap !== undefined) {
            capEl.innerText = `${strat.Effective_Max_Margin_Cap.toFixed(1)}%`;
            if (strat.Cap_Limiting_Factor) {
                capEl.title = `Limited by: ${strat.Cap_Limiting_Factor}`;
            }
        }
    }

    function renderStrategicCards(strategies, activeKey, chWt) {
        currentStrategies = strategies || {};
        selectedStrategyKey = activeKey || 'balanced';
        currentChargeableWeight = chWt || 100;

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
            card.setAttribute('aria-pressed', isSelected ? 'true' : 'false');
            card.setAttribute('aria-label', `${strat.Strategy_Name} - Quoted Selling Price ₹${formatINR(strat.Quoted_Sell_Price_INR)}`);

            const perKgPrice = Math.round(strat.Quoted_Sell_Price_INR / Math.max(1, currentChargeableWeight));
            const grossMarginPct = strat.Gross_Margin_Percentage !== undefined ? strat.Gross_Margin_Percentage : ((strat.Margin_Amount_INR / Math.max(1, strat.Quoted_Sell_Price_INR)) * 100);
            const capChip = strat.Effective_Max_Margin_Cap !== undefined ? `<span>•</span><span>Cap: <strong class="strat-meta-val" title="${strat.Cap_Limiting_Factor || ''}">${strat.Effective_Max_Margin_Cap.toFixed(1)}%</strong></span>` : '';
            const clampedBadge = strat.Is_Clamped_By_Cap ? `<span class="badge" style="background: #fef3c7; color: #92400e; font-size: 10px; margin-left: 6px;">Ceiling Reached</span>` : '';
            const winChip = (strat.History_Percentile !== undefined && strat.History_Percentile !== null)
                ? `<span>•</span><span>Hist. percentile: <strong class="strat-meta-val">P${strat.History_Percentile.toFixed(0)}</strong></span>`
                : (strat.Win_Probability !== null && strat.Win_Probability !== undefined
                    ? `<span>•</span><span>Win Prob: <strong class="strat-meta-val">${(strat.Win_Probability * 100).toFixed(1)}%</strong></span>`
                    : '');

            card.innerHTML = `
                <div class="strat-card-left">
                    <div class="strat-card-title-row">
                        <span class="strat-radio-indicator"></span>
                        <span class="strat-name">${strat.Strategy_Name}</span>
                        <span class="badge badge-${key}">${strat.Badge_Label}</span>
                        ${clampedBadge}
                    </div>
                    <div class="strat-meta-chips">
                        <span>Markup: <strong class="strat-meta-val">${strat.Margin_Percentage.toFixed(1)}%</strong></span>
                        <span>•</span>
                        <span>Gross Margin: <strong class="strat-meta-val" style="color: #0369a1;">${grossMarginPct.toFixed(1)}%</strong></span>
                        ${winChip}
                        <span>•</span>
                        <span>Gross Profit: <strong class="strat-meta-val">₹${formatINR(strat.Margin_Amount_INR)}</strong></span>
                        ${capChip}
                    </div>
                </div>
                <div class="strat-card-right">
                    <div class="strat-price">₹${formatINR(strat.Quoted_Sell_Price_INR)}</div>
                    <div class="strat-price-unit">Sell: ₹${formatINR(perKgPrice)} / kg</div>
                </div>
            `;


            card.onclick = function() { selectStrategyCard(key); };
            card.onkeydown = function(e) {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    selectStrategyCard(key);
                }
            };

            container.appendChild(card);
        });
    }

    function selectStrategyCard(key) {
        if (!currentStrategies[key]) return;
        selectedStrategyKey = key;
        const strat = currentStrategies[key];

        // Update cards selection state
        ['floor', 'balanced', 'premium'].forEach(k => {
            const el = document.getElementById(`strat_card_${k}`);
            if (el) {
                if (k === key) {
                    el.classList.add('selected');
                    el.setAttribute('aria-pressed', 'true');
                } else {
                    el.classList.remove('selected');
                    el.setAttribute('aria-pressed', 'false');
                }
            }
        });

        // Update Hero Metric and Top Summary Cards
        updateHeroMetrics(strat);

        // Highlight matching row in Market Sensitivity Frontier
        renderSensitivityTable(currentSensitivityRows, strat.Margin_Percentage);
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
            const isFloor = Math.abs(rowMargin - floorM) < 0.15;
            const isBalanced = Math.abs(rowMargin - balancedM) < 0.15;
            const isPremium = Math.abs(rowMargin - premiumM) < 0.15;
            const isSelected = selectedMargin !== undefined && Math.abs(rowMargin - selectedMargin) < 0.15;

            let rowClasses = [];
            let badgeHtml = '';

            if (isFloor) {
                rowClasses.push('row-tier-floor');
                badgeHtml = '<span class="badge badge-floor">Must-Win Floor</span>';
            } else if (isBalanced) {
                rowClasses.push('row-tier-balanced');
                badgeHtml = '<span class="badge badge-balanced">Recommended</span>';
            } else if (isPremium) {
                rowClasses.push('row-tier-premium');
                badgeHtml = '<span class="badge badge-premium">High Margin</span>';
            }

            if (isSelected) {
                rowClasses.push('row-tier-selected');
            }

            if (rowClasses.length > 0) {
                tr.className = rowClasses.join(' ');
            }

            const ratePerKg = Math.round(row.Quoted_Sell_Price_INR / Math.max(1, currentChargeableWeight));
            const profitPerKg = (row.Margin_Amount_INR / Math.max(1, currentChargeableWeight)).toFixed(1);
            const percCell = (row.History_Percentile !== undefined && row.History_Percentile !== null)
                ? `P${row.History_Percentile.toFixed(0)}`
                : (row.Win_Probability !== undefined && row.Win_Probability !== null
                    ? `${(row.Win_Probability * 100).toFixed(1)}%`
                    : '--');
            const mpkCell = (row.Margin_Per_Kg !== undefined && row.Margin_Per_Kg !== null)
                ? `₹${row.Margin_Per_Kg}/kg`
                : (row.Expected_Profit_INR !== undefined && row.Expected_Profit_INR !== null
                    ? `₹${formatINR(row.Expected_Profit_INR)}`
                    : `₹${profitPerKg}/kg`);

            tr.innerHTML = `
                <td><strong>${row.Margin_Percentage.toFixed(1)}%</strong></td>
                <td style="color: #0369a1; font-weight: 600;">${grossMarginPct}%</td>
                <td>₹${formatINR(row.Quoted_Sell_Price_INR)}</td>
                <td>₹${formatINR(ratePerKg)}/kg</td>
                <td>₹${formatINR(row.Margin_Amount_INR)}</td>
                <td>₹${profitPerKg}/kg</td>
                <td>${percCell}</td>
                <td>${mpkCell}</td>
                <td>${badgeHtml}</td>
            `;

            tr.onclick = function() {
                updateHeroMetrics(row);
                if (isFloor) selectStrategyCard('floor');
                else if (isBalanced) selectStrategyCard('balanced');
                else if (isPremium) selectStrategyCard('premium');
                else {
                    ['floor', 'balanced', 'premium'].forEach(k => {
                        const el = document.getElementById(`strat_card_${k}`);
                        if (el) { el.classList.remove('selected'); el.setAttribute('aria-pressed', 'false'); }
                    });
                    renderSensitivityTable(currentSensitivityRows, row.Margin_Percentage);
                }
            };

            tbody.appendChild(tr);
        });
    }

    window.onload = function() {
        updateDimensions();
    };
</script>
</body>
</html>
"""


# ------------------------------------------------------------------------------
# HISTORY VIEW TEMPLATE (AUTHENTIC SIGNATURE STYLING RESTORED)
# ------------------------------------------------------------------------------
HISTORY_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Recommendation History & Leads Tracker - Air Freight Margin Recommender</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@500;600;700;800&family=Mulish:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #0284c7;
            --primary-dark: #0369a1;
            --primary-light: #e0f2fe;
            --secondary: #16a34a;
            --secondary-dark: #15803d;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --border: #e2e8f0;
            --text-dark: #0f172a;
            --text-muted: #64748b;
            --radius-card: 16px;
            --radius-control: 10px;
            --radius-pill: 9999px;
            --shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.06);
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Mulish', sans-serif; }
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
                scroll-behavior: auto !important;
            }
        }
        h1, h2, h3, h4, .stat-value, .btn-action { font-family: 'Outfit', sans-serif; }
        body { background-color: var(--bg); color: var(--text-dark); padding: 32px 20px; line-height: 1.5; }
        .container { max-width: 1300px; margin: 0 auto; }
        /* SVG UI Icons */
        .ui-icon {
            display: inline-block;
            width: 15px;
            height: 15px;
            stroke-width: 2.2;
            stroke: currentColor;
            fill: none;
            stroke-linecap: round;
            stroke-linejoin: round;
            vertical-align: -2px;
            flex-shrink: 0;
        }
        .ui-icon-lg {
            display: inline-block;
            width: 44px;
            height: 44px;
            stroke-width: 1.8;
            stroke: var(--primary);
            fill: none;
            stroke-linecap: round;
            stroke-linejoin: round;
            margin-bottom: 12px;
        }

        header { margin-bottom: 24px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px; }
        .back-link { text-decoration: none; color: var(--primary-dark); font-weight: 700; font-size: 13px; display: inline-flex; align-items: center; gap: 6px; }
        .back-link:hover { text-decoration: underline; }
        h1 { font-size: 26px; font-weight: 800; color: var(--text-dark); margin-top: 4px; }
        .subtitle { color: var(--text-muted); font-size: 13px; }
        .header-actions { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
        .btn-action { text-decoration: none; padding: 10px 18px; min-height: 40px; border-radius: var(--radius-pill); font-size: 13px; font-weight: 700; cursor: pointer; border: 1.5px solid var(--border); background: #ffffff; color: var(--text-dark); transition: all 0.2s; display: inline-flex; align-items: center; gap: 6px; }
        .btn-action:hover { background: #f1f5f9; border-color: #cbd5e1; }
        .btn-action.btn-danger { color: #dc2626; border-color: #fca5a5; background: #fff; }
        .btn-action.btn-danger:hover { background: #fef2f2; }
        .btn-action.btn-primary { background: var(--primary); color: #ffffff; border-color: var(--primary); }
        .btn-action.btn-primary:hover { background: var(--primary-dark); }
        
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin-bottom: 24px; }
        .stat-card { background: var(--card-bg); border: 1px solid var(--border); border-radius: var(--radius-card); padding: 18px; box-shadow: var(--shadow); }
        .stat-title { font-size: 11px; font-weight: 700; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; display: flex; align-items: center; gap: 6px; }
        .stat-value { font-size: 24px; font-weight: 800; color: var(--text-dark); margin-top: 6px; font-family: 'Outfit', sans-serif; }
        
        .table-card { background: var(--card-bg); border: 1px solid var(--border); border-radius: var(--radius-card); padding: 22px; box-shadow: var(--shadow); }
        .table-wrap {
            overflow-x: auto;
            max-height: 580px;
            border: 1px solid var(--border);
            border-radius: var(--radius-control);
            scrollbar-width: thin;
            scrollbar-color: #0284c7 #e2e8f0;
            -webkit-overflow-scrolling: touch;
        }
        .table-wrap::-webkit-scrollbar {
            height: 8px;
            width: 8px;
        }
        .table-wrap::-webkit-scrollbar-track {
            background: #e2e8f0;
            border-radius: 4px;
        }
        .table-wrap::-webkit-scrollbar-thumb {
            background: #0284c7;
            border-radius: 4px;
        }
        .table-wrap::-webkit-scrollbar-thumb:hover {
            background: #0369a1;
        }
        table { width: 100%; min-width: 1050px; border-collapse: collapse; font-size: 12px; }
        th, td { padding: 11px 14px; text-align: left; border-bottom: 1px solid var(--border); white-space: nowrap; }
        th { background: #f8fafc; font-weight: 700; color: var(--text-muted); position: sticky; top: 0; z-index: 2; }
        tr:hover { background: #f8fafc; }
        .badge { padding: 3px 9px; border-radius: var(--radius-pill); font-size: 10.5px; font-weight: 700; display: inline-block; }
        .badge-margin { background: #e0f2fe; color: #0369a1; }
        .badge-win { background: #ecfdf5; color: #047857; }
        .status-select {
            padding: 5px 12px;
            border-radius: var(--radius-pill);
            font-size: 11.5px;
            font-weight: 700;
            border: 1.5px solid transparent;
            cursor: pointer;
            outline: none;
            transition: all 0.2s;
        }
        .status-draft { background: #e0f2fe; color: #0369a1; border-color: #bae6fd; }
        .status-negotiating { background: #fef3c7; color: #b45309; border-color: #fde68a; }
        .status-followup { background: #e0e7ff; color: #4338ca; border-color: #c7d2fe; }
        .status-review { background: #f3e8ff; color: #7e22ce; border-color: #e9d5ff; }
        .status-won { background: #dcfce7; color: #15803d; border-color: #86efac; }
        .status-lost { background: #fee2e2; color: #b91c1c; border-color: #fca5a5; }
        .empty-state { text-align: center; padding: 60px 20px; color: var(--text-muted); }
        .empty-icon { font-size: 44px; margin-bottom: 12px; opacity: 0.7; }

        /* Sleek Contextual Tooltips */
        .tip-target { position: relative; display: inline-flex; align-items: center; gap: 4px; }
        .info-dot {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 14px;
            height: 14px;
            border-radius: 50%;
            background: #e2e8f0;
            color: #64748b;
            font-size: 9.5px;
            font-weight: 700;
            font-family: 'Outfit', sans-serif;
            cursor: pointer;
            line-height: 1;
            transition: all 0.2s;
            flex-shrink: 0;
            user-select: none;
        }
        .tip-target:hover .info-dot {
            background: var(--primary);
            color: white;
        }
        .tip-bubble {
            visibility: hidden;
            opacity: 0;
            position: absolute;
            bottom: 130%;
            left: 50%;
            transform: translateX(-50%);
            background: #0f172a;
            color: #f8fafc;
            padding: 8px 11px;
            border-radius: 8px;
            font-size: 11px;
            font-weight: 500;
            font-family: 'Mulish', sans-serif;
            white-space: normal;
            width: 210px;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.2);
            z-index: 1000;
            pointer-events: none;
            transition: opacity 0.2s, visibility 0.2s;
            line-height: 1.35;
            text-transform: none;
            letter-spacing: normal;
        }
        .tip-bubble::after {
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            margin-left: -5px;
            border-width: 5px;
            border-style: solid;
            border-color: #0f172a transparent transparent transparent;
        }
        .tip-target:hover .tip-bubble {
            visibility: visible;
            opacity: 1;
        }
        th .tip-bubble {
            bottom: auto;
            top: 130%;
        }
        th .tip-bubble::after {
            top: auto;
            bottom: 100%;
            border-color: transparent transparent #0f172a transparent;
        }

        /* Pagination Controls (Strict Brand Alignment: Outfit/Mulish, Cyan/Lime, Pill Geometry) */
        .pagination-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 14px;
            margin-top: 18px;
            padding-top: 16px;
            border-top: 1px solid var(--border);
        }
        .pagination-info {
            font-size: 12.5px;
            color: var(--text-muted);
            font-family: 'Mulish', sans-serif;
        }
        .pagination-info strong {
            color: var(--text-dark);
            font-weight: 700;
        }
        .pagination-actions {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .btn-page {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 34px;
            height: 34px;
            padding: 0 12px;
            border-radius: var(--radius-pill);
            border: 1.5px solid var(--border);
            background: #ffffff;
            color: var(--text-dark);
            font-size: 12px;
            font-weight: 700;
            font-family: 'Outfit', sans-serif;
            text-decoration: none;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        .btn-page:hover:not(.disabled):not(.active) {
            background: #f1f5f9;
            border-color: var(--primary);
            color: var(--primary-dark);
        }
        .btn-page.active {
            background: var(--primary);
            border-color: var(--primary);
            color: #ffffff;
            box-shadow: 0 2px 6px rgba(35, 194, 242, 0.35);
            cursor: default;
        }
        .btn-page.disabled {
            opacity: 0.45;
            cursor: not-allowed;
            pointer-events: none;
            background: #f8fafc;
        }
        .per-page-select {
            padding: 6px 12px;
            border-radius: var(--radius-pill);
            border: 1.5px solid var(--border);
            background: #ffffff;
            font-size: 12px;
            font-family: 'Mulish', sans-serif;
            font-weight: 600;
            color: var(--text-dark);
            outline: none;
            cursor: pointer;
            transition: border-color 0.2s;
        }
        .per-page-select:focus {
            border-color: var(--primary);
        }
    </style>
</head>
<body>
<div class="container">
    <header>
        <div>
            <a href="/" class="back-link">
                <svg class="ui-icon" viewBox="0 0 24 24"><line x1="19" y1="12" x2="5" y2="12"/><polyline points="12 19 5 12 12 5"/></svg>
                Back to Margin Recommender
            </a>
            <h1>Recommendation History & Leads Tracker</h1>
            <p class="subtitle">Complete audit trail and lifecycle status tracking for all generated air freight quotations</p>
        </div>
        <div class="header-actions">
            {% if records and records|length > 0 %}
            <a href="/api/history/export" class="btn-action">
                <svg class="ui-icon" viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                Export CSV
            </a>
            <button type="button" class="btn-action btn-danger" onclick="clearHistory()">
                <svg class="ui-icon" viewBox="0 0 24 24"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/><line x1="10" y1="11" x2="10" y2="17"/><line x1="14" y1="11" x2="14" y2="17"/></svg>
                Clear History
            </button>
            {% endif %}
        </div>
    </header>

    {% if records and records|length > 0 %}
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-title">Total Inquiries</div>
            <div class="stat-value">{{ total_count }}</div>
        </div>
        <div class="stat-card">
            <div class="stat-title">
                <svg class="ui-icon" style="color: #16a34a;" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><polyline points="9 12 11 14 15 10"/></svg>
                Won Bookings
            </div>
            <div class="stat-value" style="color: #15803d;">{{ won_count }}</div>
        </div>
        <div class="stat-card">
            <div class="stat-title">Quoted Pipeline Value</div>
            <div class="stat-value" style="color: var(--primary-dark);">₹{{ "{:,.0f}".format(total_pipeline) }}</div>
        </div>
        <div class="stat-card">
            <div class="stat-title">Conversion Rate</div>
            <div class="stat-value" style="color: #0369a1;">{{ "{:.1f}%".format(win_rate) }}</div>
        </div>
    </div>

    <div class="table-card">
        <div class="table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Created At</th>
                        <th>
                            <span class="tip-target">
                                Trade Lane
                                <span class="info-dot">i</span>
                                <span class="tip-bubble">Origin (POL) to Destination (POD) airport route.</span>
                            </span>
                        </th>
                        <th>
                            <span class="tip-target">
                                Airline Buy (₹)
                                <span class="info-dot">i</span>
                                <span class="tip-bubble">Net freight cost charged to us by the airline or carrier.</span>
                            </span>
                        </th>
                        <th>
                            <span class="tip-target">
                                Quoted Sell (₹)
                                <span class="info-dot">i</span>
                                <span class="tip-bubble">Total commercial selling price quoted to the client.</span>
                            </span>
                        </th>
                        <th>
                            <span class="tip-target">
                                Gross Profit (if Won)
                                <span class="info-dot">i</span>
                                <span class="tip-bubble">Actual gross margin pocketed if the deal is won (Sell Rate - Buy Cost).</span>
                            </span>
                        </th>
                        <th>
                            <span class="tip-target">
                                Target Margin %
                                <span class="info-dot">i</span>
                                <span class="tip-bubble">Recommended markup percentage over the airline cost.</span>
                            </span>
                        </th>
                        <th>
                            <span class="tip-target">
                                Estimated Win %
                                <span class="info-dot">i</span>
                                <span class="tip-bubble">Predicted customer acceptance likelihood at this price.</span>
                            </span>
                        </th>
                        <th>Weight & Goods</th>
                        <th>Branch</th>
                        <th>
                            <span class="tip-target">
                                Lead Status
                                <span class="info-dot">i</span>
                                <span class="tip-bubble">Current status in commercial pipeline (Draft, Negotiating, Won, Lost).</span>
                            </span>
                        </th>
                    </tr>
                </thead>
                <tbody>
                    {% for r in records %}
                    <tr>
                        <td style="font-weight: 700; color: var(--text-muted);">#{{ r.id }}</td>
                        <td>{{ r.timestamp }}</td>
                        <td>
                            <strong>{{ r.origin_port }}</strong> → <br>
                            <span style="color: var(--text-muted);">{{ r.dest_port }}</span>
                        </td>
                        <td>₹{{ "{:,.0f}".format(r.airline_buy_inr) }}</td>
                        <td style="font-weight: 800; font-family: 'Outfit', sans-serif;">₹{{ "{:,.0f}".format(r.selling_price_inr) }}</td>
                        <td style="font-weight: 700; color: #16a34a; font-family: 'Outfit', sans-serif;">₹{{ "{:,.0f}".format(r.margin_amount_inr or (r.selling_price_inr - r.airline_buy_inr)) }}</td>
                        <td><span class="badge badge-margin">{{ "{:.1f}%".format(r.margin_pct) }}</span></td>
                        <td><span class="badge badge-win">{{ "{:.1f}%".format(r.win_prob * 100) }}</span></td>
                        <td>
                            <span>{{ "{:.1f} kg".format(r.chargeable_wt_kg) }}</span><br>
                            <span style="font-size: 11px; color: var(--text-muted);">{{ r.commodity }}</span>
                        </td>
                        <td>{{ r.branch }}</td>
                        <td>
                            <select class="status-select status-{{ r.status_class }}" onchange="updateLeadStatus({{ r.id }}, this.value, this)">
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

        <!-- Pagination Controls -->
        <div class="pagination-bar">
            <div class="pagination-info">
                Showing <strong>{{ start_idx }}</strong> to <strong>{{ end_idx }}</strong> of <strong>{{ total_count }}</strong> records (Page {{ page }} of {{ total_pages }})
            </div>

            <div style="display: flex; align-items: center; gap: 14px; flex-wrap: wrap;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <label for="perPageSelect" style="font-size: 12px; color: var(--text-muted); font-weight: 600;">Rows per page:</label>
                    <select id="perPageSelect" class="per-page-select" onchange="changePerPage(this.value)">
                        <option value="10" {% if per_page == 10 %}selected{% endif %}>10</option>
                        <option value="15" {% if per_page == 15 %}selected{% endif %}>15</option>
                        <option value="25" {% if per_page == 25 %}selected{% endif %}>25</option>
                        <option value="50" {% if per_page == 50 %}selected{% endif %}>50</option>
                    </select>
                </div>

                <div class="pagination-actions">
                    {% if has_prev %}
                    <a href="/history?page=1&per_page={{ per_page }}" class="btn-page" title="First Page">«</a>
                    <a href="/history?page={{ prev_page }}&per_page={{ per_page }}" class="btn-page" title="Previous Page">‹ Prev</a>
                    {% else %}
                    <span class="btn-page disabled" title="First Page">«</span>
                    <span class="btn-page disabled" title="Previous Page">‹ Prev</span>
                    {% endif %}

                    {% for p in range(1, total_pages + 1) %}
                        {% if p == page %}
                        <span class="btn-page active">{{ p }}</span>
                        {% elif p <= 3 or p > total_pages - 3 or (p >= page - 1 and p <= page + 1) %}
                        <a href="/history?page={{ p }}&per_page={{ per_page }}" class="btn-page">{{ p }}</a>
                        {% elif p == 4 or p == total_pages - 3 %}
                        <span class="btn-page disabled" style="border: none; background: transparent;">...</span>
                        {% endif %}
                    {% endfor %}

                    {% if has_next %}
                    <a href="/history?page={{ next_page }}&per_page={{ per_page }}" class="btn-page" title="Next Page">Next ›</a>
                    <a href="/history?page={{ total_pages }}&per_page={{ per_page }}" class="btn-page" title="Last Page">»</a>
                    {% else %}
                    <span class="btn-page disabled" title="Next Page">Next ›</span>
                    <span class="btn-page disabled" title="Last Page">»</span>
                    {% endif %}
                </div>
            </div>
        </div>
    </div>
    {% else %}
    <div class="table-card empty-state">
        <div class="empty-icon">
            <svg class="ui-icon-lg" style="color: #94a3b8;" viewBox="0 0 24 24"><path d="M4 20h16a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.93a2 2 0 0 1-1.66-.9l-.82-1.2A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13c0 1.1.9 2 2 2Z"/></svg>
        </div>
        <h3>No Quote Recommendations Recorded Yet</h3>
        <p style="margin-top: 8px;">Run your first recommendation on the main page to start tracking leads and negotiation outcomes.</p>
        <div style="margin-top: 20px;">
            <a href="/" class="btn-action btn-primary">
                <svg class="ui-icon" viewBox="0 0 24 24"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
                Go to Pricing Calculator
            </a>
        </div>
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
                selectEl.className = 'status-select status-' + (statusMap[newStatus] || 'draft');
            } else {
                alert('Status update failed: ' + (data.message || 'Unknown error'));
            }
        })
        .catch(err => alert('Network error: ' + err.message));
    }

    function changePerPage(newPerPage) {
        window.location.href = `/history?page=1&per_page=${newPerPage}`;
    }

    function clearHistory() {
        if (!confirm('Are you sure you want to clear all historical quote records?')) return;
        fetch('/api/history/clear', { method: 'POST' })
            .then(res => res.json())
            .then(data => {
                if (data.status === 'success') window.location.reload();
            });
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
        client_options=CLIENT_OPTIONS,
        vertical_options=VERTICAL_OPTIONS,
        commodity_options=COMMODITY_OPTIONS,
        incoterms_options=INCOTERMS_OPTIONS,
        branch_options=BRANCH_OPTIONS,
        port_labels_json=PORT_LABELS_JSON
    )


@app.route("/api/quote", methods=["POST"])
def api_quote():
    data = request.get_json(force=True, silent=True) or {}

    # Strict numeric validation (§5.2, §5.3)
    raw_buy = data.get("buy")
    try:
        buy_cost = float(str(raw_buy).replace(",", "").strip())
        if buy_cost <= 0 or not np.isfinite(buy_cost):
            raise ValueError()
    except (ValueError, TypeError):
        return jsonify({"status": "error", "message": "Invalid Airline Buy Cost. Please provide a positive numeric value."}), 400

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

    # Derived Geographic Hierarchies & Tiers
    origin_region = map_origin_region(origin)
    dest_region = map_dest_region(dest)
    regional_lane = f"{origin_region} -> {dest_region}"

    tier = get_weight_tier(chargeable_wt)
    cust_tier = map_account_tier(frequency)

    inquiry_payload = {
        "Total_Buy_INR": buy_cost,
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
        "quote_date": data.get("quote_date"),
        "customer_name": data.get("customer_name") or data.get("customer") or "",
        "inquiry_ref": data.get("inquiry_ref") or data.get("ref") or ""
    }

    quote_date = str(data.get("quote_date") or datetime.date.today().isoformat())[:10]
    try:
        month = int(quote_date[5:7])
    except (ValueError, IndexError):
        month = 6

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
        "Customer_Company": data.get("customer_name") or data.get("customer") or "",
        "Credit_Days": data.get("credit_days", 30),
        "Oversize": bool(data.get("oversize")),
        "Rate_Type": data.get("rate_type", ""),
        "Competitor_Sell_Per_Kg": comp_rate,
        "is_spot_tender": is_spot_tender,
        "pricing_mode": "spot_tender" if is_spot_tender else "standard"
    })

    try:
        rec = mf.recommend(inquiry_payload, MARGIN_TABLES, is_spot_tender=is_spot_tender)
        legacy = mf.to_legacy_response(inquiry_payload, rec, MARGIN_TABLES)
        sens = mf.sensitivity_grid(inquiry_payload, rec, MARGIN_TABLES)

        warnings = []
        if month not in TRAIN_WINDOW_MONTHS:
            warnings.append(f"Quote month {month} is outside the training window; treat as low confidence.")
        if rec["band"]["n"] < 30:
            warnings.append(f"Thin cohort (n={rec['band']['n']}); quote leans on parent cohorts.")
        warnings += [f"Approval required: {a}" for a in rec["approval_required"]]
        confidence = "Low" if warnings else "High"

        # Shadow mode: evaluate legacy ML engine alongside for monitoring
        try:
            shadow_quote, _ = engine.optimize_quote(
                inquiry_dict=inquiry_payload,
                pricing_strategy=pricing_strategy
            )
            logger.info(
                f"[SHADOW MODE] Framework Balanced={rec['tiers']['balanced']['margin_pct']}% "
                f"vs Legacy ML={shadow_quote.get('Margin_Percentage')}%)"
            )
        except Exception as shadow_err:
            logger.debug(f"[SHADOW MODE] engine.optimize_quote: {shadow_err}")

        chosen_strategy = pricing_strategy if pricing_strategy in legacy["strategic_recommendations"] else "balanced"
        optimal_quote = dict(legacy["optimal"])
        if chosen_strategy != "balanced":
            optimal_quote.update(legacy["strategic_recommendations"][chosen_strategy])

        is_preset = bool(data.get("is_preset", False))

        history_logged = log_recommendation(
            db_path=HISTORY_DB_PATH,
            payload=inquiry_payload,
            optimal=optimal_quote,
            model_version=MODEL_VERSION,
            chosen_strategy=chosen_strategy,
            is_preset=is_preset,
            override_reason=data.get("override_reason"),
            user_id=data.get("user_id")
        )

        # Weight-Break Arbitrage Evaluation (§2.8)
        arbitrage_result = check_weight_break_arbitrage(
            chargeable_wt=chargeable_wt,
            current_total_cost=buy_cost,
            next_slab_rate_per_kg=data.get("next_slab_rate") or data.get("carrier_rate_next_break"),
            slab_rates=data.get("carrier_slab_rates")
        )

        return jsonify({
            "status": "success",
            "optimal": optimal_quote,
            "strategic_recommendations": legacy["strategic_recommendations"],
            "sensitivity": sens,
            "confidence": confidence,
            "warnings": warnings,
            "signals": rec["signals"],
            "cohort_band_per_kg": rec["band"],
            "weight_break_arbitrage": arbitrage_result,
            "pricing_mode": rec.get("pricing_mode", "standard"),
            "is_spot_tender": is_spot_tender,
            "competitor_intelligence": comp_intel,
            "history_logged": history_logged,
            "model_version": MODEL_VERSION
        })
    except Exception as err:
        logger.error(f"Margin framework recommendation error: {err}", exc_info=True)
        return jsonify({"status": "error", "message": f"Optimization engine error: {str(err)}"}), 500


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

    pagination = get_history_paginated(HISTORY_DB_PATH, page=page, per_page=per_page)
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


@app.route("/api/history/export")
def api_history_export():
    records = get_all_history(HISTORY_DB_PATH)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "ID", "Timestamp", "Origin", "Destination", "Buy Cost INR",
        "Selling Price INR", "Margin Pct", "Margin Amount INR",
        "Win Probability", "Expected Profit INR", "Gross Wt Kg",
        "Chargeable Wt Kg", "Commodity", "Client Type", "Branch", "Status", "Model Version"
    ])

    for r in records:
        writer.writerow([
            r.get("id"), r.get("timestamp"), r.get("origin_port"), r.get("dest_port"),
            r.get("airline_buy_inr"), r.get("selling_price_inr"), r.get("margin_pct"),
            r.get("margin_amount_inr"), r.get("win_prob"), r.get("expected_profit_inr"),
            r.get("gross_wt_kg"), r.get("chargeable_wt_kg"), r.get("commodity"),
            r.get("client_type"), r.get("branch"), r.get("status"), r.get("model_version")
        ])

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=air_export_pricing_history.csv"}
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
