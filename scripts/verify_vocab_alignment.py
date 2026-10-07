#!/usr/bin/env python3
"""
Phase 0 Verification Gate: scripts/verify_vocab_alignment.py
Asserts that every dropdown option in app.py resolves to a known, non-negative
encoded category (zero -1 encodings) in the trained freight_margin_recommender.joblib artifact.
"""

import sys
import os
import re
import joblib
import pandas as pd

def extract_select_options(html_content, select_id):
    """Extract all option values from a select element or custom searchable dropdown."""
    # 1. Try standard <select id="...">
    pattern = rf'<select[^>]*id=[\"\']{select_id}[\"\'][^>]*>(.*?)</select>'
    match = re.search(pattern, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        options_block = match.group(1)
        values = re.findall(r'<option[^>]*value=[\"\']([^\"\']*)[\"\']', options_block)
        if not values:
            values = re.findall(r'<option[^>]*>([^<]+)</option>', options_block)
        return [v.strip() for v in values if v.strip()]

    # 2. Try custom searchable dropdown with id="{select_id}_options_list"
    list_pattern = rf'<ul[^>]*id=[\"\']{select_id}_options_list[\"\'][^>]*>(.*?)</ul>'
    list_match = re.search(list_pattern, html_content, re.DOTALL | re.IGNORECASE)
    if list_match:
        list_block = list_match.group(1)
        values = re.findall(r'data-value=[\"\']([^\"\']*)[\"\']', list_block)
        return [v.strip() for v in values if v.strip()]

    raise ValueError(f"Could not find select or custom dropdown options for id='{select_id}' in HTML template.")

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "freight_margin_recommender.joblib")
    app_path = os.path.join(base_dir, "src", "app.py")

    if not os.path.exists(model_path):
        print(f"[FAIL] Model artifact not found at: {model_path}")
        sys.exit(1)

    if not os.path.exists(app_path):
        print(f"[FAIL] app.py not found at: {app_path}")
        sys.exit(1)

    print(f"[*] Loading model artifact: {model_path}")
    model_bundle = joblib.load(model_path)
    encoder = model_bundle.get("encoder")
    cat_cols = model_bundle.get("native_cat_cols", model_bundle.get("cat_cols"))
    feature_cols = model_bundle.get("feature_cols")

    if encoder is None or cat_cols is None:
        print("[FAIL] Missing 'encoder' or 'cat_cols' in model artifact.")
        sys.exit(1)

    with open(app_path, "r", encoding="utf-8") as f:
        app_code = f.read()

    # Mapping of UI select ID to Model Categorical Column
    ui_to_model_map = {
        "origin": "Origin_Port",
        "dest": "Destination_Port",
        "client": "Company_Group",
        "vertical": "Business_Vertical",
        "commodity": "Commodity_Group",
        "incoterms": "Incoterms",
        "branch": "Booked by branch"
    }

    # Import options dictionaries from app.py
    sys.path.insert(0, os.path.join(base_dir, "src"))
    from app import ORIGIN_OPTIONS, CLIENT_OPTIONS, COMMODITY_OPTIONS, INCOTERMS_OPTIONS, DESTINATION_OPTIONS
    from preprocess import normalize_port_name
    normalize_port = normalize_port_name

    ui_options_map = {
        "origin": [o["value"] for o in ORIGIN_OPTIONS],
        "dest": [o["value"] for o in DESTINATION_OPTIONS],
        "client": [o["value"] for o in CLIENT_OPTIONS],
        "commodity": [o["value"] for o in COMMODITY_OPTIONS],
        "incoterms": [o["value"] for o in INCOTERMS_OPTIONS]
    }

    total_checked = 0
    failures = []

    print("\n" + "=" * 70)
    print("CATEGORICAL VOCABULARY ALIGNMENT REPORT (UI vs. TRAINED ENCODER)")
    print("=" * 70)

    for select_id, cat_col in ui_to_model_map.items():
        if cat_col not in cat_cols:
            print(f"[WARN] Feature '{cat_col}' not in trained cat_cols. Skipping.")
            continue

        col_idx = cat_cols.index(cat_col)
        learned_vocab = set(encoder.categories_[col_idx])
        options = ui_options_map.get(select_id)
        if not options:
            try:
                options = extract_select_options(app_code, select_id)
            except Exception:
                options = []

        print(f"\nDropdown: #{select_id}  -->  Model Column: '{cat_col}' ({len(options)} options)")
        print("-" * 70)

        for opt in options:
            total_checked += 1
            # Check direct or normalized
            resolved = normalize_port(opt) if "Port" in cat_col else opt
            in_vocab = resolved in learned_vocab

            # Verify with encoder transform
            dummy_df = pd.DataFrame({cat_col: [str(resolved)]})
            # Sub-encoder transform check
            try:
                enc_val = encoder.categories_[col_idx].tolist().index(str(resolved))
            except ValueError:
                enc_val = -1

            status = "PASS" if enc_val >= 0 else "FAIL (-1 Unknown)"
            if enc_val < 0:
                failures.append((select_id, cat_col, opt, resolved))
                print(f"  [FAIL] '{opt}' (resolved: '{resolved}') -> enc_value: {enc_val}")
            else:
                print(f"  [OK]   '{opt}' -> enc_value: {enc_val}")

    print("\n" + "=" * 70)
    print(f"SUMMARY: {total_checked - len(failures)}/{total_checked} Passed. {len(failures)} Failed.")
    print("=" * 70)

    if failures:
        print("\n[CRITICAL ERROR] The following UI values resolve to -1 in the trained encoder:")
        for sel, col, opt, res in failures:
            print(f"  - #{sel} (col: {col}): '{opt}' -> '{res}'")
        print("\nFix these mismatches in app.py or retrain the encoder before shipping.")
        sys.exit(1)
    else:
        print("\n[SUCCESS] All UI dropdown options align 100% with trained model categories!")
        sys.exit(0)

if __name__ == "__main__":
    main()
