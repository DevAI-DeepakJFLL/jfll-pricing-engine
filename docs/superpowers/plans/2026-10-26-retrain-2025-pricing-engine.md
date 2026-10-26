# 2025 Inquiries Engine Retraining & UI Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ingest the full-year 2025 inquiries dataset (`data/Shipmnts_Inquiries Jan Dec 2025.xlsx`), split 70/30 for training and out-of-time testing, expand the UI with essential commercial inputs (Quoted Carrier/Airline, Customer Company autocomplete, dual per-kg rate inputs, and quote date picker), and retrain the Pricing Engine V3 quantile models and conformal corridors with zero regressions.

**Architecture:** 
1. An automated ingestion adapter cleans and parses the 2025 raw TMS data into clean Won transactions and worked inquiries.
2. A dual 70/30 split module creates both a chronological out-of-time split (Jan–Sep 2025 Train vs mid-Sep–Dec 2025 Test) and a stratified split.
3. The geographic taxonomy in `src/taxonomy.py` is expanded with the top 50 missing international airport gateways.
4. Point-in-time (PIT) rolling customer and lane features are computed chronologically without lookahead bias.
5. `HistGradientBoostingRegressor` quantile models (P10, P50, P90) are retrained on the 70% train split and calibrated with `ConformalCorridorCalibrator`.
6. The Flask UI (`src/app.py`) is enhanced with new commercial dropdowns and interactive inputs while strictly preserving the approved minimalist monochrome design system.

**Tech Stack:** Python 3.11, pandas, numpy, scikit-learn (`HistGradientBoostingRegressor`), joblib, Flask, HTML5/CSS3 (Notion-inspired monochrome theme: Outfit/Mulish).

**Spec & Assessment:** [`DATA_ENGINEERING_AND_TRAINING_ASSESSMENT_2025.md`](file:///Users/deepak/.gemini/antigravity-cli/brain/57545377-b65f-4021-bfd7-df91f9448593/DATA_ENGINEERING_AND_TRAINING_ASSESSMENT_2025.md)

---

## Global Constraints

- **Design Governance:** Strictly preserve approved Notion-inspired monochrome theme (`#edf1f5`, `#ffffff`, `#0f172a`, `#64748b`, `Outfit` / `Mulish`). Zero reversion to legacy cyan/lime.
- **Label Purity:** Filter margin training data strictly on `Quote Status == 'Won'` (13,765 verified rows) to prevent leaking alternative rejected carrier quotes (`Inquiry Status == 'Won'` but `Quote Status == 'Lost'`).
- **Commercial Economic Filters:** Enforce $\text{Total\_Buy\_INR} > 500$, $\text{Total\_Sell\_INR} > \text{Total\_Buy\_INR}$, and $0.5 \le \text{Chargeable\_Weight\_Kg} \le 25,000$.
- **Bandwidth Discipline:** Scale-calibrated corridor spreads must maintain heavy cargo spreads $\le 1.8\text{ pp}$ and mid cargo spreads $\le 3.5\text{ pp}$.
- **Backward Compatibility:** All existing 226 unit tests and REST API payload keys must remain 100% operational.

---

## UI Inputs Expansion Specification

The following inputs will be added or enhanced in [`src/app.py`](file:///Users/deepak/Development/Jet%20Freight%20Logistics/Air%20Cargo%20Pricing%20Engine/src/app.py):

| UI Component | Field Type | Source in 2025 Data | Default Value | Options / Details |
| :--- | :--- | :--- | :--- | :--- |
| **Quoted Airline / Carrier** | Searchable Select | `Quoted Carrier` (133 airlines) | `"Any Airline"` | Top 20 airlines (Qatar Airways, Emirates, Air India, Etihad, Ethiopian, Air France, Singapore Airlines, IndiGo, Turkish Airlines, Cathay Pacific, Lufthansa, Saudia, British Airways, SriLankan, etc.) + "Other Carrier" |
| **Customer Company** | Searchable Autocomplete | `Customer Company` (1,184 accounts) | `""` (New / Spot Customer) | Autocomplete for known canonical accounts (e.g. `CUST_UNIVERSAL_CLEARING`, `CUST_TEMPCON`, etc.). If selected, dynamically pulls customer's real PIT stats. |
| **Dual Buy Rate Input** | Synchronized Numbers | `Unit Freight Buy` & `Total Buy` | ₹125,000 (Lump sum) or ₹277.78/kg | Interactive toggle/inputs: changing `Unit Rate (₹/kg)` auto-computes `Total Buy (₹)` = $\text{Unit Rate} \times \text{Chargeable Weight}$. |
| **Target Quote Date** | Date Picker | `Created At` / `quote_date` | Current Date | Allows pricing for future shipments; automatically drives `Month_Sin`, `Month_Cos`, `Air_Cargo_Season`, and `Is_Month_End`. |
| **Sales Desk / Executive** | Dropdown / Input | `Sales Person` (118 staff) | `"General Desk"` | Optional desk tag stored in SQLite audit trail. |

---

## Task Decomposition

### Task 1: Data Ingestion, Cleaning & 70/30 Split Pipeline

**Files:**
- Create: `scripts/ingest_and_split_2025.py`
- Test: `tests/test_ingest_and_split_2025.py`

**Interfaces:**
- Consumes: `data/Shipmnts_Inquiries Jan Dec 2025.xlsx`
- Produces: 
  - `data/processed/won_deals_2025_train_70.csv` (~9,635 rows)
  - `data/processed/won_deals_2025_test_30.csv` (~4,130 rows)
  - `data/processed/all_inquiries_2025_cleaned.csv` (~34,434 rows)

- [ ] **Step 1: Write unit tests for ingestion, currency parsing, and 70/30 splitting**
  Verify regex currency parsing, economic filter enforcement, monotonic date ordering, and exact 70/30 split boundary.
- [ ] **Step 2: Implement `scripts/ingest_and_split_2025.py`**
  - Read `Shipmnts_Inquiries Jan Dec 2025.xlsx`.
  - Strip commas and currency strings to pure floats for Buy, Sell, Unit Buy, Unit Sell.
  - Filter `Total_Buy_INR > 500`, `Sell > Buy`, `0.5 <= Weight <= 25000`.
  - Filter `Quote Status == 'Won'` for margin training.
  - Compute chronological 70% threshold timestamp and partition into Train (70%) and Holdout Test (30%).
  - Save clean CSVs into `data/processed/`.
- [ ] **Step 3: Run tests and verify split outputs**
  Execute `pytest tests/test_ingest_and_split_2025.py` and confirm 0 errors.

---

### Task 2: Expand Geographic & Carrier Taxonomy

**Files:**
- Modify: `src/taxonomy.py`
- Test: `tests/test_taxonomy_expanded.py`

**Interfaces:**
- Consumes: Destination port names from 2025 dataset
- Produces: Canonical `Destination_Region` with $<0.5\%$ unmapped rate

- [ ] **Step 1: Write test case verifying top 50 unmapped international gateways**
  Assert SVO (Europe), WAW (Europe), BCN (Europe), MRU (Africa), KIX (Asia-Pacific), CMN (Africa), TLV (Middle East), TAS (Europe/Central Asia), OTP (Europe), GUA (Latin America), MCI (North America) map correctly.
- [ ] **Step 2: Update `src/taxonomy.py`**
  Add dictionary of top 50 international airport names and IATA codes to destination keyword mappings.
- [ ] **Step 3: Run tests and verify unmapped rate**
  Ensure `<0.5%` unmapped destinations on the 2025 dataset.

---

### Task 3: Point-in-Time (PIT) Rolling Feature Engineering

**Files:**
- Modify: `src/features_pit.py`
- Test: `tests/test_features_pit_2025.py`

**Interfaces:**
- Consumes: Chronologically ordered `won_deals_2025_train_70.csv`
- Produces: 7 leak-free rolling features (`cust_pit_won_count_180d`, `cust_pit_median_margin_won`, `lane_pit_median_margin_won_90d`, etc.) and updated `CustomerPITRegistry`.

- [ ] **Step 1: Write test verifying zero forward-looking leakage**
  Ensure customer won stats on date $T$ only see transactions from $t < T$.
- [ ] **Step 2: Compute PIT features on 70% train set**
  Generate rolling features and export `data/processed/won_deals_2025_train_70_pit.csv` and serialized `CustomerPITRegistry`.
- [ ] **Step 3: Run tests**
  Execute `pytest tests/test_features_pit_2025.py`.

---

### Task 4: Quantile Model Retraining & Conformal Calibration

**Files:**
- Create: `scripts/train_v3_2025.py`
- Modify: `src/quantile_model.py`, `src/conformal_corridors.py`
- Test: `tests/test_quantile_retrain_2025.py`

**Interfaces:**
- Consumes: `won_deals_2025_train_70_pit.csv`
- Produces: `models/engine_v3_bundle.joblib` (retrained weights, slab quantiles, calibrator)

- [ ] **Step 1: Write test verifying quantile ordering and non-negative margin bounds**
  Verify $P_{10} \le P_{50} \le P_{90}$ across 100 test points.
- [ ] **Step 2: Retrain HistGradientBoosting Quantile Models**
  Fit P10, P50, P90 on the 70% train set with monotonic constraints on Buy Cost and Weight.
- [ ] **Step 3: Calibrate Conformal Corridors**
  Run `ConformalCorridorCalibrator` with scale-calibrated latitude ($\pm 15\%$ heavy, $\pm 20\%$ mid, $\pm 25\%$ low-budget) on the calibration partition.
- [ ] **Step 4: Serialize and save bundle**
  Save updated bundle to `models/engine_v3_bundle.joblib`.
- [ ] **Step 5: Run tests**
  Execute `pytest tests/test_quantile_retrain_2025.py`.

---

### Task 5: Out-of-Time 30% Holdout Benchmark Evaluation

**Files:**
- Create: `scripts/evaluate_2025_test_set.py`
- Output: `data/processed/EVALUATION_REPORT_2025_TEST_SET.md`

**Interfaces:**
- Consumes: `data/processed/won_deals_2025_test_30.csv`, `models/engine_v3_bundle.joblib`
- Produces: Statistical evaluation report (Pinball loss, coverage rate, median spread, win-rate simulation).

- [ ] **Step 1: Implement evaluation script**
  Evaluate predictions on all ~4,130 test quotes from the last 30% of 2025.
- [ ] **Step 2: Run benchmark evaluation**
  Verify empirical coverage $\approx 90\%$ and corridor spread discipline across heavy, mid, and small tiers.
- [ ] **Step 3: Generate evaluation artifact**
  Output `EVALUATION_REPORT_2025_TEST_SET.md`.

---

### Task 6: UI & API Input Expansion in `src/app.py`

**Files:**
- Modify: `src/app.py`, `src/engine_v3.py`
- Test: `tests/test_api_expanded_inputs.py`

**Interfaces:**
- Consumes: New UI form inputs (`carrier`, `customer_company`, `unit_buy`, `quote_date`)
- Produces: Enriched REST API response and seamless Notion-monochrome UI interface.

- [ ] **Step 1: Write test for API payload parsing with new inputs**
  Verify `/api/quote` parses `carrier`, `customer_company`, `unit_buy`, `quote_date` while retaining full backward compatibility for older payloads.
- [ ] **Step 2: Update `src/app.py` UI Template & Handlers**
  - Add "Quoted Airline / Carrier" searchable dropdown with Top 20 airlines.
  - Add "Customer Company" searchable input with auto-population of customer tier.
  - Add dual-rate interactive calculator: `Unit Rate (₹/kg)` $\leftrightarrow$ `Total Buy (₹)`.
  - Add "Target Quote Date" date picker.
  - Ensure zero CSS style breakage; adhere 100% to `#0f172a`, `#64748b`, `#ffffff`, `Outfit`/`Mulish`.
- [ ] **Step 3: Run API tests & verify UI rendering**
  Execute `pytest tests/test_api_expanded_inputs.py`.

---

### Task 7: Full Regression & System Verification

**Files:**
- Test: `scripts/verify_vocab_alignment.py`, `scripts/test_200_random_samples.py`, `tests/`

- [ ] **Step 1: Run vocabulary verification**
  Execute `./run.sh verify` (ensure 100% vocabulary alignment).
- [ ] **Step 2: Run 200 stratified random samples test**
  Execute `.venv/bin/python scripts/test_200_random_samples.py` (ensure 0 inversions, 0 errors).
- [ ] **Step 3: Run full test suite**
  Execute `.venv/bin/python -m pytest tests/` (ensure all tests pass).
- [ ] **Step 4: Synchronize secondary repository**
  Synchronize changes to `/Users/deepak/Development/Jet Freight Logistics/Air Cargo Pricing Engine`.
