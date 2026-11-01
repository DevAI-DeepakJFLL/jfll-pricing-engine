# Comprehensive Technical Documentation: All 8 PonyTail Audit Findings & Remediation Guide

**Target System:** Air Export Pricing Engine (`v3.0.0`)  
**Document Type:** Senior Engineering Audit & Remediation Guide  
**Status:** Approved for Implementation  
**Auditor:** Senior Systems Architect & Data Scientist  
**Date:** November 1, 2026  

---

## Executive Summary

A full repository audit conducted under the **PonyTail Senior Developer Audit** methodology identified **8 concrete findings** across code correctness, operational safety under load, test infrastructure, model consistency, and dead code elimination.

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    FINDINGS CLASSIFICATION                                       │
├─────────────────────┬────────────────────────────────────────────────────────────────────────────┤
│ Must Fix (1 - 3)    │ 1. Broken plain pytest execution (tests/test_api_error_handlers.py)        │
│                     │ 2. Web API re-triggers 500% margin spikes on 2-29 kg cargo (src/app.py)    │
│                     │ 3. DDL migrations executed on hot quote request path (src/db.py)           │
├─────────────────────┼────────────────────────────────────────────────────────────────────────────┤
│ Should Fix (4 - 6)  │ 4. 1,272 lines of dead V2 legacy code raising RuntimeErrors (src/engine.py)│
│                     │ 5. Engine built-in train() missing outlier cap & weights (src/engine_v3.py)│
│                     │ 6. Missing validation check on tiny buy rates < ₹10/kg (src/app.py)        │
├─────────────────────┼────────────────────────────────────────────────────────────────────────────┤
│ Nice to Have (7 - 8)│ 7. Duplicate volumetric weight calculation in quote handler (src/app.py)   │
│                     │ 8. Sliding window rate limiter unbounded IP growth between purges (sec.py) │
└─────────────────────┴────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Test Suite Crashes on Plain `pytest`

### Overview
* **File:** [`tests/test_api_error_handlers.py:L7`](file:///Users/deepak/Development/Jet%20Freight%20Logistics/Air%20Cargo%20Pricing%20Engine/tests/test_api_error_handlers.py#L7)
* **Severity:** **Must Fix** (Blocks local runs, new developer onboarding, and CI/CD pipelines)
* **Category:** Build & Test Harness

### Root Cause Analysis
Running `pytest tests/` directly from the command line without setting `PYTHONPATH=.` crashes immediately during test collection:
```text
ImportError while importing test module 'tests/test_api_error_handlers.py':
tests/test_api_error_handlers.py:7: in <module>
    from src.app import app
E   ModuleNotFoundError: No module named 'src'
```
Unlike other test files in `tests/`, `test_api_error_handlers.py` omitted inserting `PROJECT_ROOT` into `sys.path`. Additionally, the repository lacks a `pytest.ini` configuration file defining the Python path.

### Concrete Reproduction Case
```bash
cd "/Users/deepak/Development/Jet Freight Logistics/Air Cargo Pricing Engine"
.venv/bin/pytest tests/test_api_error_handlers.py
# Result: Crashes with ModuleNotFoundError: No module named 'src'
```

### Remediation & Code Diff

#### A. In [`tests/test_api_error_handlers.py`](file:///Users/deepak/Development/Jet%20Freight%20Logistics/Air%20Cargo%20Pricing%20Engine/tests/test_api_error_handlers.py):
```python
# --- BEFORE ---
import unittest
from src.app import app

# --- AFTER ---
import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.app import app
```

#### B. Create `pytest.ini` in project root:
```ini
[pytest]
pythonpath = .
testpaths = tests
```

---

## 2. Web API Re-Triggers 500% Margin Spikes on 2 kg to 29 kg Shipments

### Overview
* **File:** [`src/app.py:L726-L747`](file:///Users/deepak/Development/Jet%20Freight%20Logistics/Air%20Cargo%20Pricing%20Engine/src/app.py#L726-L747)
* **Severity:** **Must Fix** (Commercial pricing bug; causes 500%–870% margin spikes in production)
* **Category:** Data Ingestion & API Integrity

### Root Cause Analysis
During previous remediation, the threshold for rate-per-kg disambiguation (`HEAVY_CONSOL_MIN_WEIGHT_KG`) was lowered from `30.0 kg` to `2.0 kg` in [`src/validation.py:L21`](file:///Users/deepak/Development/Jet%20Freight%20Logistics/Air%20Cargo%20Pricing%20Engine/src/validation.py#L21). This solved the bug where coordinators enter unit rate (e.g. ₹311/kg) into the Total Buy field.

However, [`src/app.py`](file:///Users/deepak/Development/Jet%20Freight%20Logistics/Air%20Cargo%20Pricing%20Engine/src/app.py#L726-L747) duplicated this validation inline and **left the threshold hardcoded at 30.0 kg**:
```python
# src/app.py lines 740-747:
if (
    buy_cost is not None
    and unit_buy_rate is not None
    and chargeable_wt >= 30.0  # <--- BUG: Skips 2.0kg to 29.9kg shipments!
    and unit_buy_rate >= 10.0
    and (abs(buy_cost - unit_buy_rate) < 1.0 or (buy_cost / chargeable_wt) < 10.0)
):
    buy_cost = round(unit_buy_rate * chargeable_wt, 2)
```

### The Failure Cascade
1. User enters: `Weight = 20 kg`, `Total Buy = 311`, `Unit Buy = 311`.
2. Because `chargeable_wt (20) < 30.0`, `app.py` skips reconciliation.
3. Line 759 sets `unit_buy_rate = 311 / 20 = 15.55 INR/kg`.
4. `app.py` passes `Total_Buy_INR = 311` and `Unit_Buy_INR = 15.55` to `PricingEngineV3`.
5. Inside `PricingEngineV3`, the AWB fixed fee (₹1,500) divided by ₹311 yields an unrealistic **482% margin recommendation**.

### Remediation & Code Diff
Import `HEAVY_CONSOL_MIN_WEIGHT_KG` from [`src/validation.py`](file:///Users/deepak/Development/Jet%20Freight%20Logistics/Air%20Cargo%20Pricing%20Engine/src/validation.py#L21) and update lines 726 and 743:
```python
# --- BEFORE (src/app.py:L726, L743) ---
if (
    buy_cost is not None
    and unit_buy_rate is None
    and chargeable_wt >= 30.0
    and (buy_cost / chargeable_wt) < 10.0
    ...
):

if (
    buy_cost is not None
    and unit_buy_rate is not None
    and chargeable_wt >= 30.0
    ...
):

# --- AFTER (src/app.py) ---
from validation import HEAVY_CONSOL_MIN_WEIGHT_KG, MIN_RATE

if (
    buy_cost is not None
    and unit_buy_rate is None
    and chargeable_wt >= HEAVY_CONSOL_MIN_WEIGHT_KG
    and (buy_cost / chargeable_wt) < MIN_RATE
    ...
):

if (
    buy_cost is not None
    and unit_buy_rate is not None
    and chargeable_wt >= HEAVY_CONSOL_MIN_WEIGHT_KG
    and unit_buy_rate >= MIN_RATE
    ...
):
```

---

## 3. Database Schema Migrations Run on Every Quote Request

### Overview
* **Files:** [`src/db.py:L115`](file:///Users/deepak/Development/Jet%20Freight%20Logistics/Air%20Cargo%20Pricing%20Engine/src/db.py#L115) and [`src/app.py:L894`](file:///Users/deepak/Development/Jet%20Freight%20Logistics/Air%20Cargo%20Pricing%20Engine/src/app.py#L894)
* **Severity:** **Must Fix** (Scale / Concurrency failure; crashes quotes under real traffic)
* **Category:** Database Architecture & Concurrency

### Root Cause Analysis
Inside [`src/app.py:L894`](file:///Users/deepak/Development/Jet%20Freight%20Logistics/Air%20Cargo%20Pricing%20Engine/src/app.py#L894), every incoming `POST /api/quote` queries recent competitor rates via `get_recent_competitor_rate_benchmark(DB_PATH, origin, dest)`.

Inside `get_recent_competitor_rate_benchmark`, line 115 executes:
```python
def get_recent_competitor_rate_benchmark(...):
    ...
    migrate_outcome_columns(abs_path)  # <--- DDL migration on read path!
```
Every single quote request connects to SQLite, executes `PRAGMA table_info(recommendation_history)`, loops through columns, and attempts `CREATE INDEX IF NOT EXISTS idx_rec_history_lane ...`.

### Impact Under Real Load
SQLite allows multiple concurrent readers, but DDL (`CREATE INDEX`, `ALTER TABLE`) acquires an **exclusive database write lock**. When 10 to 50 sales coordinators submit quotes concurrently, workers block each other, timing out after 15 seconds with:
```text
sqlite3.OperationalError: database is locked
```

### Remediation & Code Diff
Remove `migrate_outcome_columns` from the hot read query. Call it strictly once during server initialization in `init_history_db`:
```python
# --- BEFORE (src/db.py:L115) ---
def get_recent_competitor_rate_benchmark(db_path: str, origin_port: str, dest_port: str, days: int = 45, max_records: int = 10):
    if not origin_port or not dest_port:
        return None
    abs_path = os.path.abspath(db_path)
    if not os.path.exists(abs_path):
        return None
    migrate_outcome_columns(abs_path)  # <--- REMOVE THIS LINE

# --- AFTER (src/db.py:L187) ---
def init_history_db(db_path: str = "data/recommendations_history.db", force: bool = False):
    abs_path = os.path.abspath(db_path)
    if not force and abs_path in _INITIALIZED_DBS:
        return
    ...
    # Run schema migrations once at startup
    migrate_outcome_columns(abs_path)
    _INITIALIZED_DBS.add(abs_path)
```

---

## 4. Dead V2 Legacy Code Raising Runtime Crashes (1,272 Lines)

### Overview
* **Files:** [`src/engine.py`](file:///Users/deepak/Development/Jet%20Freight%20Logistics/Air%20Cargo%20Pricing%20Engine/src/engine.py) (622 lines) and [`src/margin_framework.py`](file:///Users/deepak/Development/Jet%20Freight%20Logistics/Air%20Cargo%20Pricing%20Engine/src/margin_framework.py) (650 lines)
* **Severity:** **Should Fix** (Code hygiene, bloat reduction, prevention of regression errors)
* **Category:** Lean Codebase & Technical Debt

### Root Cause Analysis
During the transition to `PricingEngineV3` ([`src/engine_v3.py`](file:///Users/deepak/Development/Jet%20Freight%20Logistics/Air%20Cargo%20Pricing%20Engine/src/engine_v3.py)), the legacy V2 files were deprecated. In [`src/engine.py:L24-L34`](file:///Users/deepak/Development/Jet%20Freight%20Logistics/Air%20Cargo%20Pricing%20Engine/src/engine.py#L24-L34), a safety block was added:
```python
if os.environ.get("ALLOW_LEGACY_ENGINE") != "1":
    if "pytest" in sys.modules or "unittest" in sys.modules:
        warnings.warn("Legacy engine (v2.x). Superseded by engine_v3.py...")
    else:
        raise RuntimeError("Legacy engine (v2.x). Superseded by engine_v3.py...")
```
Furthermore, [`src/margin_framework.py`](file:///Users/deepak/Development/Jet%20Freight%20Logistics/Air%20Cargo%20Pricing%20Engine/src/margin_framework.py) contains the old `y = log(Margin/Weight)` calculation that distorted perishable margins. Neither file is used by the active application or modern test suite.

### Remediation Plan
1. Delete `src/engine.py` (-622 lines).
2. Delete `src/margin_framework.py` (-650 lines).
3. Remove legacy imports in `src/train.py` and migrate the remaining two legacy test files (`tests/test_cohort_lookup.py` and `tests/test_margin_framework.py`) to test `PricingEngineV3`.
4. **Net Lean Impact:** **-1,272 lines of dead code removed.**

---

## 5. Built-in `PricingEngineV3.train()` Missing Outlier Cap & Weights

### Overview
* **File:** [`src/engine_v3.py:L168-L175`](file:///Users/deepak/Development/Jet%20Freight%20Logistics/Air%20Cargo%20Pricing%20Engine/src/engine_v3.py#L168-L175)
* **Severity:** **Should Fix** (Model training discrepancy & stability guard)
* **Category:** Machine Learning Pipeline Consistency

### Root Cause Analysis
When retraining models via the standalone script [`scripts/train_v3_2025.py:L124-L148`](file:///Users/deepak/Development/Jet%20Freight%20Logistics/Air%20Cargo%20Pricing%20Engine/scripts/train_v3_2025.py#L124-L148), training margins are winsorized at 45.0% (`np.clip(y_raw, 0.5, 45.0)`), and sample weights are weighted with exponential recency decay ($\lambda = 0.005/\text{day}$).

However, if someone instantiates `PricingEngineV3(auto_train=True)` or calls `PricingEngineV3.train()`, the method in `src/engine_v3.py` calls:
```python
# src/engine_v3.py line 175:
self.model.fit(train_df, train_df["Margin_Percentage"])
```
This trains the tree regressor on raw, unclipped targets (where historical outliers reach up to 10,859%) and assigns equal weight to historical 2021 deals versus current 2025/2026 market rates.

### Remediation & Code Diff
Update `train()` in [`src/engine_v3.py`](file:///Users/deepak/Development/Jet%20Freight%20Logistics/Air%20Cargo%20Pricing%20Engine/src/engine_v3.py#L168):
```python
# --- BEFORE (src/engine_v3.py:L175) ---
self.model.fit(train_df, train_df["Margin_Percentage"])

# --- AFTER (src/engine_v3.py) ---
# Winsorize targets at 45.0% to match scripts/train_v3_2025.py
y_raw = train_df["Margin_Percentage"].values
y_fit = np.clip(y_raw, 0.5, 45.0)

# Exponential time-decay sample weighting
if "quote_date" in train_df.columns:
    dates = pd.to_datetime(train_df["quote_date"], errors="coerce")
    max_date = dates.max()
    days_ago = (max_date - dates).dt.days.fillna(0).values
    sample_weights = np.exp(-0.005 * days_ago)
else:
    sample_weights = None

self.model.fit(train_df, y_fit, sample_weight=sample_weights)
```

---

## 6. Missing Validation Check on Tiny Airline Buy Rates (< ₹10/kg)

### Overview
* **File:** [`src/app.py:L758-L760`](file:///Users/deepak/Development/Jet%20Freight%20Logistics/Air%20Cargo%20Pricing%20Engine/src/app.py#L758-L760)
* **Severity:** **Should Fix** (Input sanitization & typo protection)
* **Category:** Financial Input Guardrails

### Root Cause Analysis
In [`src/app.py`](file:///Users/deepak/Development/Jet%20Freight%20Logistics/Air%20Cargo%20Pricing%20Engine/src/app.py#L758-L760), if a user enters only total buy cost without unit buy rate, the system computes:
```python
if unit_buy_rate is None and buy_cost is not None:
    unit_buy_rate = round(buy_cost / max(0.1, chargeable_wt), 2)
```
If a user makes a typo and submits `buy = ₹40` for a `50 kg` cargo shipment, `unit_buy_rate` evaluates to `₹0.80/kg`. Commercial air freight costs anywhere from ₹25/kg to ₹800/kg. While `validation.py` defines `MIN_RATE = 10.0`, `app.py` fails to enforce this lower bound when deriving `unit_buy_rate` from `buy_cost`, passing the erroneous rate downstream.

### Remediation & Code Diff
Add an immediate check in [`src/app.py`](file:///Users/deepak/Development/Jet%20Freight%20Logistics/Air%20Cargo%20Pricing%20Engine/src/app.py#L760):
```python
# --- AFTER (src/app.py:L760) ---
if unit_buy_rate is None and buy_cost is not None:
    unit_buy_rate = round(buy_cost / max(0.1, chargeable_wt), 2)

if unit_buy_rate < 10.0:
    return jsonify({
        "status": "error",
        "message": f"Implied buy rate ₹{unit_buy_rate:.2f}/kg is below the commercial minimum (₹10.00/kg). Please verify total buy."
    }), 422
```

---

## 7. Volumetric Weight Computed Twice on Every Quote Request

### Overview
* **Files:** [`src/app.py:L638`](file:///Users/deepak/Development/Jet%20Freight%20Logistics/Air%20Cargo%20Pricing%20Engine/src/app.py#L638) and [`src/preprocess.py:L268`](file:///Users/deepak/Development/Jet%20Freight%20Logistics/Air%20Cargo%20Pricing%20Engine/src/preprocess.py#L268)
* **Severity:** **Nice to Have** (Elimination of duplicate logic)
* **Category:** Code Reusability & DRY

### Root Cause Analysis
In [`src/app.py:L638`](file:///Users/deepak/Development/Jet%20Freight%20Logistics/Air%20Cargo%20Pricing%20Engine/src/app.py#L638), volumetric weight is computed with inline arithmetic:
```python
vol_calc = (l_val * w_val * h_val * pcs_val) / 6000.0
```
Fifty lines later, line 687 calls:
```python
is_valid, eff_chargeable, density_ratio, err_reason = validate_shipment_physics(
    gross_wt=gross_wt, ch_wt=chargeable_wt, dimensions=dimensions
)
```
Inside `validate_shipment_physics`, line 268 re-invokes `compute_volumetric_weight(dimensions)`, recalculating the exact same value.

### Remediation
Remove the inline `(l * w * h * pcs) / 6000.0` calculation in `app.py`. Call `compute_volumetric_weight(dimensions)` directly so that if the volumetric divisor is ever configured or modified (e.g. 5,000 for express couriers vs 6,000 for standard IATA cargo), it updates in one single location.

---

## 8. In-Memory Sliding Window Rate Limiter Table Can Grow Between Purges

### Overview
* **File:** [`src/security.py:L207-L215`](file:///Users/deepak/Development/Jet%20Freight%20Logistics/Air%20Cargo%20Pricing%20Engine/src/security.py#L207-L215) and [`L238`](file:///Users/deepak/Development/Jet%20Freight%20Logistics/Air%20Cargo%20Pricing%20Engine/src/security.py#L238)
* **Severity:** **Nice to Have** (Memory safety under malicious network scans)
* **Category:** Security & Resource Management

### Root Cause Analysis
The in-memory rate limiter tracks request timestamps per IP:
```python
with _RATE_LIMIT_LOCK:
    _REQUEST_COUNTER += 1
    if _REQUEST_COUNTER % 100 == 0 or len(_RATE_LIMIT_STORE) >= 5000:
        _purge_expired_entries(now, window_seconds)
```
Purging expired timestamps only occurs every 100 requests. Under an automated vulnerability scan where an attacker cycles through spoofed `X-Forwarded-For` headers, thousands of unique IP keys can accumulate in memory between pruning intervals.

### Remediation
Add a time-based trigger in [`src/security.py`](file:///Users/deepak/Development/Jet%20Freight%20Logistics/Air%20Cargo%20Pricing%20Engine/src/security.py#L238) ensuring pruning occurs if more than 30 seconds have elapsed since the last cleanup:
```python
# --- AFTER (src/security.py:L238) ---
global _LAST_PURGE_TIME
if _REQUEST_COUNTER % 100 == 0 or len(_RATE_LIMIT_STORE) >= 5000 or (now - _LAST_PURGE_TIME) > 30.0:
    _purge_expired_entries(now, window_seconds)
    _LAST_PURGE_TIME = now
```

---

## Verification & Rollout Plan

After applying the changes for these 8 items:

| Step | Action | Success Criteria |
| :---: | :--- | :--- |
| **1** | Run `pytest tests/` (without `PYTHONPATH=.`). | All 356 tests pass cleanly with 0 collection errors. |
| **2** | Test 20 kg shipment with Buy = ₹311 via `/api/quote`. | Model recommends ~4%–6% margin; zero 500% margin spikes. |
| **3** | Execute concurrency load test (`100` parallel requests on `/api/quote`). | Zero `sqlite3.OperationalError: database is locked` errors. |
| **4** | Verify legacy file deletion. | Clean git status with `-1,272` lines of dead code removed. |
| **5** | Run model unit tests on `PricingEngineV3.train()`. | Verifies target margins are clipped at 45.0% and weighted. |
| **6** | Submit quote with `buy = 40` and `weight = 50`. | Rejects with HTTP 422: implied rate < ₹10.00/kg. |
