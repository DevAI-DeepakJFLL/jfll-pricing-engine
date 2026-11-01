# PonyTail Audit Report: Air Export Pricing Engine

**Target System:** Air Export Pricing Engine (`v3.0.0`)  
**Auditor:** Senior Developer & Pricing Systems Architect  
**Date:** November 1, 2026  

---

### What this repo does
This repository runs an AI pricing engine for air cargo export shipments. It predicts target profit margins, sets safe negotiation ranges, and enforces hard minimum price floors so quotes do not lose money. I assumed a typical freight forwarding setup: a web dashboard and API used by 20 to 100 sales coordinators quoting shipments at the same time, backed by a local SQLite database and pre-trained machine learning models.

---

### Context: The "Won vs. Lost" Quotation Rule (Preserving Modeling Discipline)

Before listing the findings, we document the established design boundary regarding quote outcomes:
* **Stage 1A Margin Quantile Model:** Must train **strictly on verified Won deals** (`Quote Status == 'Won'`). Air forwarders regularly quote 2 to 3 airline alternatives per RFQ; when a client selects one option, the rejected airline options are marked `Lost`. Ingesting lost quotes into Stage 1A contaminates the training target with unaccepted airline options, coordinator wild guesses, and exploratory RFQs (`Costing Purpose`, `Rates for exploration`).
* **Stage 1B Win Probability Model:** Must train on both **Won and Lost deals** (`Quote Status \in ['Won', 'Lost']`). Lost deals represent genuine customer price resistance, making them the correct signal for calibration of win probability curves and price elasticity.

---

### Must fix

1. **Test suite crashes on plain `pytest`** ([`tests/test_api_error_handlers.py:L7`](file:///Users/deepak/Development/Jet%20Freight%20Logistics/Air%20Cargo%20Pricing%20Engine/tests/test_api_error_handlers.py#L7))
   - **What this is:** An automated test file that verifies the API returns clean error messages for missing routes.
   - **Problem:** It imports `from src.app import app` without putting the project root in Python's search path. When you type `pytest` in your terminal or in continuous integration (CI), the test runner crashes on collection with `ModuleNotFoundError: No module named 'src'`.
   - **Fix:** Add `sys.path.insert(0, PROJECT_ROOT)` at the top of the test file, or add a `pytest.ini` file in the root folder with `pythonpath = .`.
   - **If we skip it:** Automated tests fail immediately on any fresh checkout or automated build system unless someone remembers to manually type `PYTHONPATH=.`.

2. **Web API re-triggers 500% margin spikes on 2 kg to 29 kg shipments** ([`src/app.py:L726-L747`](file:///Users/deepak/Development/Jet%20Freight%20Logistics/Air%20Cargo%20Pricing%20Engine/src/app.py#L726-L747))
   - **What this is:** Input validation in the web API that reconciles when a user types a rate-per-kg into the Total Buy box.
   - **Problem:** In [`src/validation.py`](file:///Users/deepak/Development/Jet%20Freight%20Logistics/Air%20Cargo%20Pricing%20Engine/src/validation.py#L21), the weight threshold was lowered to 2.0 kg. But [`src/app.py`](file:///Users/deepak/Development/Jet%20Freight%20Logistics/Air%20Cargo%20Pricing%20Engine/src/app.py#L743) copy-pasted the old code with `chargeable_wt >= 30.0`. If a coordinator submits a 20 kg quote where buy is ₹311/kg and total buy is typed as ₹311, `app.py` skips the fix, keeps total buy at ₹311, and corrupts the numbers before the pricing engine ever sees them.
   - **Fix:** Import `HEAVY_CONSOL_MIN_WEIGHT_KG` from [`src/validation.py`](file:///Users/deepak/Development/Jet%20Freight%20Logistics/Air%20Cargo%20Pricing%20Engine/src/validation.py#L21) and replace the hardcoded `30.0` in `app.py`.
   - **If we skip it:** The 500% to 800% margin spikes will keep happening in production on all parcels between 2 kg and 29.9 kg whenever someone types a unit rate in the web form.

3. **Database migrations run on every quote request** ([`src/db.py:L115`](file:///Users/deepak/Development/Jet%20Freight%20Logistics/Air%20Cargo%20Pricing%20Engine/src/db.py#L115) and [`src/app.py:L894`](file:///Users/deepak/Development/Jet%20Freight%20Logistics/Air%20Cargo%20Pricing%20Engine/src/app.py#L894))
   - **What this is:** A helper that reads recent competitor prices from SQLite to help sales negotiate.
   - **Problem:** Every time someone asks for a quote, `get_recent_competitor_rate_benchmark` runs `migrate_outcome_columns`. This connects to SQLite and executes `PRAGMA table_info` and `CREATE INDEX IF NOT EXISTS` on the live database.
   - **Fix:** Remove `migrate_outcome_columns` from the read query. Run it only once when the server boots up inside `init_history_db`.
   - **If we skip it:** When multiple sales coordinators quote at the same time, SQLite locks up and users get `sqlite3.OperationalError: database is locked`.

---

### Should fix

4. **1,272 lines of dead V2 legacy code that throw runtime crashes** ([`src/engine.py`](file:///Users/deepak/Development/Jet%20Freight%20Logistics/Air%20Cargo%20Pricing%20Engine/src/engine.py) and [`src/margin_framework.py`](file:///Users/deepak/Development/Jet%20Freight%20Logistics/Air%20Cargo%20Pricing%20Engine/src/margin_framework.py))
   - **What this is:** The old version 2 pricing engine and its lookup tables.
   - **Problem:** [`src/engine.py`](file:///Users/deepak/Development/Jet%20Freight%20Logistics/Air%20Cargo%20Pricing%20Engine/src/engine.py#L32) literally raises a `RuntimeError` if imported outside tests, and [`src/margin_framework.py`](file:///Users/deepak/Development/Jet%20Freight%20Logistics/Air%20Cargo%20Pricing%20Engine/src/margin_framework.py) still contains the old bug that penalized perishable foodstuff margins. The new engine ([`src/engine_v3.py`](file:///Users/deepak/Development/Jet%20Freight%20Logistics/Air%20Cargo%20Pricing%20Engine/src/engine_v3.py)) has completely replaced them.
   - **Fix:** Delete both files. Update [`src/train.py`](file:///Users/deepak/Development/Jet%20Freight%20Logistics/Air%20Cargo%20Pricing%20Engine/src/train.py) and the two old test files that still touch them.
   - **If we skip it:** New developers will waste time reading dead code, and test runs keep spitting out deprecation warnings.

5. **Built-in `PricingEngineV3.train()` is missing the outlier cap** ([`src/engine_v3.py:L168-L175`](file:///Users/deepak/Development/Jet%20Freight%20Logistics/Air%20Cargo%20Pricing%20Engine/src/engine_v3.py#L168-L175))
   - **What this is:** The training method inside the engine object used when `auto_train=True`.
   - **Problem:** The standalone script ([`scripts/train_v3_2025.py`](file:///Users/deepak/Development/Jet%20Freight%20Logistics/Air%20Cargo%20Pricing%20Engine/scripts/train_v3_2025.py#L129)) caps targets at 45.0% and weights recent deals. But `engine_v3.py` calls `model.fit(train_df, train_df["Margin_Percentage"])` on raw data with no cap and no weights.
   - **Fix:** Add `np.clip(y, 0.5, 45.0)` and the time-decay weights into `PricingEngineV3.train()`.
   - **If we skip it:** Anyone calling `engine.train()` from code will get an unclipped model that suffers from historical 10,000% outlier distortions.

6. **Missing check on tiny airline buy rates** ([`src/app.py:L758-L760`](file:///Users/deepak/Development/Jet%20Freight%20Logistics/Air%20Cargo%20Pricing%20Engine/src/app.py#L758-L760))
   - **What this is:** Logic that derives unit buy rate when a user only fills in the total buy cost.
   - **Problem:** If a user submits `buy = 40` for a `50 kg` shipment, line 759 sets `unit_buy_rate = 0.80 INR/kg`. This is impossible in air cargo, but the web layer does not reject it and sends it straight to the ML model.
   - **Fix:** Check `if unit_buy_rate < 10.0:` and return an HTTP 422 error asking the user to verify their airline cost.
   - **If we skip it:** Typos produce garbage predictions instead of prompting the user to correct their input.

---

### Nice to have

7. **Volumetric weight is computed twice on every quote** ([`src/app.py:L638`](file:///Users/deepak/Development/Jet%20Freight%20Logistics/Air%20Cargo%20Pricing%20Engine/src/app.py#L638) and [`src/preprocess.py:L268`](file:///Users/deepak/Development/Jet%20Freight%20Logistics/Air%20Cargo%20Pricing%20Engine/src/preprocess.py#L268))
   - **What this is:** Calculating chargeable weight from box length, width, and height.
   - **Problem:** `src/app.py` computes `(l * w * h * pcs) / 6000.0` by hand on line 638, and then calls `validate_shipment_physics` on line 687 which computes the exact same formula a second time.
   - **Fix:** Remove the duplicate inline calculation and let `validate_shipment_physics` do it once.
   - **If we skip it:** Unnecessary duplication; if the freight divisor ever changes (e.g., to 5000), one place will be missed.

8. **In-memory rate limiter table can grow between purges** ([`src/security.py:L238`](file:///Users/deepak/Development/Jet%20Freight%20Logistics/Air%20Cargo%20Pricing%20Engine/src/security.py#L238))
   - **What this is:** A sliding window that stops abusive scripts from spamming `/api/quote`.
   - **Problem:** Expired IP addresses are only pruned every 100 requests. An automated scanner rotating random client IPs can fill memory between pruning cycles.
   - **Fix:** Prune on a background timer or switch to an LRU dictionary structure.
   - **If we skip it:** Minor memory bloat under automated external scanning.

---

### Verdict
Mostly healthy. Fix **#2** (`app.py` 30 kg check) and **#3** (database lock under load) before scaling up user traffic.

### Lean
`-1,290 lines, -0 dependencies possible.` (Delete `src/engine.py` and `src/margin_framework.py`, and remove duplicate math in `app.py`).

### Not checked
The AWS Lambda deployment layer and cloud S3 artifact sync scripts were not run live against cloud endpoints.
