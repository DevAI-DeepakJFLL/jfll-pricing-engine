# Principal Audit Remediation & Hardening Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remediate all critical, high, and medium findings identified in the principal-level audit of the Air Export Pricing Engine across 4 strictly verified phases.

**Architecture:** 
1. Unify destination geography and fix collisions in `taxonomy.py` and `app.py`, ensuring train/serve feature consistency.
2. Harden security and authentication to fail-closed, securing all outcome and admin endpoints.
3. Replace heuristic and static components (fabricated win probabilities, pseudo-90% coverage strings, hardcoded signals) with empirical, calibrated, and auditable models (`WinModel`, `CQRCalibrator`/holdout coverage, counterfactual attributions).
4. Introduce robust input validation, transparent financial handling, thread-safe lazy engine initialization, and remove request-time database schema overhead.

**Tech Stack:** Python 3.11+, Scikit-Learn 1.9+, Flask, SQLite, Pandas, NumPy, Joblib, HMAC SHA-256.

**Spec:** Principal-Level Audit findings (Items 1.1–6.6).

## Global Constraints
- Finish and verify each phase completely before proceeding to the next phase.
- All existing tests and new regression tests must pass without errors.
- Never silently mutate financial figures; fail explicitly with HTTP 422.
- Preserve backward compatibility for legitimate API clients while rejecting malformed inputs.
- All file paths and references in documentation must be accurate and clean.

---

### Task 1: Unify Destination Taxonomy & Fix Collisions
**Files:**
- Modify: `src/taxonomy.py`
- Modify: `src/app.py`
- Modify: `src/engine_v3.py`
- Modify: `src/quantile_model.py`
- Test: `tests/test_taxonomy.py`, `tests/test_audit_remediation.py`

**Interfaces:**
- `map_destination_region(port: str) -> str`: Single source of truth for destination region mapping.
- `MarginQuantileModel.unseen_categorical_report(X: pd.DataFrame) -> Dict[str, str]`: Reports any out-of-vocabulary categories.

- [ ] **Step 1: Write failing tests for taxonomy collisions and unseen categorical reporting**
  In `tests/test_audit_remediation.py`, test:
  - San Salvador resolves to "Latin America" (not North America).
  - Atlanta, Georgia resolves to "North America" (not Europe).
  - Colombo, Dhaka, Kathmandu, Male resolve to "Indian Subcontinent".
  - Genoa and Goa resolve correctly without dictionary key collision.
  - `unseen_categorical_report` flags unseen categorical entries.
- [ ] **Step 2: Implement Patch A in `src/taxonomy.py`**
  - Delete duplicate `"GOA": "Europe"` entry.
  - Implement ambiguous suffix handling for "GEORGIA" (differentiating US state vs country).
  - Enforce word-boundary IATA matching only when whole field is the 3-letter code or properly isolated.
- [ ] **Step 3: Implement Patch B in `src/app.py`**
  - Replace `preprocess.map_dest_region` with `taxonomy.map_destination_region`.
  - Delete legacy `map_dest_region` from `src/preprocess.py`.
- [ ] **Step 4: Implement Patch C in `src/engine_v3.py` and `src/quantile_model.py`**
  - Always compute `regional_lane = f"{origin_region} -> {dest_region}"`.
  - Add `unseen_categorical_report` to `MarginQuantileModel`.
  - Add warning and set confidence to "Low" when out-of-vocabulary inputs are detected.
- [ ] **Step 5: Run tests and commit**
  - Run `pytest tests/test_taxonomy.py tests/test_audit_remediation.py`.
  - Commit as `fix(taxonomy): unify destination taxonomy and eliminate train-serve lane skew`.

---

### Task 2: Fail-Closed Security Controls & Endpoint Authentication
**Files:**
- Modify: `src/security.py`
- Modify: `src/app.py`
- Test: `tests/test_security_integrity.py`, `tests/test_audit_remediation.py`

**Interfaces:**
- `verify_artifact_integrity(filepath: str, checksums_path: Optional[str] = None) -> bool`: Fails closed in production if checksums missing.
- `require_admin_auth(f)`: Guards admin & outcome modification routes, rejects missing API key in production with HTTP 503, rejects query param tokens.

- [ ] **Step 1: Write failing tests for security fail-closed behavior**
  - `verify_artifact_integrity` raises `SecurityIntegrityError` if checksums.json is missing when `ALLOW_UNVERIFIED_ARTIFACTS!=1`.
  - `/api/history/outcome` and `/api/history/update_status` return HTTP 401 when unauthenticated.
  - Passing `?api_key=...` in query parameters is rejected.
- [ ] **Step 2: Implement fail-closed integrity and auth in `src/security.py`**
  - If `checksums.json` is missing and `ALLOW_UNVERIFIED_ARTIFACTS` is not `"1"`, raise `SecurityIntegrityError`.
  - In `require_admin_auth`, reject query parameter authentication (`request.args.get('api_key')` removed).
  - If `ADMIN_API_KEY` is unset in production (`APP_ENV=production`), return HTTP 503 Service Unavailable.
- [ ] **Step 3: Apply `@require_admin_auth` to `/api/history/outcome` and `/api/history/update_status` in `src/app.py`**
  - Validate that `id` is a valid integer, return 400 if invalid.
- [ ] **Step 4: Run tests and commit**
  - Run `pytest tests/test_security_integrity.py tests/test_audit_remediation.py`.
  - Commit as `fix(security): enforce fail-closed integrity checks and protect outcome endpoints`.

---

### Task 3: Weight-Break Arbitrage String Keys & Cost Floor Parity
**Files:**
- Modify: `src/tiers.py`
- Modify: `src/constants.py`
- Test: `tests/test_tiers.py`, `tests/test_guardrails.py`, `tests/test_audit_remediation.py`

- [ ] **Step 1: Write failing test for JSON string keys and 45kg boundary parity**
  - Pass `slab_rates={"100": 130.0}` to `check_weight_break_arbitrage` and verify real savings calculated.
  - Verify `is_estimate` is True when no carrier rates are supplied, and False when explicit rate is used.
  - Verify 45.0kg boundary consistency between `get_weight_tier` and `compute_cost_floor_price`.
- [ ] **Step 2: Update `src/tiers.py`**
  - Cast `slab_rates` keys and values to float: `{float(k): float(v) for k, v in (slab_rates or {}).items()}`.
  - Set `is_estimate = True` and prepend `"ESTIMATE (assumes 12% break discount)"` to advisory if estimated.
- [ ] **Step 3: Update `src/constants.py`**
  - Change `if wt <= 45.0:` to `if wt < 45.0:` in `compute_cost_floor_price`.
- [ ] **Step 4: Run tests and commit**
  - Run `pytest tests/test_tiers.py tests/test_guardrails.py tests/test_audit_remediation.py`.
  - Commit as `fix(tiers): normalize arbitrage slab rates and align weight tier boundary`.

---

### Task 4: Deduplicate Risk Buffer Logic
**Files:**
- Modify: `src/constants.py`
- Modify: `src/policy_guardrails.py`
- Test: `tests/test_policy_guardrails.py`, `tests/test_audit_remediation.py`

- [ ] **Step 1: Write test verifying unified risk buffer resolution**
  - Verify all commodity variants and company types resolve identical risk buffers in both modules.
- [ ] **Step 2: Implement canonical `resolve_risk_buffer` in `src/constants.py`**
  - Move the fuzzy keyword matching and GSA/NVOCC 50% discount into `resolve_risk_buffer`.
- [ ] **Step 3: Import `resolve_risk_buffer` in `src/policy_guardrails.py`**
  - Remove duplicate `_resolve_risk_buffer` function.
- [ ] **Step 4: Run tests and commit**
  - Run `pytest tests/test_policy_guardrails.py`.
  - Commit as `refactor(guardrails): unify risk buffer resolution in constants.py`.

---

### Task 5: Strict Inquiry Validation & Timezone Handling
**Files:**
- Create: `src/validation.py`
- Modify: `src/engine_v3.py`
- Modify: `src/app.py`
- Test: `tests/test_validation.py`, `tests/test_api_quote.py`

- [ ] **Step 1: Write unit tests in `tests/test_validation.py`**
  - Reject non-numeric or missing weight, zero/negative buy cost, out-of-range implied buy rate (<10 or >8000 INR/kg).
  - Require destination.
  - Parse quote date into IST (Asia/Kolkata) with strict window validation.
- [ ] **Step 2: Implement `src/validation.py`**
  - `QuoteValidationError(ValueError)`.
  - `validate_inquiry(inquiry_dict) -> Dict[str, Any]`.
- [ ] **Step 3: Wire into `src/engine_v3.py` and `src/app.py`**
  - Call `validate_inquiry` in `recommend_quote`.
  - Catch `QuoteValidationError` in `app.py` and return HTTP 422 with actionable message.
- [ ] **Step 4: Run tests and commit**
  - Commit as `feat(validation): introduce strict inquiry validation and IST date parsing`.

---

### Task 6: Transparent Financial Input Handling
**Files:**
- Modify: `src/app.py`
- Test: `tests/test_api_quote.py`

- [ ] **Step 1: Write test verifying no silent buy mutation**
  - If `buy/chargeable_wt < 10` for `wt >= 30` and `confirm_total_buy` is absent, return HTTP 422.
  - When `confirm_total_buy=true` is sent, accept the quotation.
- [ ] **Step 2: Update `src/app.py`**
  - Remove silent `buy = buy * wt` multiplication.
  - Return HTTP 422 requiring user confirmation or resubmission as `unit_buy`.
- [ ] **Step 3: Run tests and commit**
  - Commit as `fix(api): reject ambiguous financial inputs with HTTP 422 instead of silent mutation`.

---

### Task 7: Strict Physical Dimension Validation & NaN Density
**Files:**
- Modify: `src/preprocess.py`
- Modify: `src/app.py`
- Test: `tests/test_preprocess_validation.py`, `tests/test_api_quote.py`

- [ ] **Step 1: Write test for invalid dimensions and NaN density**
  - Non-positive dimensions or out-of-range pieces raise `ValueError`.
  - Express commodity uses 5,000 cm³/kg divisor.
  - When gross weight is omitted, `Cargo_Density_Ratio` is passed as `NaN` (handled natively by HGB).
- [ ] **Step 2: Update `src/preprocess.py` and `src/app.py`**
  - Strict dimension row validation in `compute_volumetric_weight`.
  - Support courier divisor in `validate_shipment_physics`.
  - Set `Cargo_Density_Ratio = np.nan` when gross weight is not provided.
- [ ] **Step 3: Run tests and commit**
  - Commit as `fix(physics): enforce strict dimension validation and native NaN density handling`.

---

### Task 8: Lazy Engine Initialization & Database Caching
**Files:**
- Modify: `src/engine_v3.py`
- Modify: `src/app.py`
- Modify: `src/db.py`
- Test: `tests/test_db_audit.py`, `tests/test_api_quote.py`

- [ ] **Step 1: Set `auto_train=False` default in `PricingEngineV3`**
- [ ] **Step 2: Implement thread-safe `get_engine()` in `src/app.py`**
- [ ] **Step 3: Cache schema migrations in `src/db.py`**
  - Use `_MIGRATED = set()` in `migrate_outcome_columns`.
- [ ] **Step 4: Run tests and commit**
  - Commit as `perf(serving): add thread-safe lazy engine loading and database migration caching`.

---

### Task 9: Implement Calibrated Monotonic WinModel
**Files:**
- Create: `src/win_model.py`
- Create: `scripts/train_win_model.py`
- Test: `tests/test_win_model.py`

- [ ] **Step 1: Implement `WinModel` in `src/win_model.py`**
  - `HistGradientBoostingClassifier` with `monotonic_cst` set to `-1` for `Margin_Percentage`.
  - Temporal split (train / calibration / test) on `all_inquiries_win_rate_v3.csv`.
  - `IsotonicRegression` probability calibration.
  - Compute ROC-AUC, Brier score, and ECE.
- [ ] **Step 2: Write `scripts/train_win_model.py` and save artifact**
- [ ] **Step 3: Unit test monotonic decrease of win probability with respect to margin**
- [ ] **Step 4: Commit as `feat(ml): implement calibrated monotonic win probability model`**

---

### Task 10: Wire WinModel & Profit-Maximizing Expected Profit into PricingEngineV3
**Files:**
- Modify: `src/engine_v3.py`
- Modify: `src/templates/index.html`
- Test: `tests/test_api_v3_quote.py`, `tests/test_forwarder_corridor_calibration.py`

- [ ] **Step 1: Integrate `WinModel` into `PricingEngineV3`**
  - Load `win_model` from bundle or standalone artifact.
  - Replace static `0.70 / 0.45 / 0.25` anchors with true predicted probabilities.
  - Calculate `Expected_Profit_INR = margin_amt * win_prob`.
  - Select balanced recommendation by maximizing expected profit.
- [ ] **Step 2: Run tests and commit**
  - Commit as `feat(engine): wire calibrated win model and expected profit optimization`.

---

### Task 11: Empirical Holdout Coverage & Release Gates
**Files:**
- Create: `src/backtest.py`
- Modify: `src/engine_v3.py`
- Test: `tests/test_conformal_calibration.py`

- [ ] **Step 1: Implement `measure_corridor_coverage` in `src/backtest.py`**
- [ ] **Step 2: Display empirical holdout coverage in `engine_v3.py`**
  - Replace static string `"90%"` with measured holdout percentage (e.g. `"89% (holdout, n=4,078)"`).
- [ ] **Step 3: Run tests and commit**
  - Commit as `feat(conformal): measure and report empirical holdout corridor coverage`.

---

### Task 12: Attribution Signals, Tier Collapse & Closed-Loop Feedback
**Files:**
- Modify: `src/engine_v3.py`
- Modify: `src/app.py`
- Modify: `src/db.py`
- Test: `tests/test_audit_remediation.py`

- [ ] **Step 1: Implement `_counterfactual_signals` in `PricingEngineV3`**
  - Compute model-based marginal effects (counterfactual commodity and category baselines).
- [ ] **Step 2: Detect and flag tier collapse**
  - If `(m_prem - m_floor) < 0.5`, warn user that cost floor dominates.
- [ ] **Step 3: Closed-loop outcome feedback**
  - In `api_history_outcome`, update in-memory `pit_registry` with won/lost quote results.
- [ ] **Step 4: Run tests and commit**
  - Commit as `feat(engine): add counterfactual attribution signals and closed-loop registry feedback`.

---

### Task 13: Shield Legacy Modules
**Files:**
- Modify: `src/engine.py`
- Modify: `src/train.py`
- Modify: `src/margin_framework.py`
- Test: `tests/test_scenarios_t1_t15.py`

- [ ] **Step 1: Add `ALLOW_LEGACY_ENGINE` guard to legacy modules**
- [ ] **Step 2: Run tests and commit**
  - Commit as `chore(legacy): guard superseded v2 modules with ALLOW_LEGACY_ENGINE`.

---

### Task 14-17: Audits, Reviews & Final Cleanup Pass
- [ ] Run Ponytail Audit (`ponytail-audit`).
- [ ] Run Security Audit (`senior-security`).
- [ ] Run Code Review (`code-reviewer`).
- [ ] Run Code Refactor Pro (`code-refactor-pro`).
