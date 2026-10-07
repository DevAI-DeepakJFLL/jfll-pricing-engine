# Codebase Review Report: Air Export Pricing Recommendation Engine

**Scope:** `README.md`, `src/{__init__,constants,tiers,preprocess,train,engine,db,app}.py`, and `Air_Export_Pricing_Combined_ML.csv` (24,665 rows × 37 columns).
**Review date:** 4 Oct 2026 | **Model version string in code:** `v2.2-panindia-20260928`

**Finding labels used throughout:** `Confirmed issue` · `Probable issue` · `Potential risk` · `Missing information / unable to verify`
**Effort:** S (<1 day) · M (days) · L (weeks) | **Priority:** P0 (do now) · P1 (this sprint) · P2 (this quarter) · P3 (later)

> **Evidence basis.** The trained artifact (`freight_margin_recommender.joblib`), the raw Excel file, `requirements.txt`, `run.sh`, and the `docs/` file were not supplied. To verify behaviour I (a) profiled the supplied CSV and (b) **re-ran the `train.py` recipe** (same features, hyperparameters, seeds, and the repo's own `engine.py`) on that CSV, using `is_won == 1` rows as the won-benchmark set. Numbers marked *(replicated)* come from that re-run. Production numbers will differ slightly, but the structural findings are properties of the code and data, not of a specific training run.

---

## 1. Executive summary

### What the application does
A Flask app (`app.py`) takes shipment details (origin, destination, **a single "Airline Buy Cost" number**, weights, client category, commodity, incoterms, branch, account frequency, revision count) and returns three quote options (Competitive Floor / Best-Recommended Balance / High-Margin Premium), a hero "recommended selling price", a win probability, and a margin-vs-win-probability table. Internally (`engine.py`) it runs a two-stage ML pipeline:

1. **Stage 1A:** a gradient-boosted regressor predicts a "market benchmark margin %" trained **only on won deals**.
2. **Stage 1B:** a calibrated gradient-boosted classifier predicts P(win) with a monotone-decreasing constraint on `Margin_Ratio = margin / benchmark`.
3. **Stage 2:** a grid search over margins (0.5% steps) maximizes expected profit = margin₹ × P(win), restricted to a corridor around the benchmark.

Every call is logged to SQLite (`db.py`) with a manual lead-status tracker.

### Why it is producing improper suggestions (top reasons, all evidenced)

| # | Reason | Label | Severity |
|---|---|---|---|
| 1 | **Price elasticity is fabricated.** The trained classifier is essentially flat in margin (P ≈ 0.24 from 1% to 9% margin, then 0.22). All downward slope in win probability comes from a hard-coded formula in `engine.py` (`logit − 1.5·excess^1.15`). The "optimizer" therefore optimizes a made-up curve. | Confirmed | Critical |
| 2 | **Historical data says the opposite of the model's assumption.** In the CSV, win rate *rises* with margin (4.5% at ≤2% margin → 77% at 10–15% → 91% at >30%). Lost quotes have a median margin of 2.05% vs 7.03% for won. The most likely explanation is that the "Total Sell" on lost quotes is not a real customer-facing offer. The monotone `-1` constraint contradicts the data, so the model learns nothing about price. | Confirmed (pattern) / Probable (cause) | Critical |
| 3 | **Outcome-leaking and non-commercial features drive win probability:** `quote_revision_count` (win rate 24% at 1 revision vs 72–89% at ≥2), `Booked by branch` (Delhi 16%, Ahmedabad 99%), `Company_Group` (Overseas Agent 97%). The UI also defaults revision count to **0**, a value that never occurs in training (minimum is 1). | Confirmed | Critical |
| 4 | **Airline buy cost is the dominant feature of the margin benchmark** (permutation importance 0.587 R² drop vs 0.223 for the next feature *(replicated)*). It is a raw ₹ total that mixes shipment size with rate level, with unknown composition, no FX conversion, and no per-kg or lane-relative normalization. | Confirmed | High |
| 5 | **The "benchmark" is the company's own past pricing behaviour on won deals,** with CV MAE ≈ 4.9 margin points *(replicated)*, which is larger than the typical predicted benchmark (3–6%). The recommended margin sits at the benchmark by construction. | Confirmed | High |
| 6 | **The three "strategies" are fixed offsets.** Floor ≈ balanced − 2.5 pts, premium ≈ balanced + 3.0 pts. `STRATEGY_THRESHOLDS` and `min_win_prob` are computed but never used. | Confirmed | High |
| 7 | **No cost floor, no operational cost, no risk buffer, no carrier/DG/surcharge/weight-break logic.** The candidate grid starts at 0.5% markup. | Confirmed | High |
| 8 | **No customer-, lane-, or cohort-specific historical lookup.** `Customer_Company` is dropped from features and not an input; there is no "max margin ever accepted for similar shipments" guardrail. | Confirmed | High |
| 9 | **Temporal blindness.** Data covers only Jan–Sep 2026; the split is random; today (Oct) and Q4 peak are outside the training window. Temporal-holdout AUC is 0.829 vs 0.930 random *(replicated)*. | Confirmed | High |

### Highest-priority fixes
1. **Stop showing the current win probability and "optimal" labels as calibrated** until F-01/F-02 are resolved (add a "heuristic / low confidence" banner). *(S)*
2. **Resolve what `Total Sell` means for Lost quotes** with the sales team, and rebuild labels from final-quote-at-outcome only. *(M)*
3. **Remove `quote_revision_count`, `Booked by branch`, and `Customer_Inquiry_Frequency` as raw features;** replace with point-in-time customer history features. *(M)*
4. **Add hard guardrails now:** ₹ minimum margin per AWB and per kg, cohort-based max/P90 cap, and a "recommendation unreliable" fallback. *(S–M)*
5. **Replace raw `Total_Buy_INR` dominance** with rate/kg, lane-relative rate index, and a margin-per-kg target. *(M)*
6. **Delete or implement** the dead strategy-threshold logic. *(S)*

---

## 2. Architecture and codebase map

### 2.1 README review (read first)
**Stated purpose:** an ML system recommending optimal margins and final quoted selling prices in air-freight forwarding. **Stack:** Python, pandas, scikit-learn, Flask, SQLite (SQLite is not mentioned in the README). **Setup:** `pip install -r requirements.txt` → `preprocess.py` → `train.py` → `app.py` on `127.0.0.1:5050`. **Data source:** one Excel file. **Users:** implied pricing/sales staff. **Business rules stated:** none.

| README gap / contradiction | Label |
|---|---|
| Tree lists only `engine/preprocess/train/app`; omits `constants.py`, `tiers.py`, `db.py`, and the SQLite history store | Confirmed |
| `preprocess.py` writes `Air_Export_Pricing_Cleaned_ML.csv` / `..._Won_Benchmark.csv`, but `train.py` defaults to `Air_Export_Pricing_Combined_ML.csv` / `..._Combined_Won_Benchmark.csv`; the "Combined" CSV also contains `Month_Name` and `Year_Month`, which `preprocess.py` never creates. **A merge step exists but is undocumented.** | Confirmed |
| No description of the two-stage model, strategies, API endpoints (`/api/quote`, `/api/history/*`), objective, or limitations | Confirmed |
| References `requirements.txt`, `run.sh`, `docs/AIR_EXPORT_PRICING_ML_DOCUMENTATION.md`; none supplied | Missing information |
| No data dictionary, no definition of "Total Buy", "Total Sell", or what "Lost" means; no tests; no deployment notes; no Python/sklearn version pin (though `app.py` hard-fails on sklearn version skew) | Confirmed |
| "Optimal" is used throughout without defining the objective | Confirmed |

### 2.2 Responsibility map

| Layer | Files | Notes |
|---|---|---|
| Frontend | `app.py` (`HTML_TEMPLATE`, `HISTORY_TEMPLATE`, ~1,000 lines of HTML/CSS/JS inside Python) | Searchable dropdowns, presets, strategy cards, sensitivity table |
| Backend / API | `app.py` (`/`, `/api/quote`, `/history`, `/api/history/update_status`, `/api/history/clear`, `/api/history/export`) | No auth, Flask dev server |
| Business logic / decisioning | `engine.py` | Optimizer, strategy extraction, retention penalty |
| Data processing | `preprocess.py`, `tiers.py` | Dedup, cleaning, regions, commodity map, tiers |
| Model training | `train.py` | Stage 1A/1B training, diagnostics, joblib export |
| Persistence | `db.py` (SQLite) | History + lead status, schema versioning |
| Configuration | `constants.py` | Margin bounds, strategy thresholds (partly dead), model version |
| Tests / CI / deploy | **none** | Confirmed absent |

### 2.3 Execution flow
```
Excel ──preprocess.py──► Cleaned_ML.csv ──(undocumented merge → "Combined")──► train.py ──► joblib
                                                                                              │
Browser form ─► app.py /api/quote ─► normalize + tier ─► engine.optimize_quote ◄──────────────┘
                                                              │
                          benchmark_reg → candidate grid → classifier → hard-coded decay → EV → 3 cards
                                                              │
                                              db.log_recommendation (balanced card only)
```

### 2.4 Architectural weaknesses
- **No cost-estimation layer.** The user types one "Airline Buy Cost" number; the system cannot tell airline base freight, surcharges, and local charges apart. Cost and commercial margin are not separated (Confirmed).
- **No cohort/historical-lookup layer;** all history is compressed into two black-box models (Confirmed).
- **Training/inference skew risks:** units, defaults, distributions (see F-03, F-07, F-21).
- **Non-deterministic inference:** `_enrich_temporal_features` uses `datetime.now()`; quote date is not an input (Confirmed).
- **Feedback loop is open:** Won/Lost statuses in SQLite never feed retraining (Confirmed).
- **Monolithic `app.py`;** catalogs, templates, API, and validation in one file (Confirmed, Low).
- **Dead code:** `STRATEGY_THRESHOLDS`, `min_win_prob`, `get_all_history` (only used by export), `selectStrategyCard` branch for `res_exp_val` element that does not exist, `PORT_LABELS` duplicated as ORIGIN_OPTIONS labels.

---

## 3. File-by-file review

### 3.1 `src/constants.py`: Low risk, high leverage
- **Purpose:** margin bounds, `MODEL_VERSION`, `STRATEGY_THRESHOLDS`.
- **Problems:**

| ID | Label | Sev | Problem / root cause | Impact | Fix | Prio/Effort |
|---|---|---|---|---|---|---|
| C-1 | Confirmed | Medium | `STRATEGY_THRESHOLDS` (floor 0.40, balanced 0.25, premium 0.15) is imported but its values never constrain any output (see F-09). | Config suggests guardrails that do not exist. | Wire into selection or delete. | P1 / S |
| C-2 | Confirmed | High | `MIN_CANDIDATE_MARGIN = 0.5` % is the only floor; no ₹ floor, no cost-based floor. | Quotes can be sent at near-cost on any shipment. | Replace with segment-level min % **and** min ₹/AWB and ₹/kg. | P0 / S |
| C-3 | Confirmed | Low | Model version `v2.2-panindia-20260928` vs `v2.0` in `train.py` banner, `db.py` defaults, DB column default. | Audit trail ambiguity. | Single version source. | P2 / S |
| C-4 | Potential risk | Medium | All margin parameters are hard-coded constants, with no per-segment overrides. | Cannot tune per lane/commodity without redeploy. | Move to versioned config (YAML/DB) with owner and effective date. | P2 / M |

### 3.2 `src/tiers.py`
- **Purpose:** weight tiers (45/100/300/500/1000 kg) and account tiers.
- `get_weight_tier` and `pd.cut(right=False)` are consistent. Breaks match standard IATA rate breaks (good).

| ID | Label | Sev | Problem | Fix | Prio/Effort |
|---|---|---|---|---|---|
| T-1 | Confirmed | High | `map_account_tier` thresholds (≥20 Key, ≥6 Regular, ≥2 Occasional) were applied to a whole-dataset inquiry count over ~9 months: **79.5% of rows (19,609/24,665) are "Key Account"**. The UI field "Account Inquiries / Yr" (default 10 → "Regular") is a different unit. | Replace with revenue/GP/recency-based segmentation computed point-in-time; document units. | P1 / M |
| T-2 | Potential risk | Medium | Tiers are labels only; no break-point logic (e.g., 95 kg billed at the 100 kg rate when cheaper). | Add break-point check as a pricing rule. | P2 / M |

### 3.3 `src/preprocess.py`
- **Purpose:** ingestion, dedup, parsing, outlier pruning, port/commodity/region mapping, features.
- **Key logic:** keep one quote per inquiry by `status_rank` (Won < Approved < Draft < Lost < Hold < Cancelled) then latest; require `Buy > 500`, `Sell ≥ Buy`, margin in [0.2%, 60%]; derive weights, density, regions, tiers, season; label `is_won = (Quote Status == "Won")`.

| ID | Label | Sev | Problem / root cause | Impact | Fix | Prio/Effort |
|---|---|---|---|---|---|---|
| P-1 | Confirmed | High | **Label contamination:** 1,556 rows (6.3%) are Draft (1,339), Cancelled (184), Hold (24), Approved (9) and are labelled `is_won = 0` as if lost. `Draft` ranks above `Lost` in `status_priority`, so an inquiry with both keeps the Draft. 7 rows are `Lost` quotes on inquiries whose Inquiry Status is Won. | Teaches the classifier that unresolved/cancelled = price loss. | Train only on resolved outcomes (Won/Lost, optionally Cancelled as separate class); reconcile with `Inquiry Status`. | P1 / S |
| P-2 | Confirmed | High | `parse_currency_field` strips `USD/EUR/GBP/CAD` and keeps the number: **no FX conversion** while columns are named `*_INR`. rate/kg 1st–99th percentile is ₹47–₹7,404, and the upper end is not credible as pure airline freight. | Mixed-currency or non-airline charges treated as INR airline cost. | Parse currency, convert at quote-date FX, or reject. | P0 / S–M (**Probable**: cannot confirm currency mix without raw data) |
| P-3 | Confirmed | High | No outlier handling on weights/buy except Buy>500: Chargeable weight max = 1.88×10¹² kg; 63 rows >20,000 kg; 65 rows ≤1 kg; 852 rows with gross > chargeable (impossible under IATA definition). `fillna(1.0)` injects 1 kg shipments. | Corrupt feature space; tree splits on garbage; rate/kg = inf for some rows. | Validate physical plausibility (chargeable ≥ gross, kg bounds by commodity), quarantine rows. | P0 / S |
| P-4 | Confirmed | High | Rows with Sell < Buy or margin <0.2% are dropped (survivorship). These are exactly "won but unprofitable" deals the business needs to learn from. | Model never sees loss-making wins; benchmark biased upward. | Keep them, flag as `below_floor_win`, model realized margin separately. (Cannot quantify without raw data.) | P1 / S |
| P-5 | Confirmed | Medium | Silent defaults: missing POL → Delhi, POD → Dubai, Incoterms → FOB, vertical → "Air Export Forwarding", company group → "Subagent". 95.7% of rows are FOB; 98.2% one vertical. | Imputation manufactures structure; two features carry almost no information. | Use explicit `Unknown`, drop constant-like features. | P2 / S |
| P-6 | Confirmed | High | Destination mapping misses: **15.5% of rows (3,817) are "Other Destination"**, e.g., Tashkent (111), Mauritius SSR (103), Bruxelles (73), Praha (62), Durban (58), Bucharest (56), Kabul (56), Lisboa (54), Dorval/Montreal (50), Kansai (49), Casablanca (49), Porto (49), Fiumicino (48), Cape Town (45), St Petersburg (45). | Regional lanes (the model's density fix) are wrong for 1 in 6.5 inquiries. | Replace substring heuristics with IATA/UN-LOCODE reference table. | P1 / M |
| P-7 | Potential risk | Medium | Substring matching in `map_commodity_group` (`"ENG"`, `"TOOL"`, `"AUTO"`, `"TIRE"`, `"MEAT"`, `"VEG"`) can misclassify (e.g., words containing "ENG"). **Dangerous goods, temperature control, live animals, valuables and oversized have no category;** DG would fall into "Other" or "General". 92.0% of rows are "General Cargo". | Special-handling cargo priced like general cargo. | Structured fields (DG class/UN no., temp range, ULD/dimensions) or controlled vocabulary. | P1 / M |
| P-8 | Confirmed | Medium | `Customer_Inquiry_Frequency = value_counts()` over the whole cleaned dataset: look-ahead (includes future and current inquiries) and window-dependent. | Leakage, unit mismatch with inference (see F-07). | Point-in-time rolling counts. | P1 / M |
| P-9 | Confirmed | Medium | `quote_revision_count` counts all quotations of the inquiry, including those made after customer engagement; strongly outcome-correlated (see F-03). | Leakage. | Use revisions *prior to this quote* or drop. | P0 / S |
| P-10 | Confirmed | Low | Origin ports like "Decimomannu", "Amalfi" appear in an India-export dataset. | Data-entry errors reach the model. | Validate origin ∈ Indian airport list. | P2 / S |

### 3.4 `src/train.py`
- **Purpose:** train Stage 1A regressor, Stage 1B calibrated classifier; export joblib.

| ID | Label | Sev | Problem / root cause | Impact | Fix | Prio/Effort |
|---|---|---|---|---|---|---|
| TR-1 | Confirmed | Critical | `monotonic_cst[-1] = -1` on `Margin_Ratio` forces a decreasing relationship that the data contradicts (see F-01/F-02). With the constraint, AUC with Margin_Ratio (0.9304) equals AUC without it (0.9303) *(replicated)*, so the feature is effectively ignored. | Model carries no price signal. | Fix labels first; only then re-impose monotonicity on a *valid* price variable. | P0 / M |
| TR-2 | Confirmed | Medium | `Margin_Ratio` for won rows is computed with a regressor **trained on those same rows** (in-sample), while lost rows are out-of-sample. Won mean/std 0.942/0.447 in-sample vs 1.010 out-of-fold *(replicated)*. | Train/inference mismatch; optimistic discrimination. | Out-of-fold benchmark predictions for all rows. | P1 / S |
| TR-3 | Confirmed | High | **Random 80/20 split** with many rows per customer and 9 months of data. Temporal holdout (train Jan–Jul, test Aug–Sep): AUC 0.829 vs 0.930 random *(replicated)*. | Reported AUC overstates real-world performance by ~0.10. | Time-based + customer-grouped validation. | P1 / S |
| TR-4 | Confirmed | Medium | `HistGradientBoosting*` is given ordinal-encoded categories **as numeric** (no `categorical_features`); alphabetical order is meaningless for 583 destination ports and 1,556 lanes. Unknown values map to −1 silently. | Spurious splits; unseen categories get arbitrary treatment. | Use native categorical support or target/frequency encoding with unknown-category fallback and a confidence flag. | P2 / S |
| TR-5 | Confirmed | Medium | Evaluation = AUC/Brier only. "§1.5 segmented check" takes the top-3 verticals, but 98% are one vertical. No calibration by margin bucket, no profit backtest, no lift vs naive baselines. | No evidence the recommendations make money. | Add backtest (see §11 and §10.6). | P1 / M |
| TR-6 | Confirmed | Medium | Redundant/proxy features: `Origin_Port`, `Origin_Region`, `Regional_Lane`, `Booked by branch`; `Customer_Inquiry_Frequency` + `Customer_Tier`; weights ×4. | Collinearity, unstable importances. | Feature pruning with permutation/ablation. | P2 / S |
| TR-7 | Confirmed | Low | Defaults point to `Combined_*` files that `preprocess.py` does not create; banner says "v2.0"; no hyper-parameter search; encoder fit on full data. | Reproducibility. | Fix pipeline contract. | P2 / S |

### 3.5 `src/engine.py`
- **Purpose:** inference-time optimization and strategy extraction.

| ID | Label | Sev | Problem / root cause | Impact | Fix | Prio/Effort |
|---|---|---|---|---|---|---|
| E-1 | Confirmed | Critical | Hard-coded elasticity: `decayed_logits = logits - 1.5 * (excess_ratios ** 1.15)`. Replicated example (DEL→LHR, 400 kg, buy ₹120k, bench 3.3%): raw model P is 0.240 at 1.2–8.7% margin and 0.217 above 11%; after decay, P = 0.240 → 0.217 → 0.080 → 0.022 → 0.005 → ≈0 at 16%+. | All displayed win probabilities and the argmax are driven by an uncalibrated constant. | Learn elasticity from valid data or run price experiments; remove overlay or fit its parameters. | P0 / M |
| E-2 | Confirmed | High | `min_win_prob` is assigned from `thresholds` and **never used** (3 occurrences, all in the assignment). `pricing_strategy` only selects which pre-computed card is "optimal"; the UI always sends `'balanced'`. | Strategy floors in docs/constants are fictional. | Implement as constraint or delete. | P1 / S |
| E-3 | Confirmed | High | Floor = nearest-to-benchmark margin ≤ balanced − 2.5; Premium = first/mid candidate ≥ balanced + 3.0. In the sweep (§5.3) premium − balanced = 3.0–3.5 pts and balanced − floor = 2.5–3.0 pts in every row. | Three options are one number ±constant, irrespective of customer, lane or risk. | Derive tiers from cohort quantiles and win-prob targets. | P1 / M |
| E-4 | Confirmed | High | Balanced = EV argmax inside corridor `[0.85·bm, max(1.5·bm, bm+4.5)]`. The "optimization" is confined to a narrow band around a regressor output, so the answer ≈ the benchmark. | Descriptive ("what we usually earned on won deals") not prescriptive. | Optimize over the full feasible range with valid P(win|margin); use corridor only as guardrail. | P1 / M |
| E-5 | Confirmed | High | Objective `E[profit] = margin₹ × P(win)` uses gross markup, not contribution after operations/risk costs; no volume/BSA/allotment value; no retention value. | Over-values low-touch big shipments, under-values small ones with fixed handling cost. | Contribution margin objective. | P1 / M |
| E-6 | Confirmed | Medium | `RetentionGuardrailConfig`: threshold 15 inquiries; in training, ~83% of rows already exceed it, and the penalty applies only above 1.8× benchmark, where P ≈ 0 after decay. | Inert guardrail that looks like relationship management. | Replace with real customer-value adjustment (§10). | P2 / S |
| E-7 | Confirmed | Medium | `datetime.now()` fills month, sin/cos, month-end, weekday. No `quote_date` input. Training months are 1–9 only. | Non-reproducible, untestable; **today (Oct) is outside the training window; Q4 "Global_Holiday_Surge" has zero observations.** | Accept `quote_date`; refuse/flag out-of-window. | P0 / S |
| E-8 | Confirmed | High | Benchmark clipped to [1, 48]; candidate range = [0.35·bm, max(36, 2.8·bm)] capped at 50%. Table always shows margins up to ≥36%, which no airfreight customer accepts for most lanes. | Clutter and implied legitimacy of absurd margins. | Cap by cohort P90. | P1 / S |
| E-9 | Confirmed | High | Unseen categorical values are encoded −1 with no warning or confidence downgrade. | Silent garbage-in for new ports/branches. | Return `confidence: low` + fallback to cohort rule. | P1 / S |

### 3.6 `src/app.py`

| ID | Label | Sev | Problem / root cause | Impact | Fix | Prio/Effort |
|---|---|---|---|---|---|---|
| A-1 | Confirmed | Critical | Form default `revision = 0` (`min=0`), payload `quote_revision_count`. Training minimum is **1** (17,621 rows at 1). Revision count is also not known when pricing a first quote. | Every default quote is out-of-support; the leakiest feature is user-controlled. | Remove feature (P0). | P0 / S |
| A-2 | Confirmed | High | `Booked by branch` is a user-selectable feature that changes P(win) via data-capture artefacts (win rates: Ahmedabad 98.7%, Hyderabad 95.6%, Bangalore 80.6%, Mumbai 35.4%, Delhi 15.7%). | Changing a dropdown changes the "win rate". Gameable. | Remove or replace with real cost/market variables. | P0 / S |
| A-3 | Confirmed | Medium | Server trusts client-provided `chargeable_wt`; no server-side recompute from dimensions, no `chargeable ≥ gross`, no 0.5 kg rounding. Density clamp [0.01, 10] vs training clip [0.1, 5]; UI "Heavy Dense Cargo" (>1.25) cannot occur by definition. | Inconsistent inputs. | Compute server-side. | P2 / S |
| A-4 | Confirmed | Medium | Only the **balanced** card is logged, even if user picks another; no customer, inquiry ref, user, final quoted price, override reason. Presets and exploratory clicks also write to history and inflate "Pipeline" and "Conversion Rate". | Audit trail and KPIs unreliable; no learning data. | Log final decision, user, customer, strategy, override. | P1 / M |
| A-5 | Confirmed | Medium | No authentication/authorization; `POST /api/history/clear` deletes all history without auth/CSRF; `int(quote_id)` unguarded → HTTP 500 on bad input; Flask dev server. | Operational/security risk if exposed beyond localhost. | Auth, RBAC, soft-delete, WSGI server. | P2 / M |
| A-6 | Confirmed | Low | Presets labelled "Quick **Historical** Test Presets" are invented inputs, not real historical quotes; no backtest harness. | False reassurance. | Replace with real anonymized historical cases. | P2 / S |
| A-7 | Confirmed | Low | "Margin %" is markup on buy `(sell−buy)/buy`, not gross margin on sell. Finance/sales may read it differently. | Misinterpretation. | Label explicitly or show both. | P2 / S |
| A-8 | Confirmed | Low | `~1,500-line` single file with embedded HTML/CSS/JS and static catalogs. | Maintainability. | Split into templates, services. | P3 / M |

### 3.7 `src/db.py`
- **Purpose:** SQLite audit store and lead statuses. Well structured (schema versioning, indexes, pagination).

| ID | Label | Sev | Problem | Fix | Prio/Effort |
|---|---|---|---|---|---|
| D-1 | Confirmed | Medium | Schema lacks customer, inquiry/quote number, airline/carrier, strategy chosen, final sell, override reason, outcome timestamp, realized margin, loss reason. The Won/Lost status is the most valuable training signal and is not connected to retraining. | Add columns + export job feeding `preprocess`. | P1 / M |
| D-2 | Confirmed | Low | `clear_history_table` is a hard delete; `get_all_history` loads the full table. | Soft delete, streaming export. | P3 / S |
| D-3 | Confirmed | Low | `model_version` default `'v2.0'` in DDL vs `v2.2…` constant. | Align. | P3 / S |

### 3.8 `src/__init__.py`: trivial, no issues. Missing: **tests, CI, requirements, run script, docs** (Missing information; Medium).

---

## 4. Business-logic review

### 4.1 Current logic, step by step
1. **Inputs** (`app.py`): route, one total "Airline Buy Cost", weights/dimensions (UI computes `L×W×H×pcs/6000`), client category, vertical, commodity, incoterms, "Account Inquiries/Yr", branch, revision count.
2. **Derivation** (`api_quote`): canonical ports, regions, regional lane, weight tier, account tier, density.
3. **Benchmark margin %** = HGB regression on won deals (target clipped 1–48%).
4. **Margin grid:** `[max(0.5, 0.35·bm) … min(50, max(36, 2.8·bm))]` step 0.5.
5. **P(win)** per margin from the calibrated classifier (feature `Margin_Ratio = margin/bm`), then logit decay `−1.5·excess^1.15`.
6. **Expected profit** = `buy × margin% × P`; ×0.85 above 1.8·bm if the account has >15 inquiries.
7. **Cards:** Balanced = max EV within corridor; Floor = ≈ balanced−2.5 pts; Premium = ≈ balanced+3 pts.
8. **UI** shows balanced as hero, logs it to SQLite.

**Actual objective being optimized:** *expected markup-₹ per inquiry within a band around the historical won-deal margin, under a hand-shaped demand curve.* It is not conversion-adjusted contribution profit, not realized margin, and not customer lifetime value.

**README vs. reality:** README promises "optimal profit margins and final quoted selling prices". Reality: a benchmark echo with synthetic elasticity and fixed ±offsets.

### 4.2 Air-export factors: coverage

| Factor | Present? | Evidence | Effect of absence |
|---|---|---|---|
| Dynamic airline base rate (₹/kg, by carrier/route/date) | **No**: single total buy | `app.py` `buy` | Cannot tell a rate spike from a bigger shipment (§5) |
| Carrier, service level, transit time, routing | **No** | not in `feature_cols` | Same lane, different carrier = same margin |
| Chargeable/actual/volumetric weight, density | Partial | `Chargeable_Weight_Kg`, ratio (clip 0.1–5) | OK; no 0.5 kg rounding, no dims in model |
| Weight-break slabs | Label only | `tiers.py` | No break-point arbitrage, no step-aware pricing |
| Commodity & special handling (DG, perishable, temp, live animals, valuables, oversize) | **Coarse** (8 buckets, DG absent) | `map_commodity_group` | DG/pharma/high-risk priced like general cargo |
| FSC/SSC, screening, THC, AWB, documentation, pickup, customs, destination charges | **No** | not modeled | No cost floor, no ops allocation |
| Capacity, seasonality, holidays, volatility | Weak: month sin/cos + 4 season labels, 9 months only | `get_air_cargo_season` | Q4 peak never learned; labels are assumptions, not validated |
| Contract/allotment vs spot | **No** | — | Cannot price BSA-backed vs spot cargo differently |
| Customer type, history, negotiation power, credit, strategic value | Weak: `Company_Group`, a frequency integer | `Customer_Inquiry_Frequency` | No customer-specific behaviour (§7) |
| Competitor pressure | **No** | — | Win model cannot see market |
| Minimum profit thresholds / risk-adjusted margin | **No** | `MIN_CANDIDATE_MARGIN = 0.5` | Loss-making/near-cost recommendations possible |
| Operational/claim/exception cost | **No** | — | Complex shipments under-priced |

### 4.3 Incorrect or weak assumptions
- **Assumption: higher margin → lower win probability, with elasticity coefficients 1.5 / 1.15.** Unvalidated; data shows the reverse (§6).
- **Assumption: won-deal margin = "market-clearing" margin.** It is the company's own policy outcome (e.g., Overseas Agent won median 3.0%, Transporter 18.0%), not market price discovery.
- **Assumption: a percentage markup is the right pricing variable.** The data shows median absolute margin ≈ ₹1.1k on ≤₹17k buys and ≈ ₹25k on >₹562k buys; real forwarders price with a minimum per AWB plus ₹/kg.
- **Assumption: "Total Buy" ≈ airline base freight.** Unverified (rate/kg tails to ₹7.4k; no FX).
- **Assumption: all non-Won = lost on price.** 1,556 rows are not resolved losses; Lost reasons are explicitly excluded as "leakage" but never analysed.

---

## 5. Airline-base-price weighting analysis

### 5.1 Where airline buy enters (all `Confirmed`)
| Entry point | Code | Transformation |
|---|---|---|
| UI `buy` | `app.py` → `inquiry_payload["Total_Buy_INR"]` | Raw ₹, no per-kg, no log, no lane-relative index |
| Stage 1A feature | `train.py` `feature_cols[0]` | Raw (trees are scale-invariant, so "scaling" is not the problem; **representation** is) |
| Stage 1B feature | `clf_features = feature_cols + [Margin_Ratio]` | Raw, plus indirectly via `Margin_Ratio` |
| Output scaling | `engine.py` `margin_amounts = base_buy * (margin/100)` | Linear: sell = buy × (1+m) |
| Cost floor | none | — |

### 5.2 Is it over-weighted? **Yes** *(replicated)*
Permutation importance, Stage 1A benchmark regressor (R² drop on held-out won deals):

| Feature | R² drop |
|---|---|
| **Total_Buy_INR** | **0.587** |
| Customer_Inquiry_Frequency | 0.223 |
| quote_revision_count | 0.055 |
| Booked by branch | 0.050 |
| Destination_Port | 0.046 |
| Origin_Port | 0.040 |
| Gross_Weight_Kg | 0.038 |
| Chargeable_Weight_Kg | 0.029 |
| Company_Group | 0.020 |

Buy cost alone outweighs the combined importance of the next eight features (≈0.50). Commodity, incoterms, vertical, weight tier and density are effectively absent from the top list. Importance is **learned, not configured**, and arises because (a) the historical pattern is "small ₹ buys carry large % margins" (corr(log buy, margin%) = −0.37; corr(log buy, log margin₹) = +0.75), (b) buy mixes weight and rate, and (c) there is no cost-structure feature to explain the margin otherwise. In the classifier, buy cost matters little (AUC drop 0.0036 when removed), which is itself evidence that the win model is dominated by other, worse features (§6).

Historical median margin by buy-cost octile (CSV):

| Buy cost octile (₹) | Median margin % | Median margin ₹ | Win rate |
|---|---|---|---|
| ≤17k | 11.1% | 1,108 | 57% |
| 17k–36k | 5.3% | 1,341 | 46% |
| 36k–65k | 3.7% | 1,804 | 42% |
| 65k–108k | 2.9% | 2,400 | 36% |
| 108k–168k | 2.8% | 3,760 | 39% |
| 168k–277k | 2.5% | 5,538 | 35% |
| 277k–562k | 2.3% | 9,452 | 29% |
| >562k | 2.1% | 25,040 | 19% |

### 5.3 Output sensitivity to buy cost *(replicated, repo `engine.py`)*
Same shipment (400 kg, DEL→LHR, Shipper, FOB, freq 10, revision 0, Delhi branch); only buy cost changes:

| Buy ₹ | ₹/kg | Benchmark % | Floor % | **Balanced %** | Premium % | Balanced P(win) | Balanced margin ₹ |
|---|---|---|---|---|---|---|---|
| 20,000 | 50 | 17.2 | 15.5 | **18.0** | 21.0 | 0.13 | 3,600 |
| 40,000 | 100 | 9.5 | 7.3 | **9.8** | 12.8 | 0.16 | 3,920 |
| 80,000 | 200 | 7.2 | 5.0 | **7.5** | 10.5 | 0.13 | 6,000 |
| 120,000 | 300 | 5.7 | 3.5 | **6.0** | 9.0 | 0.09 | 7,200 |
| 200,000 | 500 | 5.5 | 3.4 | **5.9** | 8.9 | 0.07 | 11,800 |
| 400,000 | 1,000 | 5.3 | 1.8 | **4.8** | 8.3 | 0.05 | 19,200 |
| 800,000 | 2,000 | 5.3 | 1.8 | **4.8** | 8.3 | 0.04 | 38,400 |

**Observations:**
- A **+50% airline rate** rise (80k→120k) cuts recommended margin from 7.5% to 6.0%, so the ₹ margin rises only 20% while cost rises 50%. The model reads a rate spike as a "bigger shipment", not as a market-wide cost move.
- Outputs **saturate** above ~₹200k (400k and 800k give identical percentages). The tree has no resolution where most ₹ is made.
- The **balanced recommendation has a 4–16% win probability** in every row. A quote recommended as "best balance" with a 9% chance of winning is not a practical recommendation.
- Premium and floor are constant offsets (see E-3).

### 5.4 Is cost separated from commercial margin? **No** (`Confirmed`)
There is no cost-estimation component; the airline number is both the cost base and the main predictor of the margin. A correct cost estimate can still yield an unrealistic margin because the margin is inferred from buy size, not from competitiveness, risk, or customer value.

### 5.5 Recommended balanced approach
1. Treat buy as **cost input only**: price = f(cost, margin policy), with margin modeled separately.
2. Replace raw buy with: `buy_rate_per_kg`, `rate_index = buy_rate_per_kg / lane×weight-band rolling median`, and `log(chargeable_kg)`. An elevated `rate_index` signals a market-wide spike (competitors move too) and should keep % margin stable rather than reduce it.
3. Model **margin ₹/kg** (and a ₹/AWB floor), not margin %.
4. Add cost-structure inputs (surcharges, handling) so margin is not a proxy for them.
5. Add explicit regularization tests: permuting buy must not move importance above ~25–30% of total.

---

## 6. ML / recommendation-engine review

### 6.1 Current approach
Hybrid: **regression (benchmark) → classification with monotone constraint → hand-coded demand decay → grid optimization → rule-based card extraction**. Gradient boosting is suitable for tabular data; the problems are the target definition, labels, features, validation, and the overlay.

### 6.2 Targets and labels
- **Stage 1A target:** `Margin_Percentage` of won deals only. This learns past behaviour (survivorship, circular: model of own pricing policy).
- **Stage 1B target:** `is_won` with non-final statuses mixed in as 0 (P-1).
- **The central data anomaly** (CSV, `Confirmed`):

| Margin bucket | Rows | Win rate |
|---|---|---|
| 0–2% | 4,890 | 4.5% |
| 2–4% | 9,828 | 26.6% |
| 4–6% | 3,060 | 42.9% |
| 6–8% | 1,595 | 58.9% |
| 8–10% | 1,081 | 67.6% |
| 10–15% | 1,681 | 77.0% |
| 15–20% | 868 | 85.1% |
| 20–30% | 894 | 90.4% |
| 30–60% | 768 | 90.9% |

Mean margin: won 11.2% vs lost 3.3%; 30.5% of lost quotes are ≤2% margin vs 2.3% of won. `train.py` itself prints "Lost deals mean Margin_Ratio 0.484 … aggressive spot inquiries", an unverified rationale: genuinely cheaper quotes should win more often, not less. **Probable explanation:** for lost inquiries, Total Sell is a placeholder/near-cost/system-generated price, or won quotes carry finalized extras; either way the recorded margin is **not the price the customer rejected**. **Unable to verify without the quoting-process definition.**

### 6.3 Leakage and bias *(replicated)*
| Item | Evidence | Label |
|---|---|---|
| `quote_revision_count` | win 23.7% at 1 revision vs 72.3 / 76.6 / 81.1 / 88.9% at 2–5; classifier top feature (AUC drop 0.109) | Confirmed |
| `Booked by branch`, `Origin_Port` | Ahmedabad 98.7%, Hyderabad 95.6%, Delhi 15.7%; dropping branch + origin features lowers AUC 0.930 → 0.902 | Confirmed (likely CRM capture artefacts: Probable) |
| `Company_Group` | Overseas Agent 96.8% win (837 rows) | Confirmed |
| `Customer_Inquiry_Frequency` | look-ahead count; regressor importance 0.223; win rate non-monotone across buckets (39.7, 44.0, 41.6, 39.6, **20.1**, 42.2%); likely acts as a customer-ID proxy | Confirmed / Probable (proxy) |
| In-sample `Margin_Ratio` | see TR-2 | Confirmed |
| Class balance | 37.9% won, fine | — |
| Stale rates / time | Jan–Sep 2026 only | Confirmed |

**Ablations (random split):** all features 0.930; no Margin_Ratio 0.930; honest OOF Margin_Ratio **without** monotone constraint 0.965 (price helps only when allowed to *increase* win probability); no buy cost 0.927. **Temporal split:** as trained 0.829, no Margin_Ratio 0.825, honest OOF unconstrained 0.907.

### 6.4 Evaluation, calibration, explainability, guardrails
- Metrics: AUC and Brier only; no calibration by margin or segment (Missing). No expected-profit backtest (Missing).
- Stage 1A: CV R² 0.486, MAE 4.93 pts vs a median-only baseline MAE of 7.25 *(replicated)*. A benchmark error of ~5 points on predictions of 3–6% makes `Margin_Ratio` unreliable.
- Guardrails: bounds exist, but there is no confidence level, no reason codes, no "insufficient data" fallback, no manual override flow (Confirmed).
- Determinism: models are seeded, but `datetime.now()` breaks reproducibility (E-7).
- Versioning: version string and sklearn check exist (good); no data hash, no training-window metadata.

### 6.5 Recommendation: **hybrid, rule-led with ML assist**
Given 9 months of data with questionable price labels, the system should *not* present ML-optimal prices yet. Use:
1. **Rule/cohort layer** (cost floor, cohort margin quantiles, customer adjustments): deterministic, explainable, bounded.
2. **ML as an assistant:** win probability (calibrated, price-aware, time-validated) and a cohort-margin predictor used to *adjust within the rule band*.
3. **Experimentation** (A/B quoting bands) to learn true elasticity.

---

## 7. Historical margin and customer-behaviour gaps

### 7.1 What is used today
| Signal | Used? | How |
|---|---|---|
| Average/typical margin of won deals | Indirectly | Regressor (black box) |
| Max accepted margin for similar shipments | **No** | Candidate grid reaches ≥36% |
| Same customer/lane/commodity/weight-band quantiles | **No** | No lookup exists |
| Realized (post-booking) margin | **No** | Only quoted margin |
| Quote→book conversion by margin level | **Distorted** | Data pattern inverted (§6.2) |
| Customer identity, recency, revenue, profitability | **No** | `Customer_Company` (1,123 values) excluded; UI has no customer field |
| New / repeat / dormant / strategic / price-sensitive | **No** | Only frequency integer (units mismatch) |
| Previous accept/reject/negotiate patterns | **No** | — |
| Minimum sustainable margin | **No** | — |

### 7.2 How the gaps cause bad recommendations (examples)
- **Over-pricing a loyal buyer:** a repeat customer who consistently books at 4–6% receives a "balanced" 8.2% because only a frequency integer is seen (sweep: freq 300 / revision 2 → 8.2% and P = 0.73, while freq 100 → 3.8%). The jump is a leakage artefact, not customer behaviour.
- **Under-pricing a one-off with no alternatives:** nothing flags that a shipment's cohort has historically accepted P75 = 12% margins.
- **Near-cost wins:** with a 0.5% grid floor and no ₹ floor, a ₹500k buy can be quoted at ₹2,500 margin.

### 7.3 Safe, practical design
Build a **point-in-time history store** (no look-ahead): for each quote date, compute cohort features strictly from earlier quotes.

| Feature | Definition |
|---|---|
| `cohort_p25/p50/p75/p90_won_margin_per_kg` | By lane × weight band × commodity, hierarchical fallback (lane+band → region lane+band → band) with minimum-n and shrinkage |
| `cohort_win_rate_by_margin_band` | Counts only from valid priced quotes |
| `cust_n_quotes_12m`, `cust_n_won_12m`, `cust_gp_12m`, `cust_days_since_last_won`, `cust_avg_accepted_margin`, `cust_max_accepted_margin`, `cust_reject_margin_floor` | Customer-level, rolling |
| `customer_state` | new / repeat / dormant / strategic (rules from above) |
| `realized_margin` | After charges/claims/invoice, joined by AWB |
| `negotiation_depth` | Revisions *before* final price (not after) |

---

## 8. Root-cause analysis of major issues

### F-01 | Synthetic price elasticity | Confirmed | **Critical** | P0 / M
- **Files:** `engine.py` (`optimize_quote`), `train.py` (monotone constraint)
- **Symptom:** win probability collapses to ≈0 beyond ~2× benchmark; recommended margin hugs the benchmark.
- **Root cause:** raw classifier is flat in margin; `logits − 1.5·excess^1.15` supplies the entire slope.
- **Business impact:** Displayed "Estimated Win Rate" and EV are not evidence-based; the quote ranking is arbitrary.
- **Example:** DEL→LHR 400 kg, ₹120k: raw P 0.24 at 8.7% → shown 0.022.
- **Fix:** remove overlay; estimate elasticity after fixing labels; validate with experiments; present P only with a confidence band.

### F-02 | Inverted margin/win relationship, unclear sell-price semantics | Confirmed pattern / Probable cause | **Critical** | P0 / M
- **Files:** `preprocess.py` (label/price), `train.py`
- **Symptom:** classifier ignores margin; won margin median 7.0% vs lost 2.1%.
- **Root cause:** `Total_Sell_INR` on lost quotes likely not a real offer; Draft/Cancelled/Hold labelled as losses.
- **Impact:** model cannot learn what customers rejected; strategies cannot be priced.
- **Example:** 30.5% of "lost" quotes at ≤2% margin.
- **Fix:** define Sell for each status with sales ops; use only the final customer-facing quote; add loss reasons as analysis (not features).

### F-03 | Outcome-leaking / non-commercial features | Confirmed | **Critical** | P0 / S
- **Files:** `preprocess.py` (`quote_revision_count`), `train.py` (`feature_cols`), `app.py` (revision default 0, branch selector)
- **Symptom:** win probability moves with dropdowns unrelated to price; defaults produce 5–16% P.
- **Root cause:** features correlated with outcome *after the fact* (revisions) or with CRM capture patterns (branch).
- **Impact:** wrong probabilities, gaming, out-of-support defaults.
- **Example:** same shipment priced via Ahmedabad vs Delhi branch changes P(win) drastically.
- **Fix:** drop `quote_revision_count`, `Booked by branch`; use cost/market proxies; add a leakage audit to CI.

### F-04 | Benchmark = own past behaviour on winners | Confirmed | High | P1 / M
- **Files:** `train.py` (Stage 1A), `engine.py` (`get_strategic_recommendations`)
- **Root cause:** regression only on won deals; corridor around it.
- **Impact:** never recommends beyond what was previously done; error ±4.9 pts.
- **Example:** Overseas Agent (won median 3.0%) gets permanently low margins even if the segment would accept more.
- **Fix:** cohort quantile anchors + experiments; use benchmark only as one input with confidence.

### F-05 | Airline buy as raw dominant feature | Confirmed (importance) / Probable (composition, FX) | High | P1 / M
- **Files:** `app.py`, `train.py`, `engine.py`, `preprocess.py` (`parse_currency_field`)
- **Symptom:** 50% cost rise → only 20% ₹ margin rise; saturation >₹200k.
- **Root cause:** raw ₹ total conflates weight and rate; no cost structure.
- **Impact:** rate spikes mis-handled; large shipments mis-priced.
- **Fix:** §5.5.

### F-06 | Customer features leaky and unit-inconsistent | Confirmed | High | P1 / M
- **Files:** `preprocess.py`, `tiers.py`, `app.py`, `engine.py`
- **Root cause:** whole-dataset counts; "Key Account" = 79.5%; UI "per year".
- **Impact:** retention logic and tiers meaningless; frequency acts as customer-ID proxy.
- **Fix:** §7.3.

### F-07 | No temporal validity | Confirmed | High | P0 (date input) / P1 (validation), S–M
- **Files:** `engine.py`, `train.py`, `preprocess.py`
- **Root cause:** 9 months of data, random split, `datetime.now()`.
- **Impact:** Q4 peak (now) is out-of-distribution; temporal AUC 0.829.
- **Fix:** `quote_date` input, rolling-origin validation, refuse/flag outside window, retrain monthly.

### F-08 | Strategy tiers are fixed offsets; dead thresholds | Confirmed | High | P1 / S–M
- **Files:** `engine.py`, `constants.py`
- **Impact:** three options give a false sense of choice.
- **Fix:** derive from cohort quantiles + P(win) targets; implement or delete thresholds.

### F-09 | No cost floor / ops / risk | Confirmed | High | P0 / S
- **Files:** `constants.py`, `engine.py`
- **Symptom:** margins as low as 0.5% on any shipment; the ₹ floor does not exist.
- **Impact:** profitability leakage on small/complex cargo.
- **Example:** ₹20k buy at 0.5% = ₹100 margin, below any handling cost.
- **Fix:** §10.

### F-10 | Missing air-freight variables | Confirmed | High | P2 / L
- **Files:** all
- **Variables:** carrier, service, transit, DG, temperature, ULD/oversize, surcharges, capacity, contract vs spot, competitor, credit.
- **Fix:** staged data capture (§9).

### F-11 | No historical cohort / customer guardrails | Confirmed | High | P1 / M
- See §7.

### F-12 | Data quality | Confirmed | High | P0–P1 / S–M
- **Files:** `preprocess.py`
- Corrupt weights (max 1.9×10¹² kg), gross>chargeable 852 rows, FX not converted (Probable), 15.5% unmapped destinations, label contamination, survivorship removal of ≤0.2% margin wins.
- **Fix:** validation layer with quarantine report.

---

## 9. Prioritized action plan

### Immediate (this week): bugs, wrong calculations, unsafe defaults
| # | Action | Files | Effort |
|---|---|---|---|
| 1 | Add UI banner "Experimental: win probability is not calibrated" | `app.py` | S |
| 2 | Remove `quote_revision_count` and `Booked by branch` from features; retrain | `train.py`, `preprocess.py`, `app.py` | S |
| 3 | Add ₹ minimum margin (per AWB and per kg) and cohort-P90 max cap as hard guardrails | `constants.py`, `engine.py` | S |
| 4 | Add `quote_date` parameter; remove `datetime.now()`; flag Q4 as out-of-window | `engine.py`, `app.py` | S |
| 5 | Quarantine physically impossible weights; fix currency parsing | `preprocess.py` | S–M |
| 6 | Confirm with sales ops what Total Sell means on Lost/Draft quotes | n/a | S |
| 7 | Stop labelling Draft/Cancelled/Hold/Approved as losses | `preprocess.py` | S |

### Short-term (2–6 weeks)
- Out-of-fold `Margin_Ratio`; time/customer-grouped validation; calibration by margin bucket; profit backtest (TR-2, TR-3, TR-5).
- Implement or delete `STRATEGY_THRESHOLDS`/`min_win_prob`; derive tiers from cohort quantiles.
- Log final decision, customer, user, override reason; exclude presets from KPIs (A-4, D-1).
- Replace destination substring mapping with reference table; extend commodity categories (DG, temp).
- Tests: unit + scenario tests (§11); CI leakage audit; pin dependencies; fix pipeline paths and README.
- Auth, RBAC and soft-delete for history endpoints.

### Medium-term (1–3 months)
- Point-in-time customer/cohort history store (§7.3); remove `Customer_Inquiry_Frequency`/`Customer_Tier` raw features.
- Cost model: itemized buy (base rate/kg, FSC, SSC, handling), carrier and service fields.
- Rate/kg, rate-index, margin-per-kg modeling (§5.5).
- Margin framework (§10) with approval workflow.

### Long-term (3–9+ months)
- Collect ≥12–24 months; seasonality modelling with proper holdouts.
- Controlled price experiments (randomized bands) for true elasticity; contextual bandit later.
- Competitor/market-index inputs; capacity and allotment-aware pricing.
- Monitoring: drift, calibration, margin leakage dashboards; monthly retraining with champion/challenger.

---

## 10. Proposed improved margin framework

### 10.1 Structure
```
Quote_Sell = Cost_Floor_Price
           + Target_Margin_₹
Cost_Floor_Price = (Airline_All_In_Buy + Ops_Cost + Risk_Buffer) × (1 + m_min_pct)   [and ≥ buy + ₹_min_per_AWB + ₹_min_per_kg × ch_wt]
```

| Component | Definition | Source |
|---|---|---|
| **Airline all-in buy** | base rate/kg × chargeable kg + FSC + SSC + other carrier charges | itemized input / rate sheet |
| **Operational-cost allocation** | fixed per AWB (docs, screening, AWB fee) + per-kg handling + pickup/customs where included | cost tariff table, by origin branch |
| **Risk buffer** | `r_dg + r_perishable/temp + r_valuable + r_oversize + r_fx + r_credit` as % or ₹ | rule table, reviewed quarterly |
| **Cost floor** | the larger of the % floor and the ₹ floors above | config |
| **Market competitiveness adj.** | `rate_index`/market spike factor, competitor info if available | rolling market data |
| **Customer relationship adj.** | by `customer_state` and `cust_gp_12m` | history store |
| **Historical conversion adj.** | shift within cohort quantile band by observed win rate at margin levels | history store |
| **Lane/shipment complexity adj.** | multi-leg, transshipment, peak lane, tight capacity | rule table |
| **Min/Max guardrails** | `m_min` (cost floor), `m_max = min(cohort_P90, customer_max_accepted × 1.1)` | derived |
| **Approval / override** | see 10.4 | workflow |

### 10.2 Margin score (pseudocode)
```python
def recommend(q, hist, cfg):
    cost  = q.airline_all_in + cfg.ops_cost(q) + cfg.risk_buffer(q)
    floor = max(cost * (1 + cfg.m_min_pct(q)),
                q.airline_all_in + cfg.min_margin_per_awb + cfg.min_margin_per_kg * q.ch_wt)

    cohort = hist.cohort(q.lane, q.weight_band, q.commodity)      # hierarchical fallback, min-n
    anchor = cohort.p50_margin_per_kg * q.ch_wt                    # ₹ margin anchor
    cap    = min(cohort.p90_margin_per_kg * q.ch_wt,
                 hist.customer_max_accepted(q.customer) * 1.10 if q.customer_known else inf)

    adj = (cfg.customer_adj[q.customer_state]
         * cfg.season_adj(q.quote_date, q.lane)
         * cfg.complexity_adj(q)
         * cfg.market_adj(q.rate_index))                          # each bounded, e.g. 0.8-1.25

    target_margin = clip(anchor * adj, floor - q.airline_all_in, cap)

    # Optional ML refinement inside the band, only if the win model passes calibration checks
    if win_model.is_valid(q):
        grid = linspace(floor_margin, cap, n)
        ev   = [(m - cfg.ops_cost(q) - cfg.risk_buffer(q)) * win_model.p(q, m) for m in grid]
        target_margin = grid[argmax(ev)]  # subject to p >= cfg.p_min[strategy]

    sell = q.airline_all_in + target_margin
    return Quote(sell, tiers=derive_tiers(floor, target_margin, cap),
                 confidence=hist.confidence(cohort, q), reasons=explain(...))
```

### 10.3 Decision table (examples)

| Condition | Action |
|---|---|
| Cohort n < 30 or unseen port/commodity | Use parent cohort; `confidence = Low`; show reason |
| Recommended < cost floor | Block; require approval |
| Recommended > customer's max accepted ×1.1 or > cohort P90 | Clamp and warn |
| DG / temp-controlled / valuables / oversized | Add risk buffer; require pricing-desk review |
| P(win) < 10% at recommended | Show "unlikely to convert"; offer floor tier |
| Strategic account | Allow down to floor with sales-head approval and reason code |
| Out-of-training-window date | `confidence = Low`, rules only |

### 10.4 Approval/override flow
Auto-approve within [floor, cap] and High/Medium confidence → manager approval below floor + 0% (down to a "never-below" ₹ limit) → pricing head for DG/high-risk or margin above cap. Every override stores user, reason code, and later outcome.

### 10.5 Tiers (replace the fixed ±offsets)
- **Floor tier:** `max(cost floor, cohort P25)`
- **Balanced:** `anchor × adj` (clamped)
- **Premium:** `min(cap, cohort P75)`; show the observed win rate of that band.

### 10.6 Backtest requirements
Rolling-origin on resolved quotes: report margin captured vs realized, conversion by band, share below floor, calibration by margin bucket; compare vs naive "cohort median" and the current engine.

---

## 11. Suggested test cases (realistic air-export scenarios)

| # | Scenario | Expected behaviour | Why |
|---|---|---|---|
| T1 | **Rate shock:** BOM→FRA 800 kg, buy ₹160k vs ₹192k (+20%), all else equal | Sell increases; % margin changes ≤ ±1 pt; ₹ margin does not fall | Today's engine reduces % with buy |
| T2 | **Weight-break:** 98 kg vs 101 kg DEL→DXB | No jump >X; rule flags break-point arbitrage (bill at 100 kg if cheaper) | Slab logic absent |
| T3 | **DG shipment** (UN class 3) 150 kg BLR→LHR | Higher floor via risk buffer; pricing-desk approval | DG not modeled |
| T4 | **Perishable** (fish) 1,200 kg COK→DXB in peak week | Capacity/season adjustment; temp-chain risk buffer | Peak/temp absent |
| T5 | **Key account** with 24 accepted quotes at 4–6% on lane | Recommendation ≤ P90 of customer's accepted band ×1.1 | No customer lookup |
| T6 | **New spot customer**, subagent, 45 kg | ₹ minimum per AWB applies; wider band; low confidence | ₹ floor missing |
| T7 | **Invariance:** change only `Booked by branch` or `revision` | Output unchanged | Currently changes P(win) |
| T8 | **Determinism:** same payload, different server date, explicit `quote_date` | Identical output | `datetime.now()` |
| T9 | **Q4 date** (e.g., 15 Nov) with 9-month training window | Low-confidence flag | Out-of-window |
| T10 | **Unseen port** (Tashkent) | Falls back to region cohort, confidence Low, no silent −1 | E-9 |
| T11 | **Volumetric cargo:** 3 boxes 120×80×90 cm, gross 450 kg | Chargeable computed server-side (432 vs 450 → 450); validation `chargeable ≥ gross` | A-3 |
| T12 | **Near-cost bid:** buy ₹500k | Margin ≥ ₹ floor, never 0.5% (₹2,500) | C-2 |
| T13 | **Historical margin behaviour:** cohort accepted ≤8%, rejected ≥10% | Cap near 8–9%; recommended not above | Cohort guardrail |
| T14 | **Invalid input:** buy ≤ 0, chargeable < gross, weight 1.9e12 | HTTP 400 with explicit message | P-3 |
| T15 | **Backtest gate:** 3-month holdout | Win-model calibration error by margin bucket ≤ agreed tolerance; no segment below cost floor | TR-5 |

---

## 12. Assumptions and limitations

### Information unavailable in the codebase
- Trained artifact (`freight_margin_recommender.joblib`) and its recorded metrics; `requirements.txt`, `run.sh`, `docs/AIR_EXPORT_PRICING_ML_DOCUMENTATION.md`; any tests/CI.
- Raw Excel file (`Air & Ocean Inquiries (Air Export Pricing).xlsx`) and the "Combined" merge script; columns such as `Loss Reason`, `TAT to Close`, carrier/airline.
- Definition of "Total Buy", "Total Sell", quote/inquiry lifecycle, and how Lost/Draft/Hold quotes are priced.
- Airline contract rates, allotments, spot vs contract, competitor intelligence, realized margins/claims, credit terms.
- Real production inference logs/history database.

### Business rules inferred rather than confirmed
- "Margin %" = markup on buy; "Total Buy" ≈ all-in airline cost in INR (not verified; FX not converted).
- Win = `Quote Status == "Won"`; Lost = price loss (contradicted by data patterns).
- Branch/company-group win-rate extremes are CRM capture artefacts (Probable).
- Season labels (`get_air_cargo_season`) are management assumptions, unvalidated.

### Methodological limits
- Model-behaviour figures come from my **replication** of `train.py` on the supplied CSV (won-benchmark = `is_won==1` rows), not from the production artifact; sklearn version may differ.
- Permutation importance on correlated features understates individual contributions; ablations are single-seed.
- The "Combined" CSV covers Jan–Sep 2026 only; seasonality, annual contracts, and long-term customer behaviour cannot be assessed.
- Competitive outcomes (why lost) are unobserved; the elasticity cannot be identified from this observational data without the label fix or experiments.

### Needed to validate uncertain conclusions
1. Sales-ops definition of Sell price per status; sample of 50 Lost quotes with the actual customer-facing quote.
2. Raw Excel to quantify FX mixing, dropped sub-cost wins, and weight corruption source.
3. Branch/CRM process owners to explain Ahmedabad/Hyderabad vs Delhi win-rate gap.
4. 12–24 months of history incl. AWB-level realized margins and loss reasons.
5. Carrier rate sheets, surcharge tables, ops cost tariffs for the cost-floor model.
6. Production joblib to rerun the diagnostics exactly.

---

## Recommended next steps

1. **Add a "not calibrated" banner and hard guardrails (₹ min margin, cohort P90 cap, low-confidence fallback)** this week; the current output should not be treated as optimal pricing. *(P0, S)*
2. **Meet sales ops to define Sell/Lost semantics;** rebuild labels on resolved, customer-facing quotes only. *(P0, S–M)*
3. **Remove leaking/non-commercial features** (`quote_revision_count`, `Booked by branch`, raw `Customer_Inquiry_Frequency`/`Customer_Tier`) and retrain with out-of-fold `Margin_Ratio`. *(P0, S–M)*
4. **Remove the hard-coded elasticity overlay** or fit it; stop presenting its probabilities as calibrated. *(P0, M)*
5. **Introduce `quote_date`, validate against the training window,** and flag Q4 quotes as low-confidence until Q4 data exists. *(P0, S)*
6. **Clean data:** weights, currency, destination mapping, commodity/DG fields, keep sub-cost wins. *(P1, M)*
7. **Switch to rate/kg, rate-index and margin-per-kg modelling** to fix airline-cost over-weighting. *(P1, M)*
8. **Build the point-in-time customer/cohort history store** and use it for floors, caps and tiers. *(P1–P2, M)*
9. **Implement the rule-led margin framework (§10)** with approval/override flow; use ML only inside the band. *(P2, L)*
10. **Instrument and close the feedback loop:** log final decisions and outcomes, add tests/CI/backtests, and run controlled price experiments to learn true elasticity. *(P2–P3, L)*
