# Pricing Engine: Issues by File and Recommended Fixes

**Date:** 2026-10-11
**Scope:** `Air_Export_Pricing_Combined_ML.csv` (23,048 rows), `benchmark_500_records.csv` (500 rows), and the project code (`margin_framework.py`, `train.py`, `preprocess.py`, `preprocess_job_consolidated.py`, `tune_engine_optuna.py`).
**Status:** Diagnosis and proposed fixes only. No code has been changed. Implementation waits for your approval.

---

## 1. Summary

The margin gaps are not caused by a poorly tuned model. They come from four structural problems:

1. **The label mixes two different things.** About 59% of the training rows are *Lost* quotes. Their "margin" is mostly a system default of about 2% gross margin (2.04% markup), not a pricing decision. The 500-record benchmark is half Lost, half Won. The model sits between the two populations, so it is too high on Lost rows (**+2.25 pp**) and too low on Won rows (**−2.10 pp**). The overall bias of +0.11 hides this.
2. **The production engine is a lookup with almost no room to adapt.** The six tuned "signal weights" can move a margin by at most **±1.74%** of its value. A 5% margin can shift by about ±0.09 pp. Real margins vary by about ±10 pp within a weight slab. The signals are effectively inert, so tuning them cannot reduce the gaps.
3. **Hand-set overrides contradict the data.** For example, "Other" is hardcoded at 3.5%, while its Won median is 8.75%.
4. **The benchmark cannot measure accuracy.** Every benchmark row is also in the training file. The job-ledger target is built from the same lookup the model uses. The corridor test in the harness is also mis-specified.

**The accuracy floor with today's columns is high.** On Won deals, tested out of time (train Jan–Jul, test Aug–Sep):

| Model | MAE (pp) | Bias (pp) |
|---|---|---|
| Global median | 6.81 | −2.69 |
| Current-style lookup (lane × slab × commodity) | 5.83 | −1.46 |
| Boosted trees, squared loss, current-style features | 6.03 | +1.29 |
| Boosted trees, absolute loss, current-style features | 5.17 | −0.98 |
| Boosted trees, absolute loss + point-in-time customer history | **5.02** | −1.04 |

An in-sample lookup scores 5.85 and the out-of-time lookup scores 5.83. The lookup is not overfitting. The current columns simply do not contain enough information to go much below about 5 pp on Won deals. Closer margins need new variables (Section 5, Fix 7).

### Corrections to my earlier message

These came from re-checking against the full data and code:

- **Elasticity term (RC2).** I said it caused "nearly all" negative gaps. That held in the 100-case file. In the 500-record file, heavy cargo above ₹200/kg has bias **+0.07 pp** and MAE 1.78, so it is not the main driver. The largest errors are in light parcels, "Other" commodity and Won deals with high margins. The term remains a hand-set assumption (Issue M3), but it is not the main cause.
- **Repeated margins as customer rate cards (RC4).** I said identical margins repeat across a customer's jobs. In the full data, within a customer and weight slab the most common margin covers a median of only **10%** of jobs. Only **1.7%** of customer-slab cells have a modal margin of 50% or more. Rate-card lookup performed worse (MAE 6.28) than the boosted model. Customer history helps, but as a feature, not as a fixed rate card. The 2.04% spike is the Lost-quote default, not customer policy.
- **`Customer_Inquiry_Frequency` (RC4).** I called it a full-sample count. In the training CSV it equals the full count for only 15% of rows. It also does not increase over time within a customer (correlation 0.02). Its definition is unclear (Issue D5). It is a full-sample count in the job-ledger script (Issue P2).
- **The conformal tables (RC3).** Confirmed unused in the recommendation path (Issue M5).

---

## 2. Issues by file

Severity: **Critical** (invalidates results), **High** (drives most of the gap), **Medium**, **Low**.

### Index

| ID | File | Issue | Severity |
|---|---|---|---|
| D1 | `Air_Export_Pricing_Combined_ML.csv` | Lost-quote margins are system defaults, not pricing decisions | Critical |
| D2 | same | After the worked-quote filter, win rate is 93.8% against 40.5% real | High |
| D3 | same | Margin ceiling and long tail (16% of Won above 20%) unexplained | High |
| D4 | same | Date fields inconsistent in 422 rows | Medium |
| D5 | same | `Customer_Inquiry_Frequency` and `Customer_Tier` unreliable | High |
| D6 | same | Customer names not entity-resolved (1,073 → 1,030) | Medium |
| D7 | same | Commodity taxonomy lacks DG, Valuables, Live Animals; "Other" is large and heavy-tailed | High |
| D8 | same | Missing drivers: carrier, rate type, validity, scope, sales person | High |
| B1 | `benchmark_500_records.csv` | 100% of rows exist in the training file (in-sample) | Critical |
| B2 | same | Mixes Lost (default-margin) and Won rows | Critical |
| B3 | same | Corridor test uses the wrong band (1.4% hit vs 26.8% with the app rule) | High |
| B4 | same | Stratified 100 per commodity and per budget, 5 commodities only, 9 duplicate rows | High |
| B5 | same | Model barely beats a commodity-median baseline (3.61 vs 3.56 MAE) | High |
| B6 | same | Metrics are in pp only, no ₹-weighted view | Medium |
| M1 | `margin_framework.py` | Signals can move a margin by at most ±1.74% | Critical |
| M2 | same | Hardcoded commodity overrides contradict the data | High |
| M3 | same | Hand-set sublinear elasticity (β = 0.7356, threshold 200.18) | High |
| M4 | same | Hardcoded corridor and region branches (largely dead code) | Medium |
| M5 | same | "90% conformal coverage" is a label; the corridor is a fixed ±15–25% | High |
| M6 | same | Lane tables use destination region, not port | Medium |
| M7 | same | Win probability is `None` in production | Medium |
| T1 | `train.py` | Win classifier trained on a 94%-win filtered set | High |
| T2 | same | Random shuffled 5-fold CV; no time or customer grouping | High |
| T3 | same | Stage 1A and 1B models are not used by the recommendation path | High |
| T4 | same | Temporal split affected by default-filled dates | Medium |
| T5 | same | Final classifier refit on the full data after evaluation | Low |
| P1 | `preprocess_job_consolidated.py` | Target margin set to the model's own lookup (circular) | Critical |
| P2 | same | Frequency is a full-sample count; defaults fill missing dates | High |
| P3 | same | Every row labelled Won, all buy costs estimated | High |
| O1 | `tune_engine_optuna.py` | Tunes on the circular target | Critical |
| O2 | same | `predict_one` is not the production function | High |
| O3 | same | Hardcoded local scratch paths; benchmark JSON outside the repo | Medium |
| C1 | `preprocess.py` | Substring keyword commodity mapping | High |
| C2 | same | Dest-region mapping by substring; first match wins | Medium |
| C3 | same | Margin trimmed to 0.2–60% after computing it | Medium |

---

### 2.1 `Air_Export_Pricing_Combined_ML.csv`

**D1. Lost-quote margins are defaults, not decisions (Critical).**
- Lost rows: 13,709 (59.5%). Won rows: 9,339.
- Median margin: Lost **2.04%**, Won **7.03%**.
- 2,668 rows sit exactly at 2.04% (a 2% gross margin).
- 20.3% of Lost rows have exactly 2% gross margin; only 0.4% of Won rows do.
- Median turnaround: Lost **0.018 h (about 1 minute)**, Won **25.9 h**. 30% of all rows were quoted in under a minute.
- Impact: any model or benchmark that treats Lost margins as "actual margin" learns the default, not pricing behaviour.

**D2. Filtering leaves a distorted win rate (High).**
- The `filter_worked_quotes` logic removes 13,087 of 13,709 Lost rows (95.5%). Only 622 Lost remain against 9,339 Won.
- Win rate after filtering: **93.8%**. Real win rate: **40.5%**.
- The filter uses turnaround time, a field recorded after the outcome, to define the label.
- Impact: win probability from a model trained on this set is not calibrated to reality (Issue T1).

**D3. Long tail of high Won margins (High).**
- 16.1% of Won rows (1,504) exceed 20% margin. Of these, 45% are under 45 kg and 94% are "General Cargo".
- 120 rows are at 50% or above; 14 are at exactly 50.35%.
- The 60% ceiling is applied by the cleaning code, not observed in the data (Issue C3).
- Impact: 13% of benchmark rows (actual above 12%) produce 39% of absolute error and 66% of squared error.

**D4. Inconsistent dates (Medium).**
- 422 rows have `Quote_Month = 9` but `Year_Month = 2026-06` and `Month_Name = June`.
- This matches the default-fill pattern (`"2026-06"`, `"June"`) in `preprocess_job_consolidated.py`.
- Impact: September rows are treated as June in any time split (Issue T4). Seasonality is also wrong for these rows.

**D5. Customer frequency and tier (High).**
- 279 of 1,073 customers have more than one frequency value.
- The value equals the full-file customer count for only 15.2% of rows and does not grow with time within a customer (correlation 0.02). Its definition is unknown.
- 80.2% of rows have frequency of 20 or more, so **80% of rows are labelled "Key Account"**. The tier does not separate customers.
- Impact: the customer-tier prior and corridor multiplier carry almost no information.

**D6. Customer names not entity-resolved (Medium).**
- 1,073 names collapse to 1,030 after normalising "Private Limited / Pvt Ltd / Limited / Ltd", punctuation and spacing (43 duplicates found with a simple rule).
- Impact: customer history is split across aliases, so customer features are weaker than they should be.

**D7. Commodity taxonomy (High).**
- The file has General Cargo 92%, Other 2.8%, Perishable 2.2%, Garments 1.4%, Engineering 0.6%, Pharma 0.4%, Courier 0.4%, Auto 0.1%. **No Dangerous Goods, Valuables or Live Animals**, although the code has special handling for all three.
- Won counts: Engineering **13**, Auto Parts 5, Pharma 26.
- "Other" Won margins are very spread out (p25 4.76, median 8.75, p75 17.19).
- Impact: the commodity-level results in the benchmark are not reliable. Engineering looks accurate (MAE 1.58) because 93% of its benchmark rows are Lost default-margin rows.
- Hypothesis to test: some of the 94% "General Cargo" rows above 20% margin may be special cargo labelled generically.

**D8. Missing drivers (High).**
- The file has no carrier, rate type (spot vs contract), rate validity, scope of charges, sales person, credit days or competitor rate.
- `preprocess_job_consolidated.py` extracts Carrier and Sales_Person for the job ledger, but the combined file does not contain them.
- Impact: this is the main reason for the roughly 5 pp accuracy floor.

---

### 2.2 `benchmark_500_records.csv`

Note: `Margin_Gap = Model_Target − Actual`. Positive means the model is too high.

**B1. Benchmark rows are in the training file (Critical).**
- All 500 rows match a row in the training CSV on buy cost and weight. For 98%, the margin is identical. `is_won` matches in 100%.
- The production tables are built from all Won rows, so the Won half of the benchmark is in-sample.
- Impact: the benchmark is not a holdout and cannot show how the model performs on new quotes.

**B2. Mixed label populations (Critical).**

| Group | n | MAE | Bias | Actual median | Model median |
|---|---|---|---|---|---|
| Lost | 254 | 2.78 | **+2.25** | 2.09 | 3.68 |
| Won | 246 | 4.47 | **−2.10** | 5.75 | 5.52 |

- 55 rows (11%) are exact 2% gross-margin defaults; 54 of them are Lost.
- Impact: the model is pulled toward the midpoint of two unrelated processes, creating positive gaps on Lost rows and negative gaps on Won rows.

**B3. Corridor test is mis-specified (High).**
- Reported hit rate is **1.4%**. The median corridor in the file is only **0.08 pp** wide (1.9% of the target).
- This width matches the span between the floor and premium tiers (±0.076 and +0.063 position offsets), not the app's negotiation corridor.
- Applying the app's actual corridor rule (±15/20/25% with 0.75/1.0 pp minimum) gives a **26.8% hit rate** and a median width of 1.56 pp.
- Impact: the 1.4% figure does not describe what users see. The real corridor is also far too narrow (Issue M5).
- Not verified: the script that produced the benchmark file is not in the project, so this is inferred from the numbers.

**B4. Sampling and duplicates (High).**
- 100 rows each for 5 commodities and for 5 budget segments. Pharma, Courier and Auto Parts are absent.
- Natural mix is 92% General Cargo, so results are not representative of volume.
- 9 exact duplicate rows (for example the 1,200 kg Perishable job appears twice).
- Month coverage is uneven (8 rows in May, 138 in August).

**B5. The model barely beats naive baselines (High).**

| Baseline | MAE |
|---|---|
| Global median | 3.90 |
| Commodity median | **3.56** |
| Current model | 3.61 |

- Correlation between actual and model target is 0.61; R² is 0.37. Model target std is 4.2 against actual 7.7, so the model cannot reach the tails.
- On Won rows only, the model scores MAE 4.47 against 5.56 for commodity median.

**B6. Error by segment (for reference).**

| Segment | n | MAE | Bias |
|---|---|---|---|
| Very Low Budget | 100 | **7.67** | +0.09 |
| Other (commodity) | 100 | 4.89 | −0.42 (Won rows: MAE 11.30, bias −9.24) |
| Perishable | 100 | 5.03 | −1.42 |
| General Cargo | 100 | 4.01 | +1.48 |
| < 45 kg | 48 | **9.72** | +1.07 |
| 45–100 kg | 27 | 5.59 | **+3.16** |
| Rate ≤ ₹100/kg | 45 | 4.92 | −2.48 |
| Rate ₹1,000–3,000/kg | 18 | 7.14 | −2.64 |
| Actual margin above 20% | 21 | 21.03 | −21.03 |

In ₹ terms, the net gap is +₹2.2 lakh on ₹18.7 crore of buy cost, but the sum of absolute gaps is ₹31 lakh. Medium Budget has a net −₹1.5 lakh and Very High Budget a net +₹2.9 lakh, so errors cancel in aggregate.

---

### 2.3 `margin_framework.py`

**M1. Signals have almost no effect (Critical).**
- Position is `0.5 + 0.5 × w_scale × weighted_signal`, with `w_scale = 0.2539`, so position stays within 0.5 ± 0.127.
- The band spread is ±6.85% of the benchmark margin. The maximum swing from all six signals together is therefore **±1.74%** of the margin (±0.09 pp on a 5% margin).
- Impact: customer history, commodity, competitor and market signals cannot represent the real spread. The Optuna-tuned weights (Issue O1) have no meaningful influence on results.

**M2. Hardcoded commodity overrides (High).**

| Commodity | Hardcoded parent margin | Won data median | Won n |
|---|---|---|---|
| Other | **3.5%** (slab values 2.8–12.0) | **8.75%** | 142 |
| Engineering & Machinery | **2.45%** | 5.76% | 13 |
| Perishable Foodstuff | 8.5% | 6.33% | 490 |

- The comments justify them as "eliminating over-quoting" and matching "clearing medians", but these values come from the Lost default-margin rows in the benchmark (Issue B2).
- Impact: this explains the −9.24 pp bias on Won "Other" rows.

**M3. Hand-set sublinear elasticity (High).**
- For heavy cargo above ₹200.18/kg (not Perishable), margin is multiplied by `(rate/200.18)^(0.7356 − 1)`. β was fitted to pass the T1 scenario test ("+20% buy gives less than +15% profit"), not estimated from data.
- The data show a **U-shape**, not a decline: Won median margin is 9.7% at ₹50–150/kg, falls to 4.6% at ₹200–300/kg, then rises to **13.2%** at ₹1,000–3,000/kg and 14.9% above ₹3,000/kg.
- In the 500 benchmark, heavy cargo above ₹200/kg has bias +0.07 pp, so the term is not the main cause of the gaps, but it contradicts the data above ₹500/kg.
- The file carries other elasticity values as well: `BETA_COST_PASS_THROUGH = 0.58` (0.85 for < 45 kg, 0.65 for 45–100 kg) in the fallback path, and 0.72 as the Optuna starting point. These are never reconciled.

**M4. Hardcoded region and rate branches (Medium).**
- Fixed margin tables for < 45 kg (0.1031, 0.1438, 0.1318, 0.200/0.2089), 45–100 kg (0.065, 0.075, 0.1117), Kathmandu, Oceania, US wholesale and Latin America bulk.
- These branches run only when `corridor_m is None`. The corridor lookup falls back to the global margin, so they are reachable only when origin or destination region is missing. In normal use they are dead code, but they would silently take over when region mapping fails.
- Also the AWB floor (`min(0.18, 1200 / buy)` for ≤ 75 kg and buy < ₹30,000) is an unvalidated policy rule that drives the < 45 kg and 45–100 kg gaps.

**M5. "90% confidence" corridor is not calibrated (High).**
- `build_tables` computes `conformal_corridor_q90` but it is **never used**. The corridor is a fixed ±15% (> 500 kg), ±20% (45–500 kg) or ±25% (< 45 kg), with minimum ±0.75 pp (±1.0 pp for < 45 kg).
- `confidence_coverage: "90%"` is a fixed string.
- Measured on Aug–Sep Won deals with an out-of-time model: the app-style corridor covers **31.6%**; a properly calibrated split-conformal interval of ±12.67 pp covers **90.4%**.
- The calibrated half-width (12.67 pp) is wide because the features carry little information (Issue D8).

**M6. Lane resolution (Medium).**
- Lanes are origin region to destination region. Dubai and Riyadh share one benchmark. Destination port is used only through a shrinkage band, not as a feature.
- In my out-of-time test, adding destination port did not help (MAE 5.15 against 5.02), so the gain from this alone is limited. Port-level effects should enter through shrinkage, not as raw categories.

**M7. Win probability is `None` (Medium).**
- The response always returns `win_probability=None`. The expected-value pricing in the docs is not implemented in the served engine.

---

### 2.4 `train.py`

**T1. Win classifier trained on a distorted set (High).**
- Trained after `filter_worked_quotes`: 9,339 Won, 622 Lost (93.8% win). Real win rate is 40.5%.
- Margin bucket calibration is reported on this same distorted test set.

**T2. Validation design (High).**
- Stage 1A uses `KFold(shuffle=True)`. Rows from the same customer, lane and month appear in both train and validation, which makes CV scores optimistic and ignores drift.
- No customer-grouped or rolling-time validation. No comparison to a baseline.

**T3. Stage 1A and 1B are not used in production margins (High).**
- `margin_framework.recommend` does not call `benchmark_reg` or `calibrated_clf`; only the `margin_tables` from this script are used. Training metrics therefore do not describe the margin output. I did not re-read `app.py` or `engine.py`, so confirm nothing else consumes these models.
- `Customer_Company` and customer history are not features in Stage 1A.

**T4. Temporal split uses default-filled dates (Medium).**
- The split at `Year_Month < "2026-08"` includes the 422 September rows mislabelled as June in training (Issue D4).

**T5. Final model refit on all data (Low).**
- The final classifier is refit on the full set after evaluation, so no post-fit holdout remains.

---

### 2.5 `preprocess_job_consolidated.py`

**P1. Circular target (Critical).**
- `Margin_Percentage` is set to the lane × weight-slab median from the reference Won table. `Total_Buy_INR = Sell / (1 + margin)` is then back-computed and flagged `Buy_Cost_Estimated = 1`.
- Impact: where the engine applies no adjustment, the "gap" is zero by construction. This is why the 100-case benchmark shows MAE around 0.2%.

**P2. Leakage and default fills (High).**
- `Customer_Inquiry_Frequency` is the full-sample count of each customer (uses future jobs).
- `Year_Month` fills missing dates with `"2026-06"`, `Month_Name` with `"June"`, `Quote_Month` with 6, `Quote_Turnaround_Hours` with 0.0.
- Customer name defaults to `"Unknown_Customer"`, which becomes a single fake customer.

**P3. Labels (High).**
- Every row is `is_won = 1`, `Quote Status = "Won"`, `quote_revision_count = 1` by assignment. There is no real outcome information in this file.
- It does contain real operational fields (carrier, sales person, job type) that the combined file lacks.

---

### 2.6 `tune_engine_optuna.py`

**O1. Tunes on the circular target (Critical).**
- The objective is MAE + 0.2·RMSE against `Actual_Margin` from the 100-case JSON, whose rows come from `Job_Wise_Consolidated_Cleaned_ML.csv` (Issue P1).
- Baseline printed in the script: MAE 0.2470%.
- Impact: the optimisation rewards agreement with the lookup, and the six signal weights barely move predictions anyway (Issue M1).

**O2. Not the production function (High).**
- `predict_one` omits the 3-key lane × slab × commodity hierarchy, the commodity overrides, the AWB floor, the customer-weight reassignment (n ≥ 15) and the shipper-account rules in `recommend`. Tuned parameters do not map to the shipped logic.

**O3. Reproducibility (Medium).**
- Paths point to a personal scratch folder (`/Users/deepak/.gemini/...`). The 100-case JSON used for tuning is outside the repo.
- Fixed seed 42, but 10 parameters are fitted on 100 near-duplicate cases with no holdout.

---

### 2.7 `preprocess.py`

**C1. Commodity mapping by substring (High).**
- `"PET"` (Live Animals) matches carpet, petrol or petroleum. `"AVI"` matches aviation. `"UN "` and `"CHEMICAL"` send many non-dangerous chemicals to Dangerous Goods. `"AUTO"` matches automation. `"DOG"`, `"TOOL"` and `"MOTOR"` have similar risks.
- Rules are applied first-match, so order decides the class. "Other" is never produced, yet it exists in the data.
- Impact: the commodity label, a key feature, is noisy and cannot be reproduced from the code that exists (Issue D7).

**C2. Destination region mapping (Medium).**
- First keyword hit wins, searching Middle East, then Europe, then North America and so on. Short keywords (`"nice"`, `"male"`, `"yao"`, `"sola"`, `"cali"`, `"armenia"`, `"salvador"`, `"hamilton"`, `"columbia"`) can match the wrong airport. "Armenia" is listed under Latin America and also appears as a country under Europe in the job-ledger map.

**C3. Margin trimming (Medium).**
- Rows with margin outside 0.2–60% are removed after computing margin. This hides extreme jobs from the model and from error analysis. 120 Won rows are at 50% or more, so heavy-tail handling is needed.

---

## 3. How the root causes map to the gaps

| Gap pattern (500 benchmark) | Main causes |
|---|---|
| **Positive gaps on Lost rows** (+2.25 pp; 66% of all rows have positive gap) | D1, B2: actual is a 2% default; model predicts a worked-quote margin |
| **Negative gaps on Won rows** (−2.10 pp) | M1, M2: lookup centred below the Won distribution, no way to follow customer or deal differences |
| **Large errors on light parcels** (< 45 kg MAE 9.72; 45–100 kg bias +3.16) | D3, M4: AWB floor and fixed tables; wide real spread (std 13 pp) |
| **"Other" commodity** (Won MAE 11.30, bias −9.24) | M2, D7, C1: hardcoded 3.5% against a Won median of 8.75% |
| **High-margin tail** (actual above 20%: MAE 21) | D3, D8, M1: model cannot reach the tail; missing drivers |
| **High-rate cargo** (₹1,000–3,000/kg: bias −2.64) | M3: elasticity multiplier decreases while data increases |
| **Corridors never hit** (1.4% in file; 26.8% with app rule) | B3, M5 |
| **Good-looking scores on the 100 and ledger cases** | P1, O1, B1 |

---

## 4. What I verified and what I did not

**Verified by running code on your two CSVs:** all numbers in Sections 1, 2.1, 2.2 and 3, the out-of-time model comparison, the corridor coverage test, and the swing bound in M1.

**Verified by reading the project code:** Issues in 2.3 to 2.7.

**Not verified:**
- `constants.py`, `tiers.py`, `app.py`, `engine.py`, `history.py` were not re-read in this pass. Cost-floor and cap values (which clamp results) are therefore not assessed.
- The script that generated the benchmark's min and max columns is not in the project (Issue B3 is inferred).
- The out-of-time experiments use Won deals only and a simplified feature set with default settings. They show the size of the accuracy floor, not a final model.

---

## 5. Recommended fix (a rebuild of the foundation, not a patch)

Each fix removes a root cause. They should be done in this order, because later steps depend on earlier ones.

### Fix 1. Define the target correctly
- Model **two separate things**: (a) the margin actually achieved on **Won** jobs, and (b) the probability of winning at a given margin.
- **Never use Lost default-margin rows as margin targets.** Keep Lost rows only for the win model, and only after labelling which Lost quotes were truly worked.
- Replace the turnaround-based filter with an explicit "worked quote" flag from the quoting system (Issue D1/D2). If no such flag exists, add one at source; until then, treat Lost rows as unlabelled for margin purposes and for win-rate calibration.
- Prefer realised margin from invoiced jobs (real buy cost) over quoted margin when available. Drop `Buy_Cost_Estimated` rows from any training or evaluation set (Issue P1).

### Fix 2. Build a validated data layer
- One pipeline from raw to modelling table, with **hard schema and consistency checks that fail the build**: margin identity, date consistency (`Quote_Month` equals month of `Year_Month`), no default-filled dates, outcome counts by month, no impossible turnaround, no estimated buy costs in the training set.
- **Customer master:** resolve aliases (Issue D6) with a reviewed mapping table, not only automatic rules. Assign a stable `customer_id`.
- **Commodity:** replace substring rules with a reviewed lookup table (description → group) plus word-boundary matching, versioned and unit-tested. Restore Dangerous Goods, Valuables and Live Animals as classes (Issues C1, D7). Re-label "Other" with a review of the 639 rows.
- **Region:** map by country field first, keyword as fallback, with a test that fails on ambiguous matches (Issue C2).
- Keep extremes (do not trim at 60%). Handle them with robust loss functions and flag them for review (Issues D3, C3).
- Add the missing columns from the job ledger (carrier, sales person, job type) and, where possible, rate type, validity and scope (Issue D8).

### Fix 3. Point-in-time features
- Compute customer, lane and commodity history **strictly from jobs before the quote date**, per customer ID: last N won margins, median, count, days since last job, and ₹/kg margin history.
- Replace `Customer_Inquiry_Frequency` and `Customer_Tier` with these (Issues D5, P2). Tier cut-offs set from the distribution, not a fixed threshold that puts 80% in one class.
- Use one shared function for training and serving so they cannot drift.

### Fix 4. Replace the lookup-plus-signals engine with one learned model
- **Quantile gradient boosting** (P10 / P50 / P90) on **log markup or ₹/kg margin**, with absolute-error loss for the median. In my test, absolute loss alone cut MAE from 6.03 to 5.17 and removed most of the bias swing.
- Use **monotonic constraints only where commercially justified** (for example, margin ₹ must not fall when buy cost rises), and **fit cost elasticity from data per slab**, with confidence intervals. Remove the hand-set β values and the 200.18 threshold (Issue M3).
- Add **hierarchical shrinkage** (lane → port, commodity → slab, customer) as features or priors for sparse cells, not hand-typed overrides (Issues M2, M4).
- **Delete** the hardcoded margin tables, region branches and the Optuna "signal weights". Anything that is policy (floor, cap, approval) stays as an explicit, named guardrail applied after the model, with its own tests (Issue M1).
- Keep a simple lookup as the fallback and as the baseline the model must beat.

### Fix 5. Calibrated corridors
- Use **conformalised quantile regression** on a **time-ordered calibration set**, reported per segment (weight slab, commodity, known vs new customer).
- Show the real coverage in the UI. Do not label a corridor "90%" unless coverage is measured at 90% on a recent holdout (Issue M5).
- Expect wide intervals until new features are added: with today's columns the 90% half-width is about ±12.7 pp.

### Fix 6. Evaluation that can be trusted
- **Rolling-origin backtest by quote date** (for example train to July, test August; train to August, test September), plus a **customer-grouped** holdout to measure new-customer performance.
- Report on **natural distribution** and on stratified segments, each with MAE, bias, median error, ₹-weighted error and interval coverage.
- Benchmarks must exclude all training rows, exclude estimated buy costs, exclude duplicates, and use Won rows for margin accuracy. Evaluate Lost rows only through the win model.
- **Release gate:** beat commodity-median and lookup baselines on out-of-time MAE and bias in every major segment, with no segment worse than baseline.
- Tune with **Optuna inside nested time-based CV** on the production function itself. Never tune on the reporting benchmark (Issues O1, O2, O3).

### Fix 7. Add the missing information
This is the only step that can lower the roughly 5 pp floor substantially:
- Carrier, rate type (spot or contract), rate validity, scope of charges (freight only or with origin charges), sales person, credit days, competitor rate, and customer rate-card or contract flag.
- Test each in the backtest and keep only those that improve out-of-time results.

### Fix 8. One engine, one contract
- Retire either the two-stage ML engine or the margin framework, and keep one served model with a versioned artifact, a fixed input schema, and the training code in the repo.
- If win probability is wanted, train it on the correct population (Issue T1) and check calibration against the real 40.5% base rate.
- Make the UI send the customer name or ID so customer features work (Issue M7 notes the response gap).
- Add monitoring: weekly error by segment, interval coverage, input drift, and a trigger to retrain. Q4 has no history, so show a low-confidence flag and widen intervals until Q4 data exists.

---

## 6. Plan, expected effect and acceptance checks

| Phase | Work | Effect on gaps | Acceptance check |
|---|---|---|---|
| **0. Re-baseline** (1–2 days) | Remove duplicates, Lost default rows and in-sample overlap from the benchmark; recompute current-engine MAE, bias and coverage per segment | Gives the honest starting point (for Won deals about 5.8 pp MAE out of time) | Frozen benchmark and baseline report |
| **1. Data layer** (4–6 days) | Fixes 1–3: label definition, customer master, commodity and region lookups, date repair, point-in-time features, validation tests | Removes the Lost-vs-Won positive/negative split (B2) and removes the default-margin bias | Build fails on any broken rule; all D and P issues closed |
| **2. Model** (5–7 days) | Fix 4: quantile boosting, fitted elasticity, shrinkage, guardrails | Removes the M2 (−9 pp on Won "Other") and M3 biases; out-of-time MAE about 5.0 pp, bias near 0 on the current columns | Beats both baselines in every major segment |
| **3. Calibration and validation** (3–4 days) | Fixes 5–6: conformal corridors, rolling and customer-grouped backtests | Gives corridors that really cover 90%; shows new-customer vs known-customer error | Coverage 88–92% overall and per segment on a holdout |
| **4. New variables** (depends on data) | Fix 7 | The only step that can bring MAE well below 5 pp | Each variable improves out-of-time MAE |
| **5. Deployment** (3–4 days) | Fix 8: single engine, schema contract, UI sends customer, monitoring | Prevents the gaps from coming back | Shadow run on live quotes for 2–4 weeks before cut-over |

I cannot promise a final MAE. On the current columns, the realistic result on Won deals is about 5 pp MAE with near-zero bias and honest 90% intervals. "Close to actual" at the level of a point or two needs Fix 7 data.

---

## 7. Questions I need answered before starting

1. **Source of truth for margin:** is the target the quoted margin or the invoiced margin per job? Can I get true buy cost for won jobs?
2. **Lost quotes:** is there a field that says a quote was actually worked (versus an automatic 2% default)? Why do 30% of quotes take under a minute?
3. **Customer frequency:** how is `Customer_Inquiry_Frequency` calculated, and as of what date?
4. **Missing fields:** can I get carrier, rate type, rate validity, scope of charges and sales person for historical quotes?
5. **Commodity labels:** which source produced the current `Commodity_Group` (it is not reproducible from `preprocess.py`), and where are the DG, Valuables and Live Animals jobs?
6. **Benchmark script:** can you share the script that produced `benchmark_500_records.csv`, so I can confirm how the corridor columns were made?
7. **Approval:** may I start with Phase 0 and Phase 1 only, then bring the results back for review before any model change?

---

## Appendix: key numbers (for reproduction)

| Item | Value |
|---|---|
| Training rows / Won / Lost | 23,048 / 9,339 / 13,709 |
| Lost with 2% gross-margin default | 20.3% of Lost; 12.2% of all rows |
| Lost dropped by worked-quote filter | 13,087 (kept 622; win rate after filter 93.8%) |
| Won margin: median / mean / std | 7.03 / 11.17 / 10.88 |
| Lost margin: median / mean | 2.04 / 3.05 |
| Won rows with turnaround under 1 hour | 45.6% |
| Date mismatch rows | 422 |
| Customers: raw / normalised | 1,073 / 1,030 |
| Customer-tier "Key Account" share | 80.2% |
| Benchmark MAE / bias / R² | 3.61 / +0.11 / 0.37 |
| Benchmark corridor hit: file / app rule | 1.4% / 26.8% |
| Benchmark Won vs Lost bias | −2.10 / +2.25 |
| Signal swing bound | ±1.74% of margin (position ±0.127) |
| In-sample vs out-of-time lookup MAE (Won) | 5.85 / 5.83 |
| Conformal 90% half-width (Aug–Sep Won) | ±12.67 pp, coverage 90.4% |
| App-style corridor coverage (same data) | 31.6% |
| Won median margin by rate (₹/kg) | 50–150: ~9.7%; 200–300: 4.6%; 1,000–3,000: 13.2%; above 3,000: 14.9% |
