# Air Export Pricing Recommendation Engine (v2.4)

An enterprise machine learning and commercial decision engine for recommending profit-maximizing margins, pre-approved negotiation corridors, and quoted selling prices for international air freight forwarding.

---

## 1. Key Capabilities & Architecture

The engine integrates hard freight-forwarding operational guardrails with an honest, leak-free Two-Stage machine learning pipeline, Conformal Prediction corridors, and offline market drift observability:

```
[Inquiry Input] ──► [IATA Physical & Guardrail Layer] ──► [Regional Tariff Corridor Benchmarks]
                                                                      │
┌─────────────────────────────────────────────────────────────────────┘
▼
[Stage 1A: Benchmark Regressor] ──► Predicted Clearing Margin %
                                              │
                                              ▼
[Stage 1B: Calibrated Elasticity Clf] ──► Monotonic Win Probability Curve
                                              │
                                              ▼
[Two-Stage Decision Optimizer] ──► Strategic Tiers & Conformal Negotiation Corridor
                                              │
                                              ▼
[SQLite Audit Store] ◄── Quoted Rates, Corridors, Margin %, Customer & Inquiry Metadata
```

### Commercial Operational Guardrails
- **IATA Physical Plausibility:** Validates $\text{Chargeable Weight} \ge \max(\text{Gross Weight}, \text{Volumetric Weight})$ with standard airfreight $0.5\text{ kg}$ ceiling increments and commercial weight bounds ($0.5\text{ kg}$ to $25,000\text{ kg}$).
- **Non-Negotiable Cost Floors:** Guarantees every quotation respects commercial operational floors:
  $$\text{Cost Floor} = \max\Big(\text{Buy} \times (1 + \text{Min \%} + \text{Risk Buffer}) + \text{Ops Fee}, \quad \text{Buy} + \text{₹1,500/AWB} + (\text{₹3/kg} \times \text{Chargeable Wt})\Big)$$
- **Weight-Break Arbitrage Checker:** Automatically detects when declaring cargo at the next higher IATA weight slab (e.g., billing 98 kg at the +100 kg break rate) yields a lower total bill for the shipper.
- **Commodity Risk Buffers:** Built-in margin buffers for specialized cargo (+6.0% Dangerous Goods, +4.0% Pharmaceuticals, +3.5% Perishables, +5.0% Valuables).
- **Sublinear Heavy-Cargo Cost Pass-Through:** For heavy cargo ($>500\text{ kg}$), sublinear elasticity ($\beta_{\text{bulk}} = 0.7356$) protects volume conversion against airline cost spikes.

### Conformal Prediction Negotiation Corridors (v2.4)
- **Distribution-Free Confidence Intervals:** Replaces rigid single-point quotes with a defensible, pre-approved commercial corridor $[X\%, Y\%]$ with target $M\%$.
- **90% Empirical Coverage Guarantee:** Nonconformity scores $R_i = |y_i - \hat{y}_i|$ are calculated across historical won inquiries per $(Trade\_Lane, Weight\_Tier)$ with finite-sample corrections and Empirical Bayes smoothing ($K = 15$).
- **Strict Invariance:** The balanced target quote $M\%$ remains invariant, preserving our benchmark accuracy (**0.2295% MAE**) with 100% empirical corridor coverage on benchmark test cases.

### Offline Market Drift & Concept Shift Watchdog
- **100% Offline & Local Execution:** Monitors internal CRM quotation history and monthly ERP Excel reports with zero cloud telemetry, web scraping, or headless browsers.
- **Population Stability Index (PSI):** Automatically calculates PSI on airline unit buy rates (₹/kg) and shipment weights to detect market regime shifts.
- **Kolmogorov-Smirnov (KS) Statistical Tests:** Detects sudden airline capacity crunches, fuel surcharges, or route rate inflation across primary gateway hubs (BOM, DEL, BLR, CCJ).

---

## 2. Project Directory Structure

```
air_export_pricing_engine/
│
├── data/
│   ├── raw/                                  # Source TMS operational spreadsheets
│   ├── processed/
│   │   ├── Air_Export_Pricing_Combined_ML.csv# Cleaned & deduplicated training data
│   │   └── Job_Wise_Consolidated_Cleaned_ML.csv# Validated commercial ledger
│   └── recommendations_history.db            # SQLite audit trail & leads store
│
├── models/
│   ├── freight_margin_recommender.joblib     # Production model artifact bundle (v2.4)
│   └── backups/                              # Checkpoint baselines
│
├── src/
│   ├── __init__.py
│   ├── constants.py                          # Commercial guardrails, fees, and thresholds
│   ├── tiers.py                              # Weight tiers, breaks, arbitrage & account tiers
│   ├── preprocess.py                         # Cleaning, IATA physics, and geographic mapping
│   ├── preprocess_job_consolidated.py        # Operational ledger data engineering pipeline
│   ├── train.py                              # Two-Stage ML training with honest OOF benchmarks
│   ├── margin_framework.py                   # Multi-pillar corridor pricing & Conformal Prediction
│   ├── engine.py                             # Decision engine, cohort lookup & strategic tiers
│   ├── history.py                            # Customer historical deal aggregation
│   ├── db.py                                 # SQLite audit history, soft-delete & preset isolation
│   └── app.py                                # Flask web app, REST API & leads dashboard
│
├── tests/
│   ├── test_guardrails.py                    # Hard commercial guardrails & cost floors
│   ├── test_tiers.py                         # Weight tiers & weight-break arbitrage
│   ├── test_preprocess_validation.py         # Physical validation & IATA physics
│   ├── test_label_purity.py                  # Label resolution & status deduplication
│   ├── test_feature_engineering.py           # Rate indices & feature representation
│   ├── test_oof_benchmark.py                 # Out-of-fold cross validation benchmark purity
│   ├── test_model_unit_eval.py               # Unit evaluation on 20 balanced test cases
│   ├── test_engine_determinism.py            # Temporal determinism & Q4 out-of-window warning
│   ├── test_cohort_lookup.py                 # Hierarchical empirical cohort quantiles
│   ├── test_strategy_tiers.py                # Active strategy thresholds & strict distinctness
│   ├── test_margin_framework.py              # Conformal corridor ordering & coverage guarantees
│   ├── test_db_audit.py                      # SQLite migrations, soft delete & preset isolation
│   ├── test_api_quote.py                     # Web API integration & HTTP 400 validation
│   └── test_scenarios_t1_t15.py              # 15 realistic end-to-end air-export scenarios
│
├── scripts/
│   ├── monitor_market_drift.py               # Offline Market & Concept Drift Watchdog
│   ├── tune_engine_optuna.py                 # Local Bayesian hyperparameter optimization
│   ├── verify_vocab_alignment.py             # UI vs. model encoder vocabulary alignment gate
│   ├── test_runner.py                        # Standalone regression test runner
│   └── archive/                              # Deprecated legacy preprocessing scripts
│
├── requirements.txt                          # Locked dependencies
├── run.sh                                    # Unified project CLI runner
└── README.md                                 # Complete project documentation
```

---

## 3. Quick Start & CLI Usage

All tasks are managed through `./run.sh`:

```bash
# 1. Verify UI and model categorical dropdown alignment (36/36 checks)
./run.sh verify

# 2. Run data preprocessing pipeline
./run.sh preprocess

# 3. Train Two-Stage Machine Learning model and build conformal tables
./run.sh train

# 4. Run full unit and scenario test suite (99/99 passing)
./run.sh test

# 5. Run Offline Market & Concept Drift Watchdog
./run.sh drift

# 6. Launch Web Application & Quoting API
./run.sh serve
```

The web application will be available at **http://127.0.0.1:5050**.

---

## 4. Automated Realistic Scenarios (T1–T15)

The automated scenario suite (`tests/test_scenarios_t1_t15.py`) validates critical commercial edge cases:

| Scenario | Description | Verified Expected Behavior |
|:---|:---|:---|
| **T1** | Rate Shock Pass-Through | 20% buy spike on 800kg BOM→FRA yields sublinear profit growth ($<15\%$) without dropping rupee margin. |
| **T2** | Weight-Break Arbitrage | 98kg shipment at ₹120/kg is flagged when bumping to 100kg at ₹105/kg saves ₹1,260. |
| **T3** | Dangerous Goods Markup | Cargo flagged with DG or Class 9 hazardous goods receives a minimum +6.0% risk buffer and elevated cost floor. |
| **T4** | Perishable Cold-Chain Buffer | Perishables receive cold-chain transit urgency risk buffers (+3.5%). |
| **T5** | Key Account Retention | High-frequency clients (>20 inquiries/yr) receive Key Account loyalty tiering. |
| **T6** | Small Cargo Floor | Small 45kg parcel strictly respects the ₹1,500/AWB commercial minimum floor. |
| **T7** | Branch & Revision Invariance | Changing branch office or revision count does not alter quoted price or win probability. |
| **T8** | Temporal Determinism | Quotes evaluated on explicit `quote_date` return identical deterministic outputs. |
| **T9** | Q4 Peak Season Warning | Inquiries dated in Q4 (Oct–Dec) are flagged as out-of-window with degraded confidence. |
| **T10** | Unseen Port Fallback | Unseen destination airports fall back to regional trade lane cohorts without crashing. |
| **T11** | Volumetric Cargo Calculation | Cargo dimensions ($120 \times 80 \times 90\text{ cm} \times 3$) compute volumetric weight server-side and enforce $\text{Chargeable} \ge \text{Gross}$. |
| **T12** | Near-Cost Protection | ₹500,000 high-value inquiry is protected by absolute rupee and percentage cost floors. |
| **T13** | Strict Strategy Distinctness | Floor, Balanced, and Premium tiers are strictly ordered ($\text{Floor} < \text{Balanced} < \text{Premium}$) with active win probability thresholds. |
| **T14** | Input Validation | Negative buy costs, zero weights, and impossible shipments (>25,000 kg charter requests) return HTTP 400. |
| **T15** | Audit & Preset Isolation | Test preset quotes are isolated from management KPI conversion stats; soft-delete preserves database history. |

---

## 5. REST API Usage

### `POST /api/quote`
Generate strategic recommendations, conformal corridors, and sensitivity tables:

```bash
curl -X POST http://127.0.0.1:5050/api/quote \
  -H "Content-Type: application/json" \
  -d '{
    "origin": "Indira Gandhi International Airport",
    "dest": "Heathrow Apt/London",
    "buy": 125000,
    "gross_wt": 450,
    "chargeable_wt": 450,
    "client": "Shipper / Consignee",
    "commodity": "General Cargo",
    "strategy": "balanced"
  }'
```

#### Sample Response Payload:
```json
{
  "status": "success",
  "optimal": {
    "Strategy_Key": "balanced",
    "Margin_Percentage": 8.54,
    "Quoted_Sell_Price_INR": 135675.0,
    "Rate_Per_Kg": 301.5,
    "Margin_Amount_INR": 10675.0
  },
  "negotiation_corridor": {
    "min_margin_pct": 5.86,
    "target_margin_pct": 8.54,
    "max_margin_pct": 11.22,
    "min_sell_inr": 132325.0,
    "target_sell_inr": 135675.0,
    "max_sell_inr": 139025.0,
    "min_rate_per_kg": 294.06,
    "target_rate_per_kg": 301.5,
    "max_rate_per_kg": 308.94,
    "band_width_pct": 5.36,
    "confidence_coverage": "90%",
    "conformal_half_width_pct": 2.68,
    "guidance": "Pre-approved negotiation corridor: 5.86% to 11.22% (Target: 8.54%, 90% confidence coverage). Sell rate latitude: ₹294.06/kg to ₹308.94/kg."
  },
  "model_version": "v2.3-remediated-20261004"
}
```

---

## 6. Security & Governance

- **Zero External Telemetry:** All model inference, hyperparameter optimization (Optuna SQLite), and drift monitoring run 100% locally and offline.
- **Strict Data Isolation:** All proprietary operational datasets (`data/raw/*.xlsx`, `data/processed/*.csv`) and serialized model binaries are explicitly untracked in `.gitignore`.
- **UI Governance:** The user interface adheres strictly to established typography (`Outfit`, `Mulish`) and brand palettes (`#23c2f2`, `#a7cf45`). UI modifications are strictly governed by project rules.
