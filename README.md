# Air Export Pricing Recommendation Engine (v2.3)

An enterprise machine learning and commercial decision engine for recommending profit-maximizing margins, strategic postures, and quoted selling prices for international air freight forwarding.

---

## 1. Key Capabilities & Architecture

The engine integrates hard freight-forwarding operational guardrails with an honest, leak-free Two-Stage machine learning pipeline:

```
[Inquiry Input] ──► [IATA Physical & Guardrail Layer] ──► [Empirical Cohort Quantiles]
                                                                     │
┌────────────────────────────────────────────────────────────────────┘
▼
[Stage 1A: Benchmark Regressor] ──► Predicted Clearing Margin %
                                              │
                                              ▼
[Stage 1B: Calibrated Elasticity Clf] ──► Calibrated Win Probability Curve
                                              │
                                              ▼
[Two-Stage Decision Optimizer] ──► Strict Strategic Tiers (Floor, Balanced, Premium)
                                              │
                                              ▼
[SQLite Audit Store] ◄── Quoted Rates, Margin %, Customer & Inquiry Metadata
```

### Commercial Operational Guardrails
- **IATA Physical Plausibility:** Validates $\text{Chargeable Weight} \ge \max(\text{Gross Weight}, \text{Volumetric Weight})$ with standard airfreight $0.5\text{ kg}$ ceiling increments and commercial weight bounds ($0.5\text{ kg}$ to $25,000\text{ kg}$).
- **Non-Negotiable Cost Floors:** Guarantees every quotation respects commercial operational floors:
  $$\text{Cost Floor} = \max\Big(\text{Buy} \times (1 + \text{Min \%} + \text{Risk Buffer}) + \text{Ops Fee}, \quad \text{Buy} + \text{₹1,500/AWB} + (\text{₹3/kg} \times \text{Chargeable Wt})\Big)$$
- **Weight-Break Arbitrage Checker:** Automatically detects when declaring cargo at the next higher IATA weight slab (e.g., billing 98 kg at the +100 kg break rate) yields a lower total bill for the shipper.
- **Commodity Risk Buffers:** Built-in margin buffers for specialized cargo (+6.0% Dangerous Goods, +4.0% Pharmaceuticals, +3.5% Perishables, +5.0% Valuables).
- **Out-of-Window Peak Season Safeguard:** Explicit `quote_date` input detects Q4 peak surge inquiries outside historical 9-month training distribution, flagging them with lower confidence.

### Honest Machine Learning & Purity
- **Pure Quote Resolution:** Excludes unpriced Draft, Hold, and Cancelled entries; trains strictly on genuine, resolved Won ($40.5\%$) vs. Lost ($59.5\%$) customer-facing quotes.
- **Outcome Leakage Removed:** Completely eliminated post-inquiry features (`quote_revision_count`, `Booked by branch`) from machine learning inputs.
- **Rate-Per-Kg Representation:** Replaced raw airline INR totals with `Buy_Rate_Per_Kg`, corridor-level `Lane_Rate_Index`, and `Log_Chargeable_Weight`, resolving large-shipment rate saturation.
- **Honest Out-Of-Fold (OOF) Elasticity:** Uses 5-fold cross-validated out-of-fold benchmark margin ratios without synthetic logit penalization, delivering calibrated win probability curves.
- **Empirical Cohort Anchoring:** Decision engine anchors strategy tiers against empirical won-deal margin quantiles (P25, P50, P75, P90) computed by trade corridor, weight slab, and commodity group.
- **Strict Strategic Differentiation:** Solves duplicate recommendation issues by strictly differentiating Floor, Balanced, and Premium tiers with active win rate thresholds ($P(\text{Win}) \ge 40\%$ for Floor, $\ge 25\%$ for Balanced, $\ge 15\%$ for Premium).

---

## 2. Project Directory Structure

```
air_export_pricing_engine/
│
├── data/
│   ├── raw/                                  # Source inquiries
│   ├── processed/
│   │   └── Air_Export_Pricing_Combined_ML.csv # Cleaned & deduplicated training data
│   └── recommendations_history.db            # SQLite audit trail & leads store
│
├── models/
│   └── freight_margin_recommender.joblib     # Production model artifact bundle (v2.3)
│
├── src/
│   ├── __init__.py
│   ├── constants.py                          # Commercial guardrails, fees, and thresholds
│   ├── tiers.py                              # Weight tiers, breaks, arbitrage & account tiers
│   ├── preprocess.py                         # Cleaning, IATA physics, and geographic mapping
│   ├── train.py                              # Two-Stage ML training with honest OOF benchmarks
│   ├── engine.py                             # Decision engine, cohort lookup & strategic tiers
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
│   ├── test_db_audit.py                      # SQLite migrations, soft delete & preset isolation
│   ├── test_api_quote.py                     # Web API integration & HTTP 400 validation
│   └── test_scenarios_t1_t15.py              # 15 realistic end-to-end air-export scenarios
│
├── scripts/
│   ├── verify_vocab_alignment.py             # UI vs. model encoder vocabulary alignment gate
│   └── test_runner.py                        # Standalone regression test runner
│
├── docs/
│   ├── AIR_EXPORT_PRICING_TEAM_GUIDE.md      # Sales & pricing team operational guide
│   └── evaluation_plots.png                  # Evaluation diagnostic curves
│
├── requirements.txt                          # Locked dependencies
├── run.sh                                    # Unified project CLI runner
└── README.md                                 # Project documentation
```

---

## 3. Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Preprocess Data
Clean and validate raw inquiry records into pure resolved quotes:
```bash
./run.sh preprocess
# Or: python src/preprocess.py
```

### 3. Train Model
Train Stage 1A (Benchmark Regressor) and Stage 1B (Calibrated Elasticity Classifier):
```bash
./run.sh train
# Or: python src/train.py
```

### 4. Run Test Suite
Execute the full 70-test suite, including all 15 realistic scenario tests:
```bash
./run.sh test
# Or: python -m unittest discover tests
```

### 5. Verify Vocabulary Alignment
Check 100% alignment between UI dropdowns and trained categorical features:
```bash
./run.sh verify
# Or: python scripts/verify_vocab_alignment.py
```

### 6. Launch Web UI & Quoting API
Start the Flask application:
```bash
./run.sh serve
# Or: python src/app.py
```
Open **http://127.0.0.1:5050** in your browser.

---

## 4. Automated Realistic Scenarios (T1–T15)

The automated scenario suite (`tests/test_scenarios_t1_t15.py`) validates critical commercial edge cases:

| Scenario | Description | Verified Expected Behavior |
|:---|:---|:---|
| **T1** | Rate Shock Pass-Through | 20% buy spike on 800kg BOM→FRA strictly increases quoted sell price without dropping rupee gross margin. |
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
Generate strategic recommendations and sensitivity tables:
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
Response includes `optimal` quote, `strategic_recommendations` (`floor`, `balanced`, `premium`), full `sensitivity` frontier, and audit trail status.
