# Air Export Pricing Recommendation Engine — Technical Documentation
**Version:** 2.4  
**Target Audience:** Machine Learning Engineers, Data Scientists, Software Architects, and DevOps Teams

---

## 1. System Overview & Technology Stack

The **Air Export Pricing Recommendation Engine** is an enterprise hybrid machine learning system combining econometric tariff benchmarks, empirical Bayes adjustment signals, conformal prediction confidence bounds, and strict cargo operational guardrails.

### Technology Stack
- **Programming Language:** Python 3.11+
- **Machine Learning & Statistics:** `scikit-learn` (v1.3+), `numpy`, `pandas`, `scipy` (v1.11+)
- **Hyperparameter Optimization:** `optuna` (SQLite-backed Bayesian optimization)
- **Web Application & REST API:** Flask, Jinja2, HTML5/CSS3 (vanilla architecture, zero external CDN dependencies)
- **Database & Persistence:** SQLite3 (parameterized ACID store for quotation audits and leads management), `joblib` (serialized model bundles)
- **Execution & Orchestration:** Pure offline bash CLI runner (`run.sh`), zero cloud telemetry or external scraping dependencies

---

## 2. System Architecture & Workflow Pipeline

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                   INCOMING INQUIRY                                     │
│  Origin, Destination, Gross/Vol Wt, Total Buy INR, Commodity, Client Category, Terms   │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 0: PHYSICAL VALIDATION & COMMERCIAL GUARDRAILS (src/constants.py, src/tiers.py)  │
│  - Chargeable Wt = max(Gross, Volumetric) rounded to 0.5 kg increments                 │
│  - Weight-break arbitrage evaluation                                                   │
│  - Absolute Cost Floor: max(Buy * (1 + Floor%) + Ops, Buy + ₹1,500 + ₹3/kg * Wt)       │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: REGIONAL CORRIDOR BENCHMARKS & CONFORMAL CALIBRATION (src/margin_framework.py)│
│  - Hierarchical corridor clearing median: Lane x Slab -> Origin x Slab -> Slab         │
│  - Nonconformity Scores: R_i = |Actual - Target| with Finite-Sample correction         │
│  - Empirical Bayes smoothing (K=15) for sparse trade corridor coverage quantiles       │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: MULTI-SIGNAL COMMERCIAL ADJUSTERS                                             │
│  - Weights tuned via 150-Trial Bayesian Study: Comm, Cat, Cust, Mkt, Cpx, Comp         │
│  - Sublinear bulk cargo elasticity: Beta_bulk = 0.7356 for Wt > 500 kg                 │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: DECISION OPTIMIZATION & QUOTATION STRATEGY TIERS                              │
│  - Competitive Floor Tier                                                              │
│  - Balanced Target Quote (Invariant MAE 0.2295%)                                       │
│  - Premium Selective Tier                                                              │
│  - Conformal Negotiation Corridor [Min Margin %, Target %, Max %] (90% Coverage)       │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 4: PERSISTENCE & OFFLINE OBSERVABILITY (src/db.py, scripts/monitor_market_drift) │
│  - SQLite Audit Store: soft-delete, preset isolation, conversion tracking              │
│  - Offline Market Drift Watchdog: PSI & KS-Tests on airline buy rates                  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Mathematical Formulations & Algorithms

### A. Non-Negotiable Operational Cost Floor
Every quote is guarded by hard rupee operational break-even constraints:
$$\text{Cost Floor Price (INR)} = \max\Big(\text{Buy} \times (1 + \text{Floor \%} + \text{Risk Buffer}) + \text{Handling Fee}, \; \text{Buy} + \text{₹1,500/AWB} + (\text{₹3/kg} \times \text{Chargeable Wt})\Big)$$

### B. Sublinear Elasticity on Heavy Cargo
To prevent quoting uncompetitive rates on heavy cargo ($>500\text{ kg}$) when airline buy rates increase, the system applies sublinear pass-through elasticity:
$$\text{Base Margin}_{\text{bulk}} = \text{Base Margin} \times \left(\frac{\text{Unit Rate}}{\text{Threshold}}\right)^{\beta_{\text{bulk}} - 1}$$
Where $\beta_{\text{bulk}} = 0.7356$ and $\text{Threshold} = \text{₹}200.18/\text{kg}$. On heavy air shipments (e.g., BOM $\to$ FRA $800\text{ kg}$), a $+20\%$ airline buy cost increase produces strictly $+14.38\%$ profit growth ($< 15\%$), satisfying scenario test T1.

### C. Conformal Prediction Corridor Bounds (90% Coverage)
1. **Nonconformity Score:**
   $$R_i = |y_i - \hat{y}_i|$$
   where $y_i$ is historical clearing margin percentage on won inquiries, and $\hat{y}_i$ is the corridor benchmark median.
2. **Finite-Sample Quantile with Coverage Level $1 - \alpha = 0.90$:**
   $$\hat{q}_{0.90} = \text{Quantile}\left(R, \; \frac{\lceil (n + 1)(1 - \alpha) \rceil}{n}\right)$$
3. **Empirical Bayes Smoothing:**
   $$\hat{q}_{\text{lane}} = \frac{n}{n + K} \hat{q}_{\text{lane}} + \frac{K}{n + K} \hat{q}_{\text{global}} \quad (K = 15)$$
4. **Adaptive Pre-Approved Corridor:**
   $$X\% = \max(\text{Cost Floor \%}, \; M\% - \hat{q}_{0.90})$$
   $$Y\% = \min(\text{Commercial Cap \%}, \; M\% + \hat{q}_{0.90})$$
   where $M\%$ is the balanced target quote. $M\%$ remains invariant, preserving our **0.2295% MAE** while guaranteeing **100% empirical coverage** across benchmark test inquiries.

### D. Offline Market Drift Detection (PSI & KS-Test)
1. **Population Stability Index (PSI):**
   $$\text{PSI} = \sum_{b=1}^{B} (P_b - Q_b) \ln\left(\frac{P_b}{Q_b}\right)$$
   - $\text{PSI} < 0.10$: Market Stable (No retrain needed; current: `0.0812`).
   - $0.10 \le \text{PSI} < 0.25$: Moderate Drift (Airline rate fluctuation).
   - $\text{PSI} \ge 0.25$: Significant Shift (Retraining recommended).
2. **Two-Sample Kolmogorov-Smirnov (KS) Test:**
   Evaluates continuous shift on `Unit_Buy_INR_Per_Kg` across primary hubs (BOM, DEL, BLR, CCJ) to alert on route-level fuel shocks or capacity crunches.

---

## 4. Codebase Organization & Module Structure

| Path | Purpose |
| :--- | :--- |
| [`src/margin_framework.py`](file:///Users/deepak/Downloads/air_export_pricing_engine/src/margin_framework.py) | Regional corridor benchmarks, multi-signal adjusters, Conformal Prediction corridors. |
| [`src/engine.py`](file:///Users/deepak/Downloads/air_export_pricing_engine/src/engine.py) | Two-Stage recommendation engine class and optimization loop. |
| [`src/train.py`](file:///Users/deepak/Downloads/air_export_pricing_engine/src/train.py) | Model training, out-of-fold cross-validation, and serialized joblib exporter. |
| [`src/preprocess.py`](file:///Users/deepak/Downloads/air_export_pricing_engine/src/preprocess.py) | IATA physical checks, port normalizations, cyclical date features. |
| [`src/preprocess_job_consolidated.py`](file:///Users/deepak/Downloads/air_export_pricing_engine/src/preprocess_job_consolidated.py) | Cleaning and data engineering pipeline for operational shipment ledgers. |
| [`src/tiers.py`](file:///Users/deepak/Downloads/air_export_pricing_engine/src/tiers.py) | Weight breaks, customer frequency tiering, weight-break arbitrage checks. |
| [`src/constants.py`](file:///Users/deepak/Downloads/air_export_pricing_engine/src/constants.py) | Commercial floor parameters, commodity buffers, maximum margin ladders. |
| [`src/db.py`](file:///Users/deepak/Downloads/air_export_pricing_engine/src/db.py) | SQLite audit trail, soft deletes, preset quote isolation, outcome tracking. |
| [`src/app.py`](file:///Users/deepak/Downloads/air_export_pricing_engine/src/app.py) | Web application, `/api/quote` endpoint, leads dashboard. |
| [`scripts/monitor_market_drift.py`](file:///Users/deepak/Downloads/air_export_pricing_engine/scripts/monitor_market_drift.py) | Offline market and concept drift watchdog utility. |
| [`scripts/tune_engine_optuna.py`](file:///Users/deepak/Downloads/air_export_pricing_engine/scripts/tune_engine_optuna.py) | Local Bayesian hyperparameter optimization script. |
| [`scripts/verify_vocab_alignment.py`](file:///Users/deepak/Downloads/air_export_pricing_engine/scripts/verify_vocab_alignment.py) | Dropdown vs. model encoder vocabulary alignment gate. |

---

## 5. Configuration, CLI Runner & Integration

### Unified CLI Runner (`./run.sh`)
```bash
./run.sh verify      # Runs 36/36 vocabulary alignment gate
./run.sh preprocess  # Executes data cleaning pipeline
./run.sh train       # Trains Two-Stage ML pipeline & builds conformal tables
./run.sh test        # Runs full unit and scenario test suite (99 tests)
./run.sh drift       # Executes offline market drift watchdog
./run.sh serve       # Starts web application on 127.0.0.1:5050
./run.sh pipeline    # Executes end-to-end pipeline: Preprocess -> Train -> Verify -> Test -> Serve
```

### Production REST API Endpoint
- **URL:** `POST /api/quote`
- **Request Headers:** `Content-Type: application/json`
- **Request Body:**
```json
{
  "origin": "Chhatrapati Shivaji Maharaj International Airport",
  "dest": "Frankfurt am Main",
  "buy": 160000,
  "chargeable_wt": 800,
  "client": "Shipper / Consignee",
  "commodity": "General Cargo",
  "strategy": "balanced"
}
```
- **Response Structure:**
```json
{
  "status": "success",
  "optimal": {
    "Strategy_Key": "balanced",
    "Margin_Percentage": 4.47,
    "Quoted_Sell_Price_INR": 167152.0,
    "Rate_Per_Kg": 208.94
  },
  "negotiation_corridor": {
    "min_margin_pct": 3.70,
    "target_margin_pct": 4.47,
    "max_margin_pct": 5.24,
    "confidence_coverage": "90%",
    "conformal_half_width_pct": 0.77,
    "guidance": "Pre-approved negotiation corridor: 3.70% to 5.24%..."
  },
  "strategic_recommendations": {
    "floor": { ... },
    "balanced": { ... },
    "premium": { ... }
  }
}
```

---

## 6. Testing, Security & Verification Protocols

### Verification Gates
1. **Unit & Scenario Testing:** `./run.sh test` runs 99 automated test cases covering realistic freight scenarios (T1–T15), IATA physics, database migrations, and conformal coverage bounds.
2. **Vocabulary Alignment:** `./run.sh verify` validates that all 36 UI dropdown options map 100% to the trained encoder categories.
3. **Data Security & Privacy:**
   - Pure offline execution: zero external network calls, zero web scraping, zero cloud telemetry.
   - Raw data files (`*.xlsx`, `*.csv`) and serialized binaries (`*.joblib`, `*.db`) are strictly untracked in `.gitignore`.
   - SQL queries in `src/db.py` use strict parameterized statements (`?` placeholders).

---

## 7. Known Limitations & Future Roadmap

1. **Trade Lane Sparsity:**
   - *Current Behavior:* Sparse lanes fall back to regional cohorts (`Origin_Region -> Destination_Region`) with Empirical Bayes smoothing.
   - *Roadmap:* Incorporate carrier-specific slab rate sheets when master contract rates are loaded.
2. **Carrier Contract Dynamic Integration:**
   - *Current Behavior:* Evaluates dynamic competitor quotes and airline buy rates entered per inquiry.
   - *Roadmap:* Direct offline ingestion of monthly airline Master Service Agreement (MSA) net rate sheets.
3. **Automated Retrain Triggers:**
   - *Current Behavior:* `scripts/monitor_market_drift.py` provides manual guidance when $\text{PSI} \ge 0.25$.
   - *Roadmap:* Cron-scheduled batch trigger that prompts administrators when market drift exceeds critical thresholds.
