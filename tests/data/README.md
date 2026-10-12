# Air Export Pricing Engine — Test Case Repository & Documentation

This directory contains the golden benchmark test cases and verification datasets used to evaluate and ensure the accuracy, fairness, and commercial compliance of the Air Export Pricing Recommendation Engine.

---

## 1. Test Datasets Summary

| File | Records | Format | Purpose & Description |
| :--- | :---: | :---: | :--- |
| **`benchmark_500_records.csv`** | 500 | CSV | **Segmented Production Benchmark:** 5 Commodity Types $\times$ 5 Budget Quintiles ($N=20$ per cell). Evaluates the 3-key hierarchical corridor model, checking for over-quoting and under-quoting across all commercial tiers. |
| **`benchmark_100_optimization_cases.json`** | 100 | JSON | **Corridor & Hyperparameter Evaluation:** Stratified multi-corridor inquiries across international trade lanes used to tune corridor spread, win probability calibration, and empirical Bayes shrinkage. |
| **`unit_test_cases_20.json`** | 20 | JSON | **Unit Regression Suite:** 20 stratified historical cases (40% Won, 60% Lost) used by `tests/test_model_unit_eval.py` to assert discrimination (AUC $\ge 0.85$) and calibration integrity. |

---

## 2. Benchmark 500 Records Specification

- **Stratification Design:**
  - **5 Commodities:** Engineering & Machinery, Garments / Textiles, General Cargo, Other, Perishable Foodstuff.
  - **5 Budget Segments:** Very Low Budget (<₹25k), Low Budget (₹25k–₹60k), Medium Budget (₹60k–₹150k), High Budget (₹150k–₹400k), Very High Budget (>₹400k).
  - **Sampling:** 20 records sampled deterministically per cell (seed `42`).
- **Columns Included:**
  - `Commodity_Group`, `Budget_Segment`, `Total_Buy_INR`, `Chargeable_Weight_Kg`
  - `Actual_Margin_Pct`: Historical quote margin percentage
  - `Model_Target_Margin_Pct`: Engine recommended margin percentage (Balanced tier)
  - `Model_Min_Margin_Pct`: Recommended floor margin
  - `Model_Max_Margin_Pct`: Recommended premium margin
  - `Margin_Gap`: Model Quoted - Actual Quoted
  - `Abs_Error`: Absolute prediction error
  - `Corridor_Hit`: Whether actual margin fell within recommended corridor bounds

---

## 3. How to Execute Tests

```bash
# 1. Run full unit test suite (103+ unit tests)
./run.sh test

# 2. Run the 20-case unit evaluation
python -m unittest tests/test_model_unit_eval.py

# 3. Verify vocabulary alignment with UI dropdowns
python scripts/verify_vocab_alignment.py

# 4. Run the 7 calibrated test cases from margin_engine_test_cases.md
python scripts/test_runner.py
```
