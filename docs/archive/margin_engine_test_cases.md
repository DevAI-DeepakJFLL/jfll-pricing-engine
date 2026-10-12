# Margin Engine — Test Inputs & Expected Outputs

**Document Version:** 2.4 (Enterprise Calibrated & Guardrail Enforced)  
**Reference Model:** `v2.4` (`models/freight_margin_recommender.joblib`)  
**Engine Implementation:** `src/engine.py` (`MarginRecommendationEngine`)  
**Automated Verification:** `python scripts/test_runner.py`  

---

## Case 1 — Key Account, dense >1000kg courier, Delhi → Abu Dhabi

**Commercial Context & Rationale:**
- **Customer:** SKYNET EXPRESS INDIA PRIVATE LIMITED (`Customer_Inquiry_Frequency: 32`, Key Account tier).
- **Historical Ground Truth:** In actual historical won inquiries, transactions on this lane clear around ~4.5%–5.5% margin.
- **Calibrated Engine Behavior:** The calibrated engine activates the **Account Retention Guardrail** and recommends **13.74% margin** with a **98.06% win rate**, balancing account retention with healthy profitability.

**Input**
```json
{
  "Total_Buy_INR": 120520.0,
  "Chargeable_Weight_Kg": 1048.0,
  "Gross_Weight_Kg": 1000.0,
  "Cargo_Density_Ratio": 0.9542,
  "Weight_Tier": ">1000kg",
  "Business_Vertical": "Courier",
  "Commodity_Group": "General Cargo",
  "Origin_Region": "North India",
  "Destination_Region": "Middle East",
  "Regional_Lane": "North India -> Middle East",
  "Origin_Port": "Indira Gandhi International Airport",
  "Destination_Port": "Zayed International Airport",
  "Company_Group": "Shipper / Consignee",
  "Customer_Inquiry_Frequency": 32,
  "Customer_Tier": "Key Account",
  "Booked by branch": "Delhi",
  "Incoterms": "CFR",
  "quote_revision_count": 2
}
```

**Expected Output**
```json
{
  "Benchmark_Margin_pct": 5.28,
  "Recommended_Margin_pct": 13.74,
  "Quoted_Sell_Price_INR": 137079.45,
  "Margin_Amount_INR": 16559.45,
  "Win_Probability": 0.9806,
  "Expected_Profit_INR": 13802.67
}
```

---

## Case 2 — Small <45kg parcel, occasional subagent, Bangalore → Stuttgart

**Commercial Context & Rationale:**
- **Profile:** Small 24kg courier parcel to Europe.
- **Calibrated Engine Behavior:** Low-weight parcels support higher commercial margins while being bound by the subagent wholesale category ceiling (28%). The engine recommends **27.51% margin** with a **98.55% win rate**.

**Input**
```json
{
  "Total_Buy_INR": 13092.0,
  "Chargeable_Weight_Kg": 24.0,
  "Gross_Weight_Kg": 24.0,
  "Cargo_Density_Ratio": 1.0,
  "Weight_Tier": "<45kg",
  "Business_Vertical": "Air Export Forwarding",
  "Commodity_Group": "General Cargo",
  "Origin_Region": "South India",
  "Destination_Region": "Europe",
  "Regional_Lane": "South India -> Europe",
  "Origin_Port": "Bangalore",
  "Destination_Port": "Stuttgart",
  "Company_Group": "Subagent",
  "Customer_Inquiry_Frequency": 3,
  "Customer_Tier": "Occasional",
  "Booked by branch": "Bangalore",
  "Incoterms": "FOB",
  "quote_revision_count": 2
}
```

**Expected Output**
```json
{
  "Benchmark_Margin_pct": 20.45,
  "Recommended_Margin_pct": 27.51,
  "Quoted_Sell_Price_INR": 16693.61,
  "Margin_Amount_INR": 3601.61,
  "Win_Probability": 0.9855,
  "Expected_Profit_INR": 3549.41
}
```

---

## Case 3 — Heavy 5000kg auto parts, subagent, Mumbai → Frankfurt

**Commercial Context & Rationale:**
- **Profile:** Heavy bulk automotive cargo (5 metric tons) on the high-volume West India to Europe lane.
- **Calibrated Engine Behavior:** Stage 1B probability calibration and bulk cargo pass-through protect volume conversion, recommending **12.17% margin** with a high **95.01% win rate**.

**Input**
```json
{
  "Total_Buy_INR": 500000.0,
  "Chargeable_Weight_Kg": 5000.0,
  "Gross_Weight_Kg": 5000.0,
  "Cargo_Density_Ratio": 1.0,
  "Weight_Tier": ">1000kg",
  "Business_Vertical": "Air Export Forwarding",
  "Commodity_Group": "Auto Parts",
  "Origin_Region": "West India",
  "Destination_Region": "Europe",
  "Regional_Lane": "West India -> Europe",
  "Origin_Port": "Chhatrapati Shivaji Maharaj International Airport",
  "Destination_Port": "Frankfurt am Main",
  "Company_Group": "Subagent",
  "Customer_Inquiry_Frequency": 4,
  "Customer_Tier": "Occasional",
  "Booked by branch": "Mumbai",
  "Incoterms": "FOB",
  "quote_revision_count": 1
}
```

**Expected Output**
```json
{
  "Benchmark_Margin_pct": 8.08,
  "Recommended_Margin_pct": 12.17,
  "Quoted_Sell_Price_INR": 560850.0,
  "Margin_Amount_INR": 60850.0,
  "Win_Probability": 0.9501,
  "Expected_Profit_INR": 57814.4
}
```

---

## Case 4 — Historical reference: 4kg parcel, Ahmedabad → Rio de Janeiro

**Commercial Context & Rationale:**
- **Profile:** Lightweight long-haul consignment to Latin America.
- **Calibrated Engine Behavior:** The calibrated model bounds the quote safely within the subagent ceiling ladder, recommending **27.66% margin** with a **98.92% win rate**.

**Input**
```json
{
  "Total_Buy_INR": 16500.0,
  "Chargeable_Weight_Kg": 4.0,
  "Gross_Weight_Kg": 4.0,
  "Cargo_Density_Ratio": 1.0,
  "Weight_Tier": "<45kg",
  "Business_Vertical": "Air Export Forwarding",
  "Commodity_Group": "General Cargo",
  "Origin_Region": "West India",
  "Destination_Region": "Latin America",
  "Regional_Lane": "West India -> Latin America",
  "Origin_Port": "Ahmedabad",
  "Destination_Port": "Galeao Apt/Rio de Janeiro",
  "Company_Group": "Subagent",
  "Customer_Inquiry_Frequency": 4,
  "Customer_Tier": "Occasional",
  "Booked by branch": "Ahmedabad",
  "Incoterms": "FOB",
  "quote_revision_count": 1
}
```

**Expected Output**
```json
{
  "Benchmark_Margin_pct": 15.08,
  "Recommended_Margin_pct": 27.66,
  "Quoted_Sell_Price_INR": 21063.9,
  "Margin_Amount_INR": 4563.9,
  "Win_Probability": 0.9892,
  "Expected_Profit_INR": 4514.63
}
```

---

## Case 5 — High-weight General Cargo reference: 2500kg, Delhi → Kathmandu

**Commercial Context & Rationale:**
- **Profile:** High-weight cross-border shipment to neighboring Nepal.
- **Calibrated Engine Behavior:** Correctly models competitive clearing rates for regional air freight, recommending **27.29% margin** with an expected profit of ₹18,221.82.

**Input**
```json
{
  "Total_Buy_INR": 70000.0,
  "Chargeable_Weight_Kg": 2500.0,
  "Gross_Weight_Kg": 2500.0,
  "Cargo_Density_Ratio": 1.0,
  "Weight_Tier": ">1000kg",
  "Business_Vertical": "Air Export Forwarding",
  "Commodity_Group": "General Cargo",
  "Origin_Region": "North India",
  "Destination_Region": "Asia-Pacific",
  "Regional_Lane": "North India -> Asia-Pacific",
  "Origin_Port": "Indira Gandhi International Airport",
  "Destination_Port": "Kathmandu",
  "Company_Group": "Subagent",
  "Customer_Inquiry_Frequency": 4,
  "Customer_Tier": "Occasional",
  "Booked by branch": "Delhi",
  "Incoterms": "FOB",
  "quote_revision_count": 1
}
```

**Expected Output**
```json
{
  "Benchmark_Margin_pct": 14.98,
  "Recommended_Margin_pct": 27.29,
  "Quoted_Sell_Price_INR": 89103.0,
  "Margin_Amount_INR": 19103.0,
  "Win_Probability": 0.9539,
  "Expected_Profit_INR": 18221.82
}
```

---

## Case 6 — Edge case: Unknown Origin_Port ("Mars Colony Spaceport")

**Commercial Context & Rationale:**
- **Robustness & Fallback Verification:** Tests the pipeline's handling of an unseen/out-of-vocabulary port name.
- **Calibrated Engine Behavior:** Confirms that unknown categories encode gracefully without runtime errors or numerical NaN, yielding a stable **27.64% margin** quote.

**Input**
```json
{
  "Total_Buy_INR": 50000.0,
  "Chargeable_Weight_Kg": 24.0,
  "Gross_Weight_Kg": 24.0,
  "Cargo_Density_Ratio": 1.0,
  "Weight_Tier": "<45kg",
  "Business_Vertical": "Air Export Forwarding",
  "Commodity_Group": "General Cargo",
  "Origin_Region": "South India",
  "Destination_Region": "Europe",
  "Regional_Lane": "South India -> Europe",
  "Origin_Port": "Mars Colony Spaceport",
  "Destination_Port": "Stuttgart",
  "Company_Group": "Subagent",
  "Customer_Inquiry_Frequency": 3,
  "Customer_Tier": "Occasional",
  "Booked by branch": "Bangalore",
  "Incoterms": "FOB",
  "quote_revision_count": 2
}
```

**Expected Output**
```json
{
  "Benchmark_Margin_pct": 8.18,
  "Recommended_Margin_pct": 27.64,
  "Quoted_Sell_Price_INR": 63820.0,
  "Margin_Amount_INR": 13820.0,
  "Win_Probability": 0.9698,
  "Expected_Profit_INR": 13402.28
}
```

---

## Case 7 — Edge case: Minimal buy cost (₹100), 1kg parcel (Hard Cost Floor Enforcement)

**Commercial Context & Rationale:**
- **Profile:** Minimal baseline cargo (₹100 buy cost, 1kg spot parcel, Delhi → Dubai).
- **Commercial Cost Floor Defense:** Strict enforcement of the non-negotiable **₹1,500/AWB commercial documentation and handling floor**. Quoting below break-even operational costs is prohibited, resulting in an absolute floor price of ₹1,603.50.

**Input**
```json
{
  "Total_Buy_INR": 100.0,
  "Chargeable_Weight_Kg": 1.0,
  "Gross_Weight_Kg": 1.0,
  "Cargo_Density_Ratio": 1.0,
  "Weight_Tier": "<45kg",
  "Business_Vertical": "Air Export Forwarding",
  "Commodity_Group": "General Cargo",
  "Origin_Region": "North India",
  "Destination_Region": "Middle East",
  "Regional_Lane": "North India -> Middle East",
  "Origin_Port": "Indira Gandhi International Airport",
  "Destination_Port": "Dubai",
  "Company_Group": "Shipper / Consignee",
  "Customer_Inquiry_Frequency": 1,
  "Customer_Tier": "Spot / One-Off",
  "Booked by branch": "Delhi",
  "Incoterms": "FOB",
  "quote_revision_count": 0
}
```

**Expected Output**
```json
{
  "Benchmark_Margin_pct": 1503.0,
  "Recommended_Margin_pct": 1503.5,
  "Quoted_Sell_Price_INR": 1603.5,
  "Margin_Amount_INR": 1503.5,
  "Win_Probability": 0.9221,
  "Expected_Profit_INR": 1386.39
}
```
