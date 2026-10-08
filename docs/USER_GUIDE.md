# Air Export Pricing Recommendation Engine — User Guide
**Version:** 2.4  
**Audience:** Pricing Executives, Commercial Managers, Freight Sales Representatives, Branch Operators, and Leadership

---

## 1. Executive Summary: What is This Application?

The **Air Export Pricing Recommendation Engine** is an intelligent commercial pricing assistant tailored specifically for international air freight forwarding. 

Instead of guessing freight margins or relying purely on static rate cards, pricing teams can submit cargo inquiry details and instantly receive:
1. **A Target Margin & Sell Rate (₹/kg):** The optimal balance between winning the deal and maximizing net company margin.
2. **A Pre-Approved Negotiation Corridor ($X\% \text{ to } Y\%$):** Statistically calibrated minimum and maximum boundaries within which sales executives are authorized to negotiate freely without needing management escalation.
3. **Strategic Quotation Options:** Three clearly defined pricing tiers (**Floor**, **Balanced**, and **Premium**) to match different customer relationships and competitive contexts.
4. **Automated Guardrail Checks:** Protection against quoting below cost, missing dangerous goods surcharges, or falling victim to volume weight break pricing traps.

---

## 2. How the Engine Works (In Simple Terms)

Quoting air freight requires balancing market demand, airline costs, customer loyalty, and operational risk:

```
[Inquiry Details]
       │
       ▼
[Operational Guardrails] ──► Checks IATA weight, cost floors & risk buffers
       │
       ▼
[Regional Corridor Tariff] ──► Retrieves historical clearing rates on this trade lane
       │
       ▼
[Decision Adjuster Signals] ──► Balances Commodity Risk, Customer History & Market Season
       │
       ▼
[Output Recommendations] ──► Balanced Target Quote + Pre-Approved Negotiation Corridor
```

1. **Step 1: Physical & Commercial Screening:** Checks that the cargo dimensions and weights follow standard IATA air cargo rules, applies necessary cargo risk markups (e.g., cold chain, hazardous cargo), and computes an absolute cost floor.
2. **Step 2: Regional Tariff Benchmarking:** Retrieves historical clearing margins from real won shipments across that specific origin-destination corridor and weight slab.
3. **Step 3: Signal Adjustment:** Adjusts the quote based on commercial context (is this a high-frequency customer, peak airfreight season, or high-risk dangerous cargo?).
4. **Step 4: Negotiation Corridor Calibration:** Calculates a statistically proven 90% confidence corridor ($[X\%, Y\%]$) around the target quote so the sales representative knows the safe negotiation boundaries.

---

## 3. What Inputs the Engine Evaluates

When submitting an inquiry in the Web UI or via the API, the engine evaluates the following commercial parameters:

### A. Shipment & Route Characteristics
- **Origin Airport:** Indian gateway hub (e.g., Mumbai BOM, Delhi DEL, Bangalore BLR, Calicut CCJ).
- **Destination Airport:** Final international destination gateway (e.g., London LHR, Dubai DXB, Frankfurt FRA).
- **Gross Weight vs. Dimensions:** Physical cargo scale weight versus piece dimensions ($L \times W \times H$ in cm). The engine computes Volumetric Weight using the IATA airfreight ratio ($1:6$ or $6,000\text{ cm}^3/\text{kg}$) and rounds up to the next $0.5\text{ kg}$ increment to determine the true **Chargeable Weight**.
- **Airline Buy Cost (Total INR):** What the forwarder pays to the airline carrier for the master air waybill (MAWB).

### B. Cargo & Customer Context
- **Commodity Group:** 
  - *General Cargo* (baseline).
  - *Perishable Foodstuff* (fresh fish, fruits, vegetables).
  - *Pharmaceuticals* (temperature-controlled healthcare).
  - *Dangerous Goods / Hazmat* (Class 1-9 hazardous materials).
  - *Valuables & High-Value Electronics*.
  - *Garments, Courier, Engineering & Auto Parts*.
- **Client Category (Company Group):** 
  - *Shipper / Direct Consignee* (retail direct clients).
  - *Subagent & IATA Cargo Agent* (wholesale forwarder partners).
  - *NVOCC, Co-loader, GSA, or Overseas Agent*.
- **Customer Company Name:** Evaluates historical booking volume and win rates for repeat clients.
- **Incoterms:** Commercial trade terms (e.g., FOB, CIF, EXW, CFR, DAP/DDP).
- **Air Cargo Season & Date:** Identifies quiet periods versus Q4 peak demand spikes.

---

## 4. How the Engine Generates Recommendations

The engine translates your inquiry into actionable pricing through three levels:

```
┌────────────────────────────────────────────────────────┐
│ LEVEL 1: REGIONAL CORRIDOR BENCHMARKS                  │
│ Historical market clearing rates per trade lane & slab │
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│ LEVEL 2: MULTI-SIGNAL COMMERCIAL ADJUSTERS             │
│ Commodity risk, client tier, customer loyalty, season  │
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│ LEVEL 3: STRICT SAFEGUARDS & CONFORMAL CORRIDOR        │
│ Cost floor protection, margin caps, 90% confidence band│
└────────────────────────────────────────────────────────┘
```

### The Three Strategic Tiers
For every inquiry, the engine provides three specific quotes:
- **1. Competitive Floor (Cyan):**
  - **Purpose:** Lowest defensible quote for competitive bidding or retaining price-sensitive accounts.
  - **Protection:** Guaranteed never to dip below operational break-even cost floors.
- **2. Balanced Quote (Green — Recommended Target):**
  - **Purpose:** Standard target quote maximizing total expected profit.
  - **Guidance:** Represents the realistic clearing price where the deal is likely to close profitably.
- **3. Premium Quote (Purple):**
  - **Purpose:** Selective pricing for high-complexity, urgent shipments or non-price-sensitive spot inquiries.

---

## 5. Key Factors That Influence Your Quote

| Factor | Influence on Margin | Commercial Rationale |
| :--- | :---: | :--- |
| **High Chargeable Weight ($>500\text{ kg}$)** | **Lower % Margin** | Heavy cargo operates on sublinear scale elasticity; margins taper percentage-wise to keep rupee rate per kg competitive. |
| **Dangerous Goods / Valuables** | **Higher Margin (+5% to +6%)** | Increased operational liability, specialized handling, and documentation requirements. |
| **Direct Shippers vs. Subagents** | **Higher Margin for Shippers** | Retail cargo carries higher margin yield than wholesale subagent co-loading agreements. |
| **Loyal Repeat Customer ($>15$ deals)** | **Calibrated to History** | Quote automatically adjusts toward the customer's typical won price to protect retention. |
| **Q4 Peak Shipping Surge** | **Caution Flag** | Detects peak airline capacity crunches where spot rates may be volatile. |
| **Weight-Break Opportunity** | **Smart Savings Alert** | Detects when declaring a higher weight slab (e.g., 100 kg instead of 98 kg) saves money for the client. |

---

## 6. How to Interpret and Use the Results

When an inquiry is quoted, the screen displays three key components:

### A. The Target Recommendation Box
- **Quoted Sell Price (INR):** Total amount to invoice the client.
- **Rate Per Kg (₹/kg):** Equivalent all-in air freight selling rate.
- **Margin Percentage (%):** Percentage markup over airline buy cost.
- **History Percentile:** Indicates what percentage of previously won deals in this category cleared at or below this margin.

### B. The Pre-Approved Negotiation Corridor
```
[ Min Floor: 5.86% ] ────── [ Target: 8.54% ] ────── [ Max Ceiling: 11.22% ]
₹294.06 / kg                 ₹301.50 / kg            ₹308.94 / kg
```
- **Sales Discretion:** Sales representatives can negotiate within this corridor immediately without seeking managerial approval.
- **If the Client Demands < Min Floor:** Requires formal branch manager sign-off.
- **Confidence Guarantee:** 90% of genuine clearing prices historically fall inside this window.

### C. Alerts & Special Handling Flags
- **Cost Floor Clamped:** Indicates the suggested margin was raised to prevent operating at a loss.
- **Weight-Break Arbitrage:** Displays exact weight bump details to present cost savings to the client.
- **Thin Cohort Warning:** Flags that this lane has fewer historical deals, advising caution.

---

## 7. Important Assumptions, Limitations & Guidelines

### Operational Assumptions
1. **Valid Master Buy Cost:** The system assumes the entered `Total_Buy_INR` reflects true net airline buy rates.
2. **Export Out of India:** Designed specifically for air export shipments originating from Indian commercial gateway airports.

### Key Guidelines for Sales & Pricing Teams
1. **Never Bypass Cost Floors:** The system hard-stops quotes that would cause operational cash losses.
2. **Use the Corridor:** Pitch the **Target Quote** initially. If the customer pushes back on price, concede within the pre-approved corridor down to the **Floor** without losing the deal.
3. **Monitor System Drift Alerts:** If the offline watchdog alerts that airline buy costs on a route have spiked, adjust spot quotes accordingly until base tariffs are updated.
