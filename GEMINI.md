# Air Export Pricing Engine — Project Guidelines & Rules

## 1. Strict UI & Design Governance Rule
- **NO UI CHANGES WITHOUT EXPLICIT INSTRUCTION:** Do NOT make any UI/UX changes, layout changes, component additions/removals, or styling alterations unless the user explicitly instructs you to do so.
- **ESTABLISHED DESIGN PRINCIPLES & COLOR SCHEME:** When any UI modifications or additions are explicitly requested by the user, you MUST strictly adhere to the established design principles:
  - **Typography:** Google Fonts `Outfit` (headings, large metrics, buttons, badges) and `Mulish` (body copy, form labels, inputs, table rows).
  - **Color Palette:**
    - Primary Cyan: `#23c2f2`
    - Primary Dark: `#0ea8da`
    - Secondary Lime Green: `#a7cf45`
    - Dark Text: `#0f172a`
    - Muted Text: `#64748b`
    - Background: `#f8fafc`
  - **Card Accents:** Input card top border `4px solid var(--primary)` (`#23c2f2`); Output card top border `4px solid var(--secondary)` (`#a7cf45`).
  - **Geometry:** Pill-shaped controls and badges (`border-radius: 100px`), smooth rounded cards (`border-radius: 18px`).
  - **No Plain Blue Overhauls:** Do not replace the signature cyan/lime theme with generic corporate blue styles.

## 2. Core Engine Architecture
- **Two-Stage ML Pipeline:**
  - Stage 1A: Benchmark Regressor predicting clearing margins on won inquiries.
  - Stage 1B: Calibrated Classifier with strictly negative monotonic elasticity on `Margin_Ratio`.
- **Vocabulary Alignment:** All UI dropdown options must align 100% with the trained `OrdinalEncoder` categories. Verify before shipping using `scripts/verify_vocab_alignment.py`.
- **Backend Defaults:** Default pricing strategy is `"balanced"` (maximizing Expected Value $EV = \text{Margin} \times P(\text{Win})$).
