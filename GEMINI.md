# Air Export Pricing Engine — Project Guidelines & Rules

## 1. Strict UI & Design Governance Rule
- **APPROVED DESIGN PHILOSOPHY:** The user has explicitly mandated and approved the minimalist, clean Notion-inspired UI redesign (reference: `refernce_ui.jpg` / `ui_ux_redesign_workflow_task.md`). Do NOT revert this design to the old cyan/lime theme.
- **ESTABLISHED DESIGN PRINCIPLES & COLOR SCHEME:**
  - **Typography:** Google Fonts `Outfit` (headings, large metrics, buttons, badges) and `Mulish` (body copy, form labels, inputs, table rows).
  - **Color Palette (Minimalist Notion Monochrome):**
    - Window Background: `#edf1f5`
    - Surface Card: `#ffffff`
    - Deep Slate / Dark Text: `#0f172a`
    - Muted Slate: `#64748b`
    - Light Slate / Accents: `#94a3b8`
    - Borders: `#e2e8f0` (subtle: `#f1f5f9`)
  - **Layout & Structure:**
    - Single-screen application container (`.app-window`).
    - Symmetrical 50/50 POL / POD input grid (`minmax(0, 1fr)`).
    - Single-row preset scenario bar.
    - Clean Commercial Terms card prominently featuring the Recommended Margin Range.
    - History table with pre-approved Margin Range column.
  - **Do NOT Revert:** Never revert back to the legacy cyan (`#23c2f2`) or lime green (`#a7cf45`) theme.

## 2. Core Engine Architecture
- **Two-Stage ML Pipeline:**
  - Stage 1A: Benchmark Regressor predicting clearing margins on won inquiries.
  - Stage 1B: Calibrated Classifier with strictly negative monotonic elasticity on `Margin_Ratio`.
- **Vocabulary Alignment:** All UI dropdown options must align 100% with the trained `OrdinalEncoder` categories. Verify before shipping using `scripts/verify_vocab_alignment.py`.
- **Backend Defaults:** Default pricing strategy is `"balanced"` (maximizing Expected Value $EV = \text{Margin} \times P(\text{Win})$).
