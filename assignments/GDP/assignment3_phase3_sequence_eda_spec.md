# Assignment 3 - Phase 3 Implementation

## Sequence-Focused EDA Specification

This document defines the exact EDA scope for Assignment 3 phase 3 and the decision outputs that must feed later modeling phases.

References:
- `assignments/GDP/assignment3.md`
- `assignments/GDP/assignment3_phase2_data_contract.md`
- `DGP_EDA_Beginner_Guide.md`

---

## 1) EDA Objectives (Locked)

Phase 3 EDA must answer:
- What does customer behavior look like over time and across products?
- What sequence characteristics are stable enough for modeling?
- Which preprocessing or representation choices should be adjusted before phase 4?

EDA is decision-oriented. Every plot/table must end with one explicit decision:
- keep as-is
- filter
- engineer new feature
- document risk and defer

---

## 2) Inputs and Scope

### Required input table
- `transactions_clean` from phase 2 contract.

### Sequence table to build for EDA
- `customer_events` sorted by (`CustomerID`, `InvoiceDate`, `InvoiceNo`, `StockCode`).

### Scope boundaries
- Use post-cleaning purchase stream only (according to phase 2 exclusions).
- Do not introduce target-derived features in EDA stage.
- Keep EDA reproducible with deterministic sorting and fixed random seed for sampling-based visuals.

---

## 3) Mandatory EDA Outputs

## A. Data quality and coverage summary

### A1. Row and customer retention funnel
- Table: raw rows -> after ID filter -> after return/cancellation handling -> after invalid value handling -> final rows.
- Table: unique customers retained at each step.

Decision output:
- Confirm whether retention is sufficient for sequence modeling and whether any cleaning rule needs revision.

### A2. Time coverage audit
- Plot: transaction counts by month.
- Plot: unique active customers by month.

Decision output:
- Confirm train/validation/test temporal cut points are feasible (no sparse split windows).

## B. Customer activity structure

### B1. Purchase frequency and intensity
- Histograms/box plots:
  - purchases per customer (`n_invoices`)
  - line items per customer (`n_line_items`)
  - total spend per customer (`total_spend`)

Decision output:
- Decide if winsorization/log transform is needed for heavy tails.

### B2. Basket behavior
- Distribution:
  - items per invoice
  - invoice monetary value
- Optional percentile table (p50/p75/p90/p95/p99).

Decision output:
- Define robust aggregation strategy (mean vs median-sensitive summaries).

## C. Sequence timing behavior

### C1. Interpurchase gap distribution
- Compute per-customer gaps between consecutive purchases.
- Plot overall distribution and segment by activity tiers.

Decision output:
- Confirm temporal features to include in phase 4 (for example `mean_interpurchase_days`, `recency_days`, `gap_std`).

### C2. Recency-frequency relationship
- Scatter/hex plot: recency vs frequency.
- Correlation summary (Spearman preferred for skewed distributions).

Decision output:
- Decide whether to include interaction feature(s) for recency-frequency effects.

## D. Product and sequence composition

### D1. Product long-tail
- Plot top-N `StockCode` frequency.
- Cumulative coverage curve (share explained by top-N items).

Decision output:
- Set token handling policy for rare items in later sequence modeling (for example rare bucket threshold).

### D2. Item diversity per customer
- Distribution of unique item count and diversity ratio.

Decision output:
- Confirm customer diversity features for representation families.

### D3. Co-occurrence / transition sketch
- Lightweight transition count table for frequent products (invoice-level adjacency or consecutive-event transitions).
- Heatmap for top-K products only.

Decision output:
- Confirm whether transition-derived summary features are justified for phase 4.

## E. Country and seasonality effects

### E1. Country concentration
- Table/plot: customer share and spend share by country.

Decision output:
- Decide whether country should be used as feature, grouped, or restricted to major markets for stability.

### E2. Calendar effects
- Plots:
  - day-of-week activity/spend
  - month-of-year activity/spend

Decision output:
- Decide whether calendar features are included in customer aggregates.

---

## 4) Required Artifact Set (Files/Figures/Tables)

Minimum artifacts to produce in notebook/script outputs:
- Retention funnel table
- Monthly activity table/plot
- Customer frequency/spend distribution plots
- Interpurchase gap distribution plot
- Product long-tail coverage plot
- Country concentration table
- Decision summary table

Naming convention recommendation:
- `eda_01_retention_funnel`
- `eda_02_monthly_activity`
- `eda_03_customer_activity_distribution`
- `eda_04_interpurchase_gaps`
- `eda_05_product_long_tail`
- `eda_06_country_mix`
- `eda_07_decision_summary`

---

## 5) Decision Summary Table Template (Mandatory)

| EDA finding ID | Evidence artifact | Observation | Decision | Downstream phase impacted |
|---|---|---|---|---|
| F01 | `eda_01_retention_funnel` |  |  | Phase 4/6 |
| F02 | `eda_02_monthly_activity` |  |  | Phase 6 |
| F03 | `eda_03_customer_activity_distribution` |  |  | Phase 4/7 |
| F04 | `eda_04_interpurchase_gaps` |  |  | Phase 4 |
| F05 | `eda_05_product_long_tail` |  |  | Phase 4/9 |
| F06 | `eda_06_country_mix` |  |  | Phase 4/7 |

Rule:
- No EDA section is complete without a downstream decision.

---

## 6) Leakage and Validity Checks During EDA

- Confirm all EDA statistics are computed from allowed tables per phase 2 policy.
- Do not tune target thresholds using full data.
- Any split-aware statistic that influences modeling must be recomputed on training split only before final modeling.

---

## 7) Phase 3 Done Criteria

- All mandatory outputs in section 3 are produced.
- Decision summary table is complete and links each finding to a modeling action.
- At least three feature-engineering decisions are explicitly justified from EDA evidence.
- At least one temporal split feasibility decision is documented.
- EDA outputs are ready to feed phase 4 representation design directly.
