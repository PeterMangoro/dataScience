# Assignment 3 - Phase 2 Implementation

## Data Contract and Preprocessing Rules

This document defines the canonical data model and fixed preprocessing policy for Assignment 3.

---

## 1) Canonical Schema (Raw Ingest)

Dataset source: Online Retail (UCI/Kaggle), required by `assignments/GDP/assignment3.md`.

| Column | Target type | Semantics | Required for foundation |
|---|---|---|---|
| `InvoiceNo` | string | Invoice identifier (transaction group key) | Yes |
| `StockCode` | string | Product code | Yes |
| `Description` | string | Product text description | Optional (keep) |
| `Quantity` | int | Units purchased per line item | Yes |
| `InvoiceDate` | datetime | Purchase timestamp | Yes |
| `UnitPrice` | float | Unit price at transaction time | Yes |
| `CustomerID` | string | Customer identifier | Yes |
| `Country` | string | Customer/invoice country | Yes |

### Type policy
- Read all identifier-like columns as string (`InvoiceNo`, `StockCode`, `CustomerID`).
- Parse `InvoiceDate` to timezone-naive UTC-normalized timestamp (single canonical timezone assumption for analysis).
- Keep numeric precision for `UnitPrice` as float and `Quantity` as integer.

---

## 2) Canonical Intermediate Tables

### 2.1 `transactions_clean`
Row-level cleaned line items.

Minimum columns:
- `InvoiceNo`, `StockCode`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`
- derived: `LineAmount = Quantity * UnitPrice`

### 2.2 `customer_events`
Ordered customer event stream for sequence-aware EDA and later models.

Minimum columns:
- `CustomerID`, `InvoiceDate`, `InvoiceNo`, `StockCode`, `Quantity`, `UnitPrice`, `LineAmount`, `Country`
- ordering key: (`CustomerID`, `InvoiceDate`, `InvoiceNo`, `StockCode`)

### 2.3 `customer_features_base`
Customer-level aggregated feature table for foundation baselines.

Minimum features:
- count-based: `n_invoices`, `n_line_items`, `n_unique_items`
- monetary: `total_spend`, `avg_invoice_value`, `avg_line_amount`
- temporal: `active_days`, `mean_interpurchase_days`, `recency_days`
- diversity: `item_diversity_ratio`, `country_mode` (if multi-country appears)

---

## 3) Fixed Cleaning Policy (Locked)

### 3.1 Missing identifiers
- Drop rows with missing `CustomerID` for customer-level sequence modeling.
- Keep a dropped-row count for reproducibility log.

### 3.2 Duplicates
- Exact duplicate rows across all canonical columns are removed.
- Count before/after duplicate removal is recorded.

### 3.3 Cancellations and returns
- Invoices with cancellation marker (commonly `InvoiceNo` starting with `C`) are tagged.
- Negative `Quantity` rows are treated as return/cancellation behavior.
- Foundation default: exclude cancellation/return rows from purchase-behavior baseline features, but keep a parallel audit table for transparency.

### 3.4 Invalid economic values
- Remove rows with `UnitPrice <= 0` for baseline spend features.
- Remove rows with `Quantity <= 0` from purchase stream used to build forward purchase behavior labels.

### 3.5 Timestamp normalization
- Parse all dates with explicit coercion policy.
- Rows with unparseable `InvoiceDate` are dropped and counted.
- Create derived date keys only after successful parse.

### 3.6 Stock code text cleanup
- Strip surrounding spaces and normalize case for `StockCode`.
- Preserve original `Description` text (do not use for leakage-prone target engineering in foundation phase).

---

## 4) Temporal Split and Leakage-Safe Fit Rules

### 4.1 Split policy
- Primary evaluation uses chronological split:
  - train: earliest window
  - validation: middle window
  - test: latest window
- Sensitivity check uses random stratified CV on training slice only.

### 4.2 No-lookahead constraints
- Labels for purchase-behavior classification are computed from a **future window after each reference cutoff**, never from the same window used to build input features.
- Feature windows and label windows must be non-overlapping by definition.

### 4.3 Fit/transform discipline
- Any learned preprocessing (imputation, scaling, PCA, encoder fit) is fit on train only.
- Validation/test use transform only.
- CV folds repeat this rule inside each fold.

---

## 5) Phase-2 Label Contract (Foundation Task)

Primary task: customer-level purchase-behavior classification.

Recommended label for foundation:
- `label_active_future_30d` (binary): 1 if customer has at least one valid purchase in next 30 days after cutoff; else 0.

Alternative label (keep as option):
- `label_high_value_future_30d`: 1 if future 30-day spend is above training-set percentile threshold (for example p75), else 0.

Rule:
- Thresholds are estimated from training only.
- Same threshold applied to validation/test.

---

## 6) Reproducibility Contract

- Fixed random seed constant for all randomized operations.
- Deterministic sort order before sequence generation.
- Save row-count checkpoints after each cleaning step.
- Maintain a single metric table schema across model families.

---

## 7) Data Decision Log (Initial Entries)

| Decision ID | Topic | Decision | Rationale | Impacted artifacts | Status |
|---|---|---|---|---|---|
| D01 | Missing `CustomerID` | Drop from modeling tables | Customer-level sequence cannot be built without ID | `transactions_clean`, `customer_events`, feature tables | Locked |
| D02 | Returns/cancellations | Exclude from purchase baseline; keep audit | Avoid target/feature distortion from reversal transactions | baseline features, label generation | Locked |
| D03 | Non-positive `UnitPrice` | Exclude | Invalid for spend-based features | spend features, label windows | Locked |
| D04 | Non-positive `Quantity` | Exclude from purchase stream | Not valid as positive purchase signal | sequence stream, labels | Locked |
| D05 | Date parsing failures | Drop and count | Cannot place event in temporal pipeline | all temporal artifacts | Locked |
| D06 | Learned transforms | Train-only fit | Prevent leakage and inflated metrics | evaluation pipeline | Locked |
| D07 | Split strategy | Temporal primary + CV sensitivity | Better deployment realism + robustness check | evaluation outputs | Locked |

---

## 8) Phase 2 Done Criteria

- Canonical schema and type policy are documented and unambiguous.
- Cleaning policy decisions are fixed and include rationale.
- Temporal leakage controls are explicitly defined.
- Label contract is defined with train-only threshold policy.
- Initial data decision log exists and is ready for incremental updates during execution.
