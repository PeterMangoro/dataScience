# Assignment 3 - Phase 0 and 1 Implementation

## Phase 0: Alignment and Scope Lock (Locked Decisions)

### Primary objective (for foundation work)
- Build a leakage-safe pipeline for **customer-level purchase-behavior classification**.
- Use this as the baseline task for preprocessing, EDA, unsupervised features, and evaluation design.

### Deferred objective (future extension)
- Keep **next-item prediction** for the extension phase after the foundation is complete.
- Preserve sequence-ready artifacts (customer-ordered transactions and item token mappings) so later extension does not require rework.

### Validation policy (locked)
- **Primary track:** temporal split (train on earlier window, validate/test on later window).
- **Sensitivity track:** random stratified CV on train for robustness checks.

### Selection rule (locked)
- Primary selection metric: **ROC-AUC**.
- Tie-break metric: **F1**.
- Report supporting metrics: accuracy, precision, recall, PR-AUC (or average precision).

### Leakage guardrails (locked)
- No learned transform (imputer/scaler/PCA/encoder) is fit on validation or test data.
- Temporal boundaries are fixed before modeling choices.
- Target definition uses only information available at prediction time.

---

## Phase 1: Requirement-to-Artifact Mapping and Done Criteria

Reference assignment: `assignments/GDP/assignment3.md`

| Assignment requirement | Planned artifact(s) | Done criteria |
|---|---|---|
| Data preprocessing and representation design | Data cleaning policy table + feature schema + sequence construction note | Cleaning rules are explicit (returns, missing IDs, invalid values, duplicates, timestamps), and sequence construction is reproducible from raw data |
| Aggregate transactions into customer-level sequences | Sequence table (`CustomerID`, ordered events, timestamps, item codes) + summary figure | For each customer, events are time-ordered and validated with spot checks and sequence-length distribution |
| Define transaction representation | Feature registry (raw aggregates, sequence stats, latent features) | Every feature has definition, type, and rationale; generation process is deterministic |
| Handle missing/inconsistencies | Data decision log with frequencies and treatment | Each inconsistency type has a treatment and rationale; no unresolved high-frequency issue remains undocumented |
| Unsupervised learning and representation | Clustering output + PCA/latent output + segment profile table | At least one clustering method and one latent method are run, interpreted, and compared |
| Meaningful customer segments | Segment interpretation table (size, spend/activity profile, distinguishing features) | Segments are not just numeric labels; each has a clear behavioral narrative |
| Representation quality for downstream modeling | Comparison table: raw vs latent/cluster-enhanced features | At least one downstream baseline shows whether representation helps/hurts under fixed evaluation protocol |
| Neural network modeling (later phase) | Placeholder section in report and comparison table schema | Table/report include reserved slots so NN can be appended without redesign |
| Analyze training behavior and bias-variance (later phase) | Method template for loss curves and overfitting diagnosis | Template is ready with required plots/metrics and interpretation prompts |
| Transformer-based modeling (later phase) | Placeholder section + planned comparison slots | Transformer row/section is predefined in output schema |
| Attention interpretation (later phase) | Placeholder analysis template | Template specifies what attention behavior evidence will be reported |
| Comparative analysis across methods | Canonical model comparison table + narrative bullets | Table contains consistent metrics across representation/model families; narrative includes performance, complexity, interpretability |
| Research report (2–3 pages) | Report outline with section headers and evidence mapping | Each section has linked artifacts (figure/table/metric) and concise interpretation goals |
| Python analysis code deliverable | Reproducible notebook/script structure | Pipeline runs end-to-end with fixed random seed and clear sectioning/comments |
| Code quality and reproducibility | Repro checklist | Environment assumptions, seeds, and run order are documented |

---

## Immediate outputs produced by this phase
- Scope lock and evaluation policy are finalized.
- Requirement-to-artifact matrix is finalized.
- Done criteria for grading-relevant sections are explicit and testable.
