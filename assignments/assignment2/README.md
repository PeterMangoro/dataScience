# Pima Indians Diabetes — Classification Case Study (Assignment 2)

**Author:** Peter Mangoro  
**Dataset:** Pima Indians Diabetes Database (`diabetes.csv`)  
**Deliverable:** [`pima_diabetes_assignment2.ipynb`](pima_diabetes_assignment2.ipynb)

---

## Overview

This assignment is a **supervised classification case study** on the **Pima Indians Diabetes** dataset. The goal is to predict whether a patient has diabetes from eight clinical measurements, while handling **masked missing values** (zeros that are clinically implausible), **class imbalance**, and **data leakage** in preprocessing.

The notebook moves from **EDA and clinical data audit → leakage-safe pipelines → model comparison (logistic vs random forest) → held-out test evaluation → unsupervised PCA and K-means**.

| Artifact | Description |
|----------|-------------|
| [`pima_diabetes_assignment2.ipynb`](pima_diabetes_assignment2.ipynb) | Full analysis: EDA, pipelines, CV, test eval, PCA, K-means |
| [`diabetes.csv`](diabetes.csv) | Source data (768 rows, 8 features + `Outcome`) |

---

## Problem statement

**Task:** **Binary classification** — predict `Outcome` (`1` = diabetes, `0` = no diabetes).

**Population:** Female patients, age ≥ 21, Pima Indian heritage (classic UCI / Kaggle benchmark).

**Input:** Eight numeric predictors per patient.

**Success criteria:** Strong **ROC-AUC** (ranking diabetics above non-diabetics), plus **F1**, **precision**, and **recall** under **stratified** validation—not raw accuracy alone, because the majority class dominates a naive baseline.

---

## Data

| Item | Value |
|------|--------|
| **Instances** | 768 |
| **Features** | 8 clinical measurements |
| **Target** | `Outcome` (binary) |
| **Explicit NaNs in CSV** | None |
| **Class balance** | ~**65%** non-diabetic, ~**35%** diabetic |

### Variables

| Column | Meaning |
|--------|---------|
| `Pregnancies` | Number of pregnancies |
| `Glucose` | Plasma glucose concentration |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skin fold thickness (mm) |
| `Insulin` | 2-hour serum insulin (mu U/ml) |
| `BMI` | Body mass index |
| `DiabetesPedigreeFunction` | Diabetes pedigree function |
| `Age` | Age (years) |
| `Outcome` | Class label (0/1) |

**Train/test split:** Stratified **80/20** (`random_state=9`) → **614 train** / **154 test**.  
**Cross-validation:** **5-fold stratified** on the training split only for model selection.

---

## Methodology

```mermaid
flowchart LR
  A[EDA & zero audit] --> B[0 to NaN clinical policy]
  B --> C[Leakage-safe Pipelines]
  C --> D[Logistic vs Random Forest CV]
  D --> E[Select winner refit test]
  E --> F[Exploratory PCA & K-means]
  F --> G[Supervised PCA pipelines compare]
```

### 1. Exploratory data analysis

- **Class imbalance:** ~2:1 non-diabetic vs diabetic; majority-class rule ≈ **65% accuracy** without learning.
- **Clinical zero audit:** Zeros concentrate in **Insulin** and **SkinThickness** (and nontrivial rates in **Glucose**, **BloodPressure**, **BMI**) — treated as **missingness in disguise**, not real measurements. **`Pregnancies = 0`** kept valid.
- **Distributions by class:** **Glucose** shows the strongest univariate separation; **BMI** and **Age** moderate; **Insulin** / **SkinThickness** zero-inflated.
- **Correlation heatmap:** Mostly weak-to-moderate; strongest pair **Pregnancies–Age** (~0.54); no extreme multicollinearity.

### 2. Preprocessing policy (leakage-safe)

All learned steps live inside **`sklearn.pipeline.Pipeline`** and fit on **training folds only** (per CV fold or on full train before one test eval):

| Step | Logistic regression | Random forest |
|------|---------------------|---------------|
| Recode `0 → NaN` | Glucose, BloodPressure, SkinThickness, Insulin, BMI | Same |
| **Median imputation** | Yes (per fold) | Yes (per fold) |
| **StandardScaler** | Yes | No (trees scale-invariant) |
| **Classifier** | `LogisticRegression(class_weight="balanced")` | `RandomForestClassifier(n_estimators=400, class_weight="balanced")` |

**Model selection:** Fixed hyperparameters (no grid search). Compare families on **train CV**; **one** held-out test evaluation after selection.

**Selection rule:** Maximize mean **ROC-AUC**; **F1** as tie-breaker.

### 3. Supervised model comparison (raw features)

5-fold stratified CV on `X_train` / `y_train`:

| Model | CV ROC-AUC (mean ± std) | CV F1 | CV recall | CV precision |
|-------|-------------------------|-------|-----------|--------------|
| **Logistic regression** | **0.831 ± 0.044** | **0.672** | **0.715** | 0.636 |
| Random forest | 0.818 ± 0.035 | 0.625 | 0.589 | 0.666 |

**Out-of-fold average precision (train):** Logistic **0.720** vs Random forest **0.681** (baseline prevalence ≈ 0.35).

**Winner:** **Logistic regression** (wins both ROC-AUC and F1).

### 4. Held-out test evaluation (winner refit on full train)

| Metric | Value |
|--------|--------|
| **Test ROC-AUC** | **0.853** |
| **Test accuracy** | 0.74 |
| **Diabetes recall** | 0.78 (42/54 true positives) |
| **Diabetes precision** | 0.60 |
| **Diabetes F1** | 0.68 |

Confusion matrix at default 0.5 threshold: **72 TN, 42 TP, 12 FN, 28 FP** on 154 test patients (100 non-diabetic, 54 diabetic).

### 5. Unsupervised learning

- **PCA (train-fit):** Scree and cumulative variance; **k = 7** needed for ≥90% variance; **k = 5** used in supervised PCA pipelines (~**82%** variance, parsimony choice).
- **PC1–PC2 scatter:** Partial overlap by `Outcome` — structure visible but not separable.
- **K-means (k = 3)** on first four PCs: exploratory clusters; cross-tabs vs `Outcome` show varying diabetes prevalence (descriptive only).

### 6. Supervised pipelines with PCA (`n_components = 5`)

| Pipeline | OOF AP (train) | CV ROC-AUC (mean) | CV F1 |
|----------|----------------|-------------------|-------|
| Logistic (raw) | **0.720** | **0.831** | **0.672** |
| Random forest (raw) | 0.681 | 0.818 | 0.625 |
| Logistic + PCA | 0.665 | 0.810 | 0.659 |
| RF + PCA | 0.618 | 0.788 | 0.570 |

**PCA did not improve** discrimination vs raw eight features on this dataset; compression removes target-relevant signal.

---

## Challenges

1. **Missingness disguised as zero** — CSV has no NaNs, but many clinical zeros are implausible; wrong handling would bias models.
2. **Class imbalance** — Accuracy alone is misleading (~65% naive baseline); requires stratified splits and balanced class weights.
3. **Small sample size** — 768 rows limits stable estimates; test set has only 54 diabetics, so test metrics have noise.
4. **Skew and zero inflation** — Especially **Insulin** and **SkinThickness**; motivates median imputation and nonlinear benchmark.
5. **Leakage risk** — Imputation and scaling must not use validation or test rows; enforced via `Pipeline` + per-fold fitting.
6. **Threshold tradeoffs** — Logistic favors **recall** (catch more diabetics) vs forest’s higher **precision** at 0.5; clinical deployment would tune thresholds separately.
7. **PCA limitation** — Unsupervised components maximize **X** variance, not diabetes separability; AP and AUC drop with k = 5.

---

## What we achieved

### Rigorous EDA and data hygiene

- Documented imbalance, invalid zeros, univariate signals, and correlation structure.
- Implemented a defensible **0 → NaN** policy with **`Pregnancies`** exempted.

### Leakage-safe supervised workflow

- All preprocessing inside **`Pipeline`**; model choice on **train CV only**; **single** honest test report.

### Strong diabetes screening performance

- **CV ROC-AUC ~0.83** and **test ROC-AUC ~0.85** for balanced logistic regression.
- **Recall ~0.78** on test diabetics at 0.5 threshold — finds most cases at the cost of more false alarms (precision ~0.60).

### Complete model and representation comparison

- Linear vs nonlinear classifiers on equal footing.
- Exploratory and supervised **PCA** plus **K-means** with clear conclusion: **raw features win**; PCA is for interpretation/variance understanding, not accuracy gains here.

### Key takeaway

For Pima diabetes classification, a **balanced logistic regression pipeline** (median impute → scale → logistic) outperforms random forest on **ROC-AUC** and **F1** in CV and generalizes well on holdout test data. **Clinical zero handling** and **stratified, leakage-safe validation** are as important as model choice; **PCA compression** does not beat raw features on this small tabular problem.

---

## Repository layout

```
assignments/assignment2/
├── README.md                           # This file
├── diabetes.csv                        # Dataset (768 rows)
└── pima_diabetes_assignment2.ipynb     # Main deliverable
```

---

## How to reproduce

1. From the repository root, create and activate a virtual environment (optional):

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install numpy pandas matplotlib seaborn scikit-learn ipython
   ```

   Or install shared course dependencies from [`../requirements.txt`](../requirements.txt).

2. Ensure `diabetes.csv` is in `assignments/assignment2/` (same folder as the notebook).

3. Open `pima_diabetes_assignment2.ipynb` in Jupyter or VS Code.

4. **Restart kernel** and run all cells top-to-bottom.

5. Set working directory to `assignments/assignment2/` so `diabetes.csv` loads correctly.

---

## Key takeaway

Diabetes risk in this cohort is predictable from standard clinical features when zeros are treated as missing and models are evaluated with **imbalance-aware metrics**. **Logistic regression** with **balanced class weights** and **leakage-safe imputation** is the recommended pipeline; **random forest** is a useful nonlinear comparator but does not win on CV discrimination here, and **PCA (k = 5)** reduces rather than improves predictive performance.

For full plots, ROC/PR curves, confusion matrices, and code, see **`pima_diabetes_assignment2.ipynb`**.
