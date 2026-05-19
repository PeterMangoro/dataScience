# California Housing — DGP, EDA & Modeling (Assignment 1)

**Author:** Peter Mangoro  
**Dataset:** [California Housing](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) via `sklearn.datasets.fetch_california_housing`  
**Deliverable:** [`gdp_california_housing.ipynb`](gdp_california_housing.ipynb)

---

## Overview

This assignment is a **GDP (Data Generating Process) case study** on the classic **California Housing** dataset. The goal is to understand how median house values are generated at the census block-group level, explore the data rigorously, engineer interpretable features without leakage, and compare linear models under a **bias–variance** lens—including regularization and PCA.

The notebook moves from **DGP hypothesis → EDA → leakage-safe feature engineering → model comparison → dimensionality reduction → conclusions**.

| Artifact | Description |
|----------|-------------|
| [`gdp_california_housing.ipynb`](gdp_california_housing.ipynb) | Full analysis: DGP, EDA, engineered features, model tracks, PCA, summary |

---

## Problem statement

**Task:** **Regression** — predict `MedHouseVal` (median house value in a block group, in units of **$100,000**, e.g. `2.0` ≈ $200,000).

**Input:** Eight numeric predictors per census block group (income, housing age, room/bedroom averages, population, occupancy, latitude, longitude).

**Informal DGP:**

`MedHouseVal = f(MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude) + ε`

We treat this as **associative prediction** on 1990 California census aggregates—not causal inference about individual homes. Omitted factors (school quality, crime, amenities, interest rates) enter through `ε` and can bias naive coefficient readings if correlated with observed features.

**Success criteria:** Strong holdout **R²** and low **RMSE/MAE**, with stable **5-fold cross-validation** and clear interpretation of what drives prices.

---

## Data

| Item | Value |
|------|--------|
| **Instances** | 20,640 block groups |
| **Features** | 8 numeric predictors |
| **Target** | `MedHouseVal` |
| **Missing values** | None |
| **Source** | 1990 U.S. Census (one row per block group; ~600–3,000 people per group) |

### Variables

| Column | Meaning |
|--------|---------|
| `MedInc` | Median income (tens of thousands of dollars) |
| `HouseAge` | Median house age (years) |
| `AveRooms` | Average rooms per household |
| `AveBedrms` | Average bedrooms per household |
| `Population` | Block group population |
| `AveOccup` | Average household occupancy (crowding proxy) |
| `Latitude`, `Longitude` | Centroid coordinates |
| `MedHouseVal` | Median house value ($100k units) |

**Train/validation split:** 80/20 stratified random split (`random_state=42`) → **16,512 train** / **4,128 validation**. All caps, geo reference points, scalers, and PCA are fit on **training data only**.

---

## Methodology

```mermaid
flowchart LR
  A[DGP hypothesis] --> B[EDA: marginals & scatterplots]
  B --> C[Geo & correlation analysis]
  C --> D[Leakage-safe feature engineering]
  D --> E[Raw vs engineered linear models]
  E --> F[Bias-variance: Poly2, Ridge, Lasso]
  F --> G[PCA track k=5]
  G --> H[Consolidated ranking & conclusions]
```

### 1. Data Generating Process (DGP)

- Frame the problem as regression with structured predictors plus noise.
- Hypothesize positive income effects, housing-size proxies, crowding/ density effects, and strong **location** effects in California.
- Document **omitted variables** and **sampling scope** (state-level tracts only; target capped at top of range).

### 2. Exploratory Data Analysis (EDA)

- **Marginal distributions:** Right-skew in `MedInc`, `AveRooms`, `Population`; extreme ratio outliers in `AveRooms` (up to ~140 rooms/household).
- **Scatterplots:** Strong positive `MedInc` vs `MedHouseVal`; **censoring band at 5.0** on the target; weak linear signal for some predictors alone.
- **Zoomed plots:** Typical ranges (e.g. 2–10 rooms) to see bulk behavior hidden by extremes.
- **Geographic EDA:** Spatial clusters and coastal/regional price gradients; geo correlated with income and housing mix.
- **Income deciles:** Monotonic rise in average `MedHouseVal` by `MedInc` decile; steeper at top incomes (nonlinearity).
- **Correlation matrix:** `MedInc` dominates (r ≈ **+0.688**); other marginal linear correlations weak → motivates transforms and interactions.

### 3. Leakage-safe feature engineering

Thresholds (99th percentile caps, geo center) computed on **train only**, then applied to validation:

| Engineered feature | Purpose |
|--------------------|---------|
| `logMedInc` | `log1p(MedInc)` for right-skewed income |
| `AveRooms_capped`, `AveOccup_capped` | Robust caps (q=0.99 ≈ 10.32 / 5.41) on ratio outliers |
| `RoomsPerOccup` | Space per person (capped rooms / capped occupancy) |
| `logMedInc_x_RoomsPerOccup` | Income × space interaction |
| `LatLon` | `Latitude × Longitude` interaction |
| `GeoDist` | Distance from train-set geographic center |

**Shapes:** `X_base` → 8 columns; `X_eng` → **15 columns** (8 original + 7 engineered).

### 4. Modeling tracks (consistent metrics & 5-fold CV)

**Metrics:** RMSE, MAE, R² on train, CV (mean ± std), and holdout validation.  
**Primary ranking:** **validation RMSE** (lower is better); R² as supporting evidence.

**Track A — Raw features (`X_base`):**

- OLS, degree-2 polynomial OLS, RidgeCV (scaled), LassoCV (scaled)

**Track B — Engineered features (`X_eng`):**

- OLS, RidgeCV, LassoCV

**Track C — PCA (k = 5, ~90% variance explained):**

- `StandardScaler` → `PCA` on raw features (train-fit only)
- OLS and RidgeCV on 5 principal components

---

## Challenges

1. **Target censoring** — Many tracts pile up at `MedHouseVal = 5.0`, compressing the top of the distribution and limiting linear fit at high prices.
2. **Ratio outliers** — `AveRooms` and `AveOccup` are household-level ratios; tiny household counts create extreme values that distort scatterplots and OLS.
3. **Weak marginal linear signal** — Aside from `MedInc`, most predictors have low linear correlation with the target; effects are nonlinear, interaction-driven, or geo-structured.
4. **Multicollinearity & spatial structure** — Predictors co-vary with location and income; plain OLS coefficients can be unstable without care.
5. **Bias–variance tradeoffs** — Raw polynomial expansion improves holdout error but **CV collapses** (high fold-to-fold variance); engineered features aim to capture signal without that instability.
6. **PCA vs supervision** — Retaining 90% of **X** variance does not preserve **y**-relevant structure; unsupervised compression underperforms engineered features here.

---

## What we achieved

### EDA insights that drove modeling

- Confirmed **income** as the dominant linear driver and **geography** as structured, non-monotone clusters.
- Identified **censoring**, **skewness**, and **ratio artifacts** as concrete reasons for log transforms, caps, and interactions.
- Showed that naive linear models on raw features **underfit** (high bias).

### Feature engineering lift (baseline vs engineered OLS)

| Model | Val RMSE | Val MAE | Val R² |
|-------|----------|---------|--------|
| Baseline (`X_base`) | 0.746 | 0.533 | 0.576 |
| Engineered (`X_eng`) | **0.665** | **0.470** | **0.662** |

Engineering improved validation R² by about **+0.086** (~11% relative RMSE reduction).

### Full model comparison (ranked by validation RMSE)

| Rank | Model | Val RMSE | Val R² | CV stability |
|------|-------|----------|--------|--------------|
| 1 | **Eng OLS** | **0.665** | **0.662** | Low CV std (~0.014 RMSE) |
| 2 | Eng RidgeCV | 0.665 | 0.662 | Low CV std |
| 3 | Eng LassoCV | 0.678 | 0.650 | Stable; slightly weaker |
| 4 | Raw Poly2 OLS | 0.681 | 0.646 | **Unstable CV** (std ~1.38 RMSE) |
| 5–7 | Raw Lasso / Ridge / OLS | ~0.745 | ~0.576 | Stable but high bias |
| 8–9 | PCA5 Ridge / OLS | ~0.862 | ~0.433 | Stable but much weaker |

### Key conclusions

- **Best model:** **Engineered OLS** (effectively tied with **Eng RidgeCV**) — best validation accuracy with stable cross-validation.
- **Feature engineering** mattered more than heavy regularization or PCA: interpretable logs, caps, and geo terms captured target-relevant structure.
- **Raw Poly2 OLS** illustrates **variance risk**: decent single-split validation but unreliable across folds.
- **PCA (k=5)** is stable but **loses predictive signal** (~0.43 R² vs ~0.66 for engineered models) because components maximize variance in **X**, not fit to **y**.

---

## Repository layout

```
assignments/assignment1/
├── README.md                      # This file
└── gdp_california_housing.ipynb   # Main deliverable
```

No external data download is required—the dataset loads from scikit-learn.

---

## How to reproduce

1. From the repository root, create and activate a virtual environment (optional):

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install numpy pandas matplotlib seaborn scikit-learn ipython
   ```

   Or install shared course dependencies from [`../requirements.txt`](../requirements.txt) (core packages above are sufficient for this notebook).

2. Open `gdp_california_housing.ipynb` in Jupyter or VS Code.

3. **Restart kernel** and run all cells top-to-bottom.

4. Expect generated figures (histograms, scatterplots, correlation heatmap, geo plots, model comparison tables).

---

## Key takeaway

California block-group house values are driven primarily by **local income** and **location**, with important nonlinearities, censoring at the top of the price scale, and fragile ratio-based features. **Leakage-safe engineering** (log income, capped ratios, space-per-person, geo terms) plus **linear regression** delivers the best balance of **interpretability**, **validation R² (~0.66)**, and **CV stability**—outperforming raw high-variance polynomial expansions and unsupervised PCA compression on this dataset.

For full plots, coefficient discussion, and step-by-step code, see **`gdp_california_housing.ipynb`**.
