# Feature Engineering & Dimensionality Reduction
## A Beginner's Guide in Plain English

---

## Table of Contents

1. [Quick Start (2-Minute Overview)](#quick-start-2-minute-overview)
2. [Introduction](#introduction)
3. [Why Feature Engineering Matters](#why-feature-engineering-matters)
4. [The Curse of Dimensionality](#the-curse-of-dimensionality)
5. [Types of Feature Transformations](#types-of-feature-transformations)
6. [Polynomial Feature Expansion](#polynomial-feature-expansion)
7. [Interaction Terms](#interaction-terms)
8. [Feature Scaling](#feature-scaling)
9. [Feature Selection](#feature-selection)
10. [Dimensionality Reduction: The Big Idea](#dimensionality-reduction-the-big-idea)
11. [Principal Component Analysis (PCA)](#principal-component-analysis-pca)
12. [Covariance Matrix and Eigen Decomposition](#covariance-matrix-and-eigen-decomposition)
13. [PCA Projection and Reconstruction Error](#pca-projection-and-reconstruction-error)
14. [Singular Value Decomposition (SVD)](#singular-value-decomposition-svd)
15. [Explained Variance and Choosing k](#explained-variance-and-choosing-k)
16. [Nonlinear Dimensionality Reduction](#nonlinear-dimensionality-reduction)
17. [Kernel PCA](#kernel-pca)
18. [Dimensionality Reduction in Machine Learning](#dimensionality-reduction-in-machine-learning)
19. [Tradeoffs in Dimensionality Reduction](#tradeoffs-in-dimensionality-reduction)
20. [Worked Examples](#worked-examples)
21. [Common Confusions](#common-confusions)
22. [Pitfalls and Tips](#pitfalls-and-tips)
23. [Checklists](#checklists)
24. [Putting It Together](#putting-it-together)
25. [Glossary](#glossary)
26. [Further Reading](#further-reading)

---

## Quick Start (2-Minute Overview)

**Feature engineering** is the art of turning raw inputs **X** into a better representation **Z = φ(X)** so that models can learn more easily. The right transformation can make a complex, nonlinear relationship look simple—even linear—so that a basic model can do well. A bad representation forces the model to do all the work, hurting stability and interpretability.

**Dimensionality reduction** builds a lower-dimensional summary **Z = XW** (with **k ≪ p** dimensions) that keeps the most important structure—variance, distances, or local geometry—while cutting noise and cost. **PCA** does this by finding directions of maximum variance and projecting onto them; it is optimal for linear compression (Eckart–Young). **Kernel PCA**, **t-SNE**, and **UMAP** extend the idea to nonlinear structure.

**In one sentence:** Good features and the right dimension make learning easier and more reliable; the curse of dimensionality and poor representations make it harder—so we engineer features and reduce dimensions deliberately.

---

## Introduction

### What are feature engineering and dimensionality reduction?

- **Feature engineering:** You create or transform variables (e.g. polynomials, interactions, log, domain-specific summaries) so that the relationship between **X** and **Y** is easier for your model to capture. The model sees **Z = φ(X)**, not raw **X**.
- **Dimensionality reduction:** You replace many correlated or noisy features with fewer new ones (e.g. principal components) that preserve as much useful structure as possible. You get **Z = XW** with **k** columns instead of **p**.

Together they address two core issues: **representation** (is the signal easy to learn?) and **dimension** (do we have too many features for our sample size?).

### Why they matter in data science

- **Curse of dimensionality:** In high dimensions, data become sparse, distances become less informative, and you need exponentially more samples to keep the same “density.” Reducing dimension or designing good features mitigates this.
- **Model performance:** Scaling, interactions, and polynomials help linear and distance-based methods; PCA and related methods speed up training, reduce overfitting, and can improve generalization when **n** is small relative to **p**.
- **Interpretability vs. compression:** Feature *selection* keeps original variable names; dimensionality reduction (e.g. PCA) gives you new axes that are harder to label but often more compact.

### How to use this guide

- **Read in order** for a full narrative: why features matter → curse of dimensionality → transformations → scaling → selection → DR concept → PCA → SVD → explained variance → nonlinear DR → use in ML → tradeoffs.
- **Use the Table of Contents** to jump to a topic (e.g. “When does scaling matter?” or “How do I choose k in PCA?”).
- **Equations** are in **bold** with Unicode symbols (e.g. **λ**, **Σ**, **∈**) so they read clearly without LaTeX.
- **Worked examples, checklists, and the glossary** are there to reinforce when and how to apply each idea.

---

## Why Feature Engineering Matters

Models usually work on a **transformed** representation of the inputs, not the raw ones.

**Transformed representation**

**Z = φ(X)**

So the learning problem is really about estimating **Y = f(X) + ε**. If you choose **φ** well, **f** becomes easy to approximate—even with a simple linear model. If you don’t, the model has to learn a complicated **f**, which increases variance, hurts stability, and can hurt interpretability.

**Visual idea:**

```
Raw input X          →    Y = f(X) + ε
Relationship can be highly nonlinear and hard to estimate.

Transformed Z = φ(X) →    Simpler relationship
A good φ restructures the problem so a simple model can capture the signal.
```

So feature engineering is not optional polish—it’s part of defining what the model actually “sees” and how hard the problem is.

---

## The Curse of Dimensionality

As the number of dimensions **d** grows, the “volume” of feature space grows very fast, so the same number of points becomes **sparse**. Things that work in 2D or 3D break down: distances become similar, local neighborhoods become empty or meaningless, and density estimation becomes unreliable.

**Volume growth**

**Volume ∝ r^d**

So in high dimensions, data that looked “dense” in low dimensions are spread very thin. Consequences:

- **Sparsity** increases.
- **Distance metrics** become less informative (many points end up roughly equidistant).
- **Density estimation** degrades.
- **Local neighborhoods** lose meaning.

**Practical implication:** To keep the same effective density, the amount of data you need can grow **exponentially** with **d**. For example, if you want something like 100 observations per “dimension,” in 20 dimensions you’d need on the order of **10²⁰** samples—not realistic. So **dimensionality reduction** and good **feature design** are often a statistical necessity, not just a convenience.

---

## Types of Feature Transformations

Feature transformations change variables so that algorithms can learn patterns more easily. They encode domain knowledge, reveal structure, and can linearize relationships.

| Type | Purpose |
|------|--------|
| **Polynomial** | Powers of variables to capture nonlinear curves. Tune degree to avoid overfitting. |
| **Interaction** | Products (e.g. **x₁·x₂**) so one predictor’s effect can depend on another. |
| **Logarithmic** | Compress skew, stabilize variance, linearize relationships (income, counts, finance). |
| **Domain-specific** | Expert knowledge: Fourier for signals, lags for time series, TF-IDF for text, graph summaries for networks. |

These are the main “building blocks”; combining them is part of feature engineering.

---

## Polynomial Feature Expansion

You turn a scalar **x** into a vector of powers **(1, x, x², x³, …)** so that a **linear model in these new features** can fit nonlinear curves. The model is still linear in the **parameters**, so the loss stays convex.

**Degree expansion example**

**z₁ = 1,  z₂ = x,  z₃ = x²,  z₄ = x³**

Each **zᵢ** is a separate predictor; the model fits a weighted sum of powers and thus a polynomial in **x**.

**Complexity:** For one variable it’s simple. For **p** variables and degree **d**, the number of terms grows like **O(p^d)**—the curse of dimensionality returns. So we need regularization or careful term selection for high-degree expansions.

**Bias–variance tradeoff:**

- **Degree 1:** Linear fit, high bias.
- **Degree 2–3:** Often a good compromise.
- **Degree ≥ 5:** High overfitting risk.

Choose degree (e.g. by cross-validation), not by intuition alone.

---

## Interaction Terms

In an additive model, each feature contributes independently. In reality, one variable’s effect often **depends on** another. **Interaction terms** let a linear model capture that.

**Interaction feature**

**z = x₁ · x₂**

This single new feature is just another predictor. Together with **x₁** and **x₂**, it allows a **bilinear** surface: the effect of **x₁** can change with **x₂**.

**In the model**

**ŷ = β₀ + β₁x₁ + β₂x₂ + β₃x₁x₂**

The marginal effect of **x₁** on **ŷ** is **β₁ + β₃x₂**—so it depends on **x₂**. The variables are no longer independent contributors.

---

## Feature Scaling

Scaling puts variables on comparable scales so that magnitude doesn’t dominate the model or the optimizer.

**Standardization**

**z = (x − μ) / σ**

Mean **μ**, standard deviation **σ**. After scaling, the feature has mean 0 and variance 1. This is what people usually mean by “standardized” features.

**When scaling matters most**

- **Critical:** Regularized linear models (ridge, lasso), SVMs, PCA, neural networks, and any distance-based method (e.g. k-NN). Unscaled features distort gradients and distances.
- **Less critical:** Tree-based models (random forest, gradient boosting) are invariant to monotonic rescaling of a single feature.

**Why else scale?**

- **Numerical stability:** Better condition number for **X**, fewer issues in matrix inversions and decompositions.
- **Interpretation of penalties:** In ridge/lasso, all coefficients are penalized fairly only if features are on similar scales.

**Min-max** scaling to a bounded range (e.g. [0, 1]) is an alternative when you want to preserve that property.

---

## Feature Selection

Feature **selection** keeps a **subset** of the original variables and drops the rest. Unlike dimensionality reduction, the kept features are still the original ones—so interpretability is preserved. That’s important in medicine, law, and finance.

**Three main approaches:**

| Approach | Idea | Pros | Cons |
|----------|------|------|------|
| **Filter** | Rank by a statistical criterion (e.g. correlation with **y**) before modeling. | Fast, scalable. | Ignores interactions and the model. |
| **Wrapper** | Search over subsets; train and cross-validate a model for each. | Can capture interactions. | Computationally expensive. |
| **Embedded** | Selection is part of training (e.g. Lasso’s L1 penalty drives some β to 0). | One pass; selection and estimation together. | Tied to that model class. |

Use filters for a quick screen, wrappers when you care about subset quality and can afford the cost, and embedded methods (e.g. Lasso) when you want automatic selection inside a linear model.

---

## Dimensionality Reduction: The Big Idea

Dimensionality reduction constructs a **lower-dimensional representation Z** from **X** using a **projection** (or similar map).

**Projection formula**

**Z = X W,   W ∈ ℝ^(p×k)**

So **n** observations in **p** dimensions become **n** observations in **k** dimensions, with **k ≪ p**. **W** defines what is preserved.

**Design choices:**

- **What to preserve:** Variance (PCA), distances (MDS), or topology (t-SNE, UMAP).
- **Linear vs. nonlinear:** PCA is linear; kernel PCA, t-SNE, UMAP are nonlinear.
- **Supervised vs. unsupervised:** PCA ignores **y**; methods like LDA use **y**.
- **Choice of k:** Often by cumulative explained variance (e.g. 90–95%) or by downstream task performance.

Unlike feature selection, dimensionality reduction **synthesizes** new features that can mix all original variables; you lose direct correspondence to “variable 3” or “variable 7.”

---

## Principal Component Analysis (PCA)

PCA finds **principal components**—directions in feature space along which the data have **maximum variance**. The first component has the largest variance; each next one is **orthogonal** to the previous ones and has the next largest variance. So the components are uncorrelated and form an orthogonal basis.

**Optimization (first component)**

**max_w  Var(Xw)   subject to   ‖w‖ = 1**

The unit-norm constraint avoids trivial scaling. The solution is the **leading eigenvector** of the sample covariance matrix (and can be derived with Lagrange multipliers).

**Why maximize variance?**

- High-variance directions usually carry the most information.
- Low-variance directions are often noise.
- Orthogonality removes redundancy.
- Taking them in order gives a globally optimal subspace (for linear projection).

---

## Covariance Matrix and Eigen Decomposition

The **sample covariance matrix** **Σ** summarizes pairwise (linear) relationships between features. For **centered** data matrix **X** (each column has mean 0):

**Sample covariance matrix**

**Σ = (1/n) Xᵀ X**

**Σ** is **p×p**, symmetric, and positive semi-definite. Diagonal entries are variances; off-diagonals are covariances.

**Eigen decomposition**

**Σ = V Λ Vᵀ**

- **V:** orthogonal matrix (**VᵀV = I**); its columns are the **principal directions** (eigenvectors).
- **Λ:** diagonal matrix **diag(λ₁, …, λₚ)** with **λ₁ ≥ … ≥ λₚ ≥ 0**. Each **λᵢ** is the variance of the data projected onto the **i**-th eigenvector.

So the first principal component is the eigenvector with the largest eigenvalue, and so on.

---

## PCA Projection and Reconstruction Error

After computing **V** and **Λ**, we keep the **top k** eigenvectors as columns of **Vₖ** and project:

**Z = X Vₖ,   Vₖ ∈ ℝ^(p×k)**

**Z** has **n** rows and **k** columns—the **scores** on the first **k** principal components.

**Reconstruction:** If we map back with **Vₖ**, we get **X̂ = Z Vₖᵀ = X Vₖ Vₖᵀ**. The **reconstruction error** is:

**‖X − X Vₖ Vₖᵀ‖²_F**

(the squared Frobenius norm of the residual).

**Eckart–Young theorem:** Among all rank-**k** linear projections, PCA minimizes this reconstruction error. So PCA is **optimal** for linear compression in the Frobenius sense. The residual error equals the sum of the **discarded** eigenvalues: **Σᵢ₌ₖ₊₁^p λᵢ**.

**Steps in practice:**

1. Center **X** and compute **Σ = (1/n) Xᵀ X**.
2. Eigen decomposition **Σ = V Λ Vᵀ**.
3. Choose **k** and form **Vₖ** from the top **k** columns of **V**.
4. Project: **Z = X Vₖ**.

---

## Singular Value Decomposition (SVD)

**SVD** is the numerically preferred way to compute PCA. You work directly on the (centered) data matrix **X**, which avoids forming **Σ** explicitly and often gives better numerical behavior when **p** is large or columns are collinear.

**SVD**

**X = U Σ Vᵀ**

- **U ∈ ℝ^(n×n):** left singular vectors (orthonormal in observation space).
- **Σ:** diagonal matrix of **singular values** **σ₁ ≥ … ≥ σᵣ ≥ 0** (non-negative, descending).
- **V ∈ ℝ^(p×p):** right singular vectors (orthonormal in feature space). **V** is the same as the PCA eigenvector matrix; the columns of **V** are the principal directions.

**Link to PCA:** The PCA eigenvalues and SVD singular values are related by **λᵢ = σᵢ²/n**. So you can get PCA by running SVD on **X** and using **V** and **σᵢ²/n** for eigenvalues.

---

## Explained Variance and Choosing k

**Explained variance ratio** for component **k**:

**λₖ / Σᵢ₌₁^p λᵢ**

That’s the fraction of total variance in the **k**-th component. **Cumulative** explained variance for the first **k** components is the sum of these ratios and is the usual criterion for choosing **k**.

**How to choose k:**

- **Elbow:** Plot cumulative explained variance vs. **k**; look for a bend (diminishing returns).
- **Threshold:** Keep enough components to retain e.g. **90–95%** of total variance.
- **Task-based:** Pick **k** by cross-validation or performance on the downstream task.

There is no single “right” **k**; it’s a tradeoff between compression (small **k**) and fidelity (large **k**).

---

## Nonlinear Dimensionality Reduction

When data lie on a **nonlinear manifold**, linear PCA can collapse structure: points that are close on the manifold may be far in raw space, and PCA doesn’t respect that geometry. **Nonlinear** methods try to preserve the true (nonlinear) structure.

**Manifold hypothesis:** Many high-dimensional datasets actually sit near a low-dimensional nonlinear manifold. If so, reducing dimension along that manifold can give a better representation than linear PCA.

**Common methods:**

| Method | Main idea | Typical use |
|--------|-----------|-------------|
| **Kernel PCA** | Map data (implicitly) to a high-dimensional space with a kernel, then do PCA there. | Nonlinear structure; moderate **n**. |
| **t-SNE** | Preserve local neighborhoods; emphasis on visualization. | 2D/3D plots; not for downstream modeling (non-invertible, non-parametric). |
| **UMAP** | Preserve local and global structure; scales better; supports out-of-sample projection. | Visualization and sometimes as features; often preferred over t-SNE. |

---

## Kernel PCA

**Kernel methods** let you run linear algorithms in a high-dimensional (or infinite-dimensional) feature space without explicitly computing the mapped points. You only need **inner products** in that space, which you get from a **kernel function**:

**K(xᵢ, xⱼ) = φ(xᵢ)ᵀ φ(xⱼ)**

So you never need to form **φ(x)** explicitly (the “kernel trick”).

**Kernel PCA** does PCA in that implicit feature space. Data that are nonlinearly structured in **x** can become more linear in **φ(x)**, so PCA there finds “nonlinear” principal directions.

**Common kernels:**

- **RBF:** **K(x, y) = exp(−γ ‖x − y‖²)**
- **Polynomial:** **K(x, y) = (xᵀy + c)^d**
- **Linear:** **K(x, y) = xᵀy** (reduces to standard PCA)

**Computation:** You form the **n×n** Gram matrix **K** with entries **K(xᵢ, xⱼ)**, center it in the feature space, and decompose it. The eigenvectors give the principal component scores. Cost is **O(n²)** or more, so it’s used for moderate **n**. Choosing the kernel is a modeling choice (like choosing an architecture).

---

## Dimensionality Reduction in Machine Learning

Using a reduced representation **Z** instead of **X** in ML brings several benefits:

- **Speed:** Fewer dimensions mean smaller matrices and faster training/inference (e.g. **O(k²)** vs **O(p²)** in some linear operations).
- **Noise reduction:** Dropping low-variance (e.g. low-eigenvalue) directions often removes noise and improves signal-to-noise for the next step.
- **Regularization / reduced complexity:** Fewer parameters (e.g. **p → k**) can reduce overfitting, especially when **n ≪ p**.
- **Generalization:** Working in a lower-dimensional subspace can improve out-of-sample performance when the true signal lives in that subspace.

So DR is not only for visualization—it’s a standard tool for making training feasible and improving generalization.

---

## Tradeoffs in Dimensionality Reduction

**Benefits:**

- Computational efficiency.
- Lower variance / less overfitting.
- Noise filtering.
- Visualization.
- Decorrelated features (e.g. PCA components are uncorrelated).

**Costs and limitations:**

- **Interpretability:** New dimensions (e.g. PC1, PC2) usually don’t correspond to measurable quantities; that can be a problem in regulated or explainability-sensitive domains.
- **Information loss:** Discarded variance can contain signal useful for **y**; PCA is **unsupervised** and ignores **y**, so it might remove predictive directions. Supervised DR (e.g. LDA) can help but adds complexity and overfitting risk.
- **Linearity:** PCA is linear; if the manifold is curved, linear DR can miss structure (hence kernel PCA, t-SNE, UMAP).
- **Hyperparameters:** **k**, kernel choice, perplexity (t-SNE), etc., all need tuning.

The decision to reduce dimension is always a **trade**: representation fidelity vs. tractability and generalization. Make it consciously and with the downstream task in mind.

---

## Worked Examples

### Example 1: Standardization and PCA

You have **X** with **n = 100**, **p = 5**, and columns on very different scales (e.g. one in 0–1, another in 0–10⁶).

1. **Center and standardize:** Compute **z = (x − μ)/σ** per column so each has mean 0 and variance 1. (If you don’t, the largest-scale feature will dominate the covariance matrix.)
2. **PCA on standardized data:** Compute **Σ = (1/n) Xᵀ X** (on the standardized **X**) and **Σ = V Λ Vᵀ**. Or run SVD on the centered, standardized **X**.
3. **Choose k:** Suppose the first two eigenvalues account for 85% of total variance. Take **V₂** (first two columns of **V**) and form **Z = X V₂**.
4. **Use Z:** You now have 100×2 data for visualization or as input to a model. Original features are mixed into PC1 and PC2; you can inspect **V** to see which original variables contribute most.

### Example 2: When scaling changes the model

- **Ridge regression:** Penalty is **λ Σⱼ βⱼ²**. If one feature is in millions and another in 0–1, the larger one will get a tiny coefficient and the smaller one a huge one unless you scale. So **scale first**, then fit.
- **k-NN:** Distances are **Σⱼ (xⱼ − x′ⱼ)²**. The feature with the largest scale dominates. So **scale** so each dimension contributes fairly.
- **Tree:** Splits are of the form “**xⱼ ≤ c**”. Monotonic rescaling doesn’t change the split order. So scaling is **not** required (but doesn’t hurt).

---

## Common Confusions

1. **“PCA removes the features I care about.”** PCA removes directions of **low variance**; it doesn’t know about **y**. If the predictive signal is in a low-variance direction, PCA can drop it. For prediction, consider supervised DR or feature selection.
2. **“Scaling changes the model.”** For linear models and distance-based methods, **yes**—different scales give different solutions. For trees, scaling doesn’t change splits.
3. **“More components = better.”** More components keep more variance but add dimensions and can overfit. Use explained variance or downstream performance to choose **k**.
4. **“t-SNE is for feature extraction.”** t-SNE is mainly for **visualization**; it’s not invertible and can be sensitive to perplexity. For features, prefer PCA, kernel PCA, or UMAP.
5. **“Feature selection and dimensionality reduction are the same.”** Selection **keeps** a subset of original variables (interpretable). DR **creates** new combined variables (often not directly interpretable).

---

## Pitfalls and Tips

- **Don’t fit PCA (or any DR) on the full data if you’ll use it for supervised learning.** Fit it on training data only, then apply the same **W** (or **Vₖ**) to validation/test to avoid leakage.
- **Standardize before PCA** (unless all features are already on the same scale). Otherwise, scale dominates the covariance matrix.
- **Polynomial degree:** Start with 2–3; use cross-validation. High degree + many features = overfitting.
- **Interactions:** Include main effects **x₁**, **x₂** when you add **x₁x₂**; otherwise the interpretation of the interaction is tied to the arbitrary coding of the main effects.
- **Lasso for selection:** If you need interpretable predictors, Lasso (L1) can drive some coefficients to zero and effectively perform feature selection within the model.

---

## Checklists

**Before modeling:**

- [ ] Are features on comparable scales? If using ridge, lasso, SVM, k-NN, or PCA → standardize (or min-max).
- [ ] Do you suspect interactions? Consider adding key product terms (and main effects).
- [ ] Do you have **p** large relative to **n**? Consider feature selection or dimensionality reduction to reduce overfitting.

**When using PCA:**

- [ ] Data centered (and usually standardized)?
- [ ] **k** chosen by explained variance or validation, not arbitrarily?
- [ ] PCA fitted on training data only; same transformation applied to test?

**When using nonlinear DR:**

- [ ] Visualization only → t-SNE or UMAP are fine.
- [ ] Need out-of-sample or downstream features → UMAP or kernel PCA; avoid t-SNE for that.
- [ ] Kernel PCA: **n** not too large (Gram matrix **n×n**).

---

## Putting It Together

Feature engineering and dimensionality reduction address two questions: **What representation should the model see?** and **How many dimensions can we afford?**

- **Feature engineering** (transformations, scaling, interactions, polynomials) makes the relationship between inputs and target easier to learn and keeps optimization stable.
- **Feature selection** keeps a subset of original variables and preserves interpretability.
- **Dimensionality reduction** (especially PCA) compresses the data linearly, minimizes reconstruction error (Eckart–Young), and is implemented reliably via SVD. Choose **k** by explained variance or downstream performance.
- **Nonlinear DR** (kernel PCA, t-SNE, UMAP) is for when structure is not linear; use it for visualization or when you have reason to believe in a manifold.

Always be aware of tradeoffs: interpretability vs. compression, information loss (PCA ignores **y**), and the cost of extra hyperparameters. Use the checklists above to avoid common mistakes.

---

## Glossary

- **Curse of dimensionality:** In high dimensions, data become sparse, distances become less informative, and sample needs grow quickly with dimension.
- **Eckart–Young theorem:** The best rank-**k** approximation to a matrix (in Frobenius norm) is given by the truncated SVD (top-**k** PCA projection).
- **Eigen decomposition:** Writing **Σ = V Λ Vᵀ** with **V** orthogonal and **Λ** diagonal; used in PCA.
- **Embedded (feature selection):** Selection built into the model (e.g. Lasso).
- **Explained variance ratio:** **λₖ / Σᵢ λᵢ** for component **k**; fraction of total variance in that component.
- **Feature engineering:** Designing or transforming inputs **X** into **Z = φ(X)** to simplify learning.
- **Filter (feature selection):** Ranking features by a statistic (e.g. correlation) without fitting the full model.
- **Gram matrix:** Matrix of inner products **K(xᵢ, xⱼ)**; used in kernel methods.
- **Interaction term:** A product of two (or more) predictors, e.g. **x₁x₂**.
- **Kernel trick:** Using **K(xᵢ, xⱼ)** instead of explicit **φ(x)** to run linear methods in a high-dimensional feature space.
- **Principal component:** A direction of maximum variance (eigenvector of **Σ**); components are orthogonal.
- **Reconstruction error:** **‖X − X Vₖ Vₖᵀ‖²_F**; minimized by PCA (Eckart–Young).
- **Sample covariance matrix:** **Σ = (1/n) Xᵀ X** (for centered **X**).
- **Singular Value Decomposition (SVD):** **X = U Σ Vᵀ**; numerically preferred way to compute PCA.
- **Standardization:** **z = (x − μ)/σ**; zero mean, unit variance.
- **Wrapper (feature selection):** Searching over feature subsets and evaluating each with a model (e.g. cross-validation).

---

## Further Reading

- Course slides: *W6_Feature-Engineering-and-Dimensionality-Reduction.pdf* (Prof. Yueming Xing).
- Standard references: Jolliffe, *Principal Component Analysis*; Hastie et al., *The Elements of Statistical Learning* (chapters on linear methods, regularization, and PCA).
- For nonlinear DR: UMAP documentation and papers on t-SNE and kernel PCA.
