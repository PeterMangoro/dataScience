# Linear Models & Interpretability
## A Beginner's Guide in Plain English

---

## Table of Contents

1. [Quick Start (2-Minute Overview)](#quick-start-2-minute-overview)
2. [Introduction](#introduction)
3. [The Linear Model Setup](#the-linear-model-setup)
4. [Ordinary Least Squares (OLS)](#ordinary-least-squares-ols)
5. [Geometric Interpretation and Hat Matrix](#geometric-interpretation-and-hat-matrix)
6. [Statistical Properties of OLS](#statistical-properties-of-ols)
7. [Gauss–Markov Theorem (BLUE)](#gaussmarkov-theorem-blue)
8. [Multicollinearity and Variance Inflation](#multicollinearity-and-variance-inflation)
9. [Hypothesis Testing and Confidence Intervals](#hypothesis-testing-and-confidence-intervals)
10. [Interpretability and Causality](#interpretability-and-causality)
11. [Connection to Bias–Variance and Regularization](#connection-to-biasvariance-and-regularization)
12. [Worked Examples](#worked-examples)
13. [Common Confusions](#common-confusions)
14. [Pitfalls and Tips](#pitfalls-and-tips)
15. [Checklists](#checklists)
16. [Putting It Together](#putting-it-together)
17. [Glossary](#glossary)
18. [Further Reading](#further-reading)

---

## Quick Start (2-Minute Overview)

Linear models predict an outcome **Y** from inputs **X** using a simple form:

**Y = Xβ + ε**

Here **X** is a matrix of features, **β** is a vector of coefficients, and **ε** is noise. Despite the rise of deep learning, linear models remain essential because they:

- Have **closed-form solutions** (no iterative optimizer needed).
- Offer **clear uncertainty quantification** (standard errors, confidence intervals).
- Provide **direct interpretability**: each coefficient tells you how Y changes (on average) when a predictor changes.
- Serve as **theoretical benchmarks** for more complex models.

Under standard assumptions, the ordinary least squares (OLS) estimator

**β̂ = (XᵀX)⁻¹ XᵀY**

is unbiased and has minimum variance among linear unbiased estimators (Gauss–Markov). But multicollinearity, model misspecification, and causal misinterpretation can all cause trouble.

**In one sentence:** Linear models are the simplest place where estimation, uncertainty, and interpretability all meet, and understanding them deeply helps you reason about any predictive model.

---

## Introduction

### What is a linear model in simple terms?

A **linear model** says that the expected value of Y can be written as a **weighted sum of features**:

**E[Y ∣ X] ≈ β₀ + β₁X₁ + ⋯ + β_p X_p**

You choose features (columns of X), estimate weights β from data, and then use those weights to predict Y for new X.

- If Y is a continuous outcome (e.g. price, score), this is ordinary linear regression.
- If Y is transformed or linked (e.g. via a logit link), we get generalized linear models—but this guide focuses on the classic linear case.

### Why linear models still matter

- **Interpretability:** Each βⱼ has a clear meaning: expected change in Y for a one-unit increase in Xⱼ, holding other predictors constant.
- **Tractability:** You can solve for β̂ exactly and derive exact formulas for its variance and confidence intervals.
- **Baseline:** They provide a reference point for what is possible under “ideal” conditions (Gauss–Markov) and highlight why and when more complex models are needed.
- **Bridge to regularization:** Methods like ridge and lasso are small modifications of OLS; understanding OLS helps you understand these.

### How to use this guide

- **Read in order** if you want the full story: model setup → OLS → geometry → statistical properties → multicollinearity → tests/intervals → interpretability & causality.
- **Use the Table of Contents** to jump to topics as needed (e.g. “What does β mean?” or “What is multicollinearity?”).
- Look for **Key Insight** and **Think About It** callouts for the main ideas.
- You do not need to memorize every formula; focus on what each one means and when you would use it.

---

## The Linear Model Setup

We observe n data points with p predictors. Stack them as:

- **Y ∈ ℝⁿ**: response vector.
- **X ∈ ℝⁿˣᵖ**: design matrix (row i is the feature vector for observation i).
- **β ∈ ℝᵖ**: unknown coefficient vector.
- **ε ∈ ℝⁿ**: noise vector.

The **classical linear model** is:

**Y = Xβ + ε,   ε ∼ (0, σ²I)**

Key dimensions:

- **n** — number of observations (rows of X).
- **p** — number of predictors (columns of X).
- **σ²** — noise variance (scalar).

Core assumptions:

- **Linearity in parameters:** E[Y ∣ X] is linear in β (not necessarily linear in raw features if you include transformations like x², logs, etc.).
- **Exogeneity:** **E[ε ∣ X] = 0**, so errors have mean zero given the predictors.
- **Spherical errors:** **Var(ε) = σ²I** (constant variance and uncorrelated errors).
- **Full column rank of X:** no perfect linear dependence among columns.

**Key Insight:** The geometry (columns of X, their span, and XᵀX) and the noise assumptions together make OLS both computationally simple and statistically well-behaved.

---

## Ordinary Least Squares (OLS)

The ordinary least squares estimator **β̂** chooses coefficients to make predictions Xβ̂ as close as possible to Y in squared-error sense:

**β̂ = arg min_β ‖Y − Xβ‖²**

Taking derivatives and setting them to zero gives the **normal equations**:

**XᵀX β = XᵀY**

When **XᵀX** is invertible (X has full column rank), you can solve exactly:

**β̂ = (XᵀX)⁻¹ XᵀY**

Conditions for existence and uniqueness:

- X must have **full column rank p** (no exact linear dependence among predictors).
- This implies **det(XᵀX) ≠ 0**.
- Often we also require **n ≥ p** so that full rank is possible, but n > p by itself is not enough; the real condition is invertibility of XᵀX.

**Think About It:** If two columns of X are exact copies (perfect multicollinearity), XᵀX is singular. Then there are infinitely many β that give the same fitted values, and the formula (XᵀX)⁻¹XᵀY does not exist.

---

## Geometric Interpretation and Hat Matrix

A powerful way to understand linear regression is as **orthogonal projection** in ℝⁿ.

- The columns of X span a p-dimensional **subspace** of ℝⁿ, called the **column space** of X (col(X)).
- OLS chooses β̂ so that **Ŷ = Xβ̂** is the orthogonal projection of Y onto col(X).

We can write:

**Ŷ = P Y,   where   P = X(XᵀX)⁻¹Xᵀ**

The matrix **P** is called the **hat matrix** (it puts a “hat” on Y). It has key properties:

- **Symmetric:** Pᵀ = P.
- **Idempotent:** P² = P.

The **residuals** are:

**e = Y − Ŷ = (I − P)Y**

They lie in the orthogonal complement of col(X), and satisfy:

**Xᵀ e = 0**

meaning residuals are orthogonal to every column of X.

Simple picture (very schematic):

```
Y (in ℝⁿ)
  *
  |
  |\
  | \
  |  \   e (residual, orthogonal)
  |   \
  |    * Ŷ  (projection of Y)
  +-------------------->  col(X)
```

**Key Insight:** Linear regression “uses up” all linear information in X: what is left in the residuals is uncorrelated with X. This is central for interpretation and inference.

---

## Statistical Properties of OLS

Under the standard assumptions, OLS has clean sampling properties.

### Unbiasedness

Because E[ε ∣ X] = 0 and Y = Xβ + ε,

**E[β̂ ∣ X] = β**

So OLS is **unbiased** for β: on average over repeated samples (with the same design X), it recovers the true coefficients.

### Variance

The covariance matrix of β̂ is:

**Var(β̂ ∣ X) = σ² (XᵀX)⁻¹**

Consequences:

- The **design matrix X controls precision**: well-spread, nearly orthogonal columns (well-conditioned XᵀX) lead to small variances and stable estimates.
- **Poorly conditioned XᵀX** (e.g. from multicollinearity) makes entries of (XᵀX)⁻¹ large, inflating Var(β̂) and hence standard errors.

**Key Insight:** Even when OLS is unbiased, it can have huge variance if the predictors are nearly collinear. That’s where multicollinearity and regularization come in.

---

## Gauss–Markov Theorem (BLUE)

The **Gauss–Markov theorem** is a fundamental optimality result:

> Under the assumptions of linearity, E[ε ∣ X] = 0, and Var(ε ∣ X) = σ²I, the OLS estimator β̂ is the **Best Linear Unbiased Estimator (BLUE)** of β.

“Best” here means:

- Among all estimators that are **linear in Y** and **unbiased**, OLS has the **smallest variance** for each component β̂ⱼ.

Important clarifications:

- It does **not** assume that ε is normally distributed; normality is used for exact t and F tests, not for Gauss–Markov.
- The theorem only compares **linear unbiased** estimators. Biased or nonlinear estimators may achieve lower **MSE**, tying back to the W4 bias–variance tradeoff.

**Think About It:** If you insist on an unbiased linear estimator, there is no point searching for something “better” than OLS—Gauss–Markov says you will not reduce variance.

---

## Multicollinearity and Variance Inflation

**Multicollinearity** happens when predictors are **highly correlated** or (nearly) linear combinations of each other. Then:

- **XᵀX** becomes **nearly singular**.
- Entries of **(XᵀX)⁻¹** become large.
- Consequently, **Var(β̂)** and standard errors can become huge.

This makes coefficients:

- Numerically unstable (small changes in data ⇒ big swings in β̂).
- Statistically imprecise (wide confidence intervals).
- Hard to interpret individually (signs and magnitudes may not be reliable).

### Variance Inflation Factor (VIF)

For predictor Xⱼ, the **Variance Inflation Factor** is:

**VIFⱼ = 1 / (1 − Rⱼ²)**

where **Rⱼ²** is the R² from regressing Xⱼ on all the other predictors.

- If Xⱼ is nearly a linear combination of the others, Rⱼ² is close to 1 and VIFⱼ is large.
- Rules of thumb: VIF > 5 or 10 often signals problematic multicollinearity.

**Consequences:**

- Coefficient estimates are unstable.
- Standard errors inflate, widening confidence intervals.
- Predictors may appear “insignificant” even when they matter.

**Remedies:**

- Ridge regression (L2 penalty).
- Principal component regression.
- Drop or merge redundant variables using domain knowledge.
- Collect more data with better variation in predictors.

---

## Hypothesis Testing and Confidence Intervals

### Testing a single coefficient

To test whether predictor Xⱼ contributes linearly to Y (given other predictors), we test:

- **H₀: βⱼ = 0** vs **H₁: βⱼ ≠ 0**

The usual **t-statistic** is:

**tⱼ = β̂ⱼ / SE(β̂ⱼ)**

where:

**SE(β̂ⱼ) = σ̂ √[(XᵀX)⁻¹ⱼⱼ]**

Under normal errors (or large n via CLT), tⱼ is approximately t-distributed with **n − p** degrees of freedom.

### Confidence intervals

An approximate (1 − α) confidence interval (e.g. 95%) for βⱼ is:

**β̂ⱼ ± t_{α/2, n−p} SE(β̂ⱼ)**

Interpretation:

- If we repeated the study many times and built a CI each time, about (1 − α)·100% of those intervals would contain the true βⱼ.
- Wider SE or larger critical value ⇒ wider interval ⇒ more uncertainty.

**Cautions:**

- Validity relies on model assumptions (linearity, homoscedasticity, independence).
- Testing many coefficients at once inflates the chance of at least one false positive (multiple comparisons).
- **Statistically significant** (p < 0.05) does **not** imply **practically large**.

---

## Interpretability and Causality

### What does βⱼ mean?

In the linear model, the conditional mean is:

**E[Y ∣ X] = Xβ**

The coefficient βⱼ is the **partial derivative**:

**βⱼ = ∂ E[Y ∣ X] / ∂ Xⱼ**

Informally:

> βⱼ is the expected change in Y for a one-unit increase in Xⱼ, **holding all other predictors constant**.

This “ceteris paribus” interpretation is straightforward when predictors are independent, but becomes tricky when:

- Predictors are strongly correlated (e.g. height and weight).
- Changing Xⱼ while holding others fixed leads to combinations not seen in the data (extrapolation).

So coefficients are **conditional associations**, and their real-world meaning depends on the joint distribution of X.

### Causal interpretation: exogeneity

To interpret βⱼ as a **causal effect**, we need more than the linear model and OLS algebra. A key requirement is **exogeneity**:

**E[ε ∣ X] = 0**

This encodes:

- No omitted confounders that affect both Xⱼ and Y.
- No reverse causality (Y causing Xⱼ).
- Correct functional form for the conditional mean.

If E[ε ∣ X] ≠ 0 (e.g. due to omitted variables), then OLS coefficients can be **biased**, and βⱼ may not reflect the causal effect of Xⱼ on Y.

**Key Insight:** A regression coefficient always describes conditional association. It only has a causal meaning under additional assumptions about how X and Y were generated (design, confounding, etc.).

---

## Connection to Bias–Variance and Regularization

From W4, prediction error at x decomposes as:

**E[(Y − f̂(x))²] = σ² + Bias²(f̂(x)) + Var(f̂(x))**

In linear regression with OLS:

- If the linear model is correctly specified and exogeneity holds, OLS is **unbiased** for β, so bias is mainly about model misspecification.
- The **variance** piece comes from **Var(β̂) = σ² (XᵀX)⁻¹**; when XᵀX is ill-conditioned, variance can be large.

Regularization methods such as **ridge regression** modify the estimator:

**β̂_ridge = (XᵀX + λI)⁻¹ XᵀY**

Effect:

- Introduces **bias** (shrinks coefficients toward 0).
- Often reduces **variance** dramatically, especially with multicollinearity or many predictors.
- Can yield lower overall **MSE** and better generalization.

**Think About It:** Ridge is the linear-model version of the bias–variance tradeoff: a small increase in bias can pay off with a big drop in variance.

---

## Worked Examples

### Example 1: Simple 2D projection picture

Suppose you have one predictor and an intercept, so X has two columns: a column of 1s and a column of x-values. In ℝⁿ:

- The intercept column is the constant vector (1, 1, …, 1).
- The x column is (x₁, x₂, …, xₙ).

The column space of X is all vectors of the form a·1 + b·x. OLS finds a and b so that the fitted values lie in this 2D subspace and are as close as possible (in Euclidean distance) to Y. The residuals are the vertical “leftovers” that are orthogonal to both 1 and x.

### Example 2: Multicollinearity and SE inflation (conceptual)

Imagine two predictors, X₁ and X₂, that are nearly identical (X₂ ≈ X₁). Then:

- XᵀX has off-diagonal entries close to the diagonal entries.
- The matrix is nearly singular, and its inverse has large entries.
- Var(β̂) = σ² (XᵀX)⁻¹ has correspondingly large diagonal entries.

So even if the fitted line looks reasonable and overall predictions are fine, the individual β̂₁ and β̂₂ will have large standard errors and be unstable across samples. This is multicollinearity in action.

---

## Common Confusions

1. **“We need n > p” vs “XᵀX invertible”**  
   n > p is common and helpful, but the real requirement for OLS to exist uniquely is that **X has full column rank** so that **XᵀX is invertible**. You can have n > p and still be singular if columns are linearly dependent.

2. **“Uncorrelated predictors” vs “full rank”**  
   Predictors do **not** need to be uncorrelated. They just must not be exactly linearly dependent. Correlation can be high and still full rank—though this often leads to high variance.

3. **“Normal residuals” vs OLS validity**  
   OLS does not require normal residuals to be unbiased or BLUE. Normality is mainly for exact p-values and confidence intervals.

4. **“Significant” vs “important”**  
   A very small effect can be statistically significant with large n; a large effect can be non-significant with small or noisy data. Always check magnitudes and intervals, not just p-values.

5. **“Regression = causation”**  
   A significant βⱼ does not mean Xⱼ causes Y. Causality needs exogeneity and a credible design (randomization, instruments, etc.).

---

## Pitfalls and Tips

- **Pitfall:** Ignoring multicollinearity and trusting individual coefficient signs and magnitudes in a highly correlated design.  
  **Tip:** Check VIFs, correlation matrices, and confidence intervals.

- **Pitfall:** Treating coefficients as causal effects without checking exogeneity.  
  **Tip:** Ask what variables might be omitted, whether reverse causality is plausible, and how the data were collected.

- **Pitfall:** Over-relying on p-values without considering model assumptions, effect sizes, or multiple comparisons.  
  **Tip:** Use residual diagnostics, effect-size interpretation, and, when needed, corrections for multiple testing.

---

## Checklists

### Before interpreting linear regression coefficients

- [ ] Have you checked that the linear model is a reasonable approximation (plots of Y vs X, residual plots)?
- [ ] Is there evidence of severe multicollinearity (high VIFs, near-singular XᵀX)?
- [ ] Are standard errors and confidence intervals reasonably tight (not extremely wide)?
- [ ] Are you clear about the scale of each predictor so you can interpret a “one-unit change”?

### Before treating coefficients as causal

- [ ] Is exogeneity **E[ε ∣ X] = 0** plausible (no obvious omitted confounders, no reverse causality)?
- [ ] Is the design experimental (randomized) or observational? If observational, have you considered confounding and selection?
- [ ] Are you extrapolating to predictor ranges not well represented in the data?
- [ ] Have you thought about alternative causal explanations for the association?

---

## Putting It Together

Linear models sit at the intersection of **prediction**, **uncertainty quantification**, and **interpretability**. In the simple form **Y = Xβ + ε**, OLS provides:

- A closed-form estimator **β̂ = (XᵀX)⁻¹XᵀY**.
- A geometric interpretation as projection.
- Clean statistical properties (unbiasedness, Var(β̂) = σ²(XᵀX)⁻¹, Gauss–Markov BLUE).

At the same time, real data bring multicollinearity, model misspecification, and causal ambiguity. Understanding how **XᵀX**, **Var(β̂)**, and the error assumptions interact helps you:

- Diagnose when linear regression is a good tool.
- Interpret coefficients responsibly.
- Know when to reach for regularization or more complex models.

---

## Glossary

**Design matrix (X):** Matrix whose rows are observations and columns are predictors (including an intercept column if used).

**Ordinary Least Squares (OLS):** Estimation method that chooses β̂ to minimize the sum of squared residuals ‖Y − Xβ‖².

**Residuals (e):** Differences between observed and fitted values, e = Y − Ŷ.

**Hat matrix (P):** Projection matrix P = X(XᵀX)⁻¹Xᵀ that maps Y to Ŷ = PY.

**Projection matrix:** A matrix P with P² = P that projects vectors onto a subspace; P is symmetric in the OLS setting.

**Gauss–Markov theorem:** Result stating that OLS is the Best Linear Unbiased Estimator (BLUE) under standard assumptions.

**BLUE:** Best Linear Unbiased Estimator; linear in Y, unbiased, and minimum variance among such estimators.

**Multicollinearity:** Situation where predictors are highly linearly correlated, making XᵀX nearly singular and inflating Var(β̂).

**Variance Inflation Factor (VIF):** Measure of how much the variance of β̂ⱼ is inflated due to multicollinearity; VIFⱼ = 1/(1 − Rⱼ²).

**Standard error (SE):** Estimated standard deviation of an estimator; for β̂ⱼ, SE(β̂ⱼ) = σ̂ √[(XᵀX)⁻¹ⱼⱼ].

**t-statistic:** Ratio of estimate to its SE, used for hypothesis tests about coefficients.

**Confidence interval (CI):** Interval designed to contain the true parameter with a specified long-run frequency (e.g. 95%).

**Exogeneity:** Condition E[ε ∣ X] = 0; crucial for unbiasedness and causal interpretation.

**Homoscedasticity:** Constant variance of errors Var(εᵢ ∣ X) = σ²; opposite of heteroscedasticity.

---

## Further Reading

- Course slides: `W5_Linear-Models-and-Interpretability.pdf` (the source for this guide).
- Books:
  - *An Introduction to Statistical Learning* (James et al.), chapters on linear regression.
  - *The Elements of Statistical Learning* (Hastie, Tibshirani, Friedman), linear models and regularization chapters.

As you work with more advanced models, keep linear regression as your mental baseline: ask how each new method changes the bias–variance tradeoff, interpretability of parameters, and assumptions about errors and design.

