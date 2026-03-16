# Model Evaluation, Validation & Data Leakage
## A Beginner's Guide in Plain English

---

## Table of Contents

1. [Quick Start (2-Minute Overview)](#quick-start-2-minute-overview)
2. [Introduction](#introduction)
3. [The Prediction Problem and True Error](#the-prediction-problem-and-true-error)
4. [Training Error vs. Test (Generalization) Error](#training-error-vs-test-generalization-error)
5. [Overfitting and the Bias–Variance Tradeoff](#overfitting-and-the-biasvariance-tradeoff)
6. [Holdout Validation](#holdout-validation)
7. [Limitations of a Single Split](#limitations-of-a-single-split)
8. [k-Fold Cross-Validation](#k-fold-cross-validation)
9. [The Cross-Validation Error Estimate](#the-cross-validation-error-estimate)
10. [Leave-One-Out Cross-Validation (LOOCV)](#leave-one-out-cross-validation-loocv)
11. [Evaluation Metrics for Regression](#evaluation-metrics-for-regression)
12. [Evaluation Metrics for Classification](#evaluation-metrics-for-classification)
13. [Precision, Recall, and F1](#precision-recall-and-f1)
14. [ROC Curves and AUC](#roc-curves-and-auc)
15. [What Is Data Leakage?](#what-is-data-leakage)
16. [Types of Data Leakage](#types-of-data-leakage)
17. [Leakage Example: Scaling Before Splitting](#leakage-example-scaling-before-splitting)
18. [The Correct Pipeline (Split First, Fit on Train Only)](#the-correct-pipeline-split-first-fit-on-train-only)
19. [Nested Cross-Validation](#nested-cross-validation)
20. [Practical Model Evaluation Strategy](#practical-model-evaluation-strategy)
21. [Worked Examples](#worked-examples)
22. [Common Confusions](#common-confusions)
23. [Pitfalls and Tips](#pitfalls-and-tips)
24. [Checklists](#checklists)
25. [Putting It Together](#putting-it-together)
26. [Glossary](#glossary)
27. [Further Reading](#further-reading)

---

## Quick Start (2-Minute Overview)

When we train a model, what we really care about is **how well it predicts new data**, not how perfectly it fits the training set. The key target is the **true prediction error**:

**Err = E\[ L(Y, f̂(X)) \]**

This is the expected loss on a **new** observation drawn from the same data-generating process. Training error is easy to compute but **optimistically biased**, especially for flexible models that can memorize noise.

We estimate generalization error using:

- **Holdout validation**: simple train/test split (plus possibly a validation split).
- **k-fold cross-validation**: rotate folds through the test role, average their errors.
- **Nested cross-validation**: inner loop for hyperparameter tuning; outer loop for honest performance estimation.

Metrics must match the problem:

- **Regression**: MSE, RMSE, MAE, R².
- **Classification**: accuracy, precision, recall, F1, ROC/AUC, etc.

The big danger is **data leakage**: allowing information from the test set (or the future) into training or preprocessing. Leakage makes evaluation look fantastic while real-world performance collapses.

**In one sentence:** Good model evaluation is about estimating true out-of-sample error honestly (via proper splits/CV, appropriate metrics) and ruthlessly preventing data leakage.

---

## Introduction

In supervised learning, we observe training data:

- **(Xᵢ, Yᵢ), i = 1, …, n**, drawn from some unknown distribution.

We choose a model **f̂** and a **loss function** \(L\), then ask:

- How well will **f̂** predict **new** pairs \((X, Y)\) from the same process?

This is harder than it sounds because:

- We only see a **single dataset**.
- Fitting and evaluating on the same data severely **underestimates** true error.

Evaluation and validation are about building **honest experiments** around our modeling process.

---

## The Prediction Problem and True Error

We want **f̂(x) ≈ f(x)**, where \(f\) is the unknown true function.

True prediction error:

- **Err = E\[ L(Y, f̂(X)) \]**

Key points:

- The expectation is over **new data** and over the randomness in the training sample itself.
- This is the quantity that matters in deployment, not the training loss we minimized.

We almost never know Err exactly, so we use data-splitting and cross-validation procedures as **estimators** of it.

---

## Training Error vs. Test (Generalization) Error

**Training error**:

- Computed on the same data used to fit the model.
- Always available and usually small for flexible models.

**Test (generalization) error**:

- Expected loss on fresh data.
- The real measure of performance in production.

As model complexity increases:

- Training error decreases monotonically.
- Test error typically follows a **U-shaped curve**: decreases at first (bias drops), then increases (variance and overfitting rise).

Minimizing training error alone **overfits**; model evaluation is about approximating test error and finding the sweet spot.

---

## Overfitting and the Bias–Variance Tradeoff

With increasing complexity:

- **Bias** (systematic underfitting) tends to decrease.
- **Variance** (sensitivity to noise and sample fluctuations) tends to increase.

Overfitting happens when:

- The model starts fitting idiosyncratic noise instead of the true pattern.
- Training error continues to fall, but test error starts rising.

Goal:

- Choose a model and complexity level that minimize **expected test error**, not training error.
- Use cross-validation and held-out data to estimate where the minimum lies.

---

## Holdout Validation

Simplest approach:

- Split full dataset into:
  - **Training set** (e.g. 70–80%) — used only to fit the model.
  - **Test set** (e.g. 20–30%) — untouched until final evaluation.

Estimated test error:

- Compute loss on the held-out test set after training on the training set.

Disciplined use:

- The test set is **only** for final reporting.
- Iterative choices (feature engineering, hyperparameters, model type) should not be based on test performance.

Often, practitioners use:

- **Train / validation / test** split:
  - Validation is used for tuning and selection.
  - Test is used exactly once at the end.

---

## Limitations of a Single Split

Weaknesses of plain holdout:

- High **variance** in the performance estimate:
  - A “lucky” or “unlucky” test subset can make the model look better or worse than it truly is.
- Tradeoff:
  - Small test set → noisy estimate.
  - Large test set → less data for training.

Consequences:

- Different random splits can produce very different reported errors.
- Model selection based on one split may pick the wrong model.
- Hard to assess the reliability of the estimate from a single split.

This motivates **cross-validation**, which averages over many splits.

---

## k-Fold Cross-Validation

Procedure:

1. Split data into **k** roughly equal folds.
2. For each fold **j = 1,…,k**:
   - Train the model on the other **k−1** folds.
   - Evaluate on fold **j** as the test set.
3. Average the **k** test errors.

Advantages:

- Every observation is used for **both training and testing** (but never in the same fold).
- Reduces variance compared to a single random split.
- Uses data more efficiently, especially for small datasets.

Common choices:

- **k = 5** or **k = 10** usually balances computation and estimate quality.

Bias–variance of CV estimate:

- Larger k:
  - Lower bias (training sets are closer to full data).
  - Potentially higher variance and higher computational cost.

---

## The Cross-Validation Error Estimate

Let **Errⱼ** be the test error on fold **j**. The k-fold CV estimate is:

- **CV\_k = (1/k) Σ_{j=1}^k Errⱼ**

Properties:

- Approximately **unbiased** for test error, since each model trains on most of the data.
- **Reduced variance** relative to a single split (averaging over folds).
- Computationally more expensive (train **k** models) but usually worth it for robust model selection.

CV is the standard tool for:

- Comparing competing models.
- Tuning hyperparameters.
- Estimating performance when no separate test set is available (with care).

---

## Leave-One-Out Cross-Validation (LOOCV)

LOOCV is k-fold CV with **k = n**:

- Each observation is held out exactly once.
- Each model is trained on **n−1** points and tested on the remaining one.

Pros:

- Very small bias (training sets almost full).
- Deterministic (no random split variability).

Cons:

- Often **computationally expensive** (n models).
- **High variance**: the test predictions across folds can be highly correlated, making the overall estimate noisier than 5- or 10-fold CV.

When to use:

- Very small datasets.
- Simple models for which LOOCV has analytic shortcuts.
- Situations requiring maximum data use and exact reproducibility.

In practice, 5- or 10-fold CV is usually preferred.

---

## Evaluation Metrics for Regression

Common metrics (continuous targets):

### Mean Squared Error (MSE)

- **MSE = (1/n) Σᵢ (yᵢ − ŷᵢ)²**
- Penalizes large errors heavily.
- Naturally aligned with squared-loss regression models.

### Root Mean Squared Error (RMSE)

- **RMSE = √MSE**
- Same units as the target variable → easier to interpret.

### Mean Absolute Error (MAE)

- **MAE = (1/n) Σᵢ ∣yᵢ − ŷᵢ∣**
- Treats all errors linearly.
- More robust to outliers than MSE/RMSE.

### R² (coefficient of determination)

- **R² = 1 − SS\_res / SS\_tot**
- Fraction of variance explained by the model (relative to a baseline).
- Useful, but can be misleading with non-linear models, no intercept, or in small samples.

---

## Evaluation Metrics for Classification

Basic metric:

### Accuracy

- **Accuracy = (TP + TN) / (TP + TN + FP + FN)**
- Fraction of correctly classified examples.

Limitation:

- **Misleading under class imbalance**:
  - If 95% of samples are negative, a “always negative” classifier gets 95% accuracy but is useless for detecting positives.

For imbalanced problems, we rely more on **confusion-matrix-derived** metrics:

- True positives (TP)
- False positives (FP)
- True negatives (TN)
- False negatives (FN)

From these, we build precision, recall, F1, ROC/AUC, etc.

---

## Precision, Recall, and F1

### Precision

- **Precision = TP / (TP + FP)**
- “Of all predicted positives, how many are truly positive?”
- Important when **false positives** are expensive (e.g. bad alerts, unnecessary procedures).

### Recall (Sensitivity)

- **Recall = TP / (TP + FN)**
- “Of all actual positives, how many did we catch?”
- Important when **false negatives** are expensive (e.g. missed fraud, missed disease).

### F1 Score

- **F1 = 2 · (Precision · Recall) / (Precision + Recall)**
- Harmonic mean of precision and recall.
- Useful single-number summary when dealing with class imbalance and needing a balance between precision and recall.

---

## ROC Curves and AUC

Many classifiers output **scores** or **probabilities**; by varying the decision threshold, we trade off:

- True Positive Rate (TPR, same as recall).
- False Positive Rate (FPR).

The **ROC curve** plots TPR vs FPR across thresholds.

Interpretation:

- A good classifier’s curve bows toward the **upper-left** corner.
- The **Area Under the Curve (AUC)** is:
  - A single summary between 0 and 1.
  - Interpretable as the probability that a randomly chosen positive is ranked above a randomly chosen negative.
  - Often robust to class imbalance and threshold choice.

---

## What Is Data Leakage?

**Data leakage** occurs when information from the test set (or otherwise unavailable future data) **influences model training or selection**.

Consequences:

- Evaluation metrics are **inflated** and unrealistic.
- Models that look excellent offline can fail catastrophically in production.

Typical sources:

- Preprocessing done on the full dataset (including test data).
- Features that “peek into the future” or encode the target.
- Using the test set repeatedly for model selection and tuning.

Bottom line:

- Leakage breaks the fundamental assumption that evaluation uses **unseen** data.

---

## Types of Data Leakage

### Preprocessing leakage

- Computing scalers, imputers, encoders, etc. on the **entire dataset** (train + test).
- Then applying them to both train and test.
- The test set statistics “leak” into the transformation.

### Target leakage

- Features are derived from the target variable or from information only available **after** prediction time.
- Example: using post-outcome data to predict the outcome.

### Temporal leakage

- In time series or event data, shuffling and splitting randomly can put **future** observations into the training set and earlier ones into the test set.
- This allows information from the future to influence the model used for earlier predictions.

All these forms lead to overly optimistic validation scores.

---

## Leakage Example: Scaling Before Splitting

Incorrect procedure:

1. Compute mean and standard deviation of each feature using **all data**.
2. Standardize all rows with these global statistics.
3. Split into train and test.

Problem:

- Test observations influenced the scaler, so the model has effectively “seen” the test set.
- Evaluations on this test set no longer reflect performance on truly new, unseen data.

Even if the numerical effect is small in large datasets, the **principle** is violated. This can distort model comparison and hyperparameter tuning.

---

## The Correct Pipeline (Split First, Fit on Train Only)

Leakage-safe workflow:

1. **Split first**:
   - Perform train/test split (and possibly validation) **on raw data** as the first step.
2. **Fit transformers on training data only**:
   - Compute scaling, imputation, encoding, etc. using training set.
3. **Transform, don’t refit, on test**:
   - Apply the **same learned transformations** to validation/test sets without recomputing statistics.
4. **Use pipelines**:
   - In libraries like scikit-learn, use `Pipeline` so that cross-validation refits preprocessing and model from scratch inside each training fold, preventing leakage.

Simple rules:

- The test set is **never** used to compute anything that ends up in the model.
- All preprocessing decisions are part of the training pipeline, not global operations on full data.

---

## Nested Cross-Validation

If you use cross-validation for both:

- **Hyperparameter tuning**, and
- **Performance reporting**,

in a **single** loop, you risk optimistic bias: hyperparameters are chosen to look best on the same validation folds used for reporting.

Nested CV solves this by:

- **Outer loop**:
  - Splits data into outer train/test folds.
  - Used only for **performance estimation**.
- **Inner loop** (inside each outer train fold):
  - Runs CV again to select hyperparameters.
  - Inner validation error guides tuning.

Thus, each outer test fold assesses a model whose hyperparameters were chosen **without seeing that outer test data**. Averaging outer errors gives an unbiased estimate, even after tuning.

This is the statistically correct approach for high-stakes or research-grade performance reporting.

---

## Practical Model Evaluation Strategy

Putting ideas together:

1. **Initial train/test split**  
   - Reserve a final test set **once** and set it aside.
2. **Cross-validation on the training set**  
   - Use k-fold CV for model comparison and hyperparameter search (or nested CV for strict separation).
3. **Hyperparameter tuning**  
   - Use inner CV or a validation subset; avoid peeking at test performance while tuning.
4. **Refit on full training data**  
   - Once you’ve chosen a model and hyperparameters, retrain on all training data.
5. **Final single evaluation on the test set**  
   - Evaluate exactly once, report metrics, and avoid further tuning based on this result.

Rule of thumb:

- Every time you use a dataset to make modeling decisions, it effectively becomes part of training; it can no longer serve as an **unbiased test**.

---

## Worked Examples

### Example 1: Correct vs incorrect scaling

- **Incorrect**:
  - Standardize all data, then split into train/test.
  - Result: test information leaks into the scaler; performance looks slightly better than reality.
- **Correct**:
  - Split into train/test.
  - Fit scaler on train only; apply trained scaler to test.

This pattern extends to any learned preprocessing: imputation, encoders, feature selection, PCA, etc.

### Example 2: Choosing k in k-fold CV

You try k=5 and k=10:

- k=5 yields a stable estimate with moderate computation.
- k=10 yields slightly lower average error but with higher variability across folds and roughly double computation.

For most problems, you pick k=5 or 10 depending on:

- Data size (smaller data → larger k can help).
- Computational budget.

### Example 3: Handling imbalanced classification

You have a dataset with 1% positives and 99% negatives.

- Accuracy is ~99% for a trivial classifier → misleading.
- You instead:
  - Use precision, recall, F1, and AUC.
  - Use stratified k-fold CV so each fold maintains class proportions.

This gives a more realistic picture of performance on the minority class.

---

## Common Confusions

1. **“High training accuracy means my model is good.”**  
   High training accuracy can just mean overfitting; only performance on unseen data (via proper validation) matters.

2. **“One test split is enough.”**  
   A single split can give a noisy or misleading estimate; k-fold CV usually provides a more stable picture.

3. **“LOOCV is always best because it uses almost all the data for training.”**  
   LOOCV can have higher variance and much higher computation than 5- or 10-fold CV; it’s not automatically better.

4. **“Accuracy is fine even with class imbalance.”**  
   Accuracy can be nearly useless when classes are skewed; use precision/recall/F1 and ROC/AUC instead.

5. **“Preprocessing on the full dataset is harmless.”**  
   It’s a classic form of data leakage; always fit preprocessing on the training portion only.

6. **“Using the test set to tune hyperparameters is okay as long as I’m careful.”**  
   Any tuning based on test results contaminates it; you then need a new, untouched test set for honest evaluation.

---

## Pitfalls and Tips

- **Pitfall:** Evaluating multiple models and hyperparameters directly on the test set and picking the best.  
  **Tip:** Use validation sets or inner CV for selection; keep the test set for a single final check.

- **Pitfall:** Performing feature selection or PCA on the full dataset before splitting.  
  **Tip:** Include these steps inside the CV loop or training pipeline; fit only on training data for each fold.

- **Pitfall:** Ignoring data ordering in time-series problems.  
  **Tip:** Use time-aware splits (e.g. train on past, test on future) and rolling or blocked CV.

- **Pitfall:** Relying on a single performance metric.  
  **Tip:** Report primary and secondary metrics that reflect different aspects of performance (e.g. ROC AUC + F1 + calibration).

---

## Checklists

### Before trusting your evaluation

- [ ] Is there a **clear separation** between data used for training/tuning and data used for final testing?
- [ ] Was all preprocessing (scaling, imputation, encoding, feature selection, DR) fit **only on training data** in each evaluation step?
- [ ] Is your splitting scheme appropriate for the data structure (e.g. time series vs i.i.d.)?
- [ ] Did you avoid using the test set until the very end?

### Choosing an evaluation procedure

- [ ] Small dataset:
  - [ ] Use k-fold CV (k=5 or 10), possibly nested for tuning.
  - [ ] Consider LOOCV only when n is tiny and computation is manageable.
- [ ] Medium/large dataset:
  - [ ] Use train/validation/test or train/test + CV on training.
  - [ ] For heavy tuning or publication-grade results, consider nested CV.

### Choosing metrics

- [ ] Regression:
  - [ ] Use MSE/RMSE for squared loss models.
  - [ ] Use MAE when robustness to outliers is important.
  - [ ] Report R² for variance-explained intuition (with caveats).
- [ ] Classification:
  - [ ] Check for class imbalance.
  - [ ] Use accuracy only if classes are reasonably balanced.
  - [ ] Use precision/recall/F1 and ROC/AUC when imbalance is present or consequences differ by error type.

---

## Putting It Together

Model evaluation and validation are about **estimating true generalization error** as honestly as possible while guarding against **overfitting and data leakage**.

- Holdout splits and k-fold CV provide estimates of test error.
- Appropriate metrics (MSE/MAE/RMSE/R², accuracy, precision/recall/F1, ROC/AUC) ensure you measure what matters.
- Data leakage can silently invalidate even the most sophisticated evaluation if you let test information slip into preprocessing, feature design, or tuning.
- Nested CV and disciplined pipelines provide principled ways to separate tuning from final evaluation.

Done right, evaluation turns model building from an ad hoc art into a repeatable, trustworthy process.

---

## Glossary

- **Prediction error (true error)**: **Err = E\[L(Y, f̂(X))\]**; expected loss on new data from the same process.
- **Training error**: Loss computed on the dataset used for fitting.
- **Test (generalization) error**: Loss on unseen data; what we approximate with validation procedures.
- **Holdout validation**: Single train/test (and possibly validation) split.
- **k-fold cross-validation**: Procedure that rotates which part of the data serves as the test set, averaging errors.
- **LOOCV**: k-fold CV with k = n; each observation is left out once.
- **MSE / RMSE / MAE**: Standard regression metrics; squared, root-squared, and absolute error averages.
- **Accuracy**: (TP + TN) / (TP + TN + FP + FN).
- **Precision**: TP / (TP + FP); fraction of positive predictions that are correct.
- **Recall (sensitivity)**: TP / (TP + FN); fraction of actual positives detected.
- **F1 score**: Harmonic mean of precision and recall.
- **ROC curve**: Plot of TPR vs FPR across thresholds.
- **AUC**: Area under ROC curve; probability of correct ranking of a random positive vs negative.
- **Data leakage**: Any flow of information from test (or future) data into model training or selection.
- **Preprocessing leakage**: Fitting transformations on full data, then splitting.
- **Target leakage**: Using features that encode the target or future information unavailable at prediction time.
- **Temporal leakage**: Letting future observations influence models for earlier times.
- **Nested cross-validation**: CV with an inner loop for tuning and an outer loop for unbiased performance estimation.

---

## Further Reading

- Course slides: `W8_Model-Evaluation-Validation-and-Data-Leakage.pdf` (Prof. Yueming Xing).
- Books:
  - *An Introduction to Statistical Learning* (James et al.), chapters on model assessment and selection.
  - *The Elements of Statistical Learning* (Hastie, Tibshirani, Friedman), sections on resampling methods and model selection.

