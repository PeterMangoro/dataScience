# Classification Methods
## A Beginner's Guide in Plain English

---

## Table of Contents

1. [Quick Start (2-Minute Overview)](#quick-start-2-minute-overview)
2. [Introduction](#introduction)
3. [The Classification Problem](#the-classification-problem)
4. [The Bayes Classifier (Ideal)](#the-bayes-classifier-ideal)
5. [Bayes Error Rate (Irreducible Error)](#bayes-error-rate-irreducible-error)
6. [Loss Functions in Classification](#loss-functions-in-classification)
7. [Logistic Regression (Discriminative)](#logistic-regression-discriminative)
8. [Maximum Likelihood and Cross-Entropy](#maximum-likelihood-and-cross-entropy)
9. [Decision Boundaries and Confidence](#decision-boundaries-and-confidence)
10. [LDA (Generative, Linear Boundary)](#lda-generative-linear-boundary)
11. [QDA (Generative, Quadratic Boundary)](#qda-generative-quadratic-boundary)
12. [Naive Bayes (Generative, Independence)](#naive-bayes-generative-independence)
13. [Support Vector Machines (Margin-Based)](#support-vector-machines-margin-based)
14. [Soft-Margin SVM and the Role of C](#soft-margin-svm-and-the-role-of-c)
15. [The Kernel Trick (Nonlinear Boundaries)](#the-kernel-trick-nonlinear-boundaries)
16. [Decision Trees](#decision-trees)
17. [Random Forests (Bagging)](#random-forests-bagging)
18. [Boosting (AdaBoost & Gradient Boosting)](#boosting-adaboost--gradient-boosting)
19. [Multiclass Classification Strategies](#multiclass-classification-strategies)
20. [Model Complexity vs. Generalization](#model-complexity-vs-generalization)
21. [Worked Examples](#worked-examples)
22. [Common Confusions](#common-confusions)
23. [Pitfalls and Tips](#pitfalls-and-tips)
24. [Checklists](#checklists)
25. [Putting It Together](#putting-it-together)
26. [Glossary](#glossary)
27. [Further Reading](#further-reading)

---

## Quick Start (2-Minute Overview)

**Classification** means predicting a **discrete label** \(Y\) from features \(X \in \mathbb{R}^p\). The theoretical “best possible” classifier is the **Bayes classifier**:

**f\*(x) = arg max\_k P(Y = k ∣ X = x)**

It chooses the class with the highest **posterior probability**. Even this ideal classifier has nonzero error when class distributions overlap; that lower bound is the **Bayes error rate** (irreducible error).

Most real classifiers are different ways to approximate Bayes:

- **Logistic regression**: directly models **P(Y=1 ∣ X)** with a sigmoid; trained by **maximum likelihood**, equivalent to minimizing **cross-entropy**.
- **LDA / QDA**: generative models using Gaussian assumptions for **P(X ∣ Y=k)**; LDA yields **linear** boundaries (shared covariance), QDA yields **quadratic** boundaries (class-specific covariances).
- **Naive Bayes**: generative model that assumes conditional independence of features given the class; very parameter-efficient (great for high-dimensional problems like text).
- **SVM**: chooses a separating hyperplane with **maximum margin**; soft margin uses **C** to trade off margin width vs training violations; kernels give nonlinear boundaries.
- **Trees / Random forests / Boosting**: nonparametric methods; trees are interpretable but high-variance, forests reduce variance via bagging, boosting builds strong models sequentially by focusing on mistakes.

**In one sentence:** Classification is about approximating the Bayes rule under finite data, using a model whose assumptions and complexity match your data (and controlling overfitting with regularization or ensembles).

---

## Introduction

### What are we trying to do?

We observe labeled training data:

- **(Xᵢ, Yᵢ), i = 1, …, n**
- **Xᵢ ∈ ℝᵖ**
- **Yᵢ ∈ {1, …, K}**

We want a rule that maps features to class labels:

- **f(X) : ℝᵖ → {1, …, K}**

and generalizes to new samples from the same distribution \(P(X, Y)\).

### Two big modeling philosophies

- **Discriminative**: model \(P(Y ∣ X)\) directly (e.g. logistic regression).
- **Generative**: model \(P(X ∣ Y)\) and \(P(Y)\), then use Bayes’ rule to compute \(P(Y ∣ X)\) (e.g. LDA/QDA/Naive Bayes).

### How to use this guide

- Use the **Quick Start** to remember what each method is “assuming.”
- For method choice, focus on **bias–variance** and **data regime** (small \(n\), large \(p\), separability, nonlinearity, interpretability needs).

---

## The Classification Problem

Classification starts with an unknown joint distribution **P(X, Y)**. A classifier is a decision rule \(f\) that outputs a discrete label.

Key perspective: classification is not just “predict labels”; it’s often “estimate probabilities” plus a decision rule.

---

## The Bayes Classifier (Ideal)

The **Bayes classifier** is the best possible rule *if* you knew the true posterior \(P(Y ∣ X)\):

**f\*(x) = arg max\_k P(Y = k ∣ X = x)**

Why it matters:

- It provides a **gold standard** and a conceptual target.
- All practical methods are attempts to estimate (or approximate) \(P(Y ∣ X)\) well enough to act like this rule.

---

## Bayes Error Rate (Irreducible Error)

Even with the optimal rule, you can’t do better than the **Bayes error rate** when classes overlap:

**Bayes error = E\[ 1 − max\_k P(Y = k ∣ X) \]**

Interpretation:

- If two classes overlap in feature space, no boundary can separate them perfectly.
- This is “error due to the data,” not due to your model choice.

Analogy to regression:

- Like irreducible noise \(ε\) in regression, classification has irreducible overlap-based error.

Practical implication:

- If your test error is close to the Bayes error, you’re near the ceiling.
- If you’re far above it, there’s room to improve features, model class, or training.

---

## Loss Functions in Classification

The loss function determines what “good” means during training.

### 0–1 loss (misclassification loss)

Counts mistakes:

- **L(Y, Ŷ) = I(Y ≠ Ŷ)**

Pros/cons:

- Directly matches accuracy.
- Hard to optimize (non-differentiable, combinatorial).

Many methods (trees, nearest-neighbor ideas) are best understood as approximating this objective.

### Log loss (cross-entropy)

Penalizes wrong confidence heavily:

- **L = − log P(Y ∣ X)**

Why it’s popular:

- Differentiable (and convex for logistic regression).
- Naturally trains **probabilistic** models.

### Hinge loss

Encourages correct classification with margin:

- **L = max(0, 1 − y · f(x))**

This is the defining loss for **SVMs** (with margin-based thinking).

---

## Logistic Regression (Discriminative)

Logistic regression models the probability of the positive class (binary case):

**P(Y = 1 ∣ X) = 1 / (1 + e^{−Xβ})**

Key properties:

- Outputs values in \( (0, 1) \), so they behave like probabilities.
- The model is linear in features inside the logit link, but nonlinear in the raw probability.

### Log-odds (logit) interpretation

Equivalent form:

- **log( P(Y=1 ∣ X) / (1 − P(Y=1 ∣ X)) ) = Xβ**

So each \(β_j\) is “change in log-odds per unit change in \(X_j\), holding others fixed.”

---

## Maximum Likelihood and Cross-Entropy

Logistic regression parameters are fit by **maximum likelihood**.

### Likelihood and log-likelihood (binary)

Let \(p_i = P(Y_i = 1 ∣ X_i)\). Then the log-likelihood is:

- **ℓ(β) = Σᵢ \[ Yᵢ log pᵢ + (1 − Yᵢ) log(1 − pᵢ) \]**

Important consequences:

- There’s **no closed-form** solution like OLS.
- The objective is concave in \(β\) (so it has a unique global optimum), found by numerical methods (e.g. gradient methods, Newton–Raphson / IRLS).
- Maximizing log-likelihood is equivalent to minimizing **cross-entropy**.
- Regularization (L1/L2) can be added to prevent overfitting, especially when \(p\) is large.

---

## Decision Boundaries and Confidence

In binary logistic regression, a common rule is:

- Predict class 1 if **P(Y=1 ∣ X) > 0.5**

This threshold corresponds exactly to a linear boundary:

- **Xβ = 0**

Geometry intuition:

- The decision boundary is a hyperplane in feature space.
- Points far from the boundary get probabilities near 0 or 1 (high confidence).
- Points near the boundary get probabilities near 0.5 (low confidence).

When the true boundary is nonlinear, you can add nonlinear features (polynomials/interactions), or use nonlinear models (kernels, trees, boosting).

---

## LDA (Generative, Linear Boundary)

**LDA** models the class-conditional distribution and then uses Bayes’ rule.

Assumption:

- **X ∣ Y = k ~ N(μ_k, Σ)** (class-specific means, **shared covariance**)

Why the boundary is linear:

- Sharing the covariance forces the log-likelihood ratio to be linear in \(x\).

Decision rule:

- Choose the class that maximizes a **linear discriminant score** \(δ_k(x)\) (derived from Gaussian log-densities and priors \(π_k\)).

When LDA tends to work well:

- The Gaussian assumption is “good enough.”
- You have **small sample sizes** relative to \(p\), because pooling into one shared covariance reduces variance.

---

## QDA (Generative, Quadratic Boundary)

QDA relaxes LDA’s shared-covariance assumption.

Assumption:

- **X ∣ Y = k ~ N(μ_k, Σ_k)** (each class has its own covariance)

Consequence:

- Boundaries become **quadratic** (curved) because covariance differs by class.

LDA vs QDA (tradeoff):

- **LDA**: fewer parameters → lower variance, higher bias if covariances truly differ.
- **QDA**: more parameters → lower bias, higher variance; needs more data.

Rule of thumb:

- With limited \(n\), LDA often generalizes better.
- With large \(n\) and genuinely different covariances, QDA can outperform.

---

## Naive Bayes (Generative, Independence)

Naive Bayes assumes **conditional independence** of features given the class:

- **P(X ∣ Y) = Πⱼ P(Xⱼ ∣ Y)**

Why it can work despite being “wrong”:

- Even if features are correlated, it can still classify well if correlation patterns are similar across classes.

Why it scales well:

- You estimate **p univariate** distributions per class (not a full \(p\)-dimensional covariance).
- This is especially helpful in high-dimensional settings where full covariance estimation is unstable.

Common variants:

- **Gaussian Naive Bayes** (continuous features)
- **Multinomial Naive Bayes** (counts / text)
- **Bernoulli Naive Bayes** (binary features)

---

## Support Vector Machines (Margin-Based)

SVMs aim for the **maximum-margin** separating hyperplane (binary classification), rather than modeling probabilities.

Hard-margin SVM (linearly separable case):

- Minimize **(1/2)‖w‖²**
- Subject to **yᵢ (wᵀxᵢ + b) ≥ 1** for all \(i\)

Key ideas:

- The solution depends only on a subset of points: **support vectors** (closest to the boundary).
- Larger margins tend to generalize better (structural risk minimization viewpoint).

---

## Soft-Margin SVM and the Role of C

Real data are rarely perfectly separable, so we allow violations using slack variables \(ξ_i\).

Soft-margin objective (conceptually):

- **min (1/2)‖w‖² + C Σᵢ ξᵢ**
- with \(ξ_i ≥ 0\) measuring margin violations.

How to interpret **C**:

- **Large C**: punish violations heavily → narrower margin, closer fit to training data → higher variance risk.
- **Small C**: tolerate violations → wider margin, smoother boundary → higher bias risk.

Connections:

- Soft-margin SVM is equivalent to minimizing **hinge loss** with L2 regularization.
- \(C\) plays a role similar to \(1/λ\) in ridge-style regularization.

---

## The Kernel Trick (Nonlinear Boundaries)

The SVM dual depends only on **inner products** \(x_iᵀx_j\). A **kernel** replaces these with:

- **K(xᵢ, xⱼ) = φ(xᵢ)ᵀ φ(xⱼ)**

so you can learn a linear separator in a transformed feature space without explicitly computing \(φ(x)\).

Common kernels:

- **Linear**: \(K(x, y) = xᵀy\)
- **Polynomial**: \(K(x, y) = (xᵀy + c)^d\)
- **RBF/Gaussian**: \(K(x, y) = exp( −‖x − y‖² / (2σ²) )\)

Intuition:

- Polynomial kernels capture interaction-like structure up to degree \(d\).
- RBF kernels are highly flexible (can fit very complex boundaries with enough data) but can overfit if not tuned carefully.

---

## Decision Trees

Decision trees partition feature space into rectangular regions and assign a class label per region.

How they are built:

- **Recursive greedy splitting**: at each node, choose a feature and threshold that best improves purity.
- Stop based on constraints (max depth, min leaf size) or prune afterward.

Split criteria (classification):

- **Gini impurity**:
  - **G = 1 − Σ_k p_k²**
  - \(G=0\) means a pure node (all one class).
- Entropy is an alternative and often yields similar trees.

Pros/cons:

- **Pros**: interpretable, handles nonlinearities, little preprocessing.
- **Cons**: can overfit badly (high variance) if unconstrained.

---

## Random Forests (Bagging)

Random forests reduce the high variance of single trees by averaging many de-correlated trees.

Key mechanisms:

- **Bootstrap sampling**: each tree trains on a sample with replacement.
- **Random feature subsets**: at each split, consider only a random subset of features (often \(√p\)) to reduce correlation between trees.

Prediction:

- Majority vote over trees (classification).

Additional benefit:

- Provides **feature importance** measures based on impurity reduction (and other approaches).

---

## Boosting (AdaBoost & Gradient Boosting)

Boosting builds models **sequentially**, each one focusing on mistakes made by previous models.

General form:

- **f(x) = Σ_{m=1}^M α_m h_m(x)**

where \(h_m\) are weak learners (often shallow trees).

### AdaBoost

- Reweights examples: misclassified points get higher weight.
- Can be seen as minimizing an exponential-type loss.

### Gradient boosting

- Interprets boosting as **gradient descent in function space**.
- Each new learner fits the negative gradient (“residual-like” signal) of the loss.
- Practical implementations (widely used): **XGBoost, LightGBM, CatBoost**.

Tradeoff:

- Boosting can be extremely powerful, but it needs careful regularization/tuning (learning rate, tree depth, number of trees).

---

## Multiclass Classification Strategies

Many problems have \(K > 2\) classes. Common extensions:

### One-vs-Rest (OvR)

- Train \(K\) binary classifiers, each “class k vs all others.”
- Predict the class with the highest score.

### One-vs-One (OvO)

- Train \(K(K−1)/2\) pairwise classifiers.
- Predict via majority vote across pairwise wins.

### Softmax / Multinomial logistic regression

Model all class probabilities together:

- **P(Y=k ∣ X) = exp(Xβ_k) / Σ_j exp(Xβ_j)**

Rule of thumb:

- If you want coherent, well-calibrated probabilities across classes, softmax is often preferred.
- OvR is common/default for linear SVMs and many libraries.
- OvO can help when \(K\) is small and pairwise tasks are easier/less imbalanced.

---

## Model Complexity vs. Generalization

Classification follows the same **bias–variance** logic as regression.

- **High bias / underfitting**: model too simple → train and test error both high → add features, nonlinearities, or capacity.
- **High variance / overfitting**: model too flexible → low train error but high test error → regularize, prune, use ensembles, or gather more data.
- **Optimal complexity**: best test error; found via cross-validation and regularization.

Tools for controlling complexity:

- Regularization (L1/L2, soft-margin SVM parameter \(C\), tree pruning constraints, learning-rate shrinkage in boosting).
- Cross-validation for model selection and hyperparameter tuning.

---

## Worked Examples

### Example 1: Choosing between logistic regression, LDA, and QDA

Suppose you have a binary classification task with modest features \(p\), and you suspect roughly Gaussian clusters.

- If \(n\) is small and covariance estimation is unstable, **LDA** can work well because it pools covariance.
- If you see clearly different class shapes (different covariances) *and* you have enough data, **QDA** can capture that curvature.
- If you want a simple, robust baseline and/or want probabilities without Gaussian assumptions, **logistic regression** is often the default.

### Example 2: Interpreting the SVM hyperparameter C

You train a linear SVM and observe:

- With very large **C**, training accuracy is high but validation accuracy drops → likely overfitting (narrow margin, too many violations penalized).
- With smaller **C**, training accuracy decreases slightly but validation improves → better generalization (wider margin, smoother boundary).

So you tune **C** by cross-validation, targeting best validation performance rather than perfect training fit.

### Example 3: Why random forests beat a single tree

You fit a deep decision tree and it performs extremely well on training but poorly on test.

- A single deep tree has **high variance**.
- A random forest averages many noisy trees trained on bootstrapped data and random feature splits, dramatically reducing variance.

If you need interpretability, you can:

- Keep a smaller pruned tree for explanation,
- Use a forest for accuracy, and summarize with feature importance / partial dependence / SHAP (tool-dependent).

---

## Common Confusions

1. **“Why not optimize 0–1 loss directly?”**  
   It’s non-differentiable and hard to minimize globally; surrogate losses (log loss, hinge loss) make optimization tractable and often generalize well.

2. **“Logistic regression has a linear boundary, so it’s too simple.”**  
   The boundary is linear in the feature space you feed it. With engineered features (interactions, polynomials) it can represent nonlinear boundaries.

3. **“LDA/QDA vs logistic regression: which is better?”**  
   They’re different assumptions. If Gaussian generative assumptions are good and \(n\) is small, LDA/QDA can be excellent. Logistic is often robust and a strong baseline without modeling \(P(X ∣ Y)\).

4. **“Naive Bayes assumes independence, so it must be bad.”**  
   It can still classify well in practice, especially when you mostly need correct class ranking and when high-dimensional estimation is the bottleneck.

5. **“SVM gives probabilities.”**  
   Standard SVM outputs margins/scores, not calibrated probabilities (though calibration methods exist).

6. **“Boosting always overfits.”**  
   Boosting can overfit, but with shrinkage (learning rate), shallow trees, and early stopping, it often generalizes extremely well.

---

## Pitfalls and Tips

- **Fit your model selection to your goal**: if you need probabilities and interpretability, logistic regression or calibrated probabilistic models are a better match than raw SVM scores.
- **Don’t over-interpret Bayes error**: it’s conceptual; we rarely know it exactly. Use it as an “irreducible overlap” idea, not a number you can compute without strong assumptions.
- **Watch data regime**:
  - Small \(n\), large \(p\): generative parameter estimation can be hard unless assumptions simplify (LDA, Naive Bayes).
  - Large \(n\): more flexible models (QDA, kernels, boosting) become viable.
- **Tune hyperparameters with cross-validation** (especially \(C\) in SVM, kernel parameters, tree depth/min leaf size, boosting learning rate/trees).
- **Trees**: unconstrained trees overfit; use pruning / depth limits, or switch to forests/boosting.

---

## Checklists

### Choosing a classifier quickly

- [ ] Do you need well-calibrated probabilities? → start with logistic regression / softmax (or calibrate later).
- [ ] Do you believe roughly Gaussian classes?
  - [ ] Covariances similar? → try LDA.
  - [ ] Covariances different and you have enough data? → try QDA.
- [ ] Is \(p\) huge (e.g. text counts)? → Naive Bayes is a strong baseline.
- [ ] Do you suspect a complex boundary and want strong performance? → try boosting / kernel SVM (with careful tuning).
- [ ] Do you want minimal preprocessing and nonlinear structure? → trees/forests/boosting.

### Diagnosing under/overfitting

- [ ] Training error high and test error high → underfitting → increase capacity / features.
- [ ] Training error low but test error high → overfitting → regularize (smaller \(C\) in SVM, prune trees, smaller depth, smaller learning rate, fewer features, more data, ensembles).

---

## Putting It Together

Classification is fundamentally about approximating the Bayes rule “choose the most likely class given \(X\)” under finite data.

- **Logistic regression**: simple, interpretable, probabilistic; linear boundary in feature space.
- **LDA/QDA**: generative Gaussian models; LDA is low-variance linear, QDA is higher-variance quadratic.
- **Naive Bayes**: strong simplifying assumption → very efficient estimation; great in high-dimensional domains.
- **SVMs**: margin-based; soft margin uses \(C\); kernels enable nonlinear boundaries.
- **Trees/forests/boosting**: flexible nonparametric methods; forests reduce variance; boosting reduces error sequentially and is often state-of-the-art in tabular data.

The “best” method depends on your data regime, assumptions, interpretability needs, and how you manage complexity to generalize.

---

## Glossary

- **Bayes classifier**: The optimal rule \(f\*(x)=argmax_k P(Y=k∣X=x)\) under 0–1 loss if \(P(Y∣X)\) were known.
- **Bayes error rate**: The minimum achievable classification error due to class overlap; \(E\[1−max_k P(Y=k∣X)\]\).
- **0–1 loss**: Misclassification indicator loss \(I(Y≠Ŷ)\).
- **Log loss / cross-entropy**: \(-\log P(Y∣X)\); heavily penalizes confident wrong predictions.
- **Hinge loss**: \(max(0, 1−y f(x))\); used by SVMs.
- **Discriminative model**: Models \(P(Y∣X)\) directly (e.g. logistic regression).
- **Generative model**: Models \(P(X∣Y)\) and \(P(Y)\), then computes \(P(Y∣X)\) via Bayes’ rule (e.g. LDA/QDA/Naive Bayes).
- **Logit / log-odds**: \(log(p/(1−p))\); linear in logistic regression features.
- **LDA**: Gaussian generative classifier with shared covariance → linear boundary.
- **QDA**: Gaussian generative classifier with class-specific covariance → quadratic boundary.
- **Naive Bayes**: Generative classifier with conditional independence \(P(X∣Y)=Π_j P(X_j∣Y)\).
- **Margin**: Distance between decision boundary and nearest points; SVM maximizes it.
- **Support vectors**: Training points that determine the SVM boundary.
- **Kernel trick**: Use \(K(x_i,x_j)=φ(x_i)ᵀφ(x_j)\) to learn nonlinear boundaries via inner products.
- **Gini impurity**: \(1−Σ_k p_k^2\); used for decision tree splits.
- **Bagging**: Bootstrap aggregation; used in random forests to reduce variance.
- **Boosting**: Sequentially add weak learners focusing on previous errors (AdaBoost, gradient boosting).

---

## Further Reading

- Course slides: `W7_Classification_Methods.pdf` (Prof. Yueming Xing).
- For deeper theory: *The Elements of Statistical Learning* (Hastie, Tibshirani, Friedman), chapters on classification, discriminant analysis, SVMs, and trees/ensembles.
