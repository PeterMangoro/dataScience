# Unsupervised Learning & Representation
## A Beginner's Guide in Plain English

*Based on course materials (Prof. Yueming Xing).*

---

## Table of Contents

1. [Quick Start (2-Minute Overview)](#quick-start-2-minute-overview)
2. [Introduction](#introduction)
3. [What Is Unsupervised Learning?](#what-is-unsupervised-learning)
4. [Representation Learning (Framing)](#representation-learning-framing)
5. [Types of Unsupervised Learning](#types-of-unsupervised-learning)
6. [K-Means: Objective and Assumptions](#k-means-objective-and-assumptions)
7. [K-Means Algorithm](#k-means-algorithm)
8. [Limitations of K-Means](#limitations-of-k-means)
9. [Gaussian Mixture Models (GMM)](#gaussian-mixture-models-gmm)
10. [EM Algorithm: Intuition and Steps](#em-algorithm-intuition-and-steps)
11. [DBSCAN](#dbscan)
12. [Hierarchical Clustering](#hierarchical-clustering)
13. [Linkage Criteria](#linkage-criteria)
14. [Distance Metrics](#distance-metrics)
15. [Feature Scaling for Clustering](#feature-scaling-for-clustering)
16. [Curse of Dimensionality (and Clustering)](#curse-of-dimensionality-and-clustering)
17. [Anomaly Detection](#anomaly-detection)
18. [Isolation Forest](#isolation-forest)
19. [Representation for Prediction](#representation-for-prediction)
20. [Worked Examples](#worked-examples)
21. [Common Confusions](#common-confusions)
22. [Pitfalls and Tips](#pitfalls-and-tips)
23. [Checklists](#checklists)
24. [Putting It Together](#putting-it-together)
25. [Glossary](#glossary)
26. [Further Reading](#further-reading)

---

## Quick Start (2-Minute Overview)

**Unsupervised learning** works when you only have inputs **X** and no labels **Y**. Instead of modeling **P(Y | X)** (supervised), you model **P(X)** or discover structure (clusters, low-density regions, compact codes). Evaluation is indirect because there is no single “right answer” for every point.

**Clustering** groups similar points: **K-means** minimizes within-cluster variance (hard assignments, Euclidean geometry); **GMMs** add soft assignments and elliptical clusters via **EM**; **DBSCAN** finds dense regions and noise without fixing **k**; **hierarchical** methods build a full merge tree (dendrogram).

**Distance and scale** define what “similar” means—standardize features for distance-based methods. In **high dimensions**, distances concentrate and clustering often needs **dimensionality reduction** first (see the W6 guide for PCA and related tools).

**Representation learning** turns raw **X** into a latent **Z** so downstream tasks (including supervised learning) need fewer labels. Unsupervised learning does not replace supervised learning; it **prepares** the space where prediction happens.

**In one sentence:** Unsupervised methods encode assumptions about geometry and density to uncover structure in **P(X)**; choose the algorithm, metric, and preprocessing to match that structure, then use representations to make supervised learning more data-efficient.

---

## Introduction

This guide follows the Week 10 deck on unsupervised learning and representation. It connects:

- The **problem setting** (no labels, model **P(X)** or latent structure).
- **Major families** of methods (clustering, density ideas, representations, anomalies).
- **Practical geometry**: distances, scaling, high-dimensional effects.
- How unsupervised steps **feed** modern predictive pipelines.

Read in order for a single narrative, or jump via the table of contents.

---

## What Is Unsupervised Learning?

### The core problem

You observe **X₁, …, Xₙ** from an unknown distribution, but **no Y**. Supervised learning targets **P(Y | X)**; unsupervised learning targets **P(X)** or transformations of **X** that reveal structure.

That makes the task **harder** in one sense: there is no ground-truth label to score against, and “success” depends on your goal (interpretation, downstream accuracy, stability of clusters).

### Why it matters

- **Labels are expensive**; most real-world data is unlabeled.
- Understanding **P(X)** supports better modeling, feature design, and risk control before you commit to a supervised objective.

| Idea | Meaning |
|------|--------|
| **No labels** | Only **X** is available. |
| **Model P(X)** | Learn the data distribution or its structure directly. |
| **Ambiguous success** | Without **Y**, evaluation is task-dependent (stability, silhouette, downstream metrics, expert review). |

---

## Representation Learning (Framing)

Unsupervised learning is often viewed as **representation learning**: map raw inputs to a space where:

- **Patterns** are easier to separate,
- **Noise** is reduced or compressed away,
- **Relationships** are simpler so later models are more **data-efficient**.

So the output is not only “clusters” or “scores” but a **better coordinate system** for whatever comes next.

---

## Types of Unsupervised Learning

Every method bakes in assumptions about **geometry**, **density**, **independence**, or **normality**. The taxonomy below is a useful mental map.

| Family | Goal |
|--------|------|
| **Clustering** | Partition (or soft-partition) data into groups of similar observations. |
| **Density estimation** | Model **P(X)** explicitly or implicitly. |
| **Representation learning** | Learn a compact encoding (e.g. latent vectors) of **X**. |
| **Anomaly detection** | Flag rare points that deviate from “normal” structure. |

---

## K-Means: Objective and Assumptions

**K-means** minimizes the **within-cluster sum of squared distances** to centroids—equivalently, it pushes down variance inside each cluster.

**Implicit assumptions:**

- **Squared Euclidean** distance defines dissimilarity.
- Clusters are roughly **spherical** in that metric.
- **All features count equally** (so scale matters).

If those assumptions fail, the algorithm can still **converge**, but the partition may be **misleading**.

---

## K-Means Algorithm

K-means **alternates** assignment and update. Each step lowers the objective, so it converges—but typically to a **local** minimum that depends on initialization.

1. **Initialize:** Choose **k** centroids (random, **K-means++**, etc.).
2. **Assign:** Assign each point to its **nearest** centroid (Euclidean).
3. **Update:** Set each centroid to the **mean** of its assigned points.
4. **Repeat** until assignments stop changing (or change negligibly).

**K-means++** improves initialization so centroids start spread out, reducing bad local minima compared to pure random starts.

---

## Limitations of K-Means

| Issue | What goes wrong |
|--------|------------------|
| **Non-spherical clusters** | Elongated, crescent, or interleaved groups fit poorly to “one centroid per blob.” |
| **Scaling** | Large-magnitude features dominate squared distance; **always** treat scaling as a modeling step, not an afterthought. |
| **Choosing k** | **k** must be set in advance; heuristics include **elbow** plots, **silhouette**, domain knowledge, or stability across runs. |
| **Local minima** | Poor initialization yields suboptimal partitions; use **restarts** or **K-means++**. |

---

## Gaussian Mixture Models (GMM)

A **Gaussian Mixture Model** replaces hard K-means assignments with **soft** responsibilities: each point is generated by a mixture of **k** multivariate Gaussians.

**Compared to K-means:**

- **K-means:** hard assignment, Voronoi boundaries, spherical flavor.
- **GMM:** posterior probabilities over components, **elliptical** covariances, uncertainty near boundaries.

---

## EM Algorithm: Intuition and Steps

For a GMM you want to maximize the **marginal log-likelihood** of **X**, but **component membership** is **latent**. If you knew assignments, estimating parameters would be easy; if you knew parameters, assignments would be easy—you know neither.

**Expectation–Maximization (EM)** breaks the circle:

- **E-step:** Given current parameters, compute the **posterior probability** (responsibility) that each point belongs to each component.
- **M-step:** Update means, covariances, and mixing weights using those responsibilities as **weights** (closed-form under Gaussian assumptions).

Each **E** then **M** cycle increases the observed-data log-likelihood until convergence (again, often a **local** maximum—**restarts** help).

---

## DBSCAN

**DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) discovers **dense** regions in feature space.

**Highlights:**

- **No k required** in advance—the number of clusters **emerges** from density.
- Can find **non-convex** and elongated groups.
- **Noise** is explicit: sparse points can remain unclustered.

**Hyperparameters:**

- **ε:** neighborhood radius.
- **MinPts:** minimum neighbors to call a region “dense.”

**Point types:**

- **Core:** at least **MinPts** neighbors within **ε**.
- **Border:** not core but within **ε** of a core point.
- **Noise:** neither core nor border.

Clusters grow from cores; borders attach to nearby cores, which allows **flexible shapes**.

---

## Hierarchical Clustering

**Hierarchical** clustering builds a **nested tree** of merges—a **dendrogram**. You start with each point as its own cluster and repeatedly merge the **closest** pair of clusters until one cluster remains.

| Property | Notes |
|----------|--------|
| **No fixed k** | Choose **k** by **cutting** the dendrogram at a height. |
| **Full hierarchy** | One tree encodes many granularities. |
| **Cost** | Roughly **O(n² log n)** for standard implementations—less friendly than K-means at very large **n**. |

---

## Linkage Criteria

Linkage defines **distance between two clusters** from pairwise point distances. It strongly affects cluster shape and the dendrogram.

| Linkage | Between-cluster distance | Typical behavior |
|---------|---------------------------|------------------|
| **Single** | **Minimum** pairwise distance | **Chaining**; elongated clusters; sensitive to bridges/outliers. |
| **Complete** | **Maximum** pairwise distance | More **compact** clusters; more robust to outliers than single. |
| **Average** | **Mean** pairwise distance | Common **compromise** between chaining and fragmentation. |

---

## Distance Metrics

The metric is the primitive for “similar.” Common choices:

| Metric | Idea | When it helps |
|--------|------|----------------|
| **Euclidean** | Straight-line in ℝᵖ | Continuous features; K-means / GMM default geometry. |
| **Manhattan (L1)** | Sum of absolute coordinate differences | Grid-like or robust-to-outlier-ish settings (still coordinate-dependent). |
| **Cosine** | Angle between vectors; ignores **magnitude** | Text, embeddings, sparse high-dimensional data. |

---

## Feature Scaling for Clustering

Distance-based clustering **requires** thoughtful scaling. If one feature has a huge numeric range (e.g. salary) and another is binary, Euclidean distance mostly reflects the large-scale feature.

**Standardization (Z-score):** subtract mean, divide by std → roughly zero mean, unit variance per feature. Often appropriate when features are roughly bell-shaped.

**Min–max:** map to **[0, 1]**. Simple but **outliers** can squash most of the mass into a narrow band.

Choosing the scaler is a **modeling decision**, not a universal default—but **doing nothing** is also a decision (usually a bad one for K-means).

---

## Curse of Dimensionality (and Clustering)

As **p** grows:

- **Concentration of measure:** pairwise distances can become **nearly identical**, so “nearest” is unstable and geometry stops discriminating.
- **Sparsity:** volume grows exponentially in **p**; a fixed **n** looks **empty** in high dimensions—**DBSCAN** needs much more data to find dense neighborhoods.

**Practical takeaway:** For high-dimensional **X**, **reduce or re-embed** (PCA, autoencoders, learned embeddings) **before** naive distance-based clustering when distances stop being meaningful.

---

## Anomaly Detection

An **anomaly (outlier)** is an observation that lies in a **low-probability** region of the learned structure—far from typical density or hard to reconstruct.

**Approaches (often unsupervised):**

- **Distance-based:** flag points far from **k** nearest neighbors (e.g. k-NN outlier scores).
- **Density-based:** low local density; **DBSCAN** noise labels; kernel density notions of “rare” regions.
- **Partition / ensemble-tree:** **Isolation Forest** (see next section)—isolates points using random splits; not the same math as KDE, but also flags “easy to separate” points.
- **Reconstruction-based:** autoencoders—**high reconstruction error** suggests “unlike training data.”

No labels are required to **score** unusualness; labels help only if you want to **calibrate** precision/recall for a specific harm.

---

## Isolation Forest

**Isolation Forest** (Liu, Ting, Zhou; 2008) is an **unsupervised** anomaly detector built from an **ensemble of isolation trees**. The core idea: **anomalies are few and different**, so random axis-aligned partitions tend to **isolate** them in **few splits**; inliers need **more** splits to end up alone in a leaf.

### How a single isolation tree is built

On a **subsample** of the data (for efficiency and to reduce swamping by outliers):

1. Pick a **feature** at random.
2. Pick a **split value** uniformly between the **min and max** of that feature **on the subsample**.
3. Partition points into left/right; recurse until each node holds one point or depth cap is hit.

**Path length** from root to a point’s leaf measures how quickly that point was isolated. **Short** average path length across many trees ⇒ point is easier to separate from the bulk ⇒ treated as **more anomalous**.

### Ensemble score

The algorithm trains **many** trees on different subsamples and combines path lengths into an **anomaly score** (implementations differ in normalization; scikit-learn uses a convention where **more negative** `score_samples` means **more anomalous**, or you use `decision_function` / `predict` with `contamination`).

### Why it is useful

- **Low** tuning surface compared to density methods in high **p**: no pairwise distance matrix over all points.
- Handles **different scales** of features reasonably if you follow the same preprocessing discipline as other methods (see **Feature Scaling** above—tree splits are axis-aligned but **magnitude still affects** which splits are reachable).
- **Linear-ish** time in **n** for fixed ensemble size, so it scales to larger **n** than many k-NN distance schemes.

### Assumptions and limits

- Assumes anomalies are **sparse** and **separable** by random cuts—not always true if anomalies lie **inside** the bulk distribution in every 1D projection.
- Training data should be **mostly normal**; a high fraction of anomalies (**contamination**) can distort subsample min/max and **dilute** the signal (set `contamination` or tune threshold on a clean validation slice if you have one).
- Like all unsupervised detectors, **threshold** choice is a business decision (false alarm vs miss tradeoff).

---

## Representation for Prediction

Modern pipelines rarely predict straight from raw pixels, tokens, or sensors. A common pattern:

1. **Raw data** — high-dimensional, redundant, noisy.
2. **Unsupervised representation** — PCA, autoencoder, contrastive learning, etc.—produces **Z**.
3. **Supervised model** — trained on **(Z, Y)** (or fine-tuned with few labels).

A good **Z** removes redundancy, reduces noise, and exposes latent factors; the supervised head then generalizes with **less labeled data**. So unsupervised learning **strengthens** supervised learning rather than replacing it.

---

## Worked Examples

### Example 1: K-means with mixed scales

You have features “age (0–100)” and “income (0–500k).”

1. **Standardize** each column (or otherwise put them on comparable scales).
2. Choose **k** (domain + elbow/silhouette + multiple random seeds).
3. Run K-means; **compare** cluster profiles on **original-scale** summaries for reporting (even though you clustered in standardized space).

If you skip step 1, clusters mostly separate by **income bands**.

### Example 2: DBSCAN vs K-means on two moons

Two interleaved crescents:

- **K-means** with **k = 2** often cuts the wrong way (centroid geometry).
- **DBSCAN** with tuned **ε**, **MinPts** can trace **non-convex** density—if sample size supports stable local density.

### Example 3: GMM + EM mental checklist

1. Initialize means (e.g. K-means), covariances (spherical or full), mixing weights.
2. **E-step:** responsibilities **γᵢⱼ** = **P(component j | xᵢ)**.
3. **M-step:** weighted MLE updates for each Gaussian and **πⱼ**.
4. Monitor log-likelihood; try **restarts** if plateaus look weak.

### Example 4: Isolation Forest sketch

1. Fit on **training** normal-heavy data only: `IsolationForest(n_estimators=200, contamination=0.02)` (contamination is a prior on anomaly rate if you use `predict`; tune from domain or validation).
2. **`score_samples`:** more negative ⇒ more anomalous (sklearn convention).
3. On **new** data, apply the same fitted model; adjust threshold using labeled holdout or cost curve if available.

---

## Common Confusions

1. **“Unsupervised accuracy.”** Without **Y**, there is no single accuracy; use internal indices, stability, or **downstream** supervised performance—know what each measures.
2. **“K-means found the true clusters.”** It found a **local minimum** of its objective; that may or may not match human semantics.
3. **“DBSCAN needs no parameters.”** It needs **ε** and **MinPts**; those can be as fiddly as **k**.
4. **“Hierarchical clustering is always better.”** It is **expensive** and linkage choice can dominate; K-means/GMM scale better for huge **n**.
5. **“Cosine vs Euclidean for everything.”** Cosine ignores magnitude—great for normalized text vectors, wrong when absolute scale carries signal.
6. **“Isolation Forest is density estimation.”** It does **not** fit **P(X)** explicitly; it scores how quickly a point is **separated** by random partitions. It often behaves well where density is hard to estimate, but the logic is different from KDE or DBSCAN.

---

## Pitfalls and Tips

- **Scale** before distance-based clustering; document whether you used Z-score or min–max and why.
- **Try multiple seeds** for K-means and GMM; watch **instability** as a diagnostic.
- **Validate cluster count** with several tools (domain, elbow, silhouette, prediction on a small labeled slice if available).
- In **high p**, **reduce dimensions** or use **metric learning** before trusting nearest-neighbor structure.
- For **production anomaly detection**, plan how scores become **thresholds** (cost of false alarms vs misses).
- **Isolation Forest:** if you suspect **many** anomalies in training, set **`max_samples`** and **`contamination`** deliberately or clean the training pool—otherwise min/max splits absorb the corruption.

---

## Checklists

**Before clustering:**

- [ ] Are features scaled appropriately for the chosen distance?
- [ ] Is **n** large enough relative to **p** for density methods?
- [ ] Have you set **k** (K-means/GMM) or **ε**, **MinPts** (DBSCAN) with sensitivity analysis?

**Before hierarchical clustering:**

- [ ] Chosen **linkage** with the failure modes (chaining vs compactness) in mind?
- [ ] Acceptable runtime for **n**?

**Before using outputs downstream:**

- [ ] Fit any **learned representation** on training data only; apply the same transform to validation/test to avoid leakage.
- [ ] If the end goal is **prediction**, check whether the representation **preserves** label-relevant directions (unsupervised objectives can discard them).

**Isolation Forest:**

- [ ] Training set mostly “normal,” or **`contamination`** / threshold chosen with care?
- [ ] **`n_estimators`** large enough for stable scores (often hundreds)?
- [ ] Scoring pipeline applied identically at inference time?

---

## Putting It Together

Unsupervised learning answers: **What does the data “look like” without Y?** Clustering partitions or soft-partitions points; density-based methods respect **local geometry**; **Isolation Forest** flags points that random trees isolate quickly; hierarchical methods expose **multi-scale** structure; representation learning builds **Z** for everything that follows.

The through-line is **assumptions**: Euclidean vs cosine, spherical vs elliptical vs arbitrary density, linear vs learned embeddings. Match the method to the geometry you believe in, **preprocess** deliberately, and treat unsupervised outputs as **hypotheses**—especially when they drive high-stakes decisions.

---

## Glossary

- **Anomaly / outlier:** Point in a low-density or hard-to-reconstruct region relative to a model of “normal” data.
- **Centroid (K-means):** Mean of all points assigned to a cluster.
- **DBSCAN:** Density-based clustering with **core / border / noise** roles and parameters **ε**, **MinPts**.
- **Dendrogram:** Tree of hierarchical merges; horizontal cuts pick a partition.
- **EM algorithm:** Alternating E-step (posterior over latents) and M-step (parameter updates) to increase likelihood.
- **GMM:** Mixture of Gaussians with mixing weights and component parameters; soft cluster memberships.
- **Isolation Forest:** Ensemble of random isolation trees; short average path length ⇒ more anomalous.
- **Linkage:** Rule for distance between clusters (single, complete, average, …).
- **Representation / latent space:** Compressed or structured encoding **Z** of inputs **X**.
- **Responsibility (GMM):** Posterior probability that observation **i** belongs to component **j**.
- **Unsupervised learning:** Learning from **X** without **Y**; targets structure in **P(X)** or useful transforms of **X**.

---

## Further Reading

- Course slides: *W10_Unsupervised-Learning-and-Representation.pdf* (Prof. Yueming Xing).
- Companion note in this repo: *W6_Feature_Engineering_Dimensionality_Reduction_Beginner_Guide.md* (PCA, scaling, nonlinear DR context).
- Textbook depth: Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning* (unsupervised chapters: clustering, mixtures, PCA).
- Practical clustering: scikit-learn user guides for **KMeans**, **GaussianMixture**, **DBSCAN**, and **AgglomerativeClustering**.
- Anomalies: Liu, Ting & Zhou, “Isolation Forest,” *ICDM* 2008; scikit-learn [`sklearn.ensemble.IsolationForest`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html).

---
