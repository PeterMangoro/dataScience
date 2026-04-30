# Your Final Notebook Explained (Plain English)

This document explains what [final_notebook.ipynb](final_notebook.ipynb) does, in everyday language, including the main **ideas behind the math** and how it lines up with the course rubric in [final.md](final.md).

**Important:** Exact numbers (F1, accuracy, etc.) depend on **which cells you ran** and **random seed / data subset**. The notebook stores some example outputs; after you re-run everything, use **your** printed tables as the source of truth for the report.

---

## Big picture: what problem are we solving?

We use **Amazon product reviews** (text) to predict **sentiment**:

- **Negative** reviews are labeled one way in the raw file (`__label__1` → stored as `0`).
- **Positive** reviews are labeled the other way (`__label__2` → stored as `1`).

So the core task is: **read the review text and guess whether it is positive or negative**.

That matches **Option 4** in `final.md` (Amazon Reviews, NLP, embeddings, Transformers).

---

## The main “equation” story (lay version)

You do **not** need to memorize formulas for the grade. You *do* need to know what they *mean*.

Below, formulas are written as **plain text** so they render in any Markdown preview (no LaTeX required).

### 1) What we are trying to learn (high level)

Think of a model as a **rule** that maps each review (call it **x**) to a score or probability of “positive”:

```text
score = f(x)
```

- **Classical models** (TF-IDF + logistic / SVM-style): **f** is built from **word counts and word pairs**, with weights learned from data.
- **Neural network**: **f** is a **small network** that turns word IDs into a score.
- **Transformer (DistilBERT)**: **f** is a **deep language model** that reads context (order, negation, etc.) much more richly.

**Noise (epsilon)** in plain words: reviews are messy. People exaggerate, joke, or contradict themselves. The label can be imperfect. That is “error and randomness” the model cannot fully remove.

### 2) TF-IDF + logistic regression (Phase 3 baseline) — intuition

**TF-IDF** turns each review into a long list of numbers: “how important is each word / phrase in *this* review compared to all reviews?”

**Logistic regression** learns weights so that a **weighted sum** of those numbers predicts the class.

A common way to write the predicted probability of “positive” is:

```text
P(positive | review x) = sigmoid( w · phi(x) + b )
```

Read it like this:

- **phi(x)** = the TF-IDF vector for review **x** (a big list of feature weights)
- **w** and **b** = learned weights (what the model “learns”)
- **sigmoid(...)** = the “S-curve” that squashes a score into a number between **0** and **1**

**SGDClassifier (hinge)** in your notebook is a **linear SVM-style** classifier on the same TF-IDF features: still “linear,” but a different training objective than logistic regression.

### 3) Simple neural network (Phase 5) — intuition

Your NN turns words into vectors (**embeddings**), averages them into one vector per review, then passes through small layers to produce a **logit** (a raw score).

Training uses **binary cross-entropy** (a standard loss for yes/no problems). In words:

- If the true label is positive, the model is pushed to raise the predicted probability.
- If the true label is negative, the model is pushed to lower it.

For one training example, if the true label is **y** (0 or 1) and the predicted probability is **p_hat**, the loss is often written as:

```text
loss = -( y * log(p_hat) + (1 - y) * log(1 - p_hat) )
```

(Your code uses logits internally; that is equivalent but numerically safer.)

### 4) Transformer (DistilBERT) — intuition

DistilBERT outputs **two logits** (scores for each class). A **softmax** turns logits into probabilities that sum to 1.

Training minimizes a standard classification loss (the notebook uses Hugging Face `AutoModelForSequenceClassification`, which is trained like multi-class logistic regression at the output layer).

### 5) Metrics you report (Phase 3 and Phase 5)

These are how you judge “good vs bad” in a disciplined way (as `final.md` asks).

**Accuracy:** fraction of correct guesses.

**Precision (positive class):** among reviews you called positive, how many were truly positive?

**Recall (positive class):** among reviews truly positive, how many did you catch?

**F1:** a balance between precision and recall (your notebook’s **primary** choice).

```text
F1 = 2 * (precision * recall) / (precision + recall)
```

**ROC-AUC:** how well your model ranks positives above negatives using predicted scores (your **tie-break** metric).

**PR-AUC (average precision):** especially informative when classes are imbalanced (your classes are roughly balanced, but it is still useful context).

### 6) Unsupervised part (Phase 4) — intuition

Here you are **not** predicting labels yet. You are asking:

> “Do reviews naturally fall into groups by language style or topic?”

**K-means** tries to split points into clusters by distance.

**Silhouette** is a simple “cluster quality” score (higher often means clearer separation), but it is not perfect—your notebook still interprets clusters using profiles and examples.

---

## Walkthrough: what each phase in the notebook does

### Phase 1 — Data intake + contract (`final.md` component: foundation for everything)

**Plain English:** load the files, parse each line, and prove the data looks sane.

**What the notebook checks:**

- Row counts for train and test files
- Label balance
- Text length stats (how long reviews are)
- A rule: **`test.ft.txt` is held out** for final evaluation; development uses `train.ft.txt` splits

**Aligns to `final.md`:** evaluation design + leakage prevention starts here (official test split discipline).

---

### Phase 2 — DGP narrative + EDA (`final.md` component #1 DGP + #2 EDA)

**Plain English:** tell a believable story of *how* reviews get created, then use plots and tables to check whether the data behaves like that story.

**Examples of DGP ideas in the notebook:**

- Reviews are selective (not everyone writes one)
- Text can be short or long; length may relate to tone
- Language is messy and context-dependent

**EDA tools used:**

- Class balance charts
- Length distributions
- Example shortest/longest reviews
- Top words per class (on a sample)

**Aligns to `final.md`:** DGP assumptions + EDA with interpretation (not “plot only”).

---

### Phase 3 — Baselines (`final.md` component #5 supervised: linear + classical)

**Plain English:** build two strong, fast models that are easy to explain:

1. TF-IDF + logistic regression  
2. TF-IDF + SGD hinge (SVM-style)

They share the same validation split and the same metrics, then you compare them in `baseline_results_df`.

**Representative example output captured in the notebook (your run may differ slightly):**

- Both models were around **~0.905 accuracy** in one saved output
- **SGD (hinge)** edged ahead on **F1** in the sorted table in that run

**Aligns to `final.md`:** required linear baseline + classical model + rigorous evaluation framing.

---

### Phase 4 — Unsupervised (`final.md` component #4 unsupervised + representation)

**Plain English:** two ways to discover “groups of reviews”:

1. **TF-IDF → dimensionality reduction (SVD) → k-means** (very interpretable via words)
2. **Sentence embeddings → k-means** (more “semantic,” heavier compute)

You sweep cluster counts **k** in **{4, 5, 6, 7, 8}**, compare stability, then profile clusters (sizes, label mix, example texts).

**Aligns to `final.md`:** clustering + representation learning + connect findings to supervised modeling story.

---

### Phase 5 — Neural network + Transformer (`final.md` components #5–#7)

**Plain English:**

- **NN:** a lightweight deep learning baseline that is not a Transformer.
- **Transformer:** fine-tune DistilBERT on review text for classification.

The notebook tracks **training curves** (loss / validation F1 over epochs) so you can discuss overfitting and stability—not only final accuracy.

**Representative example output captured in the notebook:**

- Transformer subset run printed on the order of **~1800+ seconds** in one output (hardware-dependent)
- The combined results table (`model_results_df`) in one saved output showed **Phase 5 Transformer** around **F1 ≈ 0.87** vs baselines around **F1 ≈ 0.91** in that snapshot—meaning **the strongest model is not always the Transformer** on the chosen subset/settings, which is a *good* story if you explain tradeoffs (speed, memory, data size, training time).

**Aligns to `final.md`:** neural nets + training practice + required Transformer + comparative thinking.

---

### Phase 6 — Leakage audit + defensible recommendation (`final.md` components #9–#10)

**Plain English:** prove you did not “cheat” by accidentally letting future information or test data leak into training, then recommend what you would actually deploy.

The notebook adds:

- A checklist for preprocessing/feature/temporal/duplicate leakage
- A duplicate overlap diagnostic on a sample split
- A final narrative template driven by `model_results_df` (ranked by F1 then ROC-AUC)

**Aligns to `final.md`:** leakage is explicitly critical + comparative recommendation with tradeoffs.

---

## How this lines up with `final.md` grading themes (quick map)

| `final.md` topic | Where it shows up in the notebook |
|---|---|
| DGP + assumptions | Phase 2 markdown + DGP-to-EDA table |
| EDA with interpretation | Phase 2 plots + “decision links” |
| Feature engineering thinking | TF-IDF features + length features + Phase 4 bridge notes |
| Unsupervised + link to supervised | Phase 4 |
| Linear + classical + NN + Transformer | Phases 3 and 5 |
| Training dynamics | Phase 5 loss/F1 plots + interpretation prompts |
| Evaluation rigor | Shared metrics + held-out test policy |
| Leakage | Phase 6 checklist + duplicate diagnostic |
| Defensible recommendation | Phase 6 + `model_results_df` |

---

## What you should say in the presentation (one paragraph)

We treated Amazon sentiment as a **predictive** text classification problem, grounded the approach with a **DGP + EDA** story, built **interpretable baselines** (TF-IDF linear models), discovered **structure** with **unsupervised clustering** in two representations, then compared **deep learning** models while documenting **training behavior**, **leakage controls**, and a **deployment-oriented** recommendation using consistent metrics.

---

## File reference

- Notebook: [final_notebook.ipynb](final_notebook.ipynb)
- Rubric/spec: [final.md](final.md)
