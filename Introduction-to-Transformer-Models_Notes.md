# Introduction to Transformer Models — Study Notes

Notes derived from **Introduction-to-Transformer-Models.pdf**: a derivation-level overview for graduate data science students and technical interviews (Vaswani et al., “Attention Is All You Need”).

**How to read this file:** Ideas are written in **plain text**, **ASCII code fences** (triple backticks + `text`), and **tables**, so the file stays readable without inline math delimiters. Optional **display** math uses `\[ ... \]`; if your preview does not render it, skip those blocks — the bullets and tables repeat the same content.

---

## Purpose and audience

- Explain each major Transformer component **with math**, under interview-style pressure.
- Framing: sequence modeling as estimating **P(y_t | x_1, ..., x_T)** (next-token / sequence likelihood) with **long-range** dependencies.
- Classical models are too local; RNNs propagate state **sequentially** and suffer training pathologies.

---

## Why RNNs struggle (BPTT)

- Update (recurrence): `h_t = f(h_{t-1}, x_t)`.
- Gradients w.r.t. early states multiply Jacobians along the chain → **vanishing** (Jacobian spectral radius below 1) or **exploding** (above 1); clipping only mitigates symptoms.
- Core issue: **information and gradients must traverse every step** — a structural bottleneck.

---

## Transformer core idea

- **Remove recurrence**; each position **directly** interacts with every other via **attention**.
- Effects: **full parallelism** in training, **O(1)** path length between any two positions (vs **O(T)** for RNNs), scales with data and compute.
- Lineage: **“Attention Is All You Need.”**

---

## Attention as a learned kernel

**Idea in words:** Attention picks up how “similar” two positions are, then uses that similarity as **weights** when mixing **value** vectors. That is structurally like a **kernel** between two inputs: a single number that says how compatible they are, built from **inner products** of feature vectors.

**Classical kernel (fixed feature map φ):**

- You choose a map **φ** (e.g. polynomial or RBF features).  
- Similarity between tokens **i** and **j** is  
  **k(x_i, x_j) = φ(x_i)^T φ(x_j)**  
  (same as dot product in feature space).  
- **φ** is **hand-designed**; the kernel is **fixed** before training.

**Attention similarity (learned linear maps into “query” and “key” space):**

- Token **i** is turned into a **query vector** **q_i**; token **j** into a **key vector** **k_j** (via **W_Q**, **W_K** on embeddings).  
- Raw score (before softmax scaling) is  
  **sim(x_i, x_j) = q_i^T k_j**  
  (dot product in **d_k**-dimensional space).  
- **W_Q** and **W_K** are **learned**; the effective “φ” for queries and keys is **not fixed** — it is whatever the network needs for the task.

| | Classical kernel | Dot-product attention score |
|--|------------------|-----------------------------|
| Compare | **φ(x_i)^T φ(x_j)** | **q_i^T k_j** with **q_i = W_Q x_i**, **k_j = W_K x_j** (schematically; **x** is really the contextual hidden state at that position) |
| Who defines the geometry? | Human-chosen **φ** | Learned **W_Q**, **W_K** |
| Data dependence | Same **φ** for all data | **q**, **k** depend on the **current sequence** |

**Parse of the parallel:** Both formulas say “similarity = dot product of two vectors built from **x_i** and **x_j**.” Kernels use one shared **φ**; attention uses **two** learned maps (**W_Q** vs **W_K**), so **q_i** and **k_j** can live in **complementary** roles (“what I’m looking for” vs “what I advertise”). After scores, softmax turns those dots into **weights** on **v_j** — that mixing step is the extra piece beyond a bare kernel matrix.

**LaTeX (for editors that render math):**

\[
\mathrm{sim}(x_i, x_j) = q_i^\top k_j
\qquad\text{parallels}\qquad
k(x_i,x_j)=\phi(x_i)^\top\phi(x_j).
\]

- Output at each position is a **weighted average of value vectors**; weights come from this **learned, position-dependent** similarity (then softmax + scale).  
- **Multiple heads** ≈ **several different “kernels”** at once (several **(W_Q, W_K, W_V)** triples), then recombined.

---

## Q, K, V and scaled dot-product attention

### Where Q, K, V come from

Token matrix **X** has one row per token: shape **n × d** (length **n**, hidden size **d**).

```text
Q = X W_Q,   K = X W_K,   V = X W_V
```

**W_Q**, **W_K**, **W_V** each have shape **d × d_k** (often **d_k = d / h** per head with **h** heads). Each position **i** gets row vectors **q_i**, **k_i**, **v_i** of length **d_k** (rows of **Q**, **K**, **V**).

### Roles (intuition)

| Symbol | Name | Role in one attention step |
|--------|------|----------------------------|
| **q_i** | Query (token **i**) | “What am I looking for right now?” — the **active** vector that **scores** every position. |
| **k_j** | Key (token **j**) | “What do I **offer** / how do I **index** myself?” — **q_i** dot-products against this to score position **j**. |
| **v_j** | Value (token **j**) | “If you attend to me, **what content** do you pull in?” — the vector that gets **mixed** into the output. |

**Database analogy:** Keys are like **index fields** you match against; values are like **payload columns** returned when a row matches; the query is the **lookup string**. Unlike a fixed database, **W_Q**, **W_K**, **W_V** are **learned**, so the model discovers its own notions of “match” and “what to pass along.”

### Full matrix form

**Readable (works without math rendering):**

```text
Attention(Q, K, V) = softmax( (Q · K^T) / sqrt(d_k) ) · V
```

- **Q**, **K**, **V** each have shape **n × d_k** (one row per token).  
- **Q · K^T** has shape **n × n**; entry **(i, j)** is the dot product **q_i^T k_j**.  
- **softmax** is applied **independently to each row** of that score matrix (after dividing every entry by **sqrt(d_k)**).  
- That **n × n** weight matrix multiplies **V** (**n × d_k**) → output **n × d_k**.

| Step | Write it | Meaning |
|------|-----------|---------|
| **Scores** | **A_ij = q_i^T k_j** | How strongly position **i** “looks at” position **j** (dot of query **i** with key **j**). |
| **Weights** | Row **i** of **P** = **softmax( (row i of A) / sqrt(d_k) )** | Nonnegative entries, **sum to 1** over **j** for each fixed **i**. Write **P_ij = α_ij**. |
| **Output** | **o_i = Σ_j α_ij v_j** | Convex mix of the **n** value vectors **v_j**; equals **row i** of the matrix product **P V**. Same as row **i** of **softmax(Q K^T / sqrt(d_k)) · V**. |

**LaTeX (display — use if your preview supports `\[ ... \]`):**

\[
\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V.
\]

\[
A_{ij} = q_i^\top k_j.
\]

\[
\alpha_i = \mathrm{softmax}\left(\frac{A_{i,:}}{\sqrt{d_k}}\right),
\qquad
o_i = \sum_j \alpha_{ij}\, v_j.
\]

**Equivalently:** **o_i** is **row i** of **softmax(Q K^T / sqrt(d_k)) V** (with **Q K^T** scaled before softmax, then multiply by **V**).

So: **keys choose the distribution**; **values supply the vectors** being averaged.

### Toy example (one query position)

Sentence tokens: **The · cat · sat** (positions **1, 2, 3**). For the token **sat** (position **i = 3**), suppose **q_sat**, **k_the**, **k_cat**, **k_sat**, **v_the**, **v_cat**, **v_sat** are already computed from embeddings.

1. **Scores** (logits for query row **i = 3**): **s_j = q_sat^T k_j** for **j ∈ {1,2,3}**.

   Suppose **q_sat** aligns strongly with **k_cat** (subject–verb pattern) and weakly with **k_the**: illustrative **(s_1, s_2, s_3) = (0, 2, 1)**.

2. **Scale** by **sqrt(d_k)** (e.g. **d_k = 2** → divide by **sqrt(2)**): **(0, sqrt(2), sqrt(2)/2)** on that row.

3. **Softmax** on that row → weights **(α_31, α_32, α_33) ≈ (0.14, 0.58, 0.28)** — most mass on **cat**.

4. **Output** for **sat**: **o_sat = α_31 v_the + α_32 v_cat + α_33 v_sat**.

So **sat**’s update is dominated by **v_cat** here. In a real model, **q, k, v** are learned from data.

### What Q and K “share” vs V

- **q_i^T k_j** is a **scalar score** — only **Q** and **K** enter the softmax; they **control routing**.  
- **V** appears **after** routing — the **substance** mixed into each output position.

If you **swapped** roles so that values were used to score keys, you would change the **meaning** of the layer (it would no longer be standard dot-product attention).

### Shapes (sanity check)

- **Q, K, V:** each **n × d_k**.  
- **Q K^T:** **n × n** (all pairwise scores).  
- **Softmax** row-wise → still **n × n**.  
- Multiply by **V:** **n × d_k** → output **n × d_k** per head (then concat / **W^O** in multi-head).

### Example questions (self-check / interview)

1. Why three different projections **W_Q**, **W_K**, **W_V** instead of one?  
2. Which matrices set the **attention weights**, and which set **what information** is blended?  
3. What object has shape **n × n** — and what does entry **(i, j)** mean **before** vs **after** row-softmax?  
4. Why **Q K^T** (not **K Q^T**) in the usual self-attention write-up? (Query row **i** scores every key **j**.)  
5. If **d_k** doubled without rescaling **W_Q**, **W_K**, why might softmax saturate — and what divisor fixes that in the formula?  
6. After **softmax(Q K^T / sqrt(d_k)) V**, write output row **i** as a literal sum over **j**.

### Worked example: explicit **X**, **W**, **Q, K, V**, and full numeric parse

**LaTeX tip:** Some `\[ ... \]` blocks below still use `bmatrix` (each row must end with **`\\`** in the source). If they collapse when copied, use the **plain tables** for **X / Q / V / Q K^T**, and the **` ```text` ** blocks for **S** and **O** (scaled scores and final output).

**Setup:** **n = 3** tokens (positions **1, 2, 3**), embedding **d = 2**, head dim **d_k = 2**, so each **W** is **2 × 2**.

**Token embedding matrix** **X** (shape **3 × 2**):

\[
X =
\begin{bmatrix}
1 & 0 \\
0 & 1 \\
1 & 1
\end{bmatrix}
\quad\text{(rows: position 1, 2, 3).}
\]

**Projection matrices** (chosen for pencil-and-paper clarity — not from training):

\[
W_Q = I_2 =
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix},
\quad
W_K = I_2,
\quad
W_V =
\begin{bmatrix}
0 & 1 \\
1 & 0
\end{bmatrix}
\quad\text{(swap value channels vs. embedding).}
\]

**Step 1 — Build Q, K, V:**

\[
Q = X W_Q =
\begin{bmatrix}
1 & 0 \\
0 & 1 \\
1 & 1
\end{bmatrix},\quad
K = X W_K = Q,
\quad
V = X W_V =
\begin{bmatrix}
0 & 1 \\
1 & 0 \\
1 & 1
\end{bmatrix}.
\]

**Plain tables** (same numbers; use if `bmatrix` / `\\` does not render):

**Q = K** (rows = positions 1…3, cols = dim 1…2):

| pos | col 1 | col 2 |
|-----|-------|-------|
| 1 | 1 | 0 |
| 2 | 0 | 1 |
| 3 | 1 | 1 |

**V = X W_V:**

| pos | col 1 | col 2 |
|-----|-------|-------|
| 1 | 0 | 1 |
| 2 | 1 | 0 |
| 3 | 1 | 1 |

Row-wise: **q_1 = k_1 = [1, 0]**, **q_2 = k_2 = [0, 1]**, **q_3 = k_3 = [1, 1]**; values **v_1 = [0, 1]**, **v_2 = [1, 0]**, **v_3 = [1, 1]**.

**Step 2 — Score matrix** **A = Q K^T** (entry **A_ij = q_i^T k_j**):

\[
Q K^\top =
\begin{bmatrix}
1 & 1 & 0 \\
0 & 0 & 1 \\
1 & 1 & 1
\end{bmatrix}.
\]

**Same matrix in plain form** (each row **i** = scores to keys **j = 1, 2, 3**):

|        | key 1 | key 2 | key 3 |
|--------|-------|-------|-------|
| query row 1 | 1 | 1 | 0 |
| query row 2 | 0 | 0 | 1 |
| query row 3 | 1 | 1 | 1 |

**Interpretation (parse dot products row by row):**

- **Row 1**, **q_1 = [1, 0]**: dot products with keys **(1, 1, 0)** → strong attention to positions **1** and **2**, none to **3**.
- **Row 2**, **q_2 = [0, 1]**: **(0, 0, 1)** → only position **3** matches.
- **Row 3**, **q_3 = [1, 1]**: **(1, 1, 1)** → ties across all keys before softmax.

**Step 3 — Scale:** **sqrt(d_k) = sqrt(2) ≈ 1.4142**. Scaled logits **S = (Q K^T) / sqrt(d_k)** (divide **every** entry of the score matrix by **sqrt(d_k)**).

**S in monospace** (no LaTeX / no `\\` needed):

```text
S ≈
[ 0.7071   0.7071   0      ]
[ 0        0        0.7071 ]
[ 0.7071   0.7071   0.7071 ]
```

**S (plain table):**

| row | col1 | col2 | col3 |
|-----|------|------|------|
| 1 | 0.7071 | 0.7071 | 0 |
| 2 | 0 | 0 | 0.7071 |
| 3 | 0.7071 | 0.7071 | 0.7071 |

**Step 4 — Row-wise softmax** **P = softmax_row(S)**. Entry **P_ij = exp(S_ij) / sum_ℓ exp(S_iℓ)**:

- **Row 1:** exp(0.7071) ≈ 2.028 twice, exp(0) = 1 → sum ≈ 5.056 → **P_1: ≈ [0.401, 0.401, 0.198]**.
- **Row 2:** logits **[0, 0, 0.7071]** → sum ≈ 4.028 → **P_2: ≈ [0.248, 0.248, 0.504]**.
- **Row 3:** three equal logits → **P_3: = [1/3, 1/3, 1/3]**.

**Step 5 — Output** **O = P V** (each row is a weighted sum of **v_1, v_2, v_3**).

**O in monospace** (rounded):

```text
O ≈
[ 0.599   0.599 ]
[ 0.752   0.752 ]
[ 0.667   0.667 ]
```

**O (plain table):**

| row | col1 | col2 |
|-----|------|------|
| 1 | 0.599 | 0.599 |
| 2 | 0.752 | 0.752 |
| 3 | 0.667 | 0.667 |

**Parse one row in full (position 2):**

\[
o_2 = 0.248\, v_1 + 0.248\, v_2 + 0.504\, v_3
= 0.248\,[0,1] + 0.248\,[1,0] + 0.504\,[1,1]
\approx [0.752,\; 0.752].
\]

So **keys** (via **Q K^T**) put **~50%** of mass on position **3** in row 2; **values** supplied the vectors that were mixed. **Forward parse:** embed → linear maps → dot scores → scale → softmax → convex mix of values.

---

## Why scale by **sqrt(d_k)** (variance argument)

- If entries of **q**, **k** are roughly i.i.d. with bounded variance, **Var(q^T k) ∝ d_k** → std grows like **sqrt(d_k)**.
- Large dot products → **saturated softmax** (nearly one-hot) → **small softmax Jacobian** → **weak gradients**.
- Dividing scores by **sqrt(d_k)** restores a healthier scale → **stable training** (like a **temperature** on logits).

**LaTeX (variance / scaling):**

\[
\mathrm{Var}(q^\top k) \propto d_k.
\]

---

## Softmax Jacobian

- For **s = softmax(z)**: partial derivative **∂s_i / ∂z_j = s_i (δ_ij − s_j)**; Jacobian matrix **diag(s) − s s^T**.
- Nearly uniform **s** → richer gradient flow; nearly one-hot → Jacobian **≈ 0**; ties to **sqrt(d_k)** scaling in attention.

---

## Multi-head attention

```text
head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
MultiHead(Q,K,V) = concat(head_1, …, head_h) W^O
```

Several **lower-dimensional subspaces** / relationship types (syntax, semantics, etc.); **W^O** **recombines** head outputs.

---

## Positional encoding (original sinusoidal)

- Self-attention alone is **permutation-equivariant** → explicit **position** information is required.
- For dimension index **i** and position **pos**:  
  **PE(pos, 2i) = sin(pos / ω_i)**, **PE(pos, 2i+1) = cos(pos / ω_i)** with **ω_i = 10000^(2i/d)**.
- Motivations: **relative position** structure, **extrapolation** beyond max training length vs fixed learned indices, outputs **bounded in [-1, 1]** so PE does not overwhelm token embeddings.

**Same formulas in a monospace block** (copies cleanly; no math renderer needed):

```text
PE(pos, 2i)   = sin(pos / omega_i)
PE(pos, 2i+1) = cos(pos / omega_i)

omega_i = 10000^((2*i)/d)

pos = token index (0, 1, 2, ...).
i   = dimension index along model width d (paired sin/cos channels).
d   = model hidden size (embedding dimension).
```

---

## One Transformer block (encoder-style)

```text
Z = LayerNorm( X + Attention(X) )
Output = LayerNorm( Z + FFN(Z) )
```

(as in the deck’s post-residual LayerNorm presentation.)

- **Residuals**: shortcut paths for gradients; easier to learn **residuals around identity**.
- **LayerNorm**: normalize **per token across features**; learned scale and shift; stabilizes deep stacks.

---

## Position-wise FFN

**Monospace (no renderer):**

```text
FFN(x) = sigma( x * W_1 + b_1 ) * W_2 + b_2
```

Here `*` is matrix multiply along the feature dimension; **x** is one token’s vector (length **d**). **W_1** maps **d → d_ff**, **W_2** maps **d_ff → d**; **sigma** is a nonlinearity (e.g. ReLU / GELU in modern models).

**d_ff** is often **≈ 4d**; same MLP at every position; **no** cross-token mixing in this sublayer alone.

- Role: **nonlinearity** after attention; without it, deep stacks would remain largely linear in retrieved values.
- Optional intuition: FFN as a form of **key–value memory** (first layer keys, second layer values).

## Complexity and tradeoffs

| | Self-attention | RNN (per step style) |
|--|----------------|----------------------|
| Compute (order) | **O(n² d)** dominant | **O(n d²)** |
| Parallelism | High (matmuls) | Sequential along time |
| Path length | **O(1)** between positions | **O(n)** |

- Long **n** motivates **sparse attention**, **linear attention**, **state-space models** (e.g. Mamba).

---

## Memory and long context

- Standard attention materializes **Q K^T** (shape **n × n**) → **O(n²)** memory; large **n** hits VRAM.
- **FlashAttention**: fused/tiled implementation; lower peak memory.
- **Sparse attention**: e.g. local window + global tokens → **O(n · w)**-style scaling for window **w**.

---

## Masking

- **Causal / autoregressive:** add a large negative (conceptually **−∞**) to scores where **j > i** so future keys get weight **0** after softmax.
- Ensures no **leakage** during teacher forcing; matches **inference**; underpins **KV caching**.
- **Encoder-only (BERT):** bidirectional, no causal mask. **Decoder-only (GPT):** causal. **Encoder–decoder:** both patterns as needed.

---

## Training objective (language modeling)

```text
L = - sum_{t=1..T} log P(y_t | y_1, ..., y_{t-1}; theta)
```

Negative log-likelihood / cross-entropy to the data distribution. Training uses **masked** self-attention + **teacher forcing** to compute steps in parallel.

- **Perplexity:** **PPL = exp(L / T)**.

---

## Why Transformers generalize (themes from the deck)

1. **Context-dependent** representations (vs static embeddings).
2. **Weight sharing** across positions (same **W_Q, W_K, W_V, W_O, W_1, W_2** everywhere).
3. **Flexible attention** — minimal hard structure; the model learns what to attend to.

---

## Interview quick reference

| Question | Short answer |
|----------|--------------|
| Why **sqrt(d_k)**? | Dot products grow with **d_k**; scaling avoids saturated softmax and tiny gradients through softmax. |
| Why multi-head? | Multiple learned subspaces / kernels; **W^O** mixes them. |
| Why positional encoding? | Attention is permutation-equivariant; order must be injected; sinusoids encode relative structure and extrapolate in length. |
| Complexity tradeoff? | Attention: **O(n² d)** compute, **O(n²)** memory for the score matrix, parallel, short paths. RNN: **O(n d²)**, sequential, long paths. Depends on **n**, **d**, and hardware. |

---

## Beyond the original Transformer

- **Sparse attention**: Longformer, BigBird.
- **Linear / subquadratic**: Performer, FNet, Mamba.
- **Mixture of experts**: Switch Transformer, Mixtral.
- **Multimodal**: CLIP, DALL·E, Flamingo, GPT-4V — same attention abstraction, different tokenizers/modalities.

---

*Source: Introduction-to-Transformer-Models.pdf (same directory as this repo’s root guides).*
