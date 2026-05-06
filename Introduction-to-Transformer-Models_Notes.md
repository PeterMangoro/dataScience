# Introduction to Transformer Models — Study Notes

Notes derived from **Introduction-to-Transformer-Models.pdf**: a derivation-level overview for graduate data science students and technical interviews (Vaswani et al., “Attention Is All You Need”).

---

## Purpose and audience

- Explain each major Transformer component **with math**, under interview-style pressure.
- Framing: sequence modeling as estimating **\(P(y_t \mid x_1,\ldots,x_T)\)** with **long-range** dependencies.
- Classical models are too local; RNNs propagate state **sequentially** and suffer training pathologies.

---

## Why RNNs struggle (BPTT)

- Update: **\(h_t = f(h_{t-1}, x_t)\)**.
- Gradients w.r.t. early states multiply Jacobians along the chain → **vanishing** (Jacobian spectral radius below 1) or **exploding** (above 1); clipping only mitigates symptoms.
- Core issue: **information and gradients must traverse every step** — a structural bottleneck.

---

## Transformer core idea

- **Remove recurrence**; each position **directly** interacts with every other via **attention**.
- Effects: **full parallelism** in training, **\(O(1)\)** path length between any two positions (vs **\(O(T)\)** for RNNs), scales with data and compute.
- Lineage: **“Attention Is All You Need.”**

---

## Attention as a learned kernel

- Similarity **\(\mathrm{sim}(x_i, x_j) = q_i^\top k_j\)** parallels kernels **\(k(x_i,x_j)=\phi(x_i)^\top\phi(x_j)\)**.
- Output at each position is a **weighted average of value vectors**; weights from a **learned, data-dependent** similarity.
- **Multiple heads** ≈ **multiple kernels** / notions of similarity.

---

## Q, K, V and scaled dot-product attention

- From token matrix **\(X \in \mathbb{R}^{n \times d}\)**:

  **\(Q = XW_Q\)**, **\(K = XW_K\)**, **\(V = XW_V\)**

- Decouples **what you query for**, **what you advertise (key)**, and **what you retrieve (value)**.
- Full attention:

  **\(\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\dfrac{QK^\top}{\sqrt{d_k}}\right) V\)**

- Steps: scores **\(A = QK^\top\)**, row-wise softmax → weights **\(\alpha\)**, then **\(\alpha V\)**.

---

## Why scale by **\(\sqrt{d_k}\)** (variance argument)

- If entries of **\(q, k\)** are roughly i.i.d. with controlled variance, **\(\mathrm{Var}(q^\top k) \propto d_k\)** → standard deviation grows like **\(\sqrt{d_k}\)**.
- Large dot products → **saturated softmax** (nearly one-hot) → **small softmax Jacobian** → **weak gradients**.
- Dividing by **\(\sqrt{d_k}\)** restores a healthier scale → **stable training** (often likened to a **temperature** on logits).

---

## Softmax Jacobian

- **\(\dfrac{\partial s_i}{\partial z_j} = s_i(\delta_{ij} - s_j)\)** for **\(s = \mathrm{softmax}(z)\)**; matrix form **\(\mathrm{diag}(s) - ss^\top\)**.
- Nearly uniform **\(s\)** → richer gradient flow; nearly one-hot → Jacobian **≈ 0**; connects directly to **\(\sqrt{d_k}\)** scaling.

---

## Multi-head attention

- **\(\mathrm{head}_i = \mathrm{Attention}(QW_i^Q, KW_i^K, VW_i^V)\)**
- **\(\mathrm{MultiHead}(Q,K,V) = \mathrm{concat}(\mathrm{head}_1,\ldots,\mathrm{head}_h) W^O\)**
- Several **lower-dimensional subspaces** / relationship types (syntax, semantics, etc.); **\(W^O\)** **recombines** head outputs.

---

## Positional encoding (original sinusoidal)

- Self-attention alone is **permutation-equivariant** → explicit **position** information is required.
- **\(PE(\mathrm{pos},2i)=\sin(\mathrm{pos}/\omega_i)\)**, **\(PE(\mathrm{pos},2i+1)=\cos(\mathrm{pos}/\omega_i)\)**, **\(\omega_i = 10000^{2i/d}\)**.
- Motivations: **relative position** structure, **extrapolation** beyond max training length vs fixed learned indices, **bounded** \([-1,1]\) so PE does not overwhelm token embeddings.

---

## One Transformer block (encoder-style)

- **\(Z = \mathrm{LayerNorm}(X + \mathrm{Attention}(X))\)**
- **\(\mathrm{Output} = \mathrm{LayerNorm}(Z + \mathrm{FFN}(Z))\)**

(as in the deck’s post-residual LayerNorm presentation.)

- **Residuals**: shortcut paths for gradients; easier to learn **residuals around identity**.
- **LayerNorm**: normalize **per token across features**; learned scale and shift; stabilizes deep stacks.

---

## Position-wise FFN

- **\(\mathrm{FFN}(x)=\sigma(xW_1+b_1)W_2+b_2\)** with **\(d_{\mathrm{ff}}\)** often **\(\approx 4d\)**; same MLP at every position; **no** cross-token mixing in this sublayer alone.
- Role: **nonlinearity** after attention; without it, deep stacks would remain largely linear in retrieved values.
- Optional intuition: FFN as a form of **key–value memory** (first layer keys, second layer values).

---

## Complexity and tradeoffs

| | Self-attention | RNN (per step style) |
|--|----------------|----------------------|
| Compute (order) | **\(O(n^2 d)\)** dominant | **\(O(n d^2)\)** |
| Parallelism | High (matmuls) | Sequential along time |
| Path length | **\(O(1)\)** between positions | **\(O(n)\)** |

- Long **\(n\)** motivates **sparse attention**, **linear attention**, **state-space models** (e.g. Mamba).

---

## Memory and long context

- Standard attention materializes **\(QK^\top \in \mathbb{R}^{n\times n}\)** → **\(O(n^2)\)** memory; large **\(n\)** hits VRAM.
- **FlashAttention**: fused/tiled implementation; lower peak memory.
- **Sparse attention**: e.g. local window + global tokens → **\(O(n \cdot w)\)**-style scaling.

---

## Masking

- **Causal / autoregressive**: add **\(-\infty\)** to scores for **\(j > i\)** so future tokens get weight 0 after softmax.
- Ensures no **leakage** during teacher forcing; matches **inference**; underpins **KV caching**.
- **Encoder-only (BERT)**: bidirectional, no causal mask. **Decoder-only (GPT)**: causal. **Encoder–decoder**: both patterns as needed.

---

## Training objective (language modeling)

- **\(L = -\sum_{t=1}^{T} \log P(y_t \mid y_1,\ldots,y_{t-1};\theta)\)** — negative log-likelihood / cross-entropy to the data distribution.
- Training uses **masked** self-attention + **teacher forcing** to compute all steps in parallel.
- **Perplexity**: **\(\mathrm{PPL} = \exp(L/T)\)**.

---

## Why Transformers generalize (themes from the deck)

1. **Context-dependent** representations (vs static embeddings).
2. **Weight sharing** across positions (same **\(W_Q,W_K,W_V,W_O,W_1,W_2\)** everywhere).
3. **Flexible attention** — minimal hard structure; the model learns what to attend to.

---

## Interview quick reference

| Question | Short answer |
|----------|--------------|
| Why **\(\sqrt{d_k}\)**? | Dot products scale with **\(d_k\)**; scaling avoids saturated softmax and vanishing gradients through softmax. |
| Why multi-head? | Multiple learned subspaces/kernels; **\(W^O\)** mixes them. |
| Why positional encoding? | Attention is permutation-equivariant; order must be injected; sinusoids give relative structure and length extrapolation. |
| Complexity tradeoff? | Attention: **\(O(n^2 d)\)** compute, **\(O(n^2)\)** memory, parallel, short paths. RNN: **\(O(nd^2)\)**, sequential, long paths. Depends on **\(n\)**, **\(d\)**, and hardware. |

---

## Beyond the original Transformer

- **Sparse attention**: Longformer, BigBird.
- **Linear / subquadratic**: Performer, FNet, Mamba.
- **Mixture of experts**: Switch Transformer, Mixtral.
- **Multimodal**: CLIP, DALL·E, Flamingo, GPT-4V — same attention abstraction, different tokenizers/modalities.

---

*Source: Introduction-to-Transformer-Models.pdf (same directory as this repo’s root guides).*
