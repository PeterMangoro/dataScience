# Assignment 3 Report: Representation Learning and Sequence Modeling in Online Retail Behavior

## 1) Problem and Dataset

This case study analyzes customer behavior sequences from the Online Retail dataset (UCI/Kaggle). The objective is to:

1. Discover latent structure in customer behavior with unsupervised methods.
2. Build sequence-based neural predictive models.
3. Evaluate whether Transformer-style attention improves balanced predictive performance relative to a neural baseline.

The final supervised task predicts cluster-derived customer segment labels from transaction sequences.

## 2) Representation Design and Preprocessing

### Data construction

The pipeline follows a customer-level sequence design:

- Raw transactions are cleaned and standardized.
- Customer events are ordered by `CustomerID`, `InvoiceDate`, and `StockCode`.
- Per-customer sequences are built with:
  - tokenized `StockCode` events,
  - numeric context features (`Quantity`, `UnitPrice`, `LineAmount`),
  - sequence masks and truncation controls.

Missing values and numeric inconsistencies are handled through coercion, NaN handling, and median imputation in representation-building steps. Feature scaling is applied consistently where required by distance-based methods and downstream models.

### Representation families

Three customer-level representation families are used:

- `raw`: compact aggregate behavioral features (frequency, spend, recency, etc.).
- `sequence`: raw features plus additional sequence-statistics (dispersion and interpurchase dynamics).
- `latent`: standardized sequence-enriched inputs compressed with PCA, then augmented with lightweight clustering metadata.

PCA retained 7 components with high explained variance (~0.949), preserving most information while reducing redundancy.

## 3) Unsupervised Learning and Segment Structure

KMeans was evaluated across `k=2..8` on all three representations using silhouette, Calinski-Harabasz, Davies-Bouldin, and inertia.

### Key findings

- `sequence` and `raw` achieved very high silhouette at `k=2`, but segment-size diagnostics showed near-degenerate partitions (large majority cluster plus tiny outlier cluster), indicating outlier separation rather than rich segmentation.
- `latent` at `k=6` produced a more informative distribution with multiple meaningful clusters and small tail groups.

Latent `k=6` class shares used for downstream labeling were:

- class 1: 0.4643
- class 2: 0.3619
- class 0: 0.0839
- class 3: 0.0832
- class 5: 0.0062
- class 4: 0.0005

This confirms severe class imbalance, but also provides a materially richer segment taxonomy than binary outlier-vs-rest splits.

## 4) Neural Network Baseline (GRU) and Training Behavior

A GRU baseline was trained on customer-disjoint train/val/test splits using identical sequence-construction protocol as the transformer.

Split summary:

- Total customers: 4,338
- Train/Val/Test: 3,036 / 651 / 651 (~70/15/15)
- Vocabulary size: 3,600
- `max_len` (p95-capped): 120

Training curves showed typical overfitting after early epochs: training loss decreased steadily while validation loss and validation macro-F1 plateaued. This motivated macro-F1-centered checkpointing rather than accuracy-centered selection.

### Imbalance sensitivity check (weighted loss)

A class-weighted loss variant improved minority-sensitive performance:

- Baseline NN: accuracy 0.6851, macro-F1 0.2498
- Weighted NN: accuracy 0.5745, macro-F1 0.2944

Interpretation: weighting reduced majority-class dominance and improved balanced class performance at expected accuracy cost.

## 5) Transformer Modeling and Attention Behavior

A Transformer encoder model was trained on the same split and input protocol.

Observed training behavior:

- Training loss and train F1 improved strongly across epochs.
- Validation loss increased substantially after early epochs (overfitting signal).
- Validation macro-F1 peaked in the low 0.33 range (best around epoch 10), then stabilized/slightly degraded.

Attention heatmap diagnostics (sample-level, head-averaged) showed selective focus on subsets of sequence positions rather than uniform attention, suggesting non-local contextual aggregation. This supports interpretability at a qualitative level, with the caveat that single-sample/head-averaged maps are descriptive rather than causal explanations.

## 6) Comparative Analysis: GRU vs Transformer

Final model comparison:

- TransformerEncoder: accuracy 0.5960, macro-F1 0.2946, train time 9.65s, params 292,566
- NeuralBaseline (GRU): accuracy 0.6851, macro-F1 0.2498, train time 6.65s, params 288,326

### Tradeoff interpretation

- GRU is better on top-line accuracy and runtime.
- Transformer is better on balanced class performance (macro-F1), with similar model size.
- Because parameter counts are close, performance differences are primarily due to sequence modeling behavior, not capacity scaling.

### Class-wise evidence

Per-class diagnostics explain the macro-F1 gain:

- GRU collapses to zero F1 on classes 0, 3, 4, and 5.
- Transformer recovers non-zero minority-class signal for classes 0 and 3.
- Both models remain weak on extreme-tail classes 4 and 5 (very low support).

This is consistent with imbalanced segmentation labels and reinforces macro-F1 as the primary decision metric.

## 7) Conclusions

1. The representation layer is the primary driver of meaningful structure: latent `k=6` provides a better segmentation target than silhouette-only `k=2` solutions that collapse into outlier partitions.
2. Under imbalanced labels, accuracy alone is misleading; macro-F1 and class-wise metrics are necessary for valid comparison.
3. The GRU baseline is computationally cheaper and achieves higher accuracy.
4. The Transformer provides better balanced performance and improved minority-class sensitivity, at moderate runtime cost.

Decision depends on objective:

- If business objective prioritizes balanced segment recognition: prefer Transformer.
- If objective prioritizes top-line accuracy and faster training: prefer GRU baseline.

Future improvements should focus on target redesign and imbalance handling (especially for ultra-rare classes), plus data/label strategies that make tail classes learnable.

Limitations: class support is highly skewed in the latent `k=6` target, including ultra-rare classes (for example, supports near 1-4 samples in the test split), which makes stable learning and evaluation difficult regardless of architecture. Consequently, macro-F1 improvements should be interpreted as directional rather than definitive for tail classes, and future iterations should prioritize rebalanced labeling/data strategies before expecting large model-only gains.

