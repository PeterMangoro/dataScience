"""Online Retail segment-recovery tab (held-out customer + attention viewer).

Pick a real held-out customer; the trained Transformer predicts their latent
behavioral segment and we show which past products it attended to. Illustrative
segment recovery / attention diagnostics (macro-F1 ~0.26-0.30; tail segments are
rarely recovered), not a production classifier.
"""

import json
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from retail_model import NUM_COLS, TransformerClassifier, build_padded

MODELS_DIR = Path(__file__).resolve().parent / "models"


def _load_json(name):
    return json.loads((MODELS_DIR / name).read_text())


_config = _load_json("retail_config.json")
_vocab = _load_json("retail_vocab.json")
_scaler = _load_json("retail_scaler.json")
_stock_desc = _load_json("retail_stock_desc.json")
_personas = {int(k): v for k, v in _load_json("retail_personas.json").items()}
_samples = _load_json("retail_samples.json")

_idx_to_stock = {v: k for k, v in _vocab.items()}
_max_len = _config["max_len"]

_model = TransformerClassifier(
    _config["vocab_size"],
    _config["num_numeric"],
    _config["num_classes"],
    _config["max_len"],
    emb_dim=_config["emb_dim"],
    num_dim=_config["num_dim"],
    n_heads=_config["n_heads"],
    ff_dim=_config["ff_dim"],
    dropout=_config["dropout"],
)
_model.load_state_dict(torch.load(MODELS_DIR / "retail_transformer.pt", map_location="cpu"))
_model.eval()


def _sample_label(s):
    return (
        f"Customer {s['customer_id']} - {s['n_invoices']} invoices, "
        f"${s['total_spend']:,.0f} - true: {s['label_name']}"
    )


_CHOICES = [_sample_label(s) for s in _samples]
_LABEL_TO_SAMPLE = {_sample_label(s): s for s in _samples}


def _persona_md(y, prefix):
    p = _personas[int(y)]
    return f"**{prefix}: {p['name']}** ({p['share'] * 100:.1f}% of customers) - {p['description']}"


def analyze(choice):
    sample = _LABEL_TO_SAMPLE[choice]
    tokens, num_feats, mask, eff_len = build_padded(sample["token_ids"], sample["num_feats"], _max_len)

    with torch.no_grad():
        logits = _model(tokens, num_feats, mask)
    probs = F.softmax(logits, dim=-1)[0]
    label_scores = {_personas[i]["name"]: float(probs[i]) for i in range(_config["num_classes"])}

    y_pred = int(probs.argmax())
    y_true = int(sample["y"])
    verdict = "correct" if y_pred == y_true else "different from clustering label"
    summary = (
        f"{_persona_md(y_true, 'True segment')}\n\n"
        f"{_persona_md(y_pred, 'Predicted segment')}\n\n"
        f"Prediction is **{verdict}**. Sequence length: {eff_len} events. "
        f"Top products: {', '.join(sample['top_products'])}."
    )

    # Head-averaged attention from the query at the last real position.
    attn = _model.last_attention.mean(dim=1)[0].cpu().numpy()
    q = max(0, eff_len - 1)
    key_mass = attn[q, :eff_len]
    order = key_mass.argsort()[::-1][:8]

    rows = []
    for rank, j in enumerate(order, start=1):
        tid = int(tokens[0, j].item())
        code = _idx_to_stock.get(tid, "?")
        desc = _stock_desc.get(code, "")
        rows.append([
            rank,
            int(j),
            round(float(key_mass[j]), 4),
            code,
            desc[:40],
            round(float(num_feats[0, j, 0]), 2),
            round(float(num_feats[0, j, 1]), 2),
            round(float(num_feats[0, j, 2]), 2),
        ])

    block = attn[:eff_len, :eff_len]
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(block, aspect="auto", cmap="viridis")
    ax.set_title("Head-averaged attention")
    ax.set_xlabel("Key position (past event)")
    ax.set_ylabel("Query position")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    return label_scores, summary, rows, fig


def build_retail_tab() -> gr.Interface:
    headers = ["rank", "pos", "attn", "StockCode", "Description"] + [f"{c}_z" for c in NUM_COLS]
    default = _CHOICES[0] if _CHOICES else None
    return gr.Interface(
        fn=analyze,
        inputs=gr.Dropdown(choices=_CHOICES, value=default, label="Held-out customer"),
        outputs=[
            gr.Label(num_top_classes=4, label="Predicted segment (probabilities)"),
            gr.Markdown(label="Result"),
            gr.Dataframe(headers=headers, label="Top attended past events", wrap=True),
            gr.Plot(label="Attention heatmap"),
        ],
        title="Online Retail - Customer Segment Recovery",
        description=(
            "A Transformer reads a customer's time-ordered purchase sequence (product token + "
            "scaled Quantity/UnitPrice/LineAmount) and predicts their latent behavioral segment "
            "(k=6 from PCA + KMeans). Pick a held-out customer to see the prediction and which past "
            "products the attention focused on. Illustrative segment recovery + attention diagnostics "
            "(macro-F1 ~0.26-0.30; rare segments are seldom recovered)."
        ),
        flagging_mode="never",
    )
