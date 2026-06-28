"""Amazon review sentiment tab.

Reuses the TF-IDF + linear (SGD log-loss) pipeline exported from the course
final notebook (final/app/train_export.py), copied into models/.
"""

from pathlib import Path

import gradio as gr
import joblib

MODEL_PATH = Path(__file__).resolve().parent / "models" / "sentiment_model.joblib"
LABELS = {0: "Negative", 1: "Positive"}

_model = joblib.load(MODEL_PATH)


def classify(text: str):
    text = (text or "").strip()
    if not text:
        return {}
    proba = _model.predict_proba([text])[0]
    return {LABELS[0]: float(proba[0]), LABELS[1]: float(proba[1])}


def build_sentiment_tab() -> gr.Interface:
    return gr.Interface(
        fn=classify,
        inputs=gr.Textbox(
            lines=6,
            label="Amazon review text",
            placeholder="Paste a product review...",
        ),
        outputs=gr.Label(num_top_classes=2, label="Predicted sentiment"),
        title="Amazon Review Sentiment",
        description=(
            "TF-IDF + linear (SGD log-loss) classifier. "
            "Predicts positive vs negative sentiment from review text. Validation F1 ~0.90."
        ),
        examples=[
            ["This product exceeded my expectations, works perfectly and arrived early."],
            ["Broke after two days. Complete waste of money, do not buy."],
            ["It's okay - does the job but the build quality feels cheap."],
        ],
        flagging_mode="never",
    )
