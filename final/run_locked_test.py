"""Run Section 7.3 locked evaluation; print JSON metrics for notebook update."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

DATA_DIR = Path("data")
TRAIN_PATH = DATA_DIR / "train.ft.txt"
TEST_PATH = DATA_DIR / "test.ft.txt"
RANDOM_SEED = 42
BASELINE_TRAIN_MAX_ROWS = 500_000

random.seed(RANDOM_SEED)


def parse_fasttext_line(line: str):
    raw = line.rstrip("\n")
    if not raw.startswith("__label__"):
        return None, None, "", True
    parts = raw.split(" ", 1)
    if len(parts) == 1:
        label_raw, text = parts[0], ""
    else:
        label_raw, text = parts[0], parts[1]
    label_map = {"__label__1": 0, "__label__2": 1}
    label = label_map.get(label_raw)
    return label_raw, label, text, label is None


def load_fasttext_as_dataframe(path: Path, max_rows: int | None = None):
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, start=1):
            _, label, text, malformed = parse_fasttext_line(line)
            if malformed:
                continue
            rows.append({"label": label, "text": text})
            if max_rows is not None and i >= max_rows:
                break
    return pd.DataFrame(rows)


def evaluate_binary_classifier(model, X_val, y_val):
    y_pred = model.predict(X_val)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_val)[:, 1]
    elif hasattr(model, "decision_function"):
        raw = model.decision_function(X_val)
        y_score = 1 / (1 + np.exp(-raw))
    else:
        y_score = y_pred
    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "precision": float(precision_score(y_val, y_pred, zero_division=0)),
        "recall": float(recall_score(y_val, y_pred, zero_division=0)),
        "f1": float(f1_score(y_val, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_val, y_score)),
        "pr_auc": float(average_precision_score(y_val, y_score)),
    }
    return metrics


def make_tfidf_sgd_pipeline():
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=5,
                    max_df=0.95,
                    sublinear_tf=True,
                    max_features=300_000,
                ),
            ),
            (
                "clf",
                SGDClassifier(
                    loss="hinge",
                    alpha=1e-5,
                    max_iter=25,
                    tol=1e-3,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def main():
    assert TRAIN_PATH.exists(), TRAIN_PATH
    assert TEST_PATH.exists(), TEST_PATH

    print("Loading train...", flush=True)
    final_train_df = load_fasttext_as_dataframe(TRAIN_PATH, max_rows=BASELINE_TRAIN_MAX_ROWS)
    print("Loading test...", flush=True)
    test_df = load_fasttext_as_dataframe(TEST_PATH)

    print(f"Train rows: {len(final_train_df):,} | Test rows: {len(test_df):,}", flush=True)
    print("Fitting pipeline...", flush=True)
    locked_pipe = make_tfidf_sgd_pipeline()
    locked_pipe.fit(final_train_df["text"], final_train_df["label"])

    print("Evaluating on test...", flush=True)
    metrics = evaluate_binary_classifier(locked_pipe, test_df["text"], test_df["label"])
    metrics["train_rows_for_refit"] = len(final_train_df)
    metrics["test_rows"] = len(test_df)

    dev_leader = {
        "stage": "development_validation",
        "model": "TF-IDF + SGDClassifier(hinge)",
        "f1": 0.9070,
        "roc_auc": 0.9659,
        "eval_split": "val_100k (500k cap)",
    }
    locked_summary = {
        "stage": "locked_test",
        "model": "TF-IDF + SGDClassifier(hinge)",
        "f1": round(metrics["f1"], 4),
        "roc_auc": round(metrics["roc_auc"], 4),
        "eval_split": "test.ft.txt (held-out)",
    }

    out = {
        "locked_test_metrics": {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
        "final_metrics_summary": [dev_leader, locked_summary],
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
