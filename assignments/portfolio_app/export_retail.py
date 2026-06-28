"""Train and export the Online Retail Transformer segment classifier.

Standalone, deterministic reimplementation of the relevant path in
assignment3/assignment3.py: clean -> customer features -> latent PCA/KMeans
segments (forced k=6) -> customer-disjoint split -> train-only vocab/scaler ->
padded sequences -> Transformer training. Saves all serving artifacts to
models/.

Run:
    python assignments/portfolio_app/export_retail.py
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

from retail_model import NUM_COLS, PAD_TOKEN, UNK_TOKEN, TransformerClassifier, build_padded

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR.parent / "assignment3" / "data.csv"
MODELS_DIR = APP_DIR / "models"

RANDOM_STATE = 42
FORCE_K = 6
EPOCHS = 12
BATCH_SIZE = 64
N_SAMPLES_PER_CLASS = 3

RAW_COLS = [
    "n_invoices",
    "total_spend",
    "avg_invoice_value",
    "recency_days",
    "mean_interpurchase_days",
    "item_diversity_ratio",
]

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
device = torch.device("cpu")


# --------------------------------------------------------------------------- #
# 1. Load + clean (ports assignment3.py lines ~315-460)
# --------------------------------------------------------------------------- #
def load_and_clean():
    try:
        df_raw = pd.read_csv(DATA_PATH, encoding="utf-8")
    except UnicodeDecodeError:
        df_raw = pd.read_csv(DATA_PATH, encoding="latin1")

    df = df_raw.copy()
    for col in ["InvoiceNo", "StockCode", "Country", "Description"]:
        df[col] = df[col].astype("string").str.strip()
    df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce").astype("Int64")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce", utc=True).dt.tz_localize(None)

    df["is_cancel_invoice"] = df["InvoiceNo"].str.startswith("C", na=False)

    df = df[df["CustomerID"].notna()].copy()
    canonical_cols = [
        "InvoiceNo", "StockCode", "Description", "Quantity",
        "InvoiceDate", "UnitPrice", "CustomerID", "Country",
    ]
    df = df.drop_duplicates(subset=canonical_cols).copy()

    transactions_clean = df.copy()
    transactions_clean["LineAmount"] = (
        transactions_clean["Quantity"].astype(float) * transactions_clean["UnitPrice"].astype(float)
    )

    transactions_purchase = transactions_clean[
        (~transactions_clean["is_cancel_invoice"])
        & (transactions_clean["Quantity"] > 0)
        & (transactions_clean["UnitPrice"] > 0)
    ].copy()

    return transactions_clean, transactions_purchase


# --------------------------------------------------------------------------- #
# 2. Customer features (ports lines ~485-555, 870-920)
# --------------------------------------------------------------------------- #
def build_customer_features(transactions_purchase):
    customer_events = transactions_purchase[
        ["CustomerID", "InvoiceDate", "InvoiceNo", "StockCode", "Quantity", "UnitPrice", "LineAmount", "Country"]
    ].copy()
    customer_events["invoice_day"] = customer_events["InvoiceDate"].dt.floor("D")

    invoice_level = (
        customer_events.groupby(["CustomerID", "InvoiceNo", "invoice_day"], as_index=False)
        .agg(invoice_value=("LineAmount", "sum"))
        .sort_values(["CustomerID", "invoice_day", "InvoiceNo"])
    )
    invoice_level["prev_day"] = invoice_level.groupby("CustomerID")["invoice_day"].shift(1)
    invoice_level["interpurchase_days"] = (invoice_level["invoice_day"] - invoice_level["prev_day"]).dt.days

    invoice_stats = invoice_level.groupby("CustomerID", as_index=False).agg(
        n_invoices=("InvoiceNo", "nunique"),
        avg_invoice_value=("invoice_value", "mean"),
        mean_interpurchase_days=("interpurchase_days", "mean"),
    )

    customer_base = customer_events.groupby("CustomerID", as_index=False).agg(
        n_line_items=("StockCode", "count"),
        n_unique_items=("StockCode", "nunique"),
        total_spend=("LineAmount", "sum"),
        first_purchase=("InvoiceDate", "min"),
        last_purchase=("InvoiceDate", "max"),
    )
    customer_features_base = customer_base.merge(invoice_stats, on="CustomerID", how="left")

    max_date = customer_events["InvoiceDate"].max()
    customer_features_base["recency_days"] = (max_date - customer_features_base["last_purchase"]).dt.days
    customer_features_base["item_diversity_ratio"] = (
        customer_features_base["n_unique_items"] / customer_features_base["avg_invoice_value"].clip(lower=1)
    )

    feature_cols = ["CustomerID"] + RAW_COLS
    customer_features_base = customer_features_base[feature_cols].copy()
    customer_features_base[RAW_COLS] = customer_features_base[RAW_COLS].apply(pd.to_numeric, errors="coerce")
    customer_features_base[RAW_COLS] = customer_features_base[RAW_COLS].fillna(
        customer_features_base[RAW_COLS].median(numeric_only=True)
    )

    # Sequence-stat representation
    seq_extra = invoice_level.groupby("CustomerID", as_index=False).agg(
        invoice_value_std=("invoice_value", "std"),
        invoice_value_median=("invoice_value", "median"),
        interpurchase_days_std=("interpurchase_days", "std"),
        interpurchase_days_median=("interpurchase_days", "median"),
    )
    seq_extra["invoice_value_std"] = seq_extra["invoice_value_std"].fillna(0)
    seq_extra["interpurchase_days_std"] = seq_extra["interpurchase_days_std"].fillna(0)
    seq_extra = seq_extra.merge(
        invoice_level.groupby("CustomerID", as_index=False).agg(invoice_value_mean=("invoice_value", "mean")),
        on="CustomerID",
        how="left",
    )
    seq_extra["invoice_value_cv"] = np.where(
        seq_extra["invoice_value_mean"].abs() > 1e-9,
        seq_extra["invoice_value_std"] / seq_extra["invoice_value_mean"].abs(),
        0.0,
    )
    seq_extra = seq_extra.drop(columns=["invoice_value_mean"])

    X_seq = customer_features_base[["CustomerID"] + RAW_COLS].merge(seq_extra, on="CustomerID", how="left")
    seq_cols = [c for c in X_seq.columns if c != "CustomerID"]
    X_seq[seq_cols] = X_seq[seq_cols].apply(pd.to_numeric, errors="coerce")
    X_seq[seq_cols] = X_seq[seq_cols].fillna(X_seq[seq_cols].median(numeric_only=True))

    return customer_events, customer_features_base, X_seq


# --------------------------------------------------------------------------- #
# 3. Latent segments, forced k=6 (ports lines ~899-920, 960-1067)
# --------------------------------------------------------------------------- #
def build_segment_labels(X_seq):
    latent_input_cols = [c for c in X_seq.columns if c != "CustomerID"]
    latent_input = X_seq[latent_input_cols].copy()

    latent_scaled = StandardScaler().fit_transform(latent_input)
    pca = PCA(n_components=0.90, random_state=RANDOM_STATE)
    latent_array = pca.fit_transform(latent_scaled)
    pca_cols = [f"pca_{i + 1}" for i in range(latent_array.shape[1])]

    X_latent = pd.DataFrame(latent_array, columns=pca_cols, index=X_seq.index)
    meta_km = KMeans(n_clusters=5, n_init=10, random_state=RANDOM_STATE)
    X_latent["cluster_id"] = meta_km.fit_predict(latent_array)
    X_latent["cluster_dist"] = meta_km.transform(latent_array).min(axis=1)

    # Final supervised labels: re-standardize full latent matrix, KMeans(k=6).
    X_lat = X_latent.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X_lat = X_lat.fillna(X_lat.median(numeric_only=True))
    scaled = StandardScaler().fit_transform(X_lat)
    km = KMeans(n_clusters=FORCE_K, n_init=20, random_state=RANDOM_STATE)
    labels = km.fit_predict(scaled)

    print(f"PCA components: {len(pca_cols)} (var {pca.explained_variance_ratio_.sum():.3f})")
    return labels


# --------------------------------------------------------------------------- #
# 4. Sequences (ports lines ~1384-1469)
# --------------------------------------------------------------------------- #
def build_sequences(customer_events, customer_target_df, train_ids):
    ev = customer_events[["CustomerID", "InvoiceDate", "StockCode", "Quantity", "UnitPrice", "LineAmount"]].copy()
    ev = ev[ev["CustomerID"].isin(customer_target_df["CustomerID"])].copy()
    ev = ev.sort_values(["CustomerID", "InvoiceDate", "StockCode"]).reset_index(drop=True)

    train_codes = ev[ev["CustomerID"].isin(train_ids)]["StockCode"].astype(str)
    unique_codes = sorted(train_codes.unique().tolist())
    stockcode_to_idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for idx, code in enumerate(unique_codes, start=2):
        stockcode_to_idx[code] = idx

    train_num = ev[ev["CustomerID"].isin(train_ids)][NUM_COLS].astype(float)
    num_mean = train_num.mean()
    num_std = train_num.std().replace(0, 1.0)

    records = []
    for cust_id, frame in ev.groupby("CustomerID", sort=False):
        token_ids = [
            stockcode_to_idx.get(str(s), stockcode_to_idx[UNK_TOKEN])
            for s in frame["StockCode"].astype(str)
        ]
        nums = ((frame[NUM_COLS].astype(float) - num_mean) / num_std).values.astype(np.float32)
        records.append({
            "CustomerID": cust_id,
            "token_ids": token_ids,
            "num_feats": nums,
            "seq_len": len(token_ids),
        })

    seq_df = pd.DataFrame(records).merge(
        customer_target_df[["CustomerID", "y"]], on="CustomerID", how="inner"
    )
    max_len = int(np.ceil(seq_df["seq_len"].quantile(0.95)))
    max_len = max(20, min(max_len, 120))

    return ev, seq_df, stockcode_to_idx, num_mean, num_std, max_len


class CustomerSeqDataset(Dataset):
    def __init__(self, frame, max_len):
        self.frame = frame.reset_index(drop=True)
        self.max_len = max_len

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        tokens, nums, mask, _ = build_padded(row["token_ids"], row["num_feats"], self.max_len)
        return {
            "tokens": tokens.squeeze(0),
            "num_feats": nums.squeeze(0),
            "mask": mask.squeeze(0),
            "label": torch.tensor(int(row["y"]), dtype=torch.long),
        }


def run_epoch(model, loader, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    loss_fn = nn.CrossEntropyLoss()
    total_loss, all_preds, all_true = 0.0, [], []
    for batch in loader:
        tokens = batch["tokens"].to(device)
        num_feats = batch["num_feats"].to(device)
        mask = batch["mask"].to(device)
        y = batch["label"].to(device)
        if train_mode:
            optimizer.zero_grad()
        logits = model(tokens, num_feats, mask)
        loss = loss_fn(logits, y)
        if train_mode:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * y.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_true.extend(y.detach().cpu().numpy().tolist())
    return {
        "loss": total_loss / len(loader.dataset),
        "acc": accuracy_score(all_true, all_preds),
        "f1": f1_score(all_true, all_preds, average="macro"),
    }


# --------------------------------------------------------------------------- #
# 5. Persona naming from segment behavioral medians
# --------------------------------------------------------------------------- #
def name_personas(customer_features_base, customer_target_df):
    joined = customer_features_base.merge(customer_target_df[["CustomerID", "y"]], on="CustomerID")
    med = joined.groupby("y")[RAW_COLS].median()
    shares = joined["y"].value_counts(normalize=True)

    personas = {}
    for y_val in sorted(med.index):
        row = med.loc[y_val]
        share = float(shares.get(y_val, 0.0))
        freq = row["n_invoices"]
        spend = row["total_spend"]
        rec = row["recency_days"]
        avg_inv = row["avg_invoice_value"]

        # Behavioral, absolute-threshold naming (robust to only 6 medians).
        if avg_inv >= 10_000:
            name = "Wholesale / bulk buyers"
        elif freq >= 20 and spend >= 10_000:
            name = "VIP loyal big spenders"
        elif freq >= 5:
            name = "Frequent regulars"
        elif rec >= 90:
            name = "Lapsed shoppers"
        elif freq <= 2:
            name = "One-off / low-engagement buyers"
        else:
            name = "Occasional buyers"

        desc = (
            f"~{row['n_invoices']:.0f} invoices, ${row['total_spend']:,.0f} spend, "
            f"{row['recency_days']:.0f}d recency, "
            f"{row['mean_interpurchase_days']:.0f}d between orders."
        )
        personas[int(y_val)] = {
            "name": name,
            "description": desc,
            "share": round(share, 4),
            "profile": {c: round(float(row[c]), 3) for c in RAW_COLS},
        }

    # Disambiguate duplicate names with a numeric suffix.
    seen = {}
    for y_val, info in personas.items():
        base = info["name"]
        if base in seen:
            seen[base] += 1
            info["name"] = f"{base} ({seen[base]})"
        else:
            seen[base] = 1
    return personas


def pick_samples(seq_df, test_ids, customer_features_base, ev, stock_desc, model, max_len, personas):
    test_frame = seq_df[seq_df["CustomerID"].isin(test_ids)].reset_index(drop=True)
    cf = customer_features_base.set_index("CustomerID")

    model.eval()
    preds = {}
    with torch.no_grad():
        for _, row in test_frame.iterrows():
            tokens, nums, mask, _ = build_padded(row["token_ids"], row["num_feats"], max_len)
            logits = model(tokens.to(device), nums.to(device), mask.to(device))
            preds[int(row["CustomerID"])] = int(logits.argmax(dim=-1).item())

    samples = []
    for y_val in sorted(test_frame["y"].unique()):
        members = test_frame[test_frame["y"] == y_val]
        correct = [c for c in members["CustomerID"] if preds.get(int(c)) == y_val]
        ordered = correct + [c for c in members["CustomerID"] if int(c) not in {int(x) for x in correct}]
        for cust in ordered[:N_SAMPLES_PER_CLASS]:
            cust = int(cust)
            row = members[members["CustomerID"] == cust].iloc[0]
            cust_ev = ev[ev["CustomerID"] == cust]
            top_codes = [c for c, _ in Counter(cust_ev["StockCode"].astype(str)).most_common(3)]
            top_products = [stock_desc.get(c, c) for c in top_codes]
            samples.append({
                "customer_id": cust,
                "y": int(y_val),
                "label_name": personas[int(y_val)]["name"],
                "predicted_correct": bool(preds.get(cust) == y_val),
                "seq_len": int(row["seq_len"]),
                "n_invoices": int(cf.loc[cust, "n_invoices"]),
                "total_spend": float(cf.loc[cust, "total_spend"]),
                "top_products": top_products,
                "token_ids": [int(t) for t in row["token_ids"][:max_len]],
                "num_feats": np.asarray(row["num_feats"][:max_len], dtype=float).round(4).tolist(),
            })
    return samples


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading and cleaning transactions...")
    transactions_clean, transactions_purchase = load_and_clean()

    print("Building customer features...")
    customer_events, customer_features_base, X_seq = build_customer_features(transactions_purchase)

    print("Discovering latent segments (k=6)...")
    labels = build_segment_labels(X_seq)
    customer_target_df = pd.DataFrame({
        "CustomerID": customer_features_base["CustomerID"].values,
        "segment_label": labels,
    })
    customer_target_df["y"] = LabelEncoder().fit_transform(customer_target_df["segment_label"])
    num_classes = int(customer_target_df["y"].nunique())
    print("Class shares:", customer_target_df["y"].value_counts(normalize=True).sort_index().round(4).to_dict())

    # Customer-disjoint split (stratify with fallback).
    all_ids = customer_target_df["CustomerID"]
    all_y = customer_target_df["y"]
    strat = all_y if all_y.value_counts().min() >= 2 else None
    train_ids, temp_ids = train_test_split(all_ids, test_size=0.30, random_state=RANDOM_STATE, stratify=strat)
    temp_y = customer_target_df.set_index("CustomerID").loc[temp_ids, "y"]
    strat2 = temp_y if (temp_y.nunique() > 1 and temp_y.value_counts().min() >= 2) else None
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.50, random_state=RANDOM_STATE, stratify=strat2)
    train_ids, val_ids, test_ids = set(train_ids), set(val_ids), set(test_ids)

    print("Building sequences + train-only vocab/scaler...")
    ev, seq_df, stockcode_to_idx, num_mean, num_std, max_len = build_sequences(
        customer_events, customer_target_df, train_ids
    )
    print(f"Customers: {len(seq_df):,} | vocab: {len(stockcode_to_idx):,} | max_len: {max_len}")

    train_ds = CustomerSeqDataset(seq_df[seq_df["CustomerID"].isin(train_ids)], max_len)
    val_ds = CustomerSeqDataset(seq_df[seq_df["CustomerID"].isin(val_ids)], max_len)
    test_ds = CustomerSeqDataset(seq_df[seq_df["CustomerID"].isin(test_ids)], max_len)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    config = {
        "vocab_size": len(stockcode_to_idx),
        "num_numeric": len(NUM_COLS),
        "num_classes": num_classes,
        "max_len": max_len,
        "emb_dim": 64,
        "num_dim": 16,
        "n_heads": 4,
        "ff_dim": 128,
        "dropout": 0.1,
        "num_cols": NUM_COLS,
    }

    print("Training Transformer...")
    model = TransformerClassifier(
        config["vocab_size"], config["num_numeric"], config["num_classes"], config["max_len"]
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_state, best_val_f1 = None, -1.0
    for epoch in range(1, EPOCHS + 1):
        tr = run_epoch(model, train_loader, optimizer=opt)
        va = run_epoch(model, val_loader, optimizer=None)
        if va["f1"] > best_val_f1:
            best_val_f1 = va["f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[TR] epoch={epoch:02d} train_loss={tr['loss']:.4f} val_loss={va['loss']:.4f} "
              f"train_f1={tr['f1']:.4f} val_f1={va['f1']:.4f}")
    model.load_state_dict(best_state)

    test = run_epoch(model, test_loader, optimizer=None)
    print(f"Transformer test acc={test['acc']:.4f} macro-F1={test['f1']:.4f}")

    # Artifacts ---------------------------------------------------------------
    stock_desc_series = (
        transactions_clean.dropna(subset=["StockCode", "Description"])
        .groupby("StockCode")["Description"]
        .agg(lambda s: s.value_counts().index[0])
    )
    stock_desc = {str(k): str(v) for k, v in stock_desc_series.items()}

    personas = name_personas(customer_features_base, customer_target_df)
    samples = pick_samples(seq_df, test_ids, customer_features_base, ev, stock_desc, model, max_len, personas)

    torch.save(model.state_dict(), MODELS_DIR / "retail_transformer.pt")
    (MODELS_DIR / "retail_config.json").write_text(json.dumps(config, indent=2))
    (MODELS_DIR / "retail_vocab.json").write_text(json.dumps(stockcode_to_idx))
    (MODELS_DIR / "retail_scaler.json").write_text(json.dumps({
        "num_mean": {c: float(num_mean[c]) for c in NUM_COLS},
        "num_std": {c: float(num_std[c]) for c in NUM_COLS},
    }, indent=2))
    (MODELS_DIR / "retail_stock_desc.json").write_text(json.dumps(stock_desc))
    (MODELS_DIR / "retail_personas.json").write_text(json.dumps(personas, indent=2))
    (MODELS_DIR / "retail_samples.json").write_text(json.dumps(samples, indent=2))

    print(f"\nSaved {len(samples)} sample customers across {num_classes} segments.")
    for f in sorted(MODELS_DIR.glob("retail_*")):
        print(f"  {f.name}: {f.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
