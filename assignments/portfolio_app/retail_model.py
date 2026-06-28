"""Shared Online Retail Transformer model + sequence helpers.

Imported by both export_retail.py (training) and tab_retail.py (serving) so the
architecture is byte-for-byte identical. The model definition is the
TransformerClassifier from assignment3/assignment3.py.
"""

import numpy as np
import torch
import torch.nn as nn

NUM_COLS = ["Quantity", "UnitPrice", "LineAmount"]
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_numeric,
        num_classes,
        max_len,
        emb_dim=64,
        num_dim=16,
        n_heads=4,
        ff_dim=128,
        dropout=0.1,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.num_proj = nn.Linear(num_numeric, num_dim)
        model_dim = emb_dim + num_dim
        self.pos_emb = nn.Embedding(max_len, model_dim)

        self.mha = nn.MultiheadAttention(model_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(model_dim)
        self.ff = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, model_dim),
        )
        self.norm2 = nn.LayerNorm(model_dim)
        self.head = nn.Sequential(
            nn.Linear(model_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )
        self.last_attention = None

    def forward(self, tokens, num_feats, mask):
        x_tok = self.token_emb(tokens)
        x_num = self.num_proj(num_feats)
        x = torch.cat([x_tok, x_num], dim=-1)

        positions = (
            torch.arange(tokens.size(1), device=tokens.device)
            .unsqueeze(0)
            .expand(tokens.size(0), -1)
        )
        x = x + self.pos_emb(positions)

        key_padding_mask = mask == 0
        attn_out, attn_weights = self.mha(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        self.last_attention = attn_weights

        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))

        mask_exp = mask.unsqueeze(-1)
        pooled = (x * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1e-6)
        return self.head(pooled)


def build_padded(token_ids, num_feats, max_len):
    """Pad/truncate one customer sequence into model tensors.

    Mirrors CustomerSeqDataset.__getitem__ / _batch_from_row in assignment3.py.
    Returns (tokens[1,L], num_feats[1,L,F], mask[1,L], eff_len).
    """
    tokens = list(token_ids[:max_len])
    nums = np.asarray(num_feats, dtype=np.float32)[:max_len]
    length = len(tokens)
    n_feat = nums.shape[1] if nums.ndim == 2 else len(NUM_COLS)

    if length < max_len:
        pad_n = max_len - length
        tokens = tokens + [0] * pad_n
        if nums.size == 0:
            nums = np.zeros((0, n_feat), dtype=np.float32)
        nums = np.vstack([nums, np.zeros((pad_n, n_feat), dtype=np.float32)])

    mask = np.zeros(max_len, dtype=np.float32)
    eff_len = min(length, max_len)
    mask[:eff_len] = 1.0

    return (
        torch.tensor(tokens, dtype=torch.long).unsqueeze(0),
        torch.tensor(nums, dtype=torch.float32).unsqueeze(0),
        torch.tensor(mask, dtype=torch.float32).unsqueeze(0),
        eff_len,
    )
