"""
PyTorch Dataset, collate function, and stratified split for SV records.
"""

import random
from collections import Counter, defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------

def stratified_split(labels: list, train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=7):
    """
    Stratified split that handles tiny and singleton classes gracefully.

    - Classes with ≥3 samples get train/val/test representation.
    - Classes with 2 samples: one in train, one in val.
    - Singleton classes: train only.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    rng = random.Random(seed)
    by_class = defaultdict(list)
    for i, y in enumerate(labels):
        by_class[y].append(i)

    train_idx, val_idx, test_idx = [], [], []

    for y, idxs in by_class.items():
        rng.shuffle(idxs)
        n = len(idxs)
        if n >= 3:
            n_train = max(1, int(round(n * train_frac)))
            n_val   = max(1, int(round(n * val_frac)))
            n_test  = max(1, n - n_train - n_val)
            while n_train + n_val + n_test > n:
                if n_train > 1: n_train -= 1
                elif n_val > 1: n_val -= 1
                else: n_test -= 1
            while n_train + n_val + n_test < n:
                n_train += 1
            train_idx += idxs[:n_train]
            val_idx   += idxs[n_train:n_train + n_val]
            test_idx  += idxs[n_train + n_val:]
        elif n == 2:
            train_idx.append(idxs[0])
            val_idx.append(idxs[1])
        else:
            train_idx.append(idxs[0])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def print_split_stats(name: str, label_ids: list, id_to_svtype: dict):
    c = Counter([id_to_svtype[y] for y in label_ids])
    print(f"\n{name}: n={sum(c.values())}")
    for k, v in sorted(c.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {k:>6s}: {v}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SVDataset(Dataset):
    """
    Dataset for Stage 1 SV type classification.

    Each record (loaded from a Cue .pkl) must contain:
        tokens     : np.ndarray (N, d_in)   pre-tokenized signal features
        attn_bias  : np.ndarray (N, N)      SR_RP read-pair contact matrix (or None)
        svtype_norm: str                     normalized SV type label
    """

    def __init__(self, records: list, svtype_to_id: dict):
        self.records      = records
        self.svtype_to_id = svtype_to_id

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]

        # --- tokens ---
        x = torch.tensor(r["tokens"], dtype=torch.float32)
        N = x.shape[0]

        # --- attention bias: log-compress SR_RP read-pair signal ---
        bias = r.get("attn_bias", None)
        if bias is None:
            bias = np.zeros((N, N), dtype=np.float32)
        else:
            bias = np.log1p(np.asarray(bias, dtype=np.float32))
        attn_bias = torch.tensor(bias, dtype=torch.float32)

        # --- label ---
        y = torch.tensor(self.svtype_to_id[r["svtype_norm"]], dtype=torch.long)

        # --- metadata (for error analysis) ---
        meta = {
            "sample_id": r.get("sample_id"),
            "svtype":    r.get("svtype_norm"),
            "chr":       r.get("chr"),
            "start":     int(r.get("start", -1)),
            "end":       int(r.get("end", -1)),
            "_pkl_path": r.get("_pkl_path", ""),
        }

        return x, attn_bias, y, meta


# ---------------------------------------------------------------------------
# Collate — pads variable-length sequences in a batch
# ---------------------------------------------------------------------------

def collate_fn(batch):
    xs, bs, ys, metas = zip(*batch)

    max_len = max(x.shape[0] for x in xs)
    d       = xs[0].shape[1]
    B       = len(xs)

    x_pad = torch.zeros(B, max_len, d,        dtype=xs[0].dtype)
    b_pad = torch.zeros(B, max_len, max_len,   dtype=bs[0].dtype)

    for i, (x, b) in enumerate(zip(xs, bs)):
        n  = x.shape[0]
        bn = min(n, b.shape[0], b.shape[1])
        x_pad[i, :n, :]    = x
        b_pad[i, :bn, :bn] = b[:bn, :bn]

    return x_pad, b_pad, torch.stack(ys), list(metas)
