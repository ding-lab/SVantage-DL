"""
Training, evaluation, and inference for Stage 2 U-Net localizer.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.heatmap_dataset import SVHeatmapDataset
from models.unet import UNetLocalizer


# ---------------------------------------------------------------------------
# Loss: spatial cross-entropy over flattened heatmap
# ---------------------------------------------------------------------------

def spatial_ce_loss(pred_logits: torch.Tensor, target_prob: torch.Tensor) -> torch.Tensor:
    """
    Spatial cross-entropy between predicted heatmap logits and soft Gaussian target.

    Flattens the H×W spatial grid into a categorical distribution over pixel
    locations and computes cross-entropy against the soft Gaussian target q.
    Equivalent to KL divergence between predicted and target spatial distributions.

    Parameters
    ----------
    pred_logits : (B, 1, H, W)   raw logits from UNetLocalizer
    target_prob : (B, 1, H, W)   soft Gaussian target (peak-normalized to 1)

    Returns
    -------
    loss : scalar
    """
    B, _, H, W = pred_logits.shape
    logp = F.log_softmax(pred_logits.view(B, -1), dim=1)  # (B, H*W)
    q    = target_prob.view(B, -1)                         # (B, H*W)
    return -(q * logp).sum(dim=1).mean()


# ---------------------------------------------------------------------------
# Soft-argmax decode: continuous bin prediction from heatmap
# ---------------------------------------------------------------------------

@torch.no_grad()
def soft_argmax_2d(pred_logits: torch.Tensor):
    """
    Decode predicted logits to continuous (i, j) bin coordinates via soft-argmax.

    Parameters
    ----------
    pred_logits : (B, 1, H, W)

    Returns
    -------
    y_pred : (B,)   predicted row index (-> breakpoint A)
    x_pred : (B,)   predicted col index (-> breakpoint B)
    """
    B, _, H, W = pred_logits.shape
    prob = torch.softmax(pred_logits.view(B, -1), dim=1).view(B, H, W)

    xs = torch.arange(W, device=pred_logits.device).float()
    ys = torch.arange(H, device=pred_logits.device).float()

    x_pred = (prob.sum(dim=1) * xs).sum(dim=1)  # col -> B coordinate
    y_pred = (prob.sum(dim=2) * ys).sum(dim=1)  # row -> A coordinate

    return y_pred, x_pred


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_localizer(
    train_pkls: list,
    keep_signals: tuple = ("SR_RP", "RD_LOW", "RD_CLIPPED"),
    epochs: int = 120,
    batch_size: int = 8,
    lr: float = 5e-4,
    sigma: float = 5.0,
) -> UNetLocalizer:
    """
    Train the U-Net localizer on a list of .pkl records.

    Parameters
    ----------
    train_pkls   : list of str   paths to training .pkl records
    keep_signals : tuple         signal channels to use as input
    epochs       : int
    batch_size   : int
    lr           : float
    sigma        : float         Gaussian spread for soft target (bins)

    Returns
    -------
    model : UNetLocalizer
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = SVHeatmapDataset(train_pkls, keep_signals=keep_signals, sigma=sigma)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                    num_workers=2, pin_memory=True)

    model = UNetLocalizer(in_ch=len(keep_signals)).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for X, target, _ in dl:
            X, target = X.to(device, non_blocking=True), target.to(device, non_blocking=True)
            pred = model(X)
            loss = spatial_ce_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"Epoch {ep:03d}/{epochs} | loss={total / len(dl):.4f}")

    return model


# ---------------------------------------------------------------------------
# Evaluation: breakpoint error in base pairs
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_localizer(
    model: UNetLocalizer,
    pkl_list: list,
    keep_signals: tuple = ("SR_RP", "RD_LOW", "RD_CLIPPED"),
    batch_size: int = 8,
) -> dict:
    """
    Evaluate breakpoint localization error in base pairs.

    Returns dict with keys:
        mean_A, mean_B, mean_max, median_max, p90_max, p95_max, worst_max
    """
    ds = SVHeatmapDataset(pkl_list, keep_signals=keep_signals)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    device = next(model.parameters()).device
    model.eval()

    errA_list, errB_list, err_max_list = [], [], []

    for X, _, meta in dl:
        X      = X.to(device)
        logits = model(X)
        y_pred, x_pred = soft_argmax_2d(logits)

        for b in range(X.shape[0]):
            bin_bp = int(meta["bin_bp"][b])
            A0     = int(meta["A0"][b])
            B0     = int(meta["B0"][b])
            trueA  = int(meta["trueA"][b])
            trueB  = int(meta["trueB"][b])

            predA = A0 + y_pred[b].item() * bin_bp
            predB = B0 + x_pred[b].item() * bin_bp

            errA = abs(predA - trueA)
            errB = abs(predB - trueB)
            errA_list.append(errA)
            errB_list.append(errB)
            err_max_list.append(max(errA, errB))

    errA    = np.array(errA_list)
    errB    = np.array(errB_list)
    err_max = np.array(err_max_list)

    return {
        "mean_A":     errA.mean(),
        "mean_B":     errB.mean(),
        "mean_max":   err_max.mean(),
        "median_max": np.median(err_max),
        "p90_max":    np.percentile(err_max, 90),
        "p95_max":    np.percentile(err_max, 95),
        "worst_max":  err_max.max(),
    }


# ---------------------------------------------------------------------------
# Inference: save predictions to CSV
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer_to_csv(
    model: UNetLocalizer,
    pkl_list: list,
    out_csv: str,
    keep_signals: tuple = ("SR_RP", "RD_LOW", "RD_CLIPPED"),
    sigma: float = 5.0,
    batch_size: int = 8,
) -> pd.DataFrame:
    """
    Run inference and save per-SV predictions to CSV.

    Columns: sample_id, chr, vcf_start, vcf_end,
             predA, predB, error_A_bp, error_B_bp, max_error_bp
    """
    device = next(model.parameters()).device
    model.eval()

    ds = SVHeatmapDataset(pkl_list, keep_signals=keep_signals, sigma=sigma)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    rows = []
    for X, target, meta in dl:
        X      = X.to(device)
        logits = model(X)

        B, _, H, W = logits.shape
        probs  = torch.softmax(logits.view(B, -1), dim=1)
        idx    = torch.argmax(probs, dim=1)
        i_pred = (idx // W).cpu().numpy()
        j_pred = (idx  % W).cpu().numpy()

        for b in range(B):
            bin_bp = meta["bin_bp"][b].item()
            A0     = meta["A0"][b].item()
            B0     = meta["B0"][b].item()
            trueA  = meta["trueA"][b].item()
            trueB  = meta["trueB"][b].item()

            predA = A0 + i_pred[b] * bin_bp
            predB = B0 + j_pred[b] * bin_bp
            errA  = abs(predA - trueA)
            errB  = abs(predB - trueB)

            rows.append({
                "sample_id":    meta["sample_id"][b],
                "chr":          meta["chr"][b],
                "vcf_start":    trueA,
                "vcf_end":      trueB,
                "predA":        predA,
                "predB":        predB,
                "error_A_bp":   errA,
                "error_B_bp":   errB,
                "max_error_bp": max(errA, errB),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    return df
