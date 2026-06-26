"""
Training and evaluation for Stage 1 SV type classifier.
"""

from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def make_class_weights(train_labels: list, n_classes: int, device: str) -> torch.Tensor:
    """Inverse-frequency class weights to handle class imbalance."""
    c     = Counter(train_labels)
    total = len(train_labels)
    w     = np.array([total / max(c.get(k, 1), 1) for k in range(n_classes)],
                     dtype=np.float32)
    w    /= w.mean()
    return torch.tensor(w, device=device)


@torch.no_grad()
def evaluate(model, loader, device: str, ce_weight=None):
    """Return (avg_loss, accuracy) on a dataloader."""
    model.eval()
    total_loss, total, correct = 0.0, 0, 0
    for x, bias, y, _ in loader:
        x, bias, y    = x.to(device), bias.to(device), y.to(device)
        sv_logits, _  = model(x, bias)
        loss          = F.cross_entropy(sv_logits, y, weight=ce_weight)
        total_loss   += loss.item() * y.size(0)
        correct      += (sv_logits.argmax(1) == y).sum().item()
        total        += y.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def collect_predictions(model, loader, id_to_svtype: dict, device: str) -> list:
    """
    Run inference and collect per-SV results for error analysis.

    Returns
    -------
    list of dicts with keys:
        sample_id, chr, vcf_start, vcf_end,
        svtype_gt, svtype_pred, confidence, pkl_path
    """
    model.eval()
    rows = []
    for x, bias, y, metas in loader:
        x, bias      = x.to(device), bias.to(device)
        sv_logits, _ = model(x, bias)
        preds        = sv_logits.argmax(1).cpu().numpy()
        probs        = F.softmax(sv_logits, dim=1).cpu().numpy()
        for i, meta in enumerate(metas):
            rows.append({
                "sample_id":     meta["sample_id"],
                "chr":           meta["chr"],
                "vcf_start":     meta["start"],
                "vcf_end":       meta["end"],
                "svtype_gt":     meta["svtype"],
                "svtype_pred":   id_to_svtype[preds[i]],
                "confidence":    float(probs[i, preds[i]]),
                "pkl_path":      meta["_pkl_path"],
            })
    return rows


def train(model, train_loader, val_loader, cfg: dict, device: str, ce_weight=None):
    """
    Train with AdamW + gradient clipping; restore best val-acc checkpoint.

    Parameters
    ----------
    cfg : dict   must have keys: lr, epochs, weight_decay, grad_clip
    """
    opt      = torch.optim.AdamW(model.parameters(),
                                  lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    best_acc = -1.0
    best_ckpt = None

    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        for x, bias, y, _ in train_loader:
            x, bias, y   = x.to(device), bias.to(device), y.to(device)
            sv_logits, _ = model(x, bias)
            loss         = F.cross_entropy(sv_logits, y, weight=ce_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()

        tr_loss, tr_acc = evaluate(model, train_loader, device, ce_weight)
        va_loss, va_acc = evaluate(model, val_loader,   device, ce_weight)
        print(f"ep {ep:03d} | train loss={tr_loss:.4f} acc={tr_acc:.3f} "
              f"| val loss={va_loss:.4f} acc={va_acc:.3f}")

        if va_acc > best_acc:
            best_acc  = va_acc
            best_ckpt = {k: v.detach().cpu().clone()
                         for k, v in model.state_dict().items()}

    if best_ckpt is not None:
        model.load_state_dict(best_ckpt)
    print(f"\nBest val acc: {best_acc:.3f}")
    return model
