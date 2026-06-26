"""
Entry point — Stage 1 SV type classification training.

Usage
-----
python scripts/run_train.py \
    --base_dir /path/to/cue/output \
    --sample_ids C3L-01469 C3N-00440 HT338 HT609 HT522 \
    --config configs/default.yaml
"""

import argparse
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data.dataset   import SVDataset, collate_fn, stratified_split, print_split_stats
from models         import Stage1Transformer
from training.train import train, evaluate, collect_predictions, make_class_weights
from utils.io       import load_records, filter_records

try:
    from sklearn.metrics import classification_report, confusion_matrix
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- data ---
    records = load_records(args.base_dir, args.sample_ids,
                           subdir=cfg["data"]["subdir"])
    records = filter_records(records)
    print(f"Records after filtering: {len(records)}")

    svtypes      = sorted(set(r["svtype_norm"] for r in records))
    svtype_to_id = {s: i for i, s in enumerate(svtypes)}
    id_to_svtype = {i: s for s, i in svtype_to_id.items()}
    print(f"SV types: {svtype_to_id}")

    y_all = [svtype_to_id[r["svtype_norm"]] for r in records]
    tr_cfg = cfg["training"]
    train_idx, val_idx, test_idx = stratified_split(
        y_all,
        train_frac=tr_cfg["train_frac"],
        val_frac=tr_cfg["val_frac"],
        test_frac=tr_cfg["test_frac"],
        seed=cfg["seed"],
    )

    train_recs = [records[i] for i in train_idx]
    val_recs   = [records[i] for i in val_idx]
    test_recs  = [records[i] for i in test_idx]

    print_split_stats("TRAIN", [svtype_to_id[r["svtype_norm"]] for r in train_recs], id_to_svtype)
    print_split_stats("VAL",   [svtype_to_id[r["svtype_norm"]] for r in val_recs],   id_to_svtype)
    print_split_stats("TEST",  [svtype_to_id[r["svtype_norm"]] for r in test_recs],  id_to_svtype)

    batch = tr_cfg["batch_size"]
    train_ds = SVDataset(train_recs, svtype_to_id)
    val_ds   = SVDataset(val_recs,   svtype_to_id)
    test_ds  = SVDataset(test_recs,  svtype_to_id)

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=batch, shuffle=False, collate_fn=collate_fn)

    # --- model ---
    d_in  = train_ds[0][0].shape[-1]
    m_cfg = cfg["model"]
    model = Stage1Transformer(
        d_in      = d_in,
        n_classes = len(svtypes),
        d_model   = m_cfg["d_model"],
        n_heads   = m_cfg["n_heads"],
        n_layers  = m_cfg["n_layers"],
        dropout   = m_cfg["dropout"],
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --- class weights ---
    train_labels = [svtype_to_id[r["svtype_norm"]] for r in train_recs]
    ce_weight    = make_class_weights(train_labels, len(svtypes), device)

    # --- train ---
    model = train(model, train_loader, val_loader, tr_cfg, device, ce_weight)

    # --- test ---
    te_loss, te_acc = evaluate(model, test_loader, device, ce_weight)
    print(f"\nTEST: acc={te_acc:.3f}  loss={te_loss:.4f}")

    if HAS_SKLEARN:
        rows   = collect_predictions(model, test_loader, id_to_svtype, device)
        y_true = [r["svtype_gt"]   for r in rows]
        y_pred = [r["svtype_pred"] for r in rows]
        print("\n=== Classification report ===")
        print(classification_report(y_true, y_pred, digits=3))
        print("\n=== Confusion matrix ===")
        print(confusion_matrix(y_true, y_pred))

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVantage-DL Stage 1 training")
    parser.add_argument("--base_dir",   required=True,               help="Root data directory (Cue output)")
    parser.add_argument("--sample_ids", required=True, nargs="+",    help="Sample IDs to load")
    parser.add_argument("--config",     default="configs/default.yaml", help="YAML config path")
    main(parser.parse_args())
