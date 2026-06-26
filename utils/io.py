"""
Data loading and filtering utilities.
"""

import os
import pickle
from glob import glob


def normalize_svtype(sv: str) -> str:
    """Map 'NA' -> 'INS'; pass everything else through."""
    return "INS" if sv == "NA" else sv


def load_records(base_dir: str, sample_ids: list, subdir: str = "sv") -> list:
    """
    Load all .pkl records produced by the tokenization pipeline.

    Expected directory layout
    -------------------------
    base_dir/
      <sample_id>/
        <subdir>/
          sv_<chr>_<idx>.pkl
          ...

    Each .pkl is a dict with at minimum:
        tokens    : np.ndarray (N, d_in)
        attn_bias : np.ndarray (N, N)   -- SR_RP log-normalized bias
        svtype    : str
        chr, start, end : positional fields
    """
    records = []
    for sid in sample_ids:
        path = os.path.join(base_dir, sid, subdir)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Missing directory: {path}")
        pkls = sorted(glob(os.path.join(path, "*.pkl")))
        if not pkls:
            raise FileNotFoundError(f"No .pkl files found under: {path}")
        for fp in pkls:
            with open(fp, "rb") as f:
                r = pickle.load(f)
            r["sample_id"] = sid
            r["_pkl_path"] = fp
            records.append(r)
    return records


def filter_records(records: list) -> list:
    """
    Drop unsupported SV types (TRA, NO_SV) and add 'svtype_norm' field.
    """
    out = []
    for r in records:
        sv = normalize_svtype(r.get("svtype", "NA"))
        if sv in ("TRA", "NO_SV"):
            continue
        r["svtype_norm"] = sv
        out.append(r)
    return out
