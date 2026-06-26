"""
Dataset for Stage 2 U-Net breakpoint localization.
"""

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


def pad_or_crop(M: np.ndarray, target: int = 200) -> np.ndarray:
    """Crop to target x target, then zero-pad if smaller."""
    M = M[:target, :target]
    h, w = M.shape
    pad_h = target - h
    pad_w = target - w
    if pad_h > 0 or pad_w > 0:
        M = np.pad(M, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
    return M


def safe_log1p(M: np.ndarray) -> np.ndarray:
    """NaN-safe log1p normalization."""
    M = M.astype(np.float32)
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    M = np.clip(M, a_min=0, a_max=None)
    return np.log1p(M)


def gaussian_2d_peak(n: int, i_gt: int, j_gt: int, sigma: float = 2.0) -> np.ndarray:
    """
    2D soft Gaussian target centered at bin (i_gt, j_gt).
    Peak value normalized to 1 (not sum-to-1, matching Cue convention).
    """
    xs = np.arange(n)
    ys = np.arange(n)
    xv, yv = np.meshgrid(xs, ys)
    heat = np.exp(-((xv - j_gt) ** 2 + (yv - i_gt) ** 2) / (2 * sigma ** 2))
    heat = heat / (heat.max() + 1e-8)
    return heat.astype(np.float32)


class SVHeatmapDataset(Dataset):
    """
    Dataset for Stage 2 coarse breakpoint localization.

    Each .pkl record (produced by the Cue tokenization pipeline) must contain:
        matrices   : dict of signal name -> np.ndarray (H, W)
        bin_size   : int   genomic bp per bin
        intervalA  : dict  {chr, start, end}
        intervalB  : dict  {chr, start, end}
        start, end : int   true VCF breakpoint coordinates

    Parameters
    ----------
    pkl_paths    : list of str
    keep_signals : tuple of str   signals to stack as input channels
    sigma        : float          Gaussian spread in bins for soft target
    target_dim   : int            spatial size H=W after pad/crop (default 200)
    """

    def __init__(
        self,
        pkl_paths: list,
        keep_signals: tuple = ("SR_RP", "RD_LOW", "RD_CLIPPED"),
        sigma: float = 2.0,
        target_dim: int = 200,
    ):
        self.pkl_paths    = pkl_paths
        self.keep_signals = tuple(keep_signals)
        self.sigma        = sigma
        self.target_dim   = target_dim

    def __len__(self):
        return len(self.pkl_paths)

    def __getitem__(self, idx):
        rec = None
        # retry up to 5 times to get an in-window sample
        for _ in range(5):
            with open(self.pkl_paths[idx], "rb") as f:
                rec = pickle.load(f)

            # stack signal channels
            mats = []
            for sig in self.keep_signals:
                if sig not in rec["matrices"]:
                    M = np.zeros((self.target_dim, self.target_dim), dtype=np.float32)
                else:
                    M = safe_log1p(rec["matrices"][sig])
                    M = pad_or_crop(M, self.target_dim)
                    M = M / (M.max() + 1e-6)
                mats.append(M)

            X = torch.from_numpy(np.stack(mats, axis=0)).float()

            bin_bp = int(rec["bin_size"])
            A0     = int(rec["intervalA"]["start"])
            B0     = int(rec["intervalB"]["start"])
            trueA  = int(rec.get("start", rec.get("vcf_start")))
            trueB  = int(rec.get("end",   rec.get("vcf_end")))

            n = self.target_dim
            i_float = (trueA - A0) / bin_bp
            j_float = (trueB - B0) / bin_bp

            if 0 <= i_float < n and 0 <= j_float < n:
                break
            # GT outside window — resample
            idx = np.random.randint(0, len(self.pkl_paths))

        i_gt   = int(i_float)
        j_gt   = int(j_float)
        heat   = gaussian_2d_peak(n, i_gt, j_gt, sigma=self.sigma)
        target = torch.from_numpy(heat).unsqueeze(0).float()

        meta = {
            "sample_id": rec["sample_id"],
            "chr":       rec["chr"],
            "bin_bp":    bin_bp,
            "A0":        A0,
            "B0":        B0,
            "trueA":     trueA,
            "trueB":     trueB,
        }

        return X, target, meta
