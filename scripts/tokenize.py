"""
Signal extraction and tokenization pipeline for SVantage-DL.

This script extracts multi-channel signal matrices from BAM files at
candidate SV loci and converts them into the token and attention bias
representations consumed by the Stage 1 transformer.

Signal extraction uses BAM alignment index functionality from the Cue
framework (https://github.com/PopicLab/cue) as a preprocessing step.
The tokenization strategy and attention bias construction implemented
here are original components of SVantage-DL.

Usage
-----
Fill in the SAMPLES dict below with sample_id -> {bam, vcf, fai} paths,
then run:

    python scripts/tokenize.py

Output
------
For each sample, one .pkl file per SV written to:
    <BASE_OUTDIR>/<sample_id>/sv/sv_<chr>_<idx:05d>.pkl

Each .pkl contains:
    tokens     : np.ndarray (N, d_in)   per-bin token features
    attn_bias  : np.ndarray (N, N)      SR_RP attention bias (float16)
    matrices   : dict                   raw signal matrices (float32)
    bin_size   : int
    intervalA/B: dict {chr, start, end}
    sample_id, svtype, chr, start, end

Tokenization strategy (strategy_c)
------------------------------------
For each bin i and each signal channel:
    features = [row.sum(), row.max(), row.mean(), argmax(row) / N]
Plus a positional feature: i / N
Final token shape: (N, n_signals * 4 + 1)

Attention bias
--------------
SR_RP matrix -> log1p -> max-normalized -> float16
Injected per-head into the transformer's multi-head self-attention.
"""

import os
import pickle

import numpy as np

# Cue imports (must have cue installed: https://github.com/PopicLab/cue)
from img import constants
from seq.aln_index import AlnIndex
from seq.intervals import GenomeInterval, GenomeIntervalPair


# ================================
# CONFIG — fill in your samples
# ================================
SAMPLES = {
    # "sample_id": {
    #     "bam": "/path/to/sample.bam",
    #     "vcf": "/path/to/sample.vcf",
    #     "fai": "/path/to/reference.fa.fai",
    # },
}

BASE_OUTDIR   = "tokens_1MB_27samples_feb20_flt"
BIN_SIZE      = 2000
INTERVAL_SIZE = 1_000_000
KEEP_SIGNALS  = ["SM", "RD_LOW", "RD_CLIPPED", "SR_RP"]


# ================================
# Cue alignment config
# ================================
class VCFConfig:
    def __init__(self, bam, fai):
        self.bam  = bam
        self.fai  = fai
        self.bin_size = BIN_SIZE
        self.signal_set        = constants.SV_SIGNAL_SET.LONG
        self.signal_set_origin = self.signal_set.name
        self.signal_mapq = {
            sig: 20 for sig in constants.SV_SIGNALS_BY_TYPE[self.signal_set]
        }
        for sig in constants.SV_SIGNAL_SCALAR:
            self.signal_mapq[sig] = 0
        self.scan_target_intervals = False
        self.allow_empty           = True
        self.stream                = False


# ================================
# VCF parser
# ================================
def parse_vcf_sv(vcf_path):
    with open(vcf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            chr1   = fields[0]
            start  = int(fields[1])
            info   = dict(
                item.split("=", 1)
                for item in fields[7].split(";")
                if "=" in item
            )
            yield dict(
                chr1=chr1,
                start=start,
                chr2=info.get("CHR2", chr1),
                end=int(info["END"]),
                svtype=info.get("SVTYPE", "NA"),
            )


# ================================
# Interval construction
# ================================
def sv_to_interval_pair(sv, aln):
    half      = INTERVAL_SIZE // 2
    chr_len_1 = aln.chr_index.chr(aln.chr_index.tid(sv["chr1"])).len
    chr_len_2 = aln.chr_index.chr(aln.chr_index.tid(sv["chr2"])).len
    return GenomeIntervalPair(
        GenomeInterval(sv["chr1"], max(0, sv["start"] - half),
                       min(chr_len_1, sv["start"] + half)),
        GenomeInterval(sv["chr2"], max(0, sv["end"] - half),
                       min(chr_len_2, sv["end"] + half)),
    )


# ================================
# Signal matrix extraction via Cue
# ================================
def extract_signal_matrices(aln, pair):
    """
    Extract raw 2D signal matrices for a candidate SV window pair.
    Uses Cue's AlnIndex scalar_apply (for depth signals) and
    intersect (for read-pair contact signals).
    """
    mats = {}
    for sig in constants.SV_SIGNALS_BY_TYPE[aln.config.signal_set]:
        if sig in constants.SV_SIGNAL_SCALAR:
            M = aln.scalar_apply(sig, pair.intervalA, pair.intervalB)
        else:
            M = aln.intersect(sig, pair.intervalA, pair.intervalB,
                              off_diagonal_only=False)
        mats[sig.name] = M
    return mats


# ================================
# Tokenization (SVantage-DL strategy_c)
# ================================
def tokenize(matrices: dict) -> np.ndarray:
    """
    Convert raw signal matrices into a 1D token sequence.

    For each bin i and each signal channel:
        [row.sum(), row.max(), row.mean(), argmax(row)/N]
    Plus positional feature: i/N

    Returns
    -------
    tokens : np.ndarray (N, n_signals * 4 + 1)
    """
    n_bins = matrices[KEEP_SIGNALS[0]].shape[0]
    tokens = []
    for i in range(n_bins):
        feats = []
        for sig in KEEP_SIGNALS:
            row = matrices[sig][i]
            feats.extend([
                row.sum(),
                row.max(),
                row.mean(),
                np.argmax(row) / n_bins,
            ])
        feats.append(i / n_bins)
        tokens.append(feats)
    return np.asarray(tokens, dtype=np.float32)


# ================================
# Attention bias construction
# ================================
def build_attention_bias(matrices: dict) -> np.ndarray:
    """
    Build the SR_RP attention bias matrix for the transformer.

    SR_RP read-pair contact matrix -> log1p -> max-normalized -> float16.
    Injected per head into multi-head self-attention as an additive bias.

    Returns
    -------
    attn_bias : np.ndarray (N, N), float16
    """
    B = np.log1p(matrices["SR_RP"])
    B = B / (B.max() + 1e-6)
    return B.astype(np.float16)


# ================================
# Main loop
# ================================
if __name__ == "__main__":
    for sample_id, paths in SAMPLES.items():
        print(f"\n=== Processing {sample_id} ===")
        out_dir = os.path.join(BASE_OUTDIR, sample_id, "sv")
        os.makedirs(out_dir, exist_ok=True)

        config    = VCFConfig(paths["bam"], paths["fai"])
        svs       = list(parse_vcf_sv(paths["vcf"]))
        aln_cache = {}

        for idx, sv in enumerate(svs):
            if sv["chr1"] != sv["chr2"]:
                continue  # skip inter-chromosomal SVs

            chrn = sv["chr1"]
            if chrn not in aln_cache:
                aln_cache[chrn] = AlnIndex.generate(chrn, config)
            aln = aln_cache[chrn]

            pair     = sv_to_interval_pair(sv, aln)
            matrices = extract_signal_matrices(aln, pair)

            record = dict(
                sample_id = sample_id,
                svtype    = sv["svtype"],
                chr       = sv["chr1"],
                start     = sv["start"],
                end       = sv["end"],
                tokens    = tokenize(matrices),
                attn_bias = build_attention_bias(matrices),
                matrices  = {k: np.asarray(matrices[k], dtype=np.float32)
                             for k in KEEP_SIGNALS},
                bin_size  = BIN_SIZE,
                intervalA = dict(chr=pair.intervalA.chr_name,
                                 start=int(pair.intervalA.start),
                                 end=int(pair.intervalA.end)),
                intervalB = dict(chr=pair.intervalB.chr_name,
                                 start=int(pair.intervalB.start),
                                 end=int(pair.intervalB.end)),
            )

            out_path = os.path.join(out_dir, f"sv_{chrn}_{idx:05d}.pkl")
            with open(out_path, "wb") as f:
                pickle.dump(record, f)

        print(f"Done: {sample_id}")

    print("\nAll samples complete.")
