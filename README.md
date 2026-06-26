# SVantage-DL

Deep learning pipeline for structural variant (SV) detection and breakpoint localization from long- and short-read sequencing data.

## Overview

SVantage-DL implements a two-stage deep learning architecture for SV analysis:

**Stage 1 — Context-modulated Transformer Encoder**
Classifies SV type (DEL / DUP / INS / INV) from tokenized signal representations of candidate SV loci. A read-pair contact matrix is converted into a per-head attention bias and injected directly into each transformer block, allowing the model to incorporate inter-locus genomic signal within the attention mechanism.

**Stage 2 — Coarse Breakpoint Localizer**
A U-Net that predicts a 2D breakpoint probability heatmap over the signal matrix. The predicted bin-pair (i, j) gives a coarse genomic coordinate estimate for each breakpoint, ready for fine-resolution refinement.

## Data preparation

### Signal extraction and tokenization

Multi-channel alignment signals are extracted from BAM files at candidate SV loci and transformed into structured token representations and pairwise contact matrices. Signal extraction draws on alignment indexing concepts from the [Cue framework](https://github.com/PopicLab/cue) (PopicLab, Broad Institute). The extracted signals are subsequently processed by SVantage-DL's preprocessing pipeline into token sequences and attention bias matrices consumed by the transformer, a representation that is agnostic to whether the underlying reads are long- or short-read data.

To run tokenization:

```bash
# Edit SAMPLES dict in scripts/tokenize.py with BAM/VCF/FAI paths
python scripts/tokenize.py
```

Output `.pkl` records are written to `<BASE_OUTDIR>/<sample_id>/sv/`.

### Expected .pkl record structure

```python
{
    "tokens":    np.ndarray,  # (N, d_in)   token features
    "attn_bias": np.ndarray,  # (N, N)      attention bias matrix, float16
    "matrices":  dict,        # raw signal matrices, float32
    "bin_size":  int,         # genomic bp per bin
    "intervalA": dict,        # {chr, start, end}
    "intervalB": dict,        # {chr, start, end}
    "sample_id": str,
    "svtype":    str,
    "chr":       str,
    "start":     int,
    "end":       int,
}
```

## Installation

```bash
git clone https://github.com/ding-lab/SVantage-DL.git
cd SVantage-DL
pip install -r requirements.txt
```

## Usage

**Stage 1 — SV type classification:**
```bash
python scripts/run_train.py \
    --base_dir /path/to/tokenized/records \
    --sample_ids SAMPLE1 SAMPLE2 SAMPLE3 \
    --config configs/default.yaml
```

**Stage 2 — Coarse breakpoint localization:**
```bash
python scripts/run_localizer.py \
    --train_glob "/path/to/train/**/*.pkl" \
    --test_glob  "/path/to/test/**/*.pkl" \
    --epochs 120 --sigma 5.0
```

All hyperparameters are in `configs/default.yaml`.

## Repository structure

```
SVantage-DL/
├── configs/
│   └── default.yaml            hyperparameter config
├── data/
│   ├── dataset.py              SVDataset, collate, stratified split (Stage 1)
│   └── heatmap_dataset.py      SVHeatmapDataset, Gaussian target (Stage 2)
├── models/
│   ├── transformer.py          Stage 1: context-modulated transformer encoder
│   └── unet.py                 Stage 2: U-Net coarse breakpoint localizer
├── training/
│   ├── train.py                Stage 1 training loop and evaluation
│   └── train_localizer.py      Stage 2 training, evaluation, inference
├── utils/
│   └── io.py                   data loading and filtering
└── scripts/
    ├── tokenize.py             signal extraction and tokenization
    ├── run_train.py            Stage 1 entry point
    └── run_localizer.py        Stage 2 entry point
```


