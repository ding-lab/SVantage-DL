"""
Entry point — Stage 2 U-Net coarse breakpoint localization training.

Usage
-----
python scripts/run_localizer.py \
    --train_glob "/path/to/cnn_train_prep_100kb/**/*.pkl" \
    --test_glob  "/path/to/cnn_test_prep_100kb/**/*.pkl" \
    --epochs 120 --sigma 5.0 --lr 5e-4
"""

import argparse
import sys
import os
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train_localizer import train_localizer, evaluate_localizer, infer_to_csv


KEEP_SIGNALS = ("SR_RP", "RD_LOW", "RD_CLIPPED")


def main(args):
    train_pkls = sorted(glob(args.train_glob, recursive=True))
    test_pkls  = sorted(glob(args.test_glob,  recursive=True))

    print(f"Train PKLs: {len(train_pkls)}")
    print(f"Test PKLs:  {len(test_pkls)}")

    model = train_localizer(
        train_pkls=train_pkls,
        keep_signals=KEEP_SIGNALS,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        sigma=args.sigma,
    )

    print("\n--- TRAIN ---")
    print(evaluate_localizer(model, train_pkls, KEEP_SIGNALS))

    print("\n--- TEST ---")
    print(evaluate_localizer(model, test_pkls, KEEP_SIGNALS))

    infer_to_csv(
        model, test_pkls,
        out_csv=args.out_csv,
        keep_signals=KEEP_SIGNALS,
        sigma=args.sigma,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVantage-DL Stage 2 training")
    parser.add_argument("--train_glob",  required=True)
    parser.add_argument("--test_glob",   required=True)
    parser.add_argument("--epochs",      type=int,   default=120)
    parser.add_argument("--batch_size",  type=int,   default=8)
    parser.add_argument("--lr",          type=float, default=5e-4)
    parser.add_argument("--sigma",       type=float, default=5.0)
    parser.add_argument("--out_csv",     default="localizer_predictions.csv")
    main(parser.parse_args())
