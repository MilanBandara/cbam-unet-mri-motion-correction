"""Generate paired training/validation data via k-space motion synthesis.

The script expects an MR-ART-style root containing the standard split layout::

    <mrart-root>/
        MRART_training_data/Original/*.png
        MRART_val_data/Original/*.png

It then writes::

    <out>/train/{Original,Corrupted}/*.png
    <out>/val/{Original,Corrupted}/*.png

Run with no arguments (after editing the defaults) or, more typically::

    python scripts/synthesize_data.py \
        --mrart-root /path/to/mrart \
        --out data/

See the paper for parameter justification (default: rigid translation+rotation,
50 phase-encoding columns swapped, width 3, uniformly random region).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/synthesize_data.py` from the repo root without install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from cbam_unet.synth import generate_split


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--mrart-root",
        type=Path,
        required=True,
        help="Root containing MRART_training_data/Original and MRART_val_data/Original.",
    )
    p.add_argument("--out", type=Path, default=Path("data"), help="Output root directory.")
    p.add_argument(
        "--train-subdir",
        default="MRART_training_data/Original",
        help="Path under --mrart-root holding the training originals.",
    )
    p.add_argument(
        "--val-subdir",
        default="MRART_val_data/Original",
        help="Path under --mrart-root holding the validation originals.",
    )
    p.add_argument("--motion-type", choices=["translation", "rotation", "both"], default="both")
    p.add_argument("--shift-x", type=float, default=2)
    p.add_argument("--shift-y", type=float, default=0)
    p.add_argument("--angle", type=float, default=3)
    p.add_argument("--lines", type=int, default=50, help="Number of phase-encoding columns swapped.")
    p.add_argument("--width", type=int, default=3, help="Width (columns) of each swapped band.")
    p.add_argument(
        "--region",
        choices=["random", "central", "peripheral", "mixed"],
        default="random",
        help="Which k-space columns may be replaced.",
    )
    p.add_argument("--seed", type=int, default=0, help="Numpy RNG seed for reproducibility.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)

    train_src = args.mrart_root / args.train_subdir
    val_src = args.mrart_root / args.val_subdir
    if not train_src.is_dir():
        raise SystemExit(f"Training source not found: {train_src}")
    if not val_src.is_dir():
        raise SystemExit(f"Validation source not found: {val_src}")

    params = dict(
        motion_type=args.motion_type,
        shift_x=args.shift_x,
        shift_y=args.shift_y,
        angle=args.angle,
        number_of_lines=args.lines,
        width=args.width,
        region=args.region,
    )

    print("Synthesis parameters:", params)
    print("Train source :", train_src)
    print("Val   source :", val_src)
    print("Output root  :", args.out)

    train_saved, train_errors = generate_split(train_src, args.out / "train", params, "TRAIN")
    val_saved, val_errors = generate_split(val_src, args.out / "val", params, "VAL")

    print("=" * 50)
    print("Dataset generation complete")
    print(f"  Train pairs saved : {train_saved:>6}  (errors: {train_errors})")
    print(f"  Val   pairs saved : {val_saved:>6}  (errors: {val_errors})")
    print(f"  Total pairs       : {train_saved + val_saved:>6}")
    print(f"  Output location   : {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
