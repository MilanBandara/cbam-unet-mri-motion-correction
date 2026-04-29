"""Run CBAM-U-Net inference on a single image or a directory of PNGs.

Examples::

    # Single image
    python scripts/infer.py \
        --weights checkpoints/weights/model_best_.weights.h5 \
        --input some_scan.png \
        --output results/predictions/

    # Directory of images
    python scripts/infer.py \
        --weights checkpoints/weights/model_best_.weights.h5 \
        --input data/test/head_motion_2/ \
        --output results/predictions_motion2/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np

from cbam_unet import artifact_removal_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, required=True, help="Path to .weights.h5 file.")
    p.add_argument("--input", type=Path, required=True, help="Single PNG or directory of PNGs.")
    p.add_argument("--output", type=Path, required=True, help="Output directory for predictions.")
    p.add_argument("--image-size", type=int, default=256)
    return p.parse_args()


def load_image(path: Path, image_size: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.resize(img, (image_size, image_size))
    return img.astype(np.float32) / 255.0


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    model = artifact_removal_model(image_resolution=args.image_size)
    model.load_weights(str(args.weights))

    if args.input.is_dir():
        files = sorted(args.input.glob("*.png"))
    else:
        files = [args.input]
    if not files:
        raise SystemExit(f"No PNGs found at {args.input}")

    print(f"Running inference on {len(files)} image(s) ...")
    for path in files:
        img = load_image(path, args.image_size)
        prediction = model.predict(img[np.newaxis, :, :, np.newaxis], verbose=0)
        out_img = (np.clip(prediction[0, :, :, 0], 0, 1) * 255).astype(np.uint8)
        out_path = args.output / f"{path.stem}_corrected.png"
        cv2.imwrite(str(out_path), out_img)
        print(f"  {path.name} -> {out_path}")

    print(f"Done. Predictions written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
