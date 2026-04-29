"""Evaluate a trained MR^2-AttUNet on validation pairs and the MR-ART real-motion test set.

Reproduces the quantitative tables in the paper:

    * Validation evaluation reads paired ``Original`` / ``Corrupted`` images
      from ``--val-dir`` and reports MSE/PSNR/SSIM.
    * Real-motion evaluation reads the ``Original``, ``head_motion_1`` and
      ``head_motion_2`` folders under ``--test-dir`` and prints a side-by-side
      summary for both motion levels.

Optionally writes qualitative sample grids next to the metrics output via
``--save-samples``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np

from cbam_unet import artifact_removal_model
from cbam_unet.metrics import mse as mse_fn, psnr as psnr_fn, ssim as ssim_fn, summarise
from cbam_unet.viz import plot_corrected_samples


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--val-dir", type=Path, default=Path("data/val"))
    p.add_argument("--test-dir", type=Path, default=Path("data/test"))
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--max-val-pairs", type=int, default=250, help="Cap on validation pairs evaluated.")
    p.add_argument("--save-samples", type=Path, default=None, help="If set, write qualitative grids here.")
    return p.parse_args()


def evaluate_pair_dir(model, ref_dir: Path, deg_dir: Path, label: str, image_size: int, limit: int | None = None):
    """Compute MSE/PSNR/SSIM for paired files in ``ref_dir`` and ``deg_dir``."""

    ref_files = sorted(p.name for p in ref_dir.iterdir() if p.suffix.lower() == ".png")
    deg_files = sorted(p.name for p in deg_dir.iterdir() if p.suffix.lower() == ".png")
    if limit is not None:
        ref_files = ref_files[:limit]
        deg_files = deg_files[:limit]
    n = min(len(ref_files), len(deg_files))

    mse_list, psnr_list, ssim_list = [], [], []
    for i in range(n):
        ref = cv2.imread(str(ref_dir / ref_files[i]), cv2.IMREAD_GRAYSCALE)
        deg = cv2.imread(str(deg_dir / deg_files[i]), cv2.IMREAD_GRAYSCALE)
        ref = cv2.resize(ref, (image_size, image_size)).astype(np.float32) / 255.0
        deg = cv2.resize(deg, (image_size, image_size)).astype(np.float32) / 255.0

        gt = ref[np.newaxis, :, :, np.newaxis]
        pred = model.predict(deg[np.newaxis, :, :, np.newaxis], verbose=0)

        mse_list.append(mse_fn(gt, pred))
        psnr_list.append(psnr_fn(gt, pred))
        ssim_list.append(ssim_fn(gt, pred))

    mse_mean, mse_std = summarise(mse_list)
    psnr_mean, psnr_std = summarise(psnr_list)
    ssim_mean, ssim_std = summarise(ssim_list)

    print(f"\n=== {label} ({n} pairs) ===")
    print(f"  Mean MSE:  {mse_mean:.6f} \u00b1 {mse_std:.6f}")
    print(f"  Mean PSNR: {psnr_mean:.4f} \u00b1 {psnr_std:.4f} dB")
    print(f"  Mean SSIM: {ssim_mean:.6f} \u00b1 {ssim_std:.6f}")

    return {
        "n": n,
        "mse": (mse_mean, mse_std),
        "psnr": (psnr_mean, psnr_std),
        "ssim": (ssim_mean, ssim_std),
    }


def main() -> int:
    args = parse_args()
    if args.save_samples is not None:
        args.save_samples.mkdir(parents=True, exist_ok=True)

    model = artifact_removal_model(image_resolution=args.image_size)
    model.load_weights(str(args.weights))

    val_corrupted = args.val_dir / "Corrupted"
    val_original = args.val_dir / "Original"
    if val_original.is_dir() and val_corrupted.is_dir():
        evaluate_pair_dir(
            model, val_original, val_corrupted, "Validation pairs", args.image_size, limit=args.max_val_pairs
        )
        if args.save_samples is not None:
            corr_files = sorted(os.listdir(val_corrupted))[: args.max_val_pairs]
            orig_files = sorted(os.listdir(val_original))[: args.max_val_pairs]
            plot_corrected_samples(
                model,
                val_corrupted, corr_files,
                val_original, orig_files,
                "Validation \u2014 Corrected Samples",
                num_samples=5,
                image_size=args.image_size,
                save_path=args.save_samples / "validation_samples.png",
            )
    else:
        print(f"Skipping validation: missing {val_original} or {val_corrupted}")

    test_original = args.test_dir / "Original"
    test_motion_1 = args.test_dir / "head_motion_1"
    test_motion_2 = args.test_dir / "head_motion_2"
    if test_original.is_dir() and (test_motion_1.is_dir() or test_motion_2.is_dir()):
        results = {}
        if test_motion_1.is_dir():
            results["m1"] = evaluate_pair_dir(
                model, test_original, test_motion_1, "Head Motion 1 (Mild)", args.image_size
            )
        if test_motion_2.is_dir():
            results["m2"] = evaluate_pair_dir(
                model, test_original, test_motion_2, "Head Motion 2 (Severe)", args.image_size
            )

        if "m1" in results and "m2" in results:
            print("\n" + "=" * 60)
            print("Summary: MR-ART Real Motion Test Data")
            print("=" * 60)
            print(f"{'Metric':<12} {'Head Motion 1':>22} {'Head Motion 2':>22}")
            print("-" * 60)
            for key in ("mse", "psnr", "ssim"):
                m1m, m1s = results["m1"][key]
                m2m, m2s = results["m2"][key]
                print(f"{key.upper():<12} {m1m:>10.6f} \u00b1 {m1s:.6f} {m2m:>10.6f} \u00b1 {m2s:.6f}")

        if args.save_samples is not None:
            orig_files = sorted(os.listdir(test_original))
            if test_motion_1.is_dir():
                m1_files = sorted(os.listdir(test_motion_1))
                plot_corrected_samples(
                    model,
                    test_motion_1, m1_files,
                    test_original, orig_files,
                    "Head Motion 1 (Mild) \u2014 Corrected Results",
                    num_samples=5,
                    image_size=args.image_size,
                    save_path=args.save_samples / "test_headmotion1_samples.png",
                )
            if test_motion_2.is_dir():
                m2_files = sorted(os.listdir(test_motion_2))
                plot_corrected_samples(
                    model,
                    test_motion_2, m2_files,
                    test_original, orig_files,
                    "Head Motion 2 (Severe) \u2014 Corrected Results",
                    num_samples=5,
                    image_size=args.image_size,
                    save_path=args.save_samples / "test_headmotion2_samples.png",
                )
    else:
        print(f"Skipping test eval: missing {test_original} or motion folders.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
