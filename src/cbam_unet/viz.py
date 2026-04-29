"""Plotting helpers for qualitative inspection and training curves."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np

from cbam_unet.metrics import psnr, ssim


def _read_gray(path: str | Path, image_size: int = 256) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.resize(img, (image_size, image_size))


def plot_corrected_samples(
    model,
    motion_dir: str | Path,
    motion_files: Sequence[str],
    original_dir: str | Path,
    original_files: Sequence[str],
    title: str,
    *,
    num_samples: int = 5,
    image_size: int = 256,
    save_path: str | Path | None = None,
):
    """Plot ``input | model output | ground truth`` rows with per-image metrics."""

    n = min(len(motion_files), len(original_files))
    if n == 0:
        raise ValueError("No paired files supplied.")
    indices = np.linspace(0, n - 1, num_samples, dtype=int)

    fig, axes = plt.subplots(num_samples, 3, figsize=(14, 4.5 * num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(indices):
        motion_img = _read_gray(Path(motion_dir) / motion_files[idx], image_size)
        original_img = _read_gray(Path(original_dir) / original_files[idx], image_size)

        input_tensor = (motion_img.astype(np.float32) / 255.0)[np.newaxis, :, :, np.newaxis]
        ground_truth = (original_img.astype(np.float32) / 255.0)[np.newaxis, :, :, np.newaxis]

        prediction = model.predict(input_tensor, verbose=0)
        predicted_img = prediction[0, :, :, 0]

        p = psnr(ground_truth, prediction)
        s = ssim(ground_truth, prediction)

        axes[row, 0].imshow(motion_img, cmap="gray")
        axes[row, 0].set_title(f"Input (Motion) [{idx}]")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(predicted_img, cmap="gray")
        axes[row, 1].set_title(f"Model Output\nPSNR={p:.2f} | SSIM={s:.4f}")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(original_img, cmap="gray")
        axes[row, 2].set_title(f"Ground Truth [{idx}]")
        axes[row, 2].axis("off")

    plt.suptitle(title, fontsize=16, y=1.01)
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_history(history_path: str | Path, save_path: str | Path | None = None):
    """Plot loss and accuracy curves from a JSON file written by ``HistorySaver``."""

    with open(history_path) as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if "loss" in history:
        axes[0].plot(history["loss"], label="train")
    if "val_loss" in history:
        axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    if "accuracy" in history:
        axes[1].plot(history["accuracy"], label="train")
    if "val_accuracy" in history:
        axes[1].plot(history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
