"""Stochastic k-space motion artifact synthesis.

This module implements the physics-informed motion simulation used to
generate paired ``(motion-free, motion-corrupted)`` training data from the
MR-ART repository, as described in the accompanying paper.

The key entry points are:

* :func:`create_artifact` — corrupt a single 2-D image in-memory.
* :func:`generate_split`  — corrupt every PNG in a folder and write paired
  outputs to disk, mirroring the structure consumed by ``cbam_unet.data``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from scipy.fft import fft2, fftshift, ifft2, ifftshift
from tqdm.auto import tqdm


def add_rotation(angle: float, image: np.ndarray) -> np.ndarray:
    """Rotate an image about its centre by ``angle`` degrees."""

    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rotation_matrix, (width, height))


def add_translation(shift_x: float, shift_y: float, image: np.ndarray) -> np.ndarray:
    """Translate an image by ``(shift_x, shift_y)`` pixels."""

    height, width = image.shape[:2]
    translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    return cv2.warpAffine(image, translation_matrix, (width, height))


def fourier_transform(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Centred 2-D FFT. Returns ``(log-magnitude, complex spectrum)``."""

    if image.ndim == 3:
        image = np.mean(image, axis=-1)
    fft_result = fft2(image)
    fft_result_shifted = fftshift(fft_result)
    magnitude_spectrum = np.log(np.abs(fft_result_shifted) + 1)
    return magnitude_spectrum, fft_result_shifted


def inverse_fourier(fft_result_shifted: np.ndarray) -> np.ndarray:
    """Undo :func:`fourier_transform` and return the magnitude image."""

    fft_result_unshifted = ifftshift(fft_result_shifted)
    return np.abs(ifft2(fft_result_unshifted))


def create_artifact(
    image: np.ndarray,
    *,
    motion_type: str = "both",
    shift_x: float = 5,
    shift_y: float = 0,
    angle: float = 7,
    number_of_lines: int = 100,
    width: int = 1,
    region: str = "random",
) -> np.ndarray:
    """Apply a motion artifact to a 2-D float image and return the corrupted result.

    Parameters
    ----------
    image
        2-D numpy array, values in ``[0, 1]``.
    motion_type
        One of ``"translation"``, ``"rotation"``, or ``"both"``.
    shift_x, shift_y
        Translation in pixels. Used when ``motion_type`` is ``"translation"``
        or ``"both"``.
    angle
        Rotation angle in degrees. Used when ``motion_type`` is ``"rotation"``
        or ``"both"``.
    number_of_lines
        Number of phase-encoding columns swapped in k-space.
    width
        Width (in columns) of each swapped band.
    region
        Which k-space columns may be replaced:

        * ``"random"``     uniformly random across all columns
        * ``"central"``    only the central 30% of k-space
        * ``"peripheral"`` only the outer 70% of k-space
        * ``"mixed"``      70% peripheral + 30% central
    """

    if motion_type == "translation":
        moved_image = add_translation(shift_x, shift_y, image)
    elif motion_type == "rotation":
        moved_image = add_rotation(angle, image)
    elif motion_type == "both":
        moved_image = add_translation(shift_x, shift_y, image)
        moved_image = add_rotation(angle, moved_image)
    else:
        raise ValueError(f"Unknown motion_type: {motion_type}")

    _, fft_original = fourier_transform(image)
    _, fft_moved = fourier_transform(moved_image)

    n_cols = image.shape[1]
    centre = n_cols // 2
    band = int(n_cols * 0.15)  # half-width of the central 30%

    if region == "random":
        line_indexes = np.random.randint(0, n_cols - 2, number_of_lines)
    elif region == "central":
        line_indexes = np.random.randint(centre - band, centre + band, number_of_lines)
    elif region == "peripheral":
        peripheral_indices = np.concatenate(
            [np.arange(0, centre - band), np.arange(centre + band, n_cols - 2)]
        )
        line_indexes = np.random.choice(peripheral_indices, size=number_of_lines)
    elif region == "mixed":
        n_periph = int(number_of_lines * 0.7)
        n_cent = number_of_lines - n_periph
        peripheral_indices = np.concatenate(
            [np.arange(0, centre - band), np.arange(centre + band, n_cols - 2)]
        )
        periph_lines = np.random.choice(peripheral_indices, size=n_periph)
        cent_lines = np.random.randint(centre - band, centre + band, n_cent)
        line_indexes = np.concatenate([periph_lines, cent_lines])
    else:
        raise ValueError(f"Unknown region: {region}")

    motion_corrupted = fft_original.copy()
    for i in line_indexes:
        motion_corrupted[:, i : i + width] = fft_moved[:, i : i + width]

    return inverse_fourier(motion_corrupted)


def generate_split(
    src_dir: str | Path,
    out_dir: str | Path,
    params: dict,
    split_name: str = "split",
) -> tuple[int, int]:
    """Read every PNG in ``src_dir`` and write paired outputs under ``out_dir``.

    The output structure is::

        out_dir/
            Original/   <original-filename>.png
            Corrupted/  <stem>_corrupted.png

    Returns ``(saved, errors)``.
    """

    src_dir = Path(src_dir)
    out_dir = Path(out_dir)
    orig_dir = out_dir / "Original"
    corr_dir = out_dir / "Corrupted"
    orig_dir.mkdir(parents=True, exist_ok=True)
    corr_dir.mkdir(parents=True, exist_ok=True)

    png_files = sorted(p.name for p in src_dir.iterdir() if p.suffix.lower() == ".png")
    print(f"[{split_name}] Found {len(png_files)} PNG files in {src_dir}")

    saved = 0
    errors = 0

    for fname in tqdm(png_files, desc=split_name):
        src_path = src_dir / fname
        img_raw = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
        if img_raw is None:
            print(f"  Warning: could not read {fname}, skipping.")
            errors += 1
            continue

        img = img_raw.astype(np.float32) / 255.0
        corrupted = create_artifact(img, **params)

        orig_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        denom = corrupted.max() if corrupted.max() > 0 else 1.0
        corr_uint8 = (np.clip(corrupted / denom, 0, 1) * 255).astype(np.uint8)

        base_name = os.path.splitext(fname)[0]
        cv2.imwrite(str(orig_dir / fname), orig_uint8)
        cv2.imwrite(str(corr_dir / f"{base_name}_corrupted.png"), corr_uint8)
        saved += 1

    print(f"[{split_name}] Done. Saved: {saved}  Errors: {errors}")
    return saved, errors
