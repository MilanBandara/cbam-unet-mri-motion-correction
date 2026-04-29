"""Image-quality metrics used to evaluate motion artifact correction.

All metrics operate on single-channel images normalised to ``[0, 1]`` and
return Python floats so they're cheap to aggregate across many examples.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import tensorflow as tf


def mse(reference: np.ndarray, prediction: np.ndarray) -> float:
    """Mean squared error between two same-shape arrays in ``[0, 1]``."""

    return float(np.mean((reference - prediction) ** 2))


def psnr(reference: np.ndarray, prediction: np.ndarray) -> float:
    """Peak signal-to-noise ratio (dB) for inputs in ``[0, 1]``.

    Returns ``+inf`` when the images are identical.
    """

    err = mse(reference, prediction)
    if err <= 0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / err))


def ssim(reference: np.ndarray, prediction: np.ndarray) -> float:
    """Structural similarity for 4-D ``(N, H, W, C)`` tensors in ``[0, 1]``."""

    ref = tf.constant(reference, dtype=tf.float32)
    pred = tf.constant(prediction, dtype=tf.float32)
    return float(tf.image.ssim(ref, pred, max_val=1.0).numpy()[0])


def summarise(values: Iterable[float]) -> tuple[float, float]:
    """Return ``(mean, std)`` for a finite iterable of floats."""

    arr = np.asarray(list(values), dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr))
