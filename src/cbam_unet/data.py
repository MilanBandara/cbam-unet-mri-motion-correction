"""Paired tf.data pipelines for motion artifact correction.

The expected on-disk layout is::

    <root>/
        Original/   *.png   (motion-free targets)
        Corrupted/  *.png   (motion-corrupted inputs)

Filename ordering must be consistent between ``Original`` and ``Corrupted`` so
that the i-th file in each folder forms a paired example. ``synthesize_data.py``
produces a layout that satisfies this contract.
"""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf


def load_image(path: tf.Tensor, image_size: int = 256) -> tf.Tensor:
    """Read a PNG, decode as grayscale, resize to ``image_size``, normalise to [0, 1]."""

    byte_img = tf.io.read_file(path)
    img = tf.io.decode_png(byte_img, channels=1)
    img = tf.image.resize(img, (image_size, image_size))
    return tf.cast(img, tf.float32) / 255.0


def _list_pngs(folder: str | Path) -> tf.data.Dataset:
    pattern = str(Path(folder) / "*.png")
    return tf.data.Dataset.list_files(pattern, shuffle=False)


def make_paired_dataset(
    split_dir: str | Path,
    *,
    image_size: int = 256,
    batch_size: int = 10,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """Build a ``(corrupted, original)`` dataset from one split folder.

    Parameters
    ----------
    split_dir
        Directory containing ``Original/`` and ``Corrupted/`` subfolders.
    image_size
        Side length the images are resized to.
    batch_size
        Mini-batch size.
    shuffle
        If True, the dataset is shuffled with a buffer covering all examples.
    """

    split_dir = Path(split_dir)
    corrupted_dir = split_dir / "Corrupted"
    original_dir = split_dir / "Original"
    if not corrupted_dir.is_dir() or not original_dir.is_dir():
        raise FileNotFoundError(
            f"Expected `Original/` and `Corrupted/` subfolders under {split_dir}"
        )

    corrupted = _list_pngs(corrupted_dir).map(
        lambda p: load_image(p, image_size), num_parallel_calls=tf.data.AUTOTUNE
    )
    original = _list_pngs(original_dir).map(
        lambda p: load_image(p, image_size), num_parallel_calls=tf.data.AUTOTUNE
    )

    paired = tf.data.Dataset.zip((corrupted, original))
    n_examples = paired.cardinality().numpy()
    if shuffle and n_examples > 0:
        paired = paired.shuffle(buffer_size=int(n_examples))
    return paired.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def count_examples(split_dir: str | Path) -> int:
    """Count paired examples available under ``split_dir/Corrupted``."""

    return len(list(Path(split_dir).glob("Corrupted/*.png")))
