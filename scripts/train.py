"""Train MR^2-AttUNet (multi-resolution CBAM U-Net) for MRI motion artifact correction.

Example::

    python scripts/train.py \
        --train-dir data/train \
        --val-dir   data/val \
        --epochs 100 \
        --output-dir checkpoints/

Weights & Biases logging is opt-in (``--wandb-project``); if omitted the run
writes only to ``--output-dir``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import tensorflow as tf
from keras.callbacks import ModelCheckpoint

from cbam_unet import artifact_removal_model
from cbam_unet.callbacks import HistorySaver, SaveWeightsEveryN
from cbam_unet.data import count_examples, make_paired_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-dir", type=Path, default=Path("data/train"))
    p.add_argument("--val-dir", type=Path, default=Path("data/val"))
    p.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--save-every-n", type=int, default=5)
    p.add_argument("--wandb-project", type=str, default=None, help="If set, log to W&B under this project.")
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def configure_gpus() -> tf.distribute.Strategy:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    print(f"Num GPUs Available: {len(gpus)}")
    strategy = tf.distribute.MirroredStrategy()
    print(f"Number of devices: {strategy.num_replicas_in_sync}")
    return strategy


def main() -> int:
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = args.output_dir / "weights"
    results_dir = args.output_dir / "results"
    weights_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    if args.wandb_project is not None:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "image_size": args.image_size,
            },
        )

    strategy = configure_gpus()

    print(f"Train images: {count_examples(args.train_dir)}")
    print(f"Val   images: {count_examples(args.val_dir)}")
    train_data = make_paired_dataset(
        args.train_dir, image_size=args.image_size, batch_size=args.batch_size, shuffle=True
    )
    val_data = make_paired_dataset(
        args.val_dir, image_size=args.image_size, batch_size=args.batch_size, shuffle=True
    )

    with strategy.scope():
        model = artifact_removal_model(image_resolution=args.image_size)
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
        model.compile(optimizer=optimizer, loss="mse", metrics=["accuracy"])

    model.summary()

    history_saver = HistorySaver(results_dir / "train_history.json", wandb_run=wandb_run)
    save_every_n = SaveWeightsEveryN(
        str(weights_dir / "model_epoch_{epoch:02d}.weights.h5"),
        n=args.save_every_n,
        wandb_run=wandb_run,
    )
    checkpoint_last = ModelCheckpoint(
        str(weights_dir / "model_last_.weights.h5"),
        save_best_only=False,
        save_weights_only=True,
        verbose=1,
    )
    checkpoint_best = ModelCheckpoint(
        str(weights_dir / "model_best_.weights.h5"),
        save_best_only=True,
        monitor="val_loss",
        mode="min",
        save_weights_only=True,
        verbose=1,
    )

    callbacks = [history_saver, checkpoint_last, checkpoint_best, save_every_n]
    if wandb_run is not None:
        from wandb.integration.keras import WandbMetricsLogger

        callbacks.append(WandbMetricsLogger())

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,
    )

    model.save(args.output_dir / "cbam_unet.keras")
    with open(results_dir / "final_history.json", "w") as f:
        json.dump({k: [float(v) for v in vs] for k, vs in history.history.items()}, f)

    if wandb_run is not None:
        wandb_run.finish()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
