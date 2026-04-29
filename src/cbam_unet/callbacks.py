"""Custom Keras callbacks used during training.

* ``HistorySaver``    streams ``model.fit`` logs to a JSON file after every epoch.
* ``SaveWeightsEveryN`` snapshots the model weights every ``n`` epochs.

Both optionally mirror their artefacts to Weights & Biases when ``wandb_run`` is
provided.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from keras.callbacks import Callback


class HistorySaver(Callback):
    """Persist a running training history to ``filename`` (JSON) each epoch."""

    def __init__(self, filename: str | Path, wandb_run: Any | None = None):
        super().__init__()
        self.filename = str(filename)
        self.history: dict[str, list[float]] = {}
        self._wandb = wandb_run

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs = logs or {}
        for key, value in logs.items():
            self.history.setdefault(key, []).append(float(value))
        Path(self.filename).parent.mkdir(parents=True, exist_ok=True)
        with open(self.filename, "w") as f:
            json.dump(self.history, f)
        if self._wandb is not None:
            try:
                self._wandb.save(self.filename, policy="live")
            except Exception:
                pass


class SaveWeightsEveryN(Callback):
    """Save model weights every ``n`` epochs to ``filepath`` (Keras format string)."""

    def __init__(self, filepath: str, n: int = 5, wandb_run: Any | None = None):
        super().__init__()
        self.filepath = filepath
        self.n = n
        self._wandb = wandb_run

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        if (epoch + 1) % self.n != 0:
            return
        out_path = self.filepath.format(epoch=epoch + 1)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(out_path)
        print(f"\nEpoch {epoch + 1}: saving model to {out_path}")
        if self._wandb is not None:
            try:
                self._wandb.save(out_path)
            except Exception:
                pass
