"""Frozen run schema. Every knob the experiment reads lives here."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class ExperimentConfig:
    """Boundary conditions for one roof-mapping run.

    Spatial size, the alpha cut on missing tiles, and the gray-level cut
    that turns an anti-aliased edge into a binary roof state are part of
    the schema, not literals inside the loader.
    """

    seed: int = 41
    data_dir: Path = Path("data")
    image_dirname: str = "images"
    label_dirname: str = "labels"
    output_dir: Path = Path("outputs")
    checkpoint_dirname: str = "checkpoints"
    prediction_dirname: str = "predictions"
    epochs: int = 40
    patience: int = 8
    min_delta: float = 1e-4
    warmup_epochs: int = 10
    batch_size: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    base_channels: int = 32
    val_fraction: float = 0.2
    num_threads: int = 4
    valid_alpha: int = 128
    mask_threshold: int = 127
    decision_threshold: float = 0.5
    dice_weight: float = 1.0
    spatial_divisor: int = 16
    in_channels: int = 3

    @property
    def image_dir(self) -> Path:
        return self.data_dir / self.image_dirname

    @property
    def label_dir(self) -> Path:
        return self.data_dir / self.label_dirname

    @property
    def checkpoint_dir(self) -> Path:
        return self.output_dir / self.checkpoint_dirname

    @property
    def prediction_dir(self) -> Path:
        return self.output_dir / self.prediction_dirname

    def pin_seeds(self) -> None:
        """Pin NumPy, Python, PyTorch, and cuDNN to ``seed``.

        ``cudnn.benchmark`` is off so convolution algorithms are not
        retuned per host. The same seed then reproduces the same masks.
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.set_num_threads(self.num_threads)
