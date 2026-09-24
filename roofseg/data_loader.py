"""Load tiles. Soft roof drawings become a hard mask."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from roofseg.config import ExperimentConfig

Symmetry = Callable[
    [np.ndarray, np.ndarray, np.ndarray, int, int],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]


@dataclass(frozen=True)
class ChannelStats:
    """Mean and std of observed RGB on the training ids only."""

    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    def as_dict(self) -> dict[str, list[float]]:
        return {"mean": list(self.mean), "std": list(self.std)}

    @classmethod
    def from_dict(cls, payload: dict[str, list[float]]) -> ChannelStats:
        return cls(mean=tuple(payload["mean"]), std=tuple(payload["std"]))


def list_ids(directory: Path) -> list[str]:
    return sorted((path.stem for path in directory.glob("*.png")), key=lambda stem: int(stem))


def split_ids(labeled_ids: list[str], val_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    """Hold out whole tiles, so one roof is not on both sides of the fit."""
    rng = np.random.default_rng(seed)
    order = np.array(labeled_ids)
    rng.shuffle(order)
    n_val = max(1, int(round(len(order) * val_fraction)))
    val_ids = sorted(order[:n_val].tolist(), key=lambda stem: int(stem))
    train_ids = sorted(order[n_val:].tolist(), key=lambda stem: int(stem))
    return train_ids, val_ids


def load_rgb(path: Path, valid_alpha: int) -> tuple[np.ndarray, np.ndarray]:
    """RGB and the observed-pixel mask. Alpha below the cut is a missing tile, black in RGB too."""
    rgba = np.array(Image.open(path).convert("RGBA"))
    validity_map = rgba[..., 3] >= valid_alpha
    return rgba[..., :3], validity_map


def discrete_mask(path: Path, threshold: int) -> np.ndarray:
    """Roof where gray is above the cut. The label edges are anti-aliased, so the fringe has to be assigned."""
    gray = np.array(Image.open(path).convert("L"))
    return (gray > threshold).astype(np.float32)


def channel_stats(image_dir: Path, ids: list[str], valid_alpha: int) -> ChannelStats:
    """Mean and std of observed RGB pixels, in [0, 1]."""
    total = np.zeros(3, dtype=np.float64)
    total_sq = np.zeros(3, dtype=np.float64)
    count = 0
    for image_id in ids:
        rgb, validity_map = load_rgb(image_dir / f"{image_id}.png", valid_alpha)
        pixels = rgb[validity_map].astype(np.float64) / 255.0
        total += pixels.sum(axis=0)
        total_sq += np.square(pixels).sum(axis=0)
        count += int(pixels.shape[0])
    if count == 0:
        raise ValueError("no observed pixels for channel statistics")
    mean = total / count
    variance = np.clip(total_sq / count - np.square(mean), 1e-8, None)
    return ChannelStats(
        mean=tuple(float(value) for value in mean),
        std=tuple(float(value) for value in np.sqrt(variance)),
    )


def occupancy(image_dir: Path, label_dir: Path, ids: list[str], config: ExperimentConfig) -> float:
    """Fraction of observed pixels that are roof. This is the null rate."""
    occupied = 0.0
    count = 0
    for image_id in ids:
        _, validity_map = load_rgb(image_dir / f"{image_id}.png", config.valid_alpha)
        mask = discrete_mask(label_dir / f"{image_id}.png", config.mask_threshold)
        occupied += float(mask[validity_map].sum())
        count += int(validity_map.sum())
    if count == 0:
        raise ValueError("no observed pixels for the null roof rate")
    return occupied / count


class RoofTileDataset(Dataset):
    """One tile, its roof mask, and the observed pixels. Pass a symmetry in if this split should be augmented."""

    def __init__(
        self,
        image_dir: Path,
        ids: list[str],
        stats: ChannelStats,
        config: ExperimentConfig,
        label_dir: Path | None = None,
        symmetry: Symmetry | None = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir) if label_dir is not None else None
        self.ids = list(ids)
        self.stats = stats
        self.config = config
        self.epoch = 0
        self._symmetry = symmetry

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        """Unobserved pixels are 0 after standardization, which is the training-set mean."""
        image_id = self.ids[index]
        rgb, validity_map = load_rgb(self.image_dir / f"{image_id}.png", self.config.valid_alpha)
        if self.label_dir is not None:
            mask = discrete_mask(self.label_dir / f"{image_id}.png", self.config.mask_threshold)
        else:
            mask = np.zeros(rgb.shape[:2], dtype=np.float32)
        if self._symmetry is not None:
            rgb, mask, validity_map = self._symmetry(
                rgb, mask, validity_map.astype(np.uint8), self.epoch, index
            )
            validity_map = validity_map.astype(bool)
        image = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor(self.stats.mean, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(self.stats.std, dtype=torch.float32).view(3, 1, 1)
        observed = torch.from_numpy(np.ascontiguousarray(validity_map)).unsqueeze(0)
        image = (image - mean) / std
        image = image * observed.to(dtype=image.dtype)
        return {
            "image": image,
            "mask": torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0),
            "validity_map": observed,
            "id": image_id,
        }
