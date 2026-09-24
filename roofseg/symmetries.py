"""Flips and quarter-turns. From straight above, a roof has no up-direction."""

from __future__ import annotations

import os

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import numpy as np


class DihedralD4(A.DualTransform):
    """Flip or not, then turn by k * 90 degrees. Same roof, different placement in the frame."""

    def __init__(self, p: float = 1.0) -> None:
        super().__init__(p=p)

    def get_params(self) -> dict[str, int]:
        return {
            "flip": int(self.random_generator.integers(0, 2)),
            "k": int(self.random_generator.integers(0, 4)),
        }

    def apply(self, img: np.ndarray, flip: int = 0, k: int = 0, **params: object) -> np.ndarray:
        if flip:
            img = np.fliplr(img)
        if k:
            img = np.rot90(img, k)
        return np.ascontiguousarray(img)

    def apply_to_mask(self, mask: np.ndarray, flip: int = 0, k: int = 0, **params: object) -> np.ndarray:
        """Same flip and turn as the image."""
        return self.apply(mask, flip=flip, k=k)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ()


def dihedral_compose(seed: int | None = None) -> A.Compose:
    """One draw, applied to the image, the roof mask, and the validity map together."""
    return A.Compose(
        [DihedralD4(p=1.0)],
        additional_targets={"validity_map": "mask"},
        seed=seed,
    )


class DihedralBatch:
    """New generator per tile, from (seed, epoch, index). Dropping one tile does not change the draw for the next."""

    def __init__(self, seed: int) -> None:
        self.seed = seed

    def apply(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        validity_map: np.ndarray,
        epoch: int,
        index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pipeline = dihedral_compose(self.seed + epoch * 10007 + index)
        out = pipeline(image=image, mask=mask, validity_map=validity_map)
        return out["image"], out["mask"], out["validity_map"]
