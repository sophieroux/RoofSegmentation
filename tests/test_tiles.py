"""Validity mask, RGB order, and D4 alignment."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from roofseg.config import ExperimentConfig
from roofseg.data_loader import ChannelStats, RoofTileDataset
from roofseg.symmetries import DihedralBatch


class TileContracts(unittest.TestCase):
    def test_unobserved_pixels_are_zero_after_standardization(self) -> None:
        config = ExperimentConfig()
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp)
            rgb = np.zeros((16, 16, 4), dtype=np.uint8)
            rgb[:, :, 0] = 200
            rgb[:, :, 3] = 255
            rgb[0, 0, 3] = 0
            rgb[0, 0, 0] = 0
            Image.fromarray(rgb, mode="RGBA").save(image_dir / "1.png")
            stats = ChannelStats(mean=(0.5, 0.5, 0.5), std=(1.0, 1.0, 1.0))
            sample = RoofTileDataset(image_dir, ["1"], stats, config)[0]
            image = sample["image"]
            validity = sample["validity_map"]
            self.assertEqual(tuple(image.shape), (3, 16, 16))
            self.assertEqual(float(image[:, 0, 0].abs().sum()), 0.0)
            self.assertFalse(bool(validity[0, 0, 0]))
            self.assertAlmostEqual(float(image[0, 0, 1]), 200 / 255 - 0.5, places=5)

    def test_d4_moves_image_mask_and_validity_together(self) -> None:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        image[1, 2] = (9, 8, 7)
        mask = np.zeros((8, 8), dtype=np.float32)
        mask[1, 2] = 1.0
        validity = np.ones((8, 8), dtype=np.uint8)
        validity[1, 2] = 0
        out_image, out_mask, out_validity = DihedralBatch(41).apply(image, mask, validity, epoch=1, index=0)
        ys, xs = np.nonzero(out_mask == 1.0)
        self.assertEqual(len(ys), 1)
        y, x = int(ys[0]), int(xs[0])
        self.assertEqual(tuple(int(v) for v in out_image[y, x]), (9, 8, 7))
        self.assertEqual(int(out_validity[y, x]), 0)


if __name__ == "__main__":
    unittest.main()
