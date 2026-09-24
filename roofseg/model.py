"""U-Net whose blocks are the same pre-activation residual unit as the
population encoder from my Msc PopulationModelingFramework project, just with convolutions instead of linears.
"""

from __future__ import annotations

import torch
from torch import nn


class ConvResBlock(nn.Module):
    """BN -> ReLU -> Conv, twice, plus a residual.

    Same order as FullyConnectedResNetBlock. The 1x1 projection is only
    there when the channel count changes, so the skip is identity inside
    a stage.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = None
        if in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.net = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.proj is None else self.proj(x)
        return self.net(x) + identity


class UNet(nn.Module):
    """Four pooling stages. Input H and W must be divisible by 16.

    Bilinear upsampling, not a transposed convolution: a roof edge is a
    pixel or two wide and the checkerboard from ConvTranspose shows up
    on it. The head emits one logit per pixel; the sigmoid stays in the
    likelihood.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        self.enc1 = ConvResBlock(in_channels, c)
        self.enc2 = ConvResBlock(c, c * 2)
        self.enc3 = ConvResBlock(c * 2, c * 4)
        self.enc4 = ConvResBlock(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvResBlock(c * 8, c * 16)

        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec4 = ConvResBlock(c * 16 + c * 8, c * 8)
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = ConvResBlock(c * 8 + c * 4, c * 4)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = ConvResBlock(c * 4 + c * 2, c * 2)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = ConvResBlock(c * 2 + c, c)
        self.head = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % 16 or x.shape[-2] % 16:
            raise ValueError(f"spatial size {tuple(x.shape[-2:])} is not divisible by 16")
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
