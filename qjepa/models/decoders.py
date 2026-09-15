"""Fresh phase-2 decoders whose only data-bearing inputs are ZI and ZU."""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import Stage, initialize_trainable, resize


class LatentCoefficientDecoder(nn.Module):
    def __init__(
        self,
        output_channels: int,
        output_size: tuple[int, ...],
        channels: tuple[int, int, int, int] = (32, 64, 96, 128),
        *,
        dim: int,
        groups: int = 8,
    ) -> None:
        super().__init__()
        c0, c1, c2, c3 = channels
        self.dim = dim
        self.output_size = output_size
        sizes = []
        for divisor in (4, 2, 1):
            sizes.append(tuple(max(1, (value + divisor - 1) // divisor) for value in output_size))
        self.sizes = tuple(sizes)
        self.up2 = Stage(c3, c2, dim=dim, groups=groups)
        self.up1 = Stage(c2, c1, dim=dim, groups=groups)
        self.up0 = Stage(c1, c0, dim=dim, groups=groups)
        conv = nn.Conv1d if dim == 1 else nn.Conv2d
        self.head = conv(c0, output_channels, 3, padding=1)
        initialize_trainable(self)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        x = self.up2(resize(latent, self.sizes[0], dim=self.dim))
        x = self.up1(resize(x, self.sizes[1], dim=self.dim))
        x = self.up0(resize(x, self.sizes[2], dim=self.dim))
        return self.head(x)


class LatentDecoders(nn.Module):
    """Predict absolute wavelet coefficients; there are no skips or residual inputs."""

    def __init__(
        self,
        image_coefficient_size: tuple[int, int] = (128, 128),
        imu_coefficient_length: int = 64,
        channels: tuple[int, int, int, int] = (32, 64, 96, 128),
        groups: int = 8,
    ) -> None:
        super().__init__()
        self.image = LatentCoefficientDecoder(
            48, image_coefficient_size, channels, dim=2, groups=groups
        )
        self.imu = LatentCoefficientDecoder(
            12, (imu_coefficient_length,), channels, dim=1, groups=groups
        )

    def forward(self, ZI: torch.Tensor, ZU: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.image(ZI), self.imu(ZU)

