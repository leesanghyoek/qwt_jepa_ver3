"""Fresh phase-2 decoders driven by ZI and ZU, optionally correcting the input."""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import Stage, Upsample, initialize_trainable, resize


class LatentCoefficientDecoder(nn.Module):
    def __init__(
        self,
        output_channels: int,
        output_size: tuple[int, ...],
        channels: tuple[int, int, int, int] = (32, 64, 96, 128),
        *,
        dim: int,
        groups: int = 8,
        residual: bool = False,
    ) -> None:
        super().__init__()
        c0, c1, c2, c3 = channels
        self.dim = dim
        self.residual = residual
        self.output_size = output_size
        sizes = []
        for divisor in (4, 2, 1):
            sizes.append(tuple(max(1, (value + divisor - 1) // divisor) for value in output_size))
        self.sizes = tuple(sizes)
        self.shuffle2 = Upsample(c3, dim=dim)
        self.up2 = Stage(c3, c2, dim=dim, groups=groups)
        self.shuffle1 = Upsample(c2, dim=dim)
        self.up1 = Stage(c2, c1, dim=dim, groups=groups)
        self.shuffle0 = Upsample(c1, dim=dim)
        self.up0 = Stage(c1, c0, dim=dim, groups=groups)
        conv = nn.Conv1d if dim == 1 else nn.Conv2d
        self.head = conv(c0, output_channels, 3, padding=1)
        initialize_trainable(self)
        if residual:
            # Bat dau o dung identity: update 0 tra lai chinh he so dau vao, nen
            # model khong the te hon input va moi buoc chi co the di len.
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, latent: torch.Tensor, base: torch.Tensor | None = None) -> torch.Tensor:
        x = self.up2(self.shuffle2(latent))
        x = self.up1(self.shuffle1(x))
        x = self.up0(self.shuffle0(x))
        if tuple(x.shape[2:]) != tuple(self.output_size):
            # Luoi khong chia het cho 8; chi con lai phan le sau ba lan nhan doi.
            x = resize(x, self.output_size, dim=self.dim)
        predicted = self.head(x)
        if not self.residual:
            return predicted
        if base is None:
            raise ValueError("Residual decoding needs the input coefficients as base")
        return base + predicted


class LatentDecoders(nn.Module):
    """Predict absolute wavelet coefficients; there are no skips or residual inputs."""

    def __init__(
        self,
        image_coefficient_size: tuple[int, int] = (128, 128),
        imu_coefficient_length: int = 64,
        channels: tuple[int, int, int, int] = (32, 64, 96, 128),
        groups: int = 8,
        residual: bool = False,
    ) -> None:
        super().__init__()
        self.residual = residual
        self.image = LatentCoefficientDecoder(
            48, image_coefficient_size, channels, dim=2, groups=groups, residual=residual
        )
        self.imu = LatentCoefficientDecoder(
            12, (imu_coefficient_length,), channels, dim=1, groups=groups, residual=residual
        )

    def forward(
        self,
        ZI: torch.Tensor,
        ZU: torch.Tensor,
        image_base: torch.Tensor | None = None,
        imu_base: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.image(ZI, image_base), self.imu(ZU, imu_base)

