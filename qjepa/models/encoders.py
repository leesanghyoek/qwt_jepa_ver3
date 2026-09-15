from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import Stage

DEFAULT_CHANNELS = (32, 64, 96, 128)


class DenseCoefficientEncoder(nn.Module):
    """Four-stage CNN returning only the dense final feature.

    Intermediate tensors remain internal and therefore cannot become decoder
    skip connections accidentally.
    """

    def __init__(
        self,
        in_channels: int,
        channels: tuple[int, int, int, int] = DEFAULT_CHANNELS,
        *,
        dim: int,
        groups: int = 8,
    ) -> None:
        super().__init__()
        c0, c1, c2, c3 = channels
        self.stages = nn.Sequential(
            Stage(in_channels, c0, dim=dim, groups=groups),
            Stage(c0, c1, dim=dim, stride=2, groups=groups),
            Stage(c1, c2, dim=dim, stride=2, groups=groups),
            Stage(c2, c3, dim=dim, stride=2, groups=groups),
        )
        self.out_channels = c3

    def forward(self, coefficients: torch.Tensor) -> torch.Tensor:
        return self.stages(coefficients)

