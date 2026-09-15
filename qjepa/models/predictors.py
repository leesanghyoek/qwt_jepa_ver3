from __future__ import annotations

import torch
import torch.nn as nn


def image_tokens(feature: torch.Tensor) -> torch.Tensor:
    return feature.flatten(2).transpose(1, 2)


def imu_tokens(feature: torch.Tensor) -> torch.Tensor:
    return feature.transpose(1, 2)


class LatentPredictor(nn.Module):
    def __init__(self, dim: int = 128, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(tokens)

