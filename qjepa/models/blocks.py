from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv(dim: int):
    if dim == 1:
        return nn.Conv1d
    if dim == 2:
        return nn.Conv2d
    raise ValueError(f"dim must be 1 or 2, got {dim}")


def _group_norm(channels: int, groups: int) -> nn.GroupNorm:
    if channels % groups:
        raise ValueError(f"{channels} channels are not divisible by {groups} groups")
    return nn.GroupNorm(groups, channels)


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, *, dim: int, stride: int = 1, groups: int = 8):
        super().__init__()
        conv = _conv(dim)
        self.net = nn.Sequential(
            conv(cin, cout, 3, stride=stride, padding=1, bias=False),
            _group_norm(cout, groups),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, channels: int, *, dim: int, groups: int = 8):
        super().__init__()
        conv = _conv(dim)
        self.conv1 = conv(channels, channels, 3, padding=1, bias=False)
        self.norm1 = _group_norm(channels, groups)
        self.conv2 = conv(channels, channels, 3, padding=1, bias=False)
        self.norm2 = _group_norm(channels, groups)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.norm1(self.conv1(x)))
        return self.act(x + self.norm2(self.conv2(y)))


class Stage(nn.Module):
    def __init__(self, cin: int, cout: int, *, dim: int, stride: int = 1, groups: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(cin, cout, dim=dim, stride=stride, groups=groups),
            ResBlock(cout, dim=dim, groups=groups),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def resize(x: torch.Tensor, size: tuple[int, ...], *, dim: int) -> torch.Tensor:
    mode = "linear" if dim == 1 else "bilinear"
    return F.interpolate(x, size=size, mode=mode, align_corners=False)


def initialize_trainable(module: nn.Module) -> None:
    """Explicit Kaiming initialization used for fresh phase-2 decoders."""
    for child in module.modules():
        if isinstance(child, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(child.weight, nonlinearity="relu")
            if child.bias is not None:
                nn.init.zeros_(child.bias)

