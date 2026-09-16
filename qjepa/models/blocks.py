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


class Upsample(nn.Module):
    """Sub-pixel convolution: conv mo rong kenh roi trai kenh ra khong gian.

    Noi suy bilinear la bo loc thong thap — no khong the sinh ra tan so cao,
    nen moi chi tiet nho hon o latent deu mat vinh vien du decoder co sau bao
    nhieu. Phep nang nay hoc duoc, va no quyet dinh duong net co song hay khong.
    """

    def __init__(self, channels: int, *, dim: int, factor: int = 2) -> None:
        super().__init__()
        if factor < 1:
            raise ValueError(f"Upsample factor must be positive, got {factor}")
        self.dim = dim
        self.factor = factor
        self.conv = _conv(dim)(channels, channels * factor**dim, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.factor == 1:
            return x
        x = self.conv(x)
        if self.dim == 2:
            return F.pixel_shuffle(x, self.factor)
        batch, channels, length = x.shape
        x = x.reshape(batch, channels // self.factor, self.factor, length)
        return x.permute(0, 1, 3, 2).reshape(batch, channels // self.factor, length * self.factor)


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

