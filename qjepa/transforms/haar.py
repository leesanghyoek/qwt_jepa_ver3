"""One-level orthonormal Haar DWT along the IMU time axis."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .layout import TransformLayout, fp32_transform

BACKEND = "haar1d"
REVISION = "1.0.0"
SQRT2 = math.sqrt(2.0)
CHANNEL_ORDER = ("ax", "ay", "az", "gx", "gy", "gz")


class HaarTransform1D(nn.Module):
    def __init__(self, channels: int = 6, levels: int = 1) -> None:
        super().__init__()
        if levels != 1:
            raise ValueError("Only one Haar level is supported")
        self.channels = channels
        self.levels = levels

    @property
    def coeff_channels(self) -> int:
        return 2 * self.channels

    def layout_for(self, shape: tuple[int, int, int]) -> TransformLayout:
        b, c, length = shape
        if c != self.channels or length % 2:
            raise ValueError(f"Expected [B,{self.channels},even L], got {shape}")
        return TransformLayout(
            backend=BACKEND,
            revision=REVISION,
            original_shape=shape,
            coefficient_shape=(b, 2 * c, length // 2),
            levels=1,
            boundary_mode="none_even_length",
            scale_convention="orthonormal_sqrt2",
            band_order=("approx", "detail"),
            channel_order=CHANNEL_ORDER[:c],
        )

    @fp32_transform
    def analysis(self, x: torch.Tensor) -> tuple[torch.Tensor, TransformLayout]:
        if x.ndim != 3:
            raise ValueError(f"Expected [B,C,L], got {tuple(x.shape)}")
        layout = self.layout_for(tuple(x.shape))
        even, odd = x[..., 0::2], x[..., 1::2]
        coeff = torch.cat([(even + odd) / SQRT2, (even - odd) / SQRT2], dim=1)
        return coeff, layout

    @fp32_transform
    def synthesis(self, coeff: torch.Tensor, layout: TransformLayout) -> torch.Tensor:
        layout.require(BACKEND, REVISION)
        b, c, length = layout.original_shape
        if tuple(coeff.shape) != (b, 2 * c, length // 2):
            raise ValueError(f"Coefficient shape {tuple(coeff.shape)} does not match layout")
        approx, detail = coeff[:, :c], coeff[:, c:]
        even = (approx + detail) / SQRT2
        odd = (approx - detail) / SQRT2
        return torch.stack((even, odd), dim=-1).reshape(b, c, length)

