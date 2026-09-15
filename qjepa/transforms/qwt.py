"""Experimental one-level redundant shifted-db4 transform for RGB.

Each RGB channel is processed independently by four separable db4 trees. The
trees are packed into slots named ``(real, i, j, k)``. Every component is
an orthonormal transform under periodic extension; synthesis averages the four
reconstructions. Packing is RGB x 4 bands x 4 components = 48 real channels.

The paired trees use integer shifts of the same filters. Hilbert-pair/QWT
reference equivalence has NOT been established. The legacy class/backend names
are retained for API compatibility; round-trip tests only verify invertibility.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layout import TransformLayout, fp32_transform

BACKEND = "qwt_dualtree_db4"
REVISION = "1.0.0"
BAND_ORDER = ("approx", "detail_y", "detail_x", "detail_xy")
COMPONENT_ORDER = ("real", "i", "j", "k")
CHANNEL_ORDER = ("R", "G", "B")
DB4_H0 = (
    0.23037781330885523,
    0.7148465705525415,
    0.6308807679295904,
    -0.02798376941698385,
    -0.18703481171888114,
    0.030841381835986965,
    0.032883011666982945,
    -0.010597401784997278,
)


def _qmf(h0: torch.Tensor) -> torch.Tensor:
    sign = torch.tensor([(-1.0) ** n for n in range(h0.numel())], dtype=h0.dtype)
    return sign * h0.flip(0)


def _analysis_1d(x: torch.Tensor, filt: torch.Tensor, offset: int) -> torch.Tensor:
    n, taps = x.shape[-1], filt.numel()
    shape = x.shape
    flat = x.reshape(-1, 1, n)
    needed = offset + taps - 1
    repetitions = math.ceil(needed / n) + 2
    periodic = flat.repeat(1, 1, repetitions)[..., : n + needed]
    if offset:
        periodic = periodic[..., offset:]
    out = F.conv1d(periodic, filt.view(1, 1, taps), stride=2)
    return out.reshape(*shape[:-1], n // 2)


def _synthesis_1d(x: torch.Tensor, filt: torch.Tensor, offset: int, n: int) -> torch.Tensor:
    shape = x.shape
    wide = F.conv_transpose1d(
        x.reshape(-1, 1, shape[-1]), filt.view(1, 1, -1), stride=2
    )
    out = wide.new_zeros(wide.shape[0], 1, n)
    for start in range(0, wide.shape[-1], n):
        chunk = wide[..., start : start + n]
        index = (torch.arange(chunk.shape[-1], device=x.device) + start + offset) % n
        out.index_add_(-1, index, chunk)
    return out.reshape(*shape[:-1], n)


def _analysis_axis(x: torch.Tensor, h0: torch.Tensor, h1: torch.Tensor, offset: int, axis: int):
    if axis == -2:
        x = x.transpose(-1, -2)
    low, high = _analysis_1d(x, h0, offset), _analysis_1d(x, h1, offset)
    if axis == -2:
        low, high = low.transpose(-1, -2), high.transpose(-1, -2)
    return low, high


def _synthesis_axis(
    low: torch.Tensor,
    high: torch.Tensor,
    h0: torch.Tensor,
    h1: torch.Tensor,
    offset: int,
    size: int,
    axis: int,
) -> torch.Tensor:
    if axis == -2:
        low, high = low.transpose(-1, -2), high.transpose(-1, -2)
    out = _synthesis_1d(low, h0, offset, size) + _synthesis_1d(high, h1, offset, size)
    return out.transpose(-1, -2) if axis == -2 else out


class QuaternionWaveletTransform2D(nn.Module):
    def __init__(self, levels: int = 1) -> None:
        super().__init__()
        if levels != 1:
            raise ValueError("Only one QWT level is supported")
        # Keep canonical taps in float64 so numerical reference tests do not
        # inherit float32 rounding. They are cast to the input dtype per call.
        h0 = torch.tensor(DB4_H0, dtype=torch.float64)
        self.register_buffer("h0", h0)
        self.register_buffer("h1", _qmf(h0))
        self.levels = levels

    @property
    def coeff_channels(self) -> int:
        return 48

    def layout_for(self, shape: tuple[int, int, int, int]) -> TransformLayout:
        b, c, h, w = shape
        if c != 3 or h % 2 or w % 2:
            raise ValueError(f"Expected [B,3,even H,even W], got {shape}")
        return TransformLayout(
            backend=BACKEND,
            revision=REVISION,
            original_shape=shape,
            coefficient_shape=(b, 48, h // 2, w // 2),
            levels=1,
            boundary_mode="periodic",
            scale_convention="orthonormal_db4_mean_of_four_trees",
            band_order=BAND_ORDER,
            component_order=COMPONENT_ORDER,
            channel_order=CHANNEL_ORDER,
            extra={"filter": "db4", "tree_shift_samples": 1},
        )

    @fp32_transform
    def analysis(self, x: torch.Tensor) -> tuple[torch.Tensor, TransformLayout]:
        if x.ndim != 4:
            raise ValueError(f"Expected [B,3,H,W], got {tuple(x.shape)}")
        layout = self.layout_for(tuple(x.shape))
        b, _, h, w = x.shape
        h0, h1 = self.h0.to(x.dtype), self.h1.to(x.dtype)
        out = x.new_empty(b, 3, 4, 4, h // 2, w // 2)
        for tree_x, offset_x in enumerate((0, 1)):
            low_x, high_x = _analysis_axis(x, h0, h1, offset_x, -1)
            for tree_y, offset_y in enumerate((0, 1)):
                ll, lh = _analysis_axis(low_x, h0, h1, offset_y, -2)
                hl, hh = _analysis_axis(high_x, h0, h1, offset_y, -2)
                component = tree_x + 2 * tree_y
                out[:, :, 0, component] = ll
                out[:, :, 1, component] = lh
                out[:, :, 2, component] = hl
                out[:, :, 3, component] = hh
        return out.reshape(b, 48, h // 2, w // 2), layout

    @fp32_transform
    def synthesis(self, coeff: torch.Tensor, layout: TransformLayout) -> torch.Tensor:
        layout.require(BACKEND, REVISION)
        b, c, h, w = layout.original_shape
        if c != 3 or tuple(coeff.shape) != (b, 48, h // 2, w // 2):
            raise ValueError(f"Coefficient shape {tuple(coeff.shape)} does not match layout")
        h0, h1 = self.h0.to(coeff.dtype), self.h1.to(coeff.dtype)
        packed = coeff.reshape(b, 3, 4, 4, h // 2, w // 2)
        result = torch.zeros((b, 3, h, w), dtype=coeff.dtype, device=coeff.device)
        for tree_x, offset_x in enumerate((0, 1)):
            for tree_y, offset_y in enumerate((0, 1)):
                component = tree_x + 2 * tree_y
                ll = packed[:, :, 0, component]
                lh = packed[:, :, 1, component]
                hl = packed[:, :, 2, component]
                hh = packed[:, :, 3, component]
                low_x = _synthesis_axis(ll, lh, h0, h1, offset_y, h, -2)
                high_x = _synthesis_axis(hl, hh, h0, h1, offset_y, h, -2)
                result.add_(_synthesis_axis(low_x, high_x, h0, h1, offset_x, w, -1))
        return result / 4.0
