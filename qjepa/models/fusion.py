"""Shared gated multimodal fusion while retaining dense spatial/time grids."""

from __future__ import annotations

import torch
import torch.nn as nn


def build_time_metadata(image_time: torch.Tensor, imu_times: torch.Tensor) -> torch.Tensor:
    if image_time.ndim != 1 or imu_times.ndim != 2 or image_time.shape[0] != imu_times.shape[0]:
        raise ValueError("Expected image_time [B] and imu_times [B,L]")
    if imu_times.shape[1] < 2 or not torch.isfinite(imu_times).all() or not torch.isfinite(image_time).all():
        raise ValueError("Need finite camera timestamps and at least two finite IMU timestamps")
    image_time, imu_times = image_time.double(), imu_times.double()
    dt = torch.diff(imu_times, dim=-1)
    span = imu_times[:, -1] - imu_times[:, 0]
    median_dt = dt.median(dim=-1).values
    if (dt <= 0).any() or (span <= 0).any() or (median_dt <= 0).any():
        raise ValueError("IMU timestamps must be finite and strictly increasing")
    if ((image_time < imu_times[:, 0]) | (image_time > imu_times[:, -1])).any():
        raise ValueError("Camera timestamp must lie inside its IMU window")
    offset = image_time - imu_times[:, 0] - 0.5 * span
    return torch.stack(
        (
            offset / span,
            torch.log(span),
            torch.log(median_dt / 0.01),
        ),
        dim=-1,
    ).float()


class SharedGatedFusion(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        hidden: int = 256,
        imu_bins: int = 4,
        metadata_dim: int = 3,
        gate_bias: float = -2.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.metadata_dim = metadata_dim
        self.image_norm = nn.LayerNorm(dim)
        self.imu_pool = nn.AdaptiveAvgPool1d(imu_bins)
        self.imu_project = nn.Linear(dim * imu_bins, dim)
        self.imu_norm = nn.LayerNorm(dim)
        input_dim = 2 * dim + metadata_dim
        self.shared = nn.Sequential(nn.Linear(input_dim, hidden), nn.SiLU(), nn.Linear(hidden, dim))
        self.shared_norm = nn.LayerNorm(dim)
        self.image_gate = nn.Linear(input_dim, dim)
        self.imu_gate = nn.Linear(input_dim, dim)
        self.image_delta = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1), nn.SiLU(), nn.Conv2d(dim, dim, 1)
        )
        self.imu_delta = nn.Sequential(
            nn.Conv1d(2 * dim, dim, 1), nn.SiLU(), nn.Conv1d(dim, dim, 1)
        )
        for gate in (self.image_gate, self.imu_gate):
            nn.init.zeros_(gate.weight)
            nn.init.constant_(gate.bias, gate_bias)

    def forward(
        self, image_feature: torch.Tensor, imu_feature: torch.Tensor, metadata: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if metadata.shape != (image_feature.shape[0], self.metadata_dim):
            raise ValueError(f"Expected metadata [B,{self.metadata_dim}], got {tuple(metadata.shape)}")
        image_summary = self.image_norm(image_feature.mean(dim=(-2, -1)))
        imu_summary = self.imu_norm(self.imu_project(self.imu_pool(imu_feature).flatten(1)))
        inputs = torch.cat((image_summary, imu_summary, metadata), dim=-1)
        shared = self.shared_norm(self.shared(inputs))
        image_shared = shared[:, :, None, None].expand(-1, -1, *image_feature.shape[-2:])
        imu_shared = shared[:, :, None].expand(-1, -1, imu_feature.shape[-1])
        zi = image_feature + torch.sigmoid(self.image_gate(inputs))[:, :, None, None] * self.image_delta(
            torch.cat((image_feature, image_shared), dim=1)
        )
        zu = imu_feature + torch.sigmoid(self.imu_gate(inputs))[:, :, None] * self.imu_delta(
            torch.cat((imu_feature, imu_shared), dim=1)
        )
        return zi, zu
