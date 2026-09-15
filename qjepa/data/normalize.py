"""IMU normalization fitted only on unique clean training timestamps."""

from __future__ import annotations

import torch
import torch.nn as nn

STD_FLOOR = (1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4)


class ImuNormalizer(nn.Module):
    def __init__(
        self,
        mean: torch.Tensor | list[float] | None = None,
        std: torch.Tensor | list[float] | None = None,
        channels: int = 6,
    ) -> None:
        super().__init__()
        floor = torch.tensor(STD_FLOOR[:channels], dtype=torch.float32)
        mu = torch.zeros(channels) if mean is None else torch.as_tensor(mean, dtype=torch.float32)
        sigma = torch.ones(channels) if std is None else torch.as_tensor(std, dtype=torch.float32)
        if mu.shape != (channels,) or sigma.shape != (channels,):
            raise ValueError(f"IMU mean/std must each contain {channels} values")
        if not torch.isfinite(mu).all() or not torch.isfinite(sigma).all():
            raise ValueError("IMU normalization contains NaN/Inf")
        self.register_buffer("mean", mu)
        self.register_buffer("scale", torch.maximum(sigma, floor))
        self.register_buffer("fitted", torch.tensor(mean is not None and std is not None))

    def normalize(self, imu_phys: torch.Tensor) -> torch.Tensor:
        if imu_phys.ndim != 3 or imu_phys.shape[1] != self.mean.numel():
            raise ValueError(f"Expected [B,{self.mean.numel()},L], got {tuple(imu_phys.shape)}")
        return (imu_phys - self.mean[None, :, None]) / self.scale[None, :, None]

    def denormalize(self, imu_norm: torch.Tensor) -> torch.Tensor:
        return imu_norm * self.scale[None, :, None] + self.mean[None, :, None]

    def metadata(self) -> dict[str, list[float] | bool]:
        return {
            "mean": self.mean.detach().cpu().tolist(),
            "scale": self.scale.detach().cpu().tolist(),
            "fitted": bool(self.fitted),
        }

