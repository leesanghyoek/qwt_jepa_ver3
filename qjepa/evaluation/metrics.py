from __future__ import annotations

import torch
import torch.nn.functional as F


def _ssim(restored: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """Channel-wise local SSIM with data_range=1."""
    size = min(11, restored.shape[-2], restored.shape[-1])
    if size % 2 == 0:
        size -= 1
    coordinate = torch.arange(size, dtype=restored.dtype, device=restored.device) - size // 2
    sigma = max(0.5, 1.5 * size / 11.0)
    one_dimensional = torch.exp(-coordinate.square() / (2.0 * sigma * sigma))
    one_dimensional /= one_dimensional.sum()
    kernel = torch.outer(one_dimensional, one_dimensional)
    kernel = kernel.expand(restored.shape[1], 1, size, size)
    padding = size // 2
    mean_x = F.conv2d(restored, kernel, padding=padding, groups=restored.shape[1])
    mean_y = F.conv2d(clean, kernel, padding=padding, groups=clean.shape[1])
    variance_x = F.conv2d(restored.square(), kernel, padding=padding, groups=restored.shape[1]) - mean_x.square()
    variance_y = F.conv2d(clean.square(), kernel, padding=padding, groups=clean.shape[1]) - mean_y.square()
    covariance = F.conv2d(restored * clean, kernel, padding=padding, groups=restored.shape[1]) - mean_x * mean_y
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mean_x * mean_y + c1) * (2 * covariance + c2)) / (
        (mean_x.square() + mean_y.square() + c1) * (variance_x + variance_y + c2)
    )
    return score.mean()


def image_metrics(restored: torch.Tensor, clean: torch.Tensor) -> dict[str, float]:
    restored = restored.clamp(0.0, 1.0)
    clean = clean.clamp(0.0, 1.0)
    error = restored - clean
    mae = error.abs().mean()
    mse = error.square().mean()
    psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
    return {
        "image_mae": float(mae),
        "image_psnr_db": float(psnr),
        "image_ssim": float(_ssim(restored, clean)),
    }


def imu_metrics(
    restored_phys: torch.Tensor,
    clean_phys: torch.Tensor,
    timestamps: torch.Tensor | None = None,
) -> dict[str, float | list[float]]:
    error = restored_phys - clean_phys
    rmse_axis = error.square().mean(dim=(0, 2)).sqrt()
    mae_axis = error.abs().mean(dim=(0, 2))
    bias_axis = error.mean(dim=(0, 2))
    metrics: dict[str, float | list[float]] = {
        "imu_rmse_axis": rmse_axis.detach().cpu().tolist(),
        "imu_mae_axis": mae_axis.detach().cpu().tolist(),
        "imu_bias_axis": bias_axis.detach().cpu().tolist(),
        "accel_rmse": float(error[:, :3].square().mean().sqrt()),
        "gyro_rmse": float(error[:, 3:].square().mean().sqrt()),
    }
    if timestamps is not None:
        dt = torch.diff(timestamps, dim=-1)
        if (dt <= 0).any():
            raise ValueError("Timestamps must increase for IMU variation metrics")
        restored_rate = torch.diff(restored_phys, dim=-1) / dt[:, None, :]
        clean_rate = torch.diff(clean_phys, dim=-1) / dt[:, None, :]
        rate_error = restored_rate - clean_rate
        metrics["accel_variation_rmse"] = float(rate_error[:, :3].square().mean().sqrt())
        metrics["gyro_variation_rmse"] = float(rate_error[:, 3:].square().mean().sqrt())
    return metrics


def pooled_effective_rank(feature: torch.Tensor) -> float:
    if feature.ndim == 4:
        pooled = feature.mean(dim=(-2, -1))
    elif feature.ndim == 3:
        pooled = feature.mean(dim=-1)
    else:
        raise ValueError("Expected a dense image or IMU feature")
    pooled = pooled.float() - pooled.float().mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(pooled)
    if singular.sum() <= 1e-12:
        return 0.0
    probability = singular / singular.sum().clamp_min(1e-12)
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum()
    return float(torch.exp(entropy))


def latent_diagnostics(feature: torch.Tensor) -> dict[str, float]:
    tokens = feature.flatten(2).transpose(1, 2)
    same_position_std = tokens.std(dim=0, unbiased=True).mean()
    normalized = F.layer_norm(tokens.float(), (tokens.shape[-1],))
    normalized_feature = normalized.transpose(1, 2).reshape_as(feature)
    return {
        "raw_rms": float(feature.float().square().mean().sqrt()),
        "same_position_std": float(same_position_std),
        "pooled_effective_rank": pooled_effective_rank(feature),
        "normalized_same_position_std": float(normalized.std(dim=0, unbiased=True).mean()),
        "normalized_pooled_effective_rank": pooled_effective_rank(normalized_feature),
    }
