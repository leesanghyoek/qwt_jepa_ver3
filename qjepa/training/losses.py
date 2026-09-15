"""Losses for the strictly separated latent and restoration phases."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..models.predictors import image_tokens, imu_tokens


def layer_norm_no_affine(tokens: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return F.layer_norm(tokens, (tokens.shape[-1],), weight=None, bias=None, eps=eps)


def jepa_latent_loss(
    predicted_image: torch.Tensor,
    predicted_imu: torch.Tensor,
    target_image_dense: torch.Tensor,
    target_imu_dense: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_image = image_tokens(target_image_dense).detach()
    target_imu = imu_tokens(target_imu_dense).detach()
    image_term = F.smooth_l1_loss(
        layer_norm_no_affine(predicted_image), layer_norm_no_affine(target_image), beta=beta
    )
    imu_term = F.smooth_l1_loss(
        layer_norm_no_affine(predicted_imu), layer_norm_no_affine(target_imu), beta=beta
    )
    return 0.5 * (image_term + imu_term), image_term, imu_term


def dense_positions(feature: torch.Tensor, indices: torch.Tensor | None = None) -> torch.Tensor:
    """Convert channel-first dense features to aligned [B,K,D] positions."""
    if feature.ndim == 4:
        tokens = image_tokens(feature)
    elif feature.ndim == 3:
        tokens = imu_tokens(feature)
    else:
        raise ValueError(f"Expected [B,D,H,W] or [B,D,L], got {tuple(feature.shape)}")
    return tokens if indices is None else tokens.index_select(1, indices.to(tokens.device))


def variance_covariance_loss(
    positions: torch.Tensor,
    gamma: float = 1.0,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """VICReg-style statistics over B at the same spatial/temporal position."""
    if positions.ndim != 3 or positions.shape[0] < 2:
        raise ValueError("Expected H[B,K,D] with B >= 2")
    values = positions.float()
    centred = values - values.mean(dim=0, keepdim=True)
    variance = centred.square().sum(dim=0) / (values.shape[0] - 1)
    variance_loss = F.relu(gamma - torch.sqrt(variance + eps)).mean()
    covariance = torch.einsum("bkd,bke->kde", centred, centred) / (values.shape[0] - 1)
    diagonal = torch.eye(values.shape[-1], dtype=torch.bool, device=values.device)[None]
    off_diagonal = covariance.masked_fill(diagonal, 0.0)
    covariance_loss = off_diagonal.square().sum(dim=(-2, -1)).mean() / values.shape[-1]
    return variance_loss, covariance_loss


def phase2_reconstruction_loss(
    image_restored: torch.Tensor,
    image_clean: torch.Tensor,
    imu_restored_normalized: torch.Tensor,
    imu_clean_normalized: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    image = F.l1_loss(image_restored, image_clean)
    accel = F.smooth_l1_loss(
        imu_restored_normalized[:, :3], imu_clean_normalized[:, :3], beta=beta
    )
    gyro = F.smooth_l1_loss(
        imu_restored_normalized[:, 3:], imu_clean_normalized[:, 3:], beta=beta
    )
    total = image + 0.5 * (accel + gyro)
    return total, {"image_l1": image, "imu_accel_smooth_l1": accel, "imu_gyro_smooth_l1": gyro}

