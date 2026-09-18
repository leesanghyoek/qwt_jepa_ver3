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


def detail_band_l1(predicted: torch.Tensor, target: torch.Tensor, dim: int) -> torch.Tensor:
    """L1 tren rieng cac bang chi tiet — tuc duong net.

    QWT xep he so thanh [mau, bang, thanh phan] roi dep thanh 48 kenh; bang 0 la
    LL con bang 1..3 la LH/HL/HH. Haar cho IMU xep approx truoc, detail sau.
    """
    if predicted.shape != target.shape:
        raise ValueError(f"Coefficients {tuple(predicted.shape)} != {tuple(target.shape)}")
    if dim == 2:
        batch, channels, height, width = predicted.shape
        if channels % 16:
            raise ValueError(f"Expected 16 bands*components per colour, got {channels}")
        shape = (batch, channels // 16, 4, 4, height, width)
        return F.l1_loss(predicted.reshape(shape)[:, :, 1:], target.reshape(shape)[:, :, 1:])
    half = predicted.shape[1] // 2
    return F.l1_loss(predicted[:, half:], target[:, half:])


def phase2_reconstruction_loss(
    image_restored: torch.Tensor,
    image_clean: torch.Tensor,
    imu_restored_normalized: torch.Tensor,
    imu_clean_normalized: torch.Tensor,
    beta: float = 1.0,
    image_coefficients: torch.Tensor | None = None,
    image_coefficient_target: torch.Tensor | None = None,
    imu_coefficients: torch.Tensor | None = None,
    imu_coefficient_target: torch.Tensor | None = None,
    detail_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    image = F.l1_loss(image_restored, image_clean)
    accel = F.smooth_l1_loss(
        imu_restored_normalized[:, :3], imu_clean_normalized[:, :3], beta=beta
    )
    gyro = F.smooth_l1_loss(
        imu_restored_normalized[:, 3:], imu_clean_normalized[:, 3:], beta=beta
    )
    total = image + 0.5 * (accel + gyro)
    parts = {"image_l1": image, "imu_accel_smooth_l1": accel, "imu_gyro_smooth_l1": gyro}
    if detail_weight > 0 and image_coefficients is not None:
        # L1 tren pixel toi uu ve trung vi co dieu kien, ma nghiem do chinh la
        # anh mo. So hang nay phat rieng phan duong net bi mat.
        image_detail = detail_band_l1(image_coefficients, image_coefficient_target, dim=2)
        imu_detail = detail_band_l1(imu_coefficients, imu_coefficient_target, dim=1)
        total = total + detail_weight * (image_detail + 0.5 * imu_detail)
        parts["image_detail_l1"] = image_detail
        parts["imu_detail_l1"] = imu_detail
    return total, parts



def phase1_reconstruction_loss(
    image_predicted: torch.Tensor,
    image_target: torch.Tensor,
    imu_predicted: torch.Tensor,
    imu_target: torch.Tensor,
    detail_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Neo latent vao he so clean that, voi trong so them cho bang chi tiet.

    Day la so hang duy nhat trong phase 1 khong tu quy chieu: JEPA so bieu dien
    hoc duoc voi teacher EMA cua chinh no, con variance/covariance chi la thong
    ke cua bieu dien. Khong co so hang nay, ca he co the troi ve mot bieu dien
    tho ma loss van giam.

    QWT xep he so thanh [mau, bang, thanh phan] roi dep thanh 48 kenh; bang 0 la
    LL con bang 1..3 la LH/HL/HH, tuc duong net. Haar cho IMU xep approx truoc,
    detail sau. Ca hai deu duoc can them o phan chi tiet.
    """
    if image_predicted.shape != image_target.shape:
        raise ValueError(f"Image coefficients {tuple(image_predicted.shape)} != {tuple(image_target.shape)}")
    if imu_predicted.shape != imu_target.shape:
        raise ValueError(f"IMU coefficients {tuple(imu_predicted.shape)} != {tuple(imu_target.shape)}")
    image = F.l1_loss(image_predicted, image_target)
    image_detail = detail_band_l1(image_predicted, image_target, dim=2)
    imu = F.l1_loss(imu_predicted, imu_target)
    imu_detail = detail_band_l1(imu_predicted, imu_target, dim=1)
    total = (image + detail_weight * image_detail) + 0.5 * (imu + detail_weight * imu_detail)
    return total, {
        "reconstruction_image": image,
        "reconstruction_image_detail": image_detail,
        "reconstruction_imu": imu,
        "reconstruction_imu_detail": imu_detail,
    }
