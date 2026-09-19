"""Configuration loading, validation, and object factories."""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .corruptions import (
    ImuCorruptionConfig,
    LowLightImageCorruptionConfig,
    LowLightImageCorruptor,
    TrajectoryImuCorruptor,
)
from .data.normalize import ImuNormalizer
from .models import LatentDecoders, LatentPretrainingModel, MultimodalBackbone


def _merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("Top-level YAML config must be a mapping")
    parent = raw.pop("extends", None)
    if parent:
        parent_path = (path.parent / parent).resolve()
        config = _merge(load_config(parent_path), raw)
    else:
        config = raw
    config["_config_path"] = str(path)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    if config.get("pipeline_version") != 3:
        raise ValueError("pipeline_version must be exactly 3")
    run_kind = config.get("run_kind", "main")
    if run_kind not in {"main", "smoke"}:
        raise ValueError("run_kind must be main or smoke")
    data = config["data"]
    phase1 = config["phase1"]
    phase2 = config["phase2"]
    model = config["model"]
    if str(config["runtime"].get("gpu_count", "auto")) not in {"auto", "1", "2"}:
        raise ValueError("runtime.gpu_count must be auto, 1 or 2")
    if model.get("image_transform") != "qwt_dualtree_db4" or model.get("imu_transform") != "haar1d":
        raise ValueError("v3 supports qwt_dualtree_db4 for RGB and haar1d for IMU")
    if model.get("time_metadata_dim") != 3 or model.get("imu_summary_bins") != 4:
        raise ValueError("v3 fusion requires three time metadata values and four IMU summary bins")
    if run_kind == "main":
        if data.get("image_size") != [256, 256] or data.get("imu_window") != 128:
            raise ValueError("Main v3 requires one RGB 256x256 frame and an IMU window of 128")
        if phase1.get("batch_size", 0) < max(8, phase1.get("minimum_statistics_batch", 8)):
            raise ValueError("Phase-1 physical batch must meet minimum_statistics_batch")
        if config["monitor"].get("validation_bank_size", 64) < 64:
            raise ValueError("Main validation_bank_size must be at least 64 (or all available samples)")
    if config["monitor"].get("validation_bank_size", 64) < 2:
        raise ValueError("Validation bank needs at least two samples")
    for phase in (phase1, phase2):
        for key in ("batch_size", "gradient_accumulation", "max_successful_updates"):
            if not isinstance(phase.get(key), int) or phase[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
    for key in ("validation_batches", "log_every_updates", "checkpoint_every_updates"):
        if config["runtime"].get(key, 0) < 1:
            raise ValueError(f"runtime.{key} must be positive")
    # Phase 1 khong co duong khoi phuc trong khong gian pixel; chi he so.
    if phase1.get("reconstruction_loss_weight", 0.0) != 0.0:
        raise ValueError("phase1.reconstruction_loss_weight must be 0.0; use the coefficient weight")
    coefficient_weight = phase1.get("coefficient_reconstruction_loss_weight", 0.0)
    if bool(phase1.get("decoder_enabled", False)):
        if coefficient_weight <= 0:
            raise ValueError(
                "phase1.decoder_enabled needs coefficient_reconstruction_loss_weight > 0,"
                " otherwise the decoder trains without steering the latent"
            )
        if not 0.0 <= phase1.get("reconstruction_detail_weight", 0.0):
            raise ValueError("phase1.reconstruction_detail_weight cannot be negative")
    elif coefficient_weight != 0.0:
        raise ValueError("phase1.coefficient_reconstruction_loss_weight needs decoder_enabled")
    if phase1.get("gradient_accumulation", 1) != 1:
        raise ValueError("Phase 1 uses real batch statistics; gradient_accumulation must be 1")
    if not phase1.get("online_clean_forward_for_regularization", False):
        raise ValueError("Phase 1 requires the gradient-enabled clean online branch")
    if phase1.get("variance_weight", 0) <= 0 or phase1.get("covariance_weight", 0) <= 0:
        raise ValueError("Main latent training requires explicit variance and covariance losses")
    if phase1.get("jepa_weight") != 1.0 or phase1.get("precision") != "fp32":
        raise ValueError("Supported phase-1 recipe requires jepa_weight=1 and FP32")
    required_maps = {"FI", "FU", "ZI", "ZU", "FI_clean", "FU_clean", "ZI_clean", "ZU_clean"}
    if set(phase1.get("regularized_maps", ())) != required_maps:
        raise ValueError("phase1.regularized_maps must contain all eight raw feature maps")
    required_phase2 = {"freeze_backbone": True}
    for key, required in required_phase2.items():
        if phase2.get(key) != required:
            raise ValueError(f"phase2.{key} must be {required!r}")
    # decoder_input phai noi that ve viec decoder nhan gi. Bat skip ma van khai
    # "chi latent" la dung loai noi doi ma cac kiem tra o day sinh ra de chan.
    skips = bool(phase2.get("encoder_skips", False))
    expected_input = "latent_plus_encoder_skips" if skips else "fused_dense_latent_only"
    if phase2.get("decoder_input") != expected_input:
        raise ValueError(
            f"phase2.decoder_input must be {expected_input!r} when encoder_skips is {skips}"
        )
    # Phai khai TUONG MINH kieu merge. Neu de mac dinh, mot config cu (khong co
    # khoa nay) van hash giong het mot checkpoint cu, roi lang le dung kien truc
    # moi — va loi chi lo ra o load_state_dict, sau khi da dung sai model.
    if skips and "skip_gating" not in phase2:
        raise ValueError(
            "phase2.encoder_skips needs an explicit phase2.skip_gating so the"
            " configuration hash records which merge the checkpoint was trained with"
        )
    # Hai khoa nay phai noi cung mot chuyen, neu khong config se noi doi ve
    # viec decoder that su lam gi.
    residual = bool(phase2.get("input_coefficient_residual", False))
    expected_output = "input_residual" if residual else "absolute_prediction"
    if phase2.get("output_coefficients") != expected_output:
        raise ValueError(
            f"phase2.output_coefficients must be {expected_output!r} when"
            f" input_coefficient_residual is {residual}"
        )
    # Cung ly do voi skip_gating: de mac dinh thi mot config cu hash giong het mot
    # checkpoint cu roi lang le dung kien truc khac.
    if residual and "residual_sees_input" not in phase2:
        raise ValueError(
            "phase2.input_coefficient_residual needs an explicit"
            " phase2.residual_sees_input so the hash records which head was trained"
        )
    if phase2.get("reconstruction_detail_weight", 0.0) < 0:
        raise ValueError("phase2.reconstruction_detail_weight cannot be negative")
    if phase2.get("imu_variation_weight", 0.0) < 0:
        raise ValueError("phase2.imu_variation_weight cannot be negative")
    if phase2.get("detail_energy_weight", 0.0) < 0:
        raise ValueError("phase2.detail_energy_weight cannot be negative")
    if phase2.get("smooth_l1_beta", 0.0) <= 0:
        raise ValueError("phase2.smooth_l1_beta must be positive")
    if phase2.get("jepa_loss_weight") != 0.0 or phase2.get("sensitivity_loss_weight") != 0.0:
        raise ValueError("Phase 2 cannot optimize latent/Jacobian losses")
    if phase2.get("reconstruction_loss_weight") != 1.0 or phase2.get("precision") != "fp32":
        raise ValueError("Supported phase-2 recipe requires reconstruction weight 1 and FP32")
    if data.get("split_unit") != "trajectory":
        raise ValueError("Data split unit must be trajectory")
    if data.get("minimum_trajectories_per_batch", 0) < 1:
        raise ValueError("minimum_trajectories_per_batch must be positive")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def build_normalizer(metadata: dict[str, Any]) -> ImuNormalizer:
    normalization = metadata["normalization"]
    return ImuNormalizer(normalization["mean"], normalization["std"])


def build_backbone(config: dict[str, Any]) -> MultimodalBackbone:
    model = config["model"]
    channels = tuple(model["encoder_channels"])
    return MultimodalBackbone(
        channels=channels,
        embedding_dim=model["embedding_dim"],
        fusion_hidden=model["fusion_hidden_dim"],
        imu_summary_bins=model["imu_summary_bins"],
        time_metadata_dim=model["time_metadata_dim"],
        gate_bias=model["gate_bias_init"],
        groups=model["groupnorm_groups"],
    )


def build_phase1_model(config: dict[str, Any], normalizer: ImuNormalizer) -> LatentPretrainingModel:
    enabled = bool(config["phase1"].get("decoder_enabled", False))
    return LatentPretrainingModel(
        backbone=build_backbone(config),
        normalizer=normalizer,
        predictor_hidden=config["model"]["predictor_hidden_dim"],
        # Neo phase 1 khong bao gio nhan skip: neu no co duong vong tu encoder thi
        # no thoa man duoc neo ma khong ep gi vao latent — dung cai ma neo sinh ra
        # de ngan. Cung ly do voi viec no giu he so tuyet doi thay vi residual.
        decoders=build_decoders(config, residual=False, skips=False) if enabled else None,
    )


def build_decoders(
    config: dict[str, Any], *, residual: bool | None = None, skips: bool | None = None
) -> LatentDecoders:
    image_size = config["data"]["image_size"]
    channels = tuple(config["model"]["encoder_channels"])
    if residual is None:
        residual = bool(config["phase2"].get("input_coefficient_residual", False))
    if skips is None:
        skips = bool(config["phase2"].get("encoder_skips", False))
    return LatentDecoders(
        image_coefficient_size=(image_size[0] // 2, image_size[1] // 2),
        imu_coefficient_length=config["data"]["imu_window"] // 2,
        channels=channels,
        groups=config["model"]["groupnorm_groups"],
        residual=residual,
        # Thu tu tu tho den min, khop voi thu tu encoder tra ve.
        skip_channels=(channels[2], channels[1], channels[0]) if skips else None,
        # .get chu khong phai [...]: ham nay cung doc config NAM TRONG checkpoint,
        # va checkpoint train truoc khi khoa ra doi thi khong co no. "Khong co"
        # nghia la kien truc truoc do, tuc False — dung gia tri tai tao lai dung
        # mang da train. validate_config van bat khai tuong minh cho config MOI,
        # nen khong co duong nao doi kien truc am tham.
        skip_gating=bool(config["phase2"].get("skip_gating", False)) if skips else True,
        sees_input=bool(config["phase2"].get("residual_sees_input", False)) if residual else False,
    )


def build_corruptors(config: dict[str, Any]):
    image_values = config["corruption"]["image"]
    imu_values = config["corruption"]["imu"]
    image_cfg = LowLightImageCorruptionConfig(**image_values)
    imu_cfg = ImuCorruptionConfig(**imu_values)
    seed = config["data"]["corruption_seed"]
    return LowLightImageCorruptor(image_cfg, seed), TrajectoryImuCorruptor(imu_cfg, seed)


def serializable_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}
