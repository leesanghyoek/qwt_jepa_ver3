"""The image detail term must stop rewarding blur.

A controlled A/B showed that giving the phase-2 decoder more of the input (the
encoder skip path) did not move the detail error at all, and a latent-only
decoder hit the same floor: the floor belonged to the loss. These tests pin the
replacement term to the property that removes that floor -- under edge-position
uncertainty its optimum is the true edge strength, not a shrunk one.
"""

from __future__ import annotations

import math

import pytest
import torch

from qjepa.config import build_decoders, build_phase1_model, load_config, seed_everything, validate_config
from qjepa.data import ImuNormalizer
from qjepa.models import RestorationSystem
from qjepa.training.losses import (
    detail_band_l1,
    detail_modulus,
    detail_modulus_l1,
    phase2_reconstruction_loss,
)
from qjepa.training.phase2 import Phase2Trainer
from qjepa.transforms.qwt import QuaternionWaveletTransform2D

QWT = QuaternionWaveletTransform2D(backend="qwt_dualtree_hilbert")
SCALES = (0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5)


def _shift(image: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    height, width = image.shape[-2:]
    fy = torch.fft.fftfreq(height, dtype=torch.float64).reshape(-1, 1)
    fx = torch.fft.fftfreq(width, dtype=torch.float64).reshape(1, -1)
    phase = torch.exp(-2j * math.pi * (fx * dx + fy * dy))
    return torch.fft.ifft2(torch.fft.fft2(image.to(torch.complex128)) * phase).real


def _texture(seed: int, size: int = 64) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    spectrum = torch.fft.fft2(torch.randn(1, 3, size, size, generator=generator, dtype=torch.float64))
    fy = torch.fft.fftfreq(size, dtype=torch.float64).reshape(-1, 1)
    fx = torch.fft.fftfreq(size, dtype=torch.float64).reshape(1, -1)
    radius = (fx.square() + fy.square()).sqrt()
    spectrum = spectrum * ((radius > 0.12) & (radius < 0.45)).to(spectrum.dtype)
    image = torch.fft.ifft2(spectrum).real
    return (image - image.amin()) / (image.amax() - image.amin())


def _scale_detail(coefficients: torch.Tensor, scale: float) -> torch.Tensor:
    batch, channels, height, width = coefficients.shape
    packed = coefficients.reshape(batch, channels // 16, 4, 4, height, width).clone()
    packed[:, :, 1:] *= scale
    return packed.reshape(batch, channels, height, width)


def _preferred_scale(loss, image: torch.Tensor) -> float:
    """The detail strength a decoder trained with `loss` would converge to.

    The truth is the image shifted by an unknown sub-pixel amount -- an edge whose
    exact position the blurry input cannot pin down. Minimising the expected loss
    over that uncertainty is what training does.
    """
    coefficients, _ = QWT.analysis(image.float())
    truths = [QWT.analysis(_shift(image, dx, dy).float())[0]
              for dx in (-0.75, -0.25, 0.25, 0.75) for dy in (-0.75, 0.75)]
    expected = [sum(float(loss(_scale_detail(coefficients, s), t)) for t in truths)
                for s in SCALES]
    return SCALES[expected.index(min(expected))]


def test_modulus_is_zero_for_identical_coefficients_and_positive_otherwise() -> None:
    coefficients, _ = QWT.analysis(torch.rand(2, 3, 32, 32))
    assert float(detail_modulus_l1(coefficients, coefficients)) == pytest.approx(0.0, abs=1e-7)
    assert float(detail_modulus_l1(coefficients * 0.5, coefficients)) > 0.0


def test_modulus_covers_only_the_detail_bands() -> None:
    coefficients, _ = QWT.analysis(torch.rand(1, 3, 32, 32))
    assert detail_modulus(coefficients).shape == (1, 3, 3, 16, 16)
    # Changing the approximation band must not change the detail modulus.
    packed = coefficients.reshape(1, 3, 4, 4, 16, 16).clone()
    packed[:, :, 0] += 5.0
    assert float(detail_modulus_l1(packed.reshape(1, 48, 16, 16), coefficients)) == pytest.approx(0.0, abs=1e-7)


def test_gradient_stays_finite_where_the_modulus_vanishes() -> None:
    """Flat regions have |q| -> 0, where a bare sqrt has an infinite derivative."""
    flat = torch.zeros(1, 48, 8, 8, requires_grad=True)
    target, _ = QWT.analysis(torch.rand(1, 3, 16, 16))
    detail_modulus_l1(flat, target).backward()
    assert torch.isfinite(flat.grad).all()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_coefficient_l1_rewards_blur_under_positional_uncertainty(seed: int) -> None:
    """The failure being fixed, pinned so it cannot be quietly un-diagnosed."""
    coefficient_l1 = lambda p, t: detail_band_l1(p, t, dim=2)
    assert _preferred_scale(coefficient_l1, _texture(seed)) <= 0.5


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_modulus_l1_prefers_the_true_edge_strength(seed: int) -> None:
    """Same uncertainty, same candidates: the modulus term picks the real edges.

    Measured on real TartanAir frames too: 0.5x vs 1.0x with edges uncertain by
    +-0.75 px, and 0.25x vs 1.25x on the actual corrupted input.
    """
    assert 0.9 <= _preferred_scale(detail_modulus_l1, _texture(seed)) <= 1.1


def _loss_kwargs(mode: str, restored: torch.Tensor, clean: torch.Tensor):
    restored_c, _ = QWT.analysis(restored)
    clean_c, _ = QWT.analysis(clean)
    imu = torch.randn(2, 6, 16)
    imu_c = torch.randn(2, 12, 8)
    return dict(
        image_restored=restored, image_clean=clean,
        imu_restored_normalized=imu, imu_clean_normalized=imu,
        image_coefficients=restored_c, image_coefficient_target=clean_c,
        imu_coefficients=imu_c, imu_coefficient_target=imu_c,
        detail_weight=2.0, image_detail_loss=mode,
    )


def test_modulus_mode_still_logs_the_coefficient_term_for_comparison() -> None:
    """Every earlier run reports image_detail_l1; new runs must stay comparable."""
    restored, clean = torch.rand(2, 3, 32, 32), torch.rand(2, 3, 32, 32)
    total, parts = phase2_reconstruction_loss(**_loss_kwargs("modulus", restored, clean))
    assert {"image_detail_l1", "image_detail_modulus_l1"} <= set(parts)
    assert not parts["image_detail_l1"].requires_grad
    _, coefficient_parts = phase2_reconstruction_loss(**_loss_kwargs("coefficient", restored, clean))
    assert "image_detail_modulus_l1" not in coefficient_parts
    assert float(parts["image_detail_l1"]) == pytest.approx(
        float(coefficient_parts["image_detail_l1"]), rel=1e-6)


def test_unknown_mode_is_rejected() -> None:
    restored, clean = torch.rand(2, 3, 32, 32), torch.rand(2, 3, 32, 32)
    with pytest.raises(ValueError, match="image_detail_loss"):
        phase2_reconstruction_loss(**_loss_kwargs("perceptual", restored, clean))
    config = load_config("configs/smoke.yaml")
    config["phase2"]["image_detail_loss"] = "perceptual"
    with pytest.raises(ValueError, match="image_detail_loss"):
        validate_config(config)


def test_configs_written_before_the_term_keep_their_meaning() -> None:
    """A saved config without the key must train exactly as it did when saved."""
    config = load_config("configs/smoke.yaml")
    config["phase2"].pop("image_detail_loss", None)
    assert config["phase2"].get("image_detail_loss", "coefficient") == "coefficient"


def test_phase2_step_runs_with_the_modulus_term() -> None:
    from qjepa.cli import _synthetic_batch

    config = load_config("configs/smoke.yaml")
    config["phase2"]["image_detail_loss"] = "modulus"
    config["phase2"]["reconstruction_detail_weight"] = 2.0
    seed_everything(config["phase1"]["initialization_seed"])
    batch = _synthetic_batch(config)
    model = build_phase1_model(config, ImuNormalizer())
    seed_everything(config["phase2"]["decoder_initialization_seed"])
    system = RestorationSystem(model.backbone, model.normalizer, build_decoders(config))
    trainer = Phase2Trainer(system, config, torch.device("cpu"), "synthetic")
    micro = {k: v[:1] if isinstance(v, torch.Tensor) else v[:1] for k, v in batch.items()}
    metrics = trainer.step([micro] * config["phase2"]["gradient_accumulation"])
    assert metrics["skipped"] is False
    assert metrics["image_detail_modulus_l1"] > 0.0
    assert metrics["image_detail_l1"] > 0.0
    trainer.assert_backbone_frozen()
