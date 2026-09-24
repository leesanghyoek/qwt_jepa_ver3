"""Image detail terms must be scored on what reaches the pixels.

The QWT keeps four trees and synthesis averages them, so the decoder's 48
coefficient channels are 4x redundant. Kaggle run p5 lowered its modulus term by
29% while the per-coefficient detail error did not move and the image stayed
soft: on a real frame, a perturbation confined to the null space of synthesis
lowers the modulus term 63%, the coefficient L1 41% and the energy gap 68%, with
the image unchanged to 1e-7. These tests pin that loophole and its closure.
"""

from __future__ import annotations

import pytest
import torch

from qjepa.config import build_decoders, build_phase1_model, load_config, seed_everything, validate_config
from qjepa.data import ImuNormalizer
from qjepa.evaluation.metrics import image_metrics
from qjepa.models import RestorationSystem
from qjepa.training.losses import invisible_detail_fraction, phase2_reconstruction_loss
from qjepa.training.phase2 import Phase2Trainer
from qjepa.transforms.qwt import QuaternionWaveletTransform2D

QWT = QuaternionWaveletTransform2D(backend="qwt_dualtree_hilbert")


def _null_component(coefficients: torch.Tensor, layout) -> torch.Tensor:
    """The part of ``coefficients`` synthesis throws away."""
    return coefficients - QWT.analysis(QWT.synthesis(coefficients, layout))[0]


def _detail_loss_gradient(mode: str, scores_image: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Gradient the image detail terms send to the decoder's coefficients."""
    generator = torch.Generator().manual_seed(0)
    clean = torch.rand(2, 3, 32, 32, generator=generator)
    noisy = (clean + 0.1 * torch.randn(2, 3, 32, 32, generator=generator)).clamp(0, 1)
    target, layout = QWT.analysis(clean)
    start, _ = QWT.analysis(noisy)
    # Stand-in for a decoder output that already carries some invisible energy.
    decoder = (start + _null_component(0.05 * torch.randn(start.shape, generator=generator), layout))
    decoder.requires_grad_(True)
    image = QWT.synthesis(decoder, layout)
    scored = QWT.analysis(image)[0] if scores_image else decoder
    imu = torch.zeros(2, 6, 16)
    imu_c = torch.zeros(2, 12, 8)
    _, parts = phase2_reconstruction_loss(
        image, clean, imu, imu, image_coefficients=scored, image_coefficient_target=target,
        imu_coefficients=imu_c, imu_coefficient_target=imu_c, detail_weight=2.0,
        detail_energy_weight=1.0, image_detail_loss=mode)
    detail = parts["image_detail_modulus_l1" if mode == "modulus" else "image_detail_l1"]
    (detail + parts["image_detail_energy"]).backward()
    return decoder.grad, _null_component(decoder.grad, layout)


@pytest.mark.parametrize("mode", ["coefficient", "modulus"])
def test_decoder_coefficients_leak_gradient_into_the_invisible_part(mode: str) -> None:
    """The loophole p5 used, pinned so it cannot be quietly un-diagnosed.

    Half or more of the gradient (0.50 coefficient, 0.67 modulus, measured)
    pushes energy the image never shows; the closed path below sends ~1e-7.
    """
    gradient, invisible = _detail_loss_gradient(mode, scores_image=False)
    assert float(invisible.norm() / gradient.norm()) > 0.25


@pytest.mark.parametrize("mode", ["coefficient", "modulus"])
def test_restored_image_sends_no_gradient_into_the_invisible_part(mode: str) -> None:
    gradient, invisible = _detail_loss_gradient(mode, scores_image=True)
    assert float(gradient.norm()) > 0.0
    assert float(invisible.norm() / gradient.norm()) < 1e-4


def test_invisible_fraction_is_zero_for_analysed_images_and_grows_with_null_energy() -> None:
    image = torch.rand(1, 3, 32, 32)
    coefficients, layout = QWT.analysis(image)
    visible = QWT.analysis(QWT.synthesis(coefficients, layout))[0]
    assert float(invisible_detail_fraction(coefficients, visible)) < 1e-8
    padded = coefficients + _null_component(torch.randn_like(coefficients), layout)
    assert torch.allclose(QWT.synthesis(padded, layout), image, atol=1e-5)
    visible = QWT.analysis(QWT.synthesis(padded, layout))[0]
    assert 0.5 < float(invisible_detail_fraction(padded, visible)) < 1.0


def test_unknown_source_is_rejected_and_old_configs_keep_their_meaning() -> None:
    config = load_config("configs/smoke.yaml")
    config["phase2"]["image_detail_source"] = "pixels"
    with pytest.raises(ValueError, match="image_detail_source"):
        validate_config(config)
    config["phase2"].pop("image_detail_source")
    assert config["phase2"].get("image_detail_source", "decoder_coefficients") == "decoder_coefficients"


@pytest.mark.parametrize("source", ["decoder_coefficients", "restored_image"])
def test_phase2_step_logs_the_invisible_fraction(source: str) -> None:
    from qjepa.cli import _synthetic_batch

    config = load_config("configs/smoke.yaml")
    config["phase2"].update(image_detail_loss="modulus", reconstruction_detail_weight=2.0,
                            detail_energy_weight=1.0, image_detail_source=source)
    seed_everything(config["phase1"]["initialization_seed"])
    batch = _synthetic_batch(config)
    model = build_phase1_model(config, ImuNormalizer())
    seed_everything(config["phase2"]["decoder_initialization_seed"])
    system = RestorationSystem(model.backbone, model.normalizer, build_decoders(config))
    trainer = Phase2Trainer(system, config, torch.device("cpu"), "synthetic")
    micro = {k: v[:1] for k, v in batch.items()}
    metrics = trainer.step([micro] * config["phase2"]["gradient_accumulation"])
    assert metrics["skipped"] is False
    assert 0.0 <= metrics["image_detail_invisible_fraction"] <= 1.0
    assert metrics["image_detail_modulus_l1"] > 0.0
    trainer.assert_backbone_frozen()


def test_stripe_metric_sees_what_psnr_does_not() -> None:
    """A smooth offset in one detail band is period-2 px stripes in the image.

    PSNR barely registers it; the stripe ratio does, while blur reads as lost edge
    power instead. This is the pattern a phase-blind detail reward buys.
    """
    generator = torch.Generator().manual_seed(0)
    clean = torch.nn.functional.avg_pool2d(torch.rand(1, 3, 128, 128, generator=generator), 5, 1, 2)
    coefficients, layout = QWT.analysis(clean)
    packed = coefficients.reshape(1, 3, 4, 4, 64, 64).clone()
    packed[:, :, 1] += 0.01
    striped = QWT.synthesis(packed.reshape(1, 48, 64, 64), layout)
    blurred = torch.nn.functional.avg_pool2d(clean, 5, 1, 2)

    same, stripes, blur = (image_metrics(x, clean) for x in (clean, striped, blurred))
    assert stripes["image_psnr_db"] > 40.0
    assert stripes["image_stripe_power"] > 2.0 * same["image_stripe_power"]
    assert stripes["image_edge_power"] == pytest.approx(1.0, abs=0.05)
    assert blur["image_edge_power"] < 0.5
    assert blur["image_stripe_power"] < 0.5 * same["image_stripe_power"]
