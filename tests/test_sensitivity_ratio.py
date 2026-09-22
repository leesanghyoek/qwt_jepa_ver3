"""The anisotropic sensitivity term, and the property that lets it carry weight.

The old term minimised one gain along a random direction, which is a request to
be less sensitive to everything and therefore a pull toward collapse. It survived
only at a weight of 1e-4, where it did nothing. These tests pin the two claims
that justify the replacement: the loss is scale free, so collapsing cannot lower
it, and it separates noise from signal rather than treating both alike.
"""

from __future__ import annotations

import math

import pytest
import torch

from qjepa.config import build_phase1_model, load_config, seed_everything
from qjepa.data import ImuNormalizer
from qjepa.training.phase1 import Phase1Trainer
from qjepa.training.sensitivity import (
    corruption_direction,
    detail_direction,
    directional_gain,
    sensitivity_ratio_loss,
    sensitivity_weight,
)


def _features(seed: int = 0, shape=(4, 8, 6, 6)) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    base = torch.randn(shape, generator=generator)
    perturbed = base + 0.1 * torch.randn(shape, generator=generator)
    energy = torch.full((shape[0],), 0.01)
    return base, perturbed, energy


def test_direction_helpers_are_unit_rms() -> None:
    clean = torch.rand(3, 3, 16, 16)
    noisy = clean + 0.05 * torch.randn(3, 3, 16, 16)
    for direction, _ in (corruption_direction(clean, noisy), detail_direction(clean)):
        rms = direction.square().flatten(1).mean(dim=1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)


def test_detail_direction_is_high_frequency() -> None:
    """It must pick out the part a deblurring decoder has to rebuild."""
    smooth = torch.linspace(0, 1, 32).reshape(1, 1, 1, 32).expand(2, 3, 32, 32).contiguous()
    textured = smooth + 0.2 * torch.randn(2, 3, 32, 32)
    smooth_response = (smooth - smooth.mean()).abs().mean()
    direction, _ = detail_direction(textured)
    # A ramp has almost no local detail, so the direction must be dominated by the
    # texture rather than by the underlying gradient.
    assert float(direction.abs().mean()) > float(smooth_response) * 0.1


def test_detail_direction_handles_imu_rank_three() -> None:
    imu = torch.randn(2, 6, 128)
    direction, valid = detail_direction(imu)
    assert direction.shape == imu.shape
    assert bool(valid.all())


def test_ratio_loss_is_scale_free() -> None:
    """The whole point: shrinking the representation must not lower the loss.

    This is what the old isotropic penalty could not offer, and why it could never
    be given a weight that mattered.
    """
    base, perturbed, energy = _features()
    noise_gain = directional_gain(base, perturbed, energy)
    signal_gain = directional_gain(base, base + 3.0 * (perturbed - base), energy)
    reference, _ = sensitivity_ratio_loss(noise_gain, signal_gain)
    for factor in (0.01, 0.1, 10.0, 100.0):
        scaled, _ = sensitivity_ratio_loss(noise_gain * factor, signal_gain * factor)
        assert float(scaled) == pytest.approx(float(reference), abs=1e-4)


def test_ratio_loss_falls_when_noise_sensitivity_falls() -> None:
    base, perturbed, energy = _features()
    signal_gain = directional_gain(base, perturbed, energy)
    deaf, _ = sensitivity_ratio_loss(signal_gain * 0.1, signal_gain)
    even, _ = sensitivity_ratio_loss(signal_gain, signal_gain)
    sensitive, _ = sensitivity_ratio_loss(signal_gain * 10.0, signal_gain)
    assert float(deaf) < float(even) < float(sensitive)
    assert float(even) == pytest.approx(0.0, abs=1e-5)


def test_report_carries_both_gains_not_just_the_ratio() -> None:
    """A ratio with no magnitudes beside it cannot be read; collapse looks fine."""
    base, perturbed, energy = _features()
    gain = directional_gain(base, perturbed, energy)
    _, report = sensitivity_ratio_loss(gain, gain * 2.0)
    assert set(report) == {
        "sensitivity_noise_gain", "sensitivity_signal_gain", "sensitivity_ratio",
        "sensitivity_valid_fraction",
    }
    assert report["sensitivity_ratio"] == pytest.approx(0.5, rel=1e-3)


def test_weight_ramp_starts_at_zero_and_saturates() -> None:
    assert sensitivity_weight(0, start_after=500, ramp_updates=1000, maximum=0.05) == 0.0
    assert sensitivity_weight(499, start_after=500, ramp_updates=1000, maximum=0.05) == 0.0
    middle = sensitivity_weight(1000, start_after=500, ramp_updates=1000, maximum=0.05)
    assert 0.0 < middle < 0.05
    assert sensitivity_weight(9999, start_after=500, ramp_updates=1000, maximum=0.05) == 0.05


def test_phase1_step_reports_both_gains_once_the_term_is_live() -> None:
    """Integration: the term must actually run inside a step, not just in isolation."""
    from qjepa.cli import _synthetic_batch

    config = load_config("configs/smoke.yaml")
    config["encoder_sensitivity"]["enabled"] = True
    config["encoder_sensitivity"]["start_after_updates"] = 0
    config["encoder_sensitivity"]["ramp_updates"] = 1
    config["encoder_sensitivity"]["weight_max"] = 0.05
    seed_everything(config["phase1"]["initialization_seed"])
    batch = _synthetic_batch(config)
    model = build_phase1_model(config, ImuNormalizer())
    trainer = Phase1Trainer(model, config, torch.device("cpu"), "synthetic")

    metrics = trainer.step(batch)
    assert metrics["skipped"] is False
    assert metrics["encoder_sensitivity_weight"] > 0
    for key in ("sensitivity_noise_gain", "sensitivity_signal_gain", "sensitivity_ratio"):
        assert key in metrics and metrics[key] > 0.0
    assert metrics["encoder_source"] == "image"

    # The next update alternates to the IMU branch, so both paths are exercised.
    second = trainer.step(batch)
    assert second["encoder_source"] == "imu"
    assert second["sensitivity_signal_gain"] > 0.0


def test_uncorrupted_samples_are_dropped_instead_of_crashing() -> None:
    """clean_probability leaves some samples untouched; that must not kill an update."""
    clean = torch.rand(4, 3, 16, 16)
    noisy = clean.clone()
    noisy[1:] += 0.05 * torch.randn(3, 3, 16, 16)
    direction, valid = corruption_direction(clean, noisy)
    assert valid.tolist() == [False, True, True, True]
    assert torch.isfinite(direction).all()
    rms = direction.square().flatten(1).mean(dim=1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)


def test_fully_clean_batch_yields_a_zero_term_not_an_exception() -> None:
    gains = torch.full((3,), 2.0)
    loss, report = sensitivity_ratio_loss(gains, gains, torch.zeros(3, dtype=torch.bool))
    assert float(loss) == 0.0
    assert report["sensitivity_valid_fraction"] == 0.0


def test_floor_stops_the_term_pushing_once_the_ratio_is_good_enough() -> None:
    """Without a floor the objective is unbounded below and can run away."""
    signal = torch.full((4,), 1.0)
    floor = math.log(0.02)
    deaf, _ = sensitivity_ratio_loss(signal * 1e-4, signal, floor_log_ratio=floor)
    deafer, _ = sensitivity_ratio_loss(signal * 1e-8, signal, floor_log_ratio=floor)
    assert float(deaf) == pytest.approx(floor, abs=1e-6)
    assert float(deafer) == pytest.approx(floor, abs=1e-6)
    above, _ = sensitivity_ratio_loss(signal * 0.5, signal, floor_log_ratio=floor)
    assert float(above) > floor
