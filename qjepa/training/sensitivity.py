"""Directional sensitivity of the pre-fusion encoders.

The previous version probed one Rademacher direction and minimised the gain along
it. That is an isotropic contraction penalty: it asks the encoder to be less
sensitive to *everything*. Restoration wants the opposite of half of that -- the
encoder must stay sharply sensitive to the signal it has to rebuild while going
deaf to the corruption. Minimising total sensitivity pushes toward the collapsed
solution, which is why it only ever survived at a weight of 1e-4, where it did
nothing measurable either way.

What is measured here instead are two gains around the same clean operating
point, along two directions that mean something:

    g_noise  -- the direction the corruption actually moved this sample,
                (noisy - clean), so it is this batch's real noise, not a
                random vector that mostly misses it.
    g_signal -- the high-frequency content of the clean sample, which is the
                part a deblurring decoder has to recover.

The loss is the log ratio

    L = log(g_noise + eps) - log(g_signal + eps)

which is scale free. Collapsing the encoder drives both gains to zero and leaves
the ratio where it was, so unlike the old term this one does not reward collapse
and can be given a weight that actually moves the training. The two gains are
reported separately: the ratio is the number to watch, and it is meaningless
without the two magnitudes beside it.

The base point is the CLEAN input, whose features phase 1 already computes for
the variance/covariance term, so the extra cost is two encoder forwards rather
than three.
"""

from __future__ import annotations

import hashlib

import torch
import torch.nn.functional as F


def measurement_normalize(feature: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    if feature.ndim not in (3, 4):
        raise ValueError("Expected channel-first dense feature")
    moved = feature.float().movedim(1, -1)
    return F.layer_norm(moved, (moved.shape[-1],), weight=None, bias=None, eps=eps).movedim(-1, 1)


def stateless_rademacher(x: torch.Tensor, seed: int, *context: object) -> torch.Tensor:
    raw = "|".join(str(value) for value in (seed, *context)).encode()
    derived = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") >> 1
    generator = torch.Generator(device="cpu").manual_seed(derived)
    bits = torch.randint(0, 2, x.shape, dtype=torch.int8, generator=generator)
    return bits.to(device=x.device, dtype=x.dtype).mul_(2).sub_(1)


def _unit_rms(
    direction: torch.Tensor, minimum_energy: float, fallback: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale each sample to unit RMS and report which samples carried a direction.

    Degenerate rows are real, not hypothetical: ``clean_probability`` leaves 2% of
    frames and 5% of IMU windows uncorrupted, so ``noisy - clean`` is exactly zero
    for them. Raising on those would kill the update; instead they are given a
    fallback so the probe forward stays well defined, and reported as invalid so
    they are dropped from the loss.
    """
    scale = direction.float().square().flatten(1).mean(dim=1).sqrt()
    valid = torch.isfinite(scale) & (scale > minimum_energy)
    shape = (-1,) + (1,) * (direction.ndim - 1)
    if fallback is None:
        fallback = torch.ones_like(direction)
    safe = torch.where(valid.reshape(shape), direction, fallback)
    safe_scale = safe.float().square().flatten(1).mean(dim=1).sqrt().clamp_min(minimum_energy)
    return safe / safe_scale.reshape(shape).to(direction.dtype), valid


def corruption_direction(
    clean: torch.Tensor, noisy: torch.Tensor, *, minimum_energy: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """The direction the corruption actually moved this sample, plus a validity mask."""
    if clean.shape != noisy.shape:
        raise ValueError("Clean and noisy inputs must have identical shapes")
    residual = noisy.float() - clean.float()
    # An uncorrupted sample has no noise direction; fall back to its own detail so
    # the probe is still a sensible step, and let the mask exclude it downstream.
    fallback, _ = detail_direction(clean, minimum_energy=minimum_energy)
    return _unit_rms(residual, minimum_energy, fallback)


def detail_direction(
    clean: torch.Tensor, *, minimum_energy: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """High-frequency content of the clean sample: what restoration has to recover.

    A box lowpass is subtracted rather than a Gaussian so the operator stays cheap
    and has no parameters. Images are blurred over 3x3, IMU over three samples.
    """
    values = clean.float()
    if values.ndim == 4:
        weight = values.new_full((values.shape[1], 1, 3, 3), 1.0 / 9.0)
        low = F.conv2d(F.pad(values, (1, 1, 1, 1), mode="reflect"), weight, groups=values.shape[1])
    elif values.ndim == 3:
        weight = values.new_full((values.shape[1], 1, 3), 1.0 / 3.0)
        low = F.conv1d(F.pad(values, (1, 1), mode="reflect"), weight, groups=values.shape[1])
    else:
        raise ValueError(f"Expected [B,C,H,W] or [B,C,L], got {tuple(values.shape)}")
    return _unit_rms(values - low, minimum_energy)


def _log_ratio(noise_gain: torch.Tensor, signal_gain: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.log(noise_gain + eps) - torch.log(signal_gain + eps)


def make_probe(
    x: torch.Tensor,
    source: str,
    direction: torch.Tensor,
    *,
    image_epsilon: float = 1.0 / 255.0,
    imu_epsilon: float = 0.01,
    alpha: float = 1.0,
    minimum_energy: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Step ``x`` along ``direction`` and report the energy actually realised.

    Images are clamped back into [0,1], so the realised step can be shorter than
    the requested one. The denominator uses the realised energy, never the
    requested one, or a clamped probe would read as more sensitive than it is.
    """
    if source not in ("image", "imu") or direction.shape != x.shape:
        raise ValueError("Invalid finite-difference source or direction")
    epsilon = image_epsilon if source == "image" else imu_epsilon
    proposed = x + alpha * epsilon * direction
    perturbed = proposed.clamp(0.0, 1.0) if source == "image" else proposed
    energy = (perturbed.float() - x.float()).square().flatten(1).mean(dim=1)
    if not torch.isfinite(energy).all() or (energy <= minimum_energy).any():
        raise ValueError("Finite-difference perturbation has invalid energy")
    clipped = float((proposed != perturbed).float().mean()) if source == "image" else 0.0
    return perturbed, energy, clipped


def directional_gain(
    base_feature: torch.Tensor,
    perturbed_feature: torch.Tensor,
    input_energy: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Per-sample ``||dh||^2 / ||dx||^2`` of the normalised pre-fusion feature."""
    if base_feature.shape != perturbed_feature.shape:
        raise ValueError("Base and perturbed features must have identical shapes")
    delta = measurement_normalize(perturbed_feature, eps) - measurement_normalize(base_feature, eps)
    gain = delta.square().flatten(1).mean(dim=1) / input_energy
    if not torch.isfinite(gain).all():
        raise FloatingPointError("Non-finite encoder sensitivity")
    return gain


def sensitivity_ratio_loss(
    noise_gain: torch.Tensor,
    signal_gain: torch.Tensor,
    valid: torch.Tensor | None = None,
    floor_log_ratio: float | None = None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Log ratio of noise sensitivity to signal sensitivity.

    Scale free by construction, so shrinking the whole representation does not
    lower it; only becoming *relatively* deafer to noise than to signal does.

    ``floor_log_ratio`` clamps the term from below. Without it the objective is
    unbounded -- nothing stops the optimiser driving the ratio toward zero forever
    -- and an unbounded auxiliary term competing with JEPA over thousands of
    updates is exactly how a run quietly goes sideways. With it the term stops
    pushing once the encoder is deaf enough, and becomes a guard rail.

    ``valid`` drops samples that had no corruption to measure; see ``_unit_rms``.
    """
    if noise_gain.shape != signal_gain.shape:
        raise ValueError("Both gains must be measured on the same batch")
    if valid is None:
        valid = torch.ones_like(noise_gain, dtype=torch.bool)
    if valid.shape != noise_gain.shape:
        raise ValueError("Validity mask must match the per-sample gains")
    per_sample = _log_ratio(noise_gain, signal_gain, eps)
    if floor_log_ratio is not None:
        per_sample = per_sample.clamp_min(floor_log_ratio)
    count = int(valid.sum())
    if count == 0:
        loss = per_sample.sum() * 0.0
    else:
        loss = per_sample[valid].mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite sensitivity ratio")
    selected = valid if count else torch.ones_like(valid)
    report = {
        "sensitivity_noise_gain": float(noise_gain[selected].mean().detach()),
        "sensitivity_signal_gain": float(signal_gain[selected].mean().detach()),
        "sensitivity_ratio": float(
            (noise_gain[selected] / (signal_gain[selected] + eps)).mean().detach()
        ),
        "sensitivity_valid_fraction": count / max(1, valid.numel()),
    }
    return loss, report


def encoder_sensitivity_loss(
    base_feature: torch.Tensor,
    perturbed_feature: torch.Tensor,
    input_energy: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Legacy isotropic gain, kept so the old behaviour stays runnable for ablation."""
    gain = directional_gain(base_feature, perturbed_feature, input_energy, eps)
    return gain.mean(), gain.detach()


def sensitivity_weight(
    successful_updates: int,
    *,
    start_after: int = 500,
    ramp_updates: int = 1000,
    maximum: float = 1e-4,
) -> float:
    if successful_updates < start_after:
        return 0.0
    return maximum * min((successful_updates - start_after + 1) / max(1, ramp_updates), 1.0)
