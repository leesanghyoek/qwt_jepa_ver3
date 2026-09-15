"""Finite-difference sensitivity of pre-fusion encoder features."""

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


def encoder_sensitivity_loss(
    base_feature: torch.Tensor,
    perturbed_feature: torch.Tensor,
    input_energy: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    if base_feature.shape != perturbed_feature.shape:
        raise ValueError("Base and perturbed features must have identical shapes")
    delta = measurement_normalize(perturbed_feature, eps) - measurement_normalize(base_feature, eps)
    gain = delta.square().flatten(1).mean(dim=1) / input_energy
    if not torch.isfinite(gain).all():
        raise FloatingPointError("Non-finite encoder sensitivity")
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

