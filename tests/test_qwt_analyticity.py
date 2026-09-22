"""Measure whether the transform is actually a quaternion wavelet transform.

The repository previously tested only invertibility, which every orthogonal
change of basis passes and which says nothing about the analytic property that
gives a QWT its name. These tests measure that property directly, so the claim in
the module docstring is checked rather than asserted.

Two independent measurements:

* ``negative_frequency_energy`` -- the fraction of the complex wavelet's energy
  sitting on negative frequencies. An exactly analytic pair scores 0, a single
  real wavelet scores 0.5. This is a property of the filters alone.
* the quaternion modulus under sub-pixel translation -- for an analytic pair the
  modulus is nearly invariant while the phases carry the displacement. This is
  the property the restoration pipeline actually consumes.
"""

from __future__ import annotations

import math

import pytest
import torch

from qjepa.transforms.qwt import (
    FILTER_BANKS,
    QuaternionWaveletTransform2D,
    quaternion_modulus,
    quaternion_phases,
)

BACKENDS = sorted(FILTER_BANKS)


def negative_frequency_energy(transform: QuaternionWaveletTransform2D, size: int = 2048) -> float:
    """Energy of ``W_A + i W_B`` on negative frequencies, as a fraction of the total.

    The equivalent level-1 analysis wavelet of a tree is its highpass filter placed
    at that tree's offset, so the pair can be read straight off the module.
    """
    trees = transform._trees(torch.float64)
    spectra = []
    for _, h1, offset in trees:
        placed = torch.zeros(size, dtype=torch.float64)
        index = (torch.arange(h1.numel()) + offset) % size
        placed[index] = h1
        spectra.append(torch.fft.fft(placed))
    combined = spectra[0] + 1j * spectra[1]
    half = size // 2
    positive = combined[1:half].abs().square().sum()
    negative = combined[half + 1:].abs().square().sum()
    return float(negative / (positive + negative))


def _fractional_shift(image: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """Exact sub-pixel translation of a periodic image, via the Fourier shift theorem."""
    height, width = image.shape[-2:]
    fy = torch.fft.fftfreq(height, dtype=torch.float64).reshape(-1, 1)
    fx = torch.fft.fftfreq(width, dtype=torch.float64).reshape(1, -1)
    phase = torch.exp(-2j * math.pi * (fx * dx + fy * dy))
    return torch.fft.ifft2(torch.fft.fft2(image.to(torch.complex128)) * phase).real


def _textured_image(size: int = 64, seed: int = 0) -> torch.Tensor:
    """Band-limited noise: broadband, unlike a single grating that any shift can fake."""
    generator = torch.Generator().manual_seed(seed)
    spectrum = torch.fft.fft2(torch.randn(1, 3, size, size, generator=generator, dtype=torch.float64))
    fy = torch.fft.fftfreq(size, dtype=torch.float64).reshape(-1, 1)
    fx = torch.fft.fftfreq(size, dtype=torch.float64).reshape(1, -1)
    radius = (fx.square() + fy.square()).sqrt()
    # Keep the mid band where the level-1 detail subbands actually live.
    spectrum = spectrum * ((radius > 0.12) & (radius < 0.45)).to(spectrum.dtype)
    image = torch.fft.ifft2(spectrum).real
    image = image - image.amin()
    return image / image.amax().clamp_min(1e-12)


def modulus_ripple(
    transform: QuaternionWaveletTransform2D, shifts=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
) -> float:
    """Per-position spread of the quaternion modulus under sub-pixel translation.

    Measured position by position, and only where there is an edge to measure: the
    spatial *mean* modulus of a homogeneous texture is stable for any orthonormal
    transform, so averaging first would compare nothing. Restricting to the top
    quintile of modulus keeps the ratio away from near-zero denominators too.
    """
    image = _textured_image()
    moduli = []
    for shift in shifts:
        shifted = _fractional_shift(image, shift, 0.0)
        coefficients, _ = transform.analysis(shifted.to(torch.float32))
        moduli.append(quaternion_modulus(coefficients)[:, :, 1:].double())
    stacked = torch.stack(moduli)
    strong = stacked[0] > stacked[0].quantile(0.8)
    relative = stacked.std(dim=0) / stacked.mean(dim=0).clamp_min(1e-9)
    return float(relative[strong].mean())


@pytest.mark.parametrize("backend", BACKENDS)
def test_round_trip_is_exact(backend: str) -> None:
    transform = QuaternionWaveletTransform2D(backend=backend)
    image = torch.rand(2, 3, 32, 32)
    coefficients, layout = transform.analysis(image)
    assert coefficients.shape == (2, 48, 16, 16)
    restored = transform.synthesis(coefficients, layout)
    assert torch.allclose(restored, image, atol=1e-5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_each_tree_is_orthonormal(backend: str) -> None:
    """Double-shift orthogonality; without it a tree would not invert on its own."""
    transform = QuaternionWaveletTransform2D(backend=backend)
    for h0 in (transform.h0_a, transform.h0_b):
        assert torch.isclose(h0.sum(), torch.tensor(math.sqrt(2.0), dtype=h0.dtype), atol=1e-6)
        assert torch.isclose(h0.square().sum(), torch.tensor(1.0, dtype=h0.dtype), atol=1e-6)
        for lag in range(1, h0.numel() // 2):
            overlap = (h0[2 * lag:] * h0[: h0.numel() - 2 * lag]).sum()
            assert abs(float(overlap)) < 1e-6


def test_single_tree_scores_one_half() -> None:
    """Control: one real wavelet splits its energy evenly, so the metric reads 0.5."""
    transform = QuaternionWaveletTransform2D(backend="qwt_dualtree_db4")
    transform.offsets = (0, 0)  # both trees identical -> not a pair at all
    assert negative_frequency_energy(transform) == pytest.approx(0.5, abs=1e-6)


def test_shifted_db4_is_not_a_hilbert_pair() -> None:
    """The legacy backend is pinned at the structural ceiling of an integer shift.

    Two trees running the SAME filter at an integer offset give
    ``W_B = W_A e^{-jwd}``, so the residual depends only on the shift. Measured
    identical (0.1804) for db2..db20, sym4..sym20 and coif3/coif5, which is why no
    change of wavelet can rescue this backend -- only a genuine pair can.
    """
    legacy = QuaternionWaveletTransform2D(backend="qwt_dualtree_db4")
    assert negative_frequency_energy(legacy) == pytest.approx(0.1804, abs=5e-3)


def test_designed_pair_beats_the_integer_shift_ceiling() -> None:
    designed = QuaternionWaveletTransform2D(backend="qwt_dualtree_hilbert")
    legacy = QuaternionWaveletTransform2D(backend="qwt_dualtree_db4")
    designed_score = negative_frequency_energy(designed)
    legacy_score = negative_frequency_energy(legacy)
    assert designed_score < legacy_score, (
        f"designed pair {designed_score:.4f} did not improve on the integer-shift "
        f"ceiling {legacy_score:.4f}"
    )


def test_modulus_is_more_shift_stable_than_the_legacy_backend() -> None:
    """The property the pipeline consumes, not just a property of the filters."""
    designed = modulus_ripple(QuaternionWaveletTransform2D(backend="qwt_dualtree_hilbert"))
    legacy = modulus_ripple(QuaternionWaveletTransform2D(backend="qwt_dualtree_db4"))
    # Measured at the time of writing: 0.0957 designed vs 0.1180 legacy.
    assert designed < legacy, (
        f"quaternion modulus ripple {designed:.4f} is not below the legacy {legacy:.4f}"
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_phase_advances_with_displacement(backend: str) -> None:
    """Phase must track translation; a constant phase carries no displacement."""
    transform = QuaternionWaveletTransform2D(backend=backend)
    image = _textured_image()
    phases = []
    for shift in (0.0, 0.5, 1.0, 1.5):
        coefficients, _ = transform.analysis(
            _fractional_shift(image, shift, 0.0).to(torch.float32)
        )
        phases.append(quaternion_phases(coefficients)[:, :, 1:, 0].double())
    spread = torch.stack(phases).std(dim=0).mean()
    assert float(spread) > 1e-3, "phi_x does not move when the image translates"
