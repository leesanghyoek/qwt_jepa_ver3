"""Design the level-1 Hilbert filter pair used by the QWT, and print it.

Why this is an enumeration rather than a fit: every spectral factor of one
Daubechies halfband filter has the *same* magnitude response and is exactly
orthonormal, and they differ only in phase. The set of orthonormal filters of a
given length is therefore finite and small, so the pair whose phase difference
best approximates the half-sample delay can be found by listing them.

This is what separates the two backends in ``qjepa/transforms/qwt.py``. Running
the same filter in both trees at an integer offset gives
``W_B(w) = W_A(w) e^{-jwd}``: the residual is decided by the shift alone, which is
why it measures 0.1804 for every orthonormal wavelet ever tried and cannot be
improved by choosing a nicer one. Two *different* spectral factors can do better,
and this script reports by how much.

    python3 tools/design_hilbert_pair.py --orders 4 6 7 8 10

Paste the printed tuples into ``HILBERT_TREE_A`` / ``HILBERT_TREE_B`` /
``HILBERT_OFFSETS``, then run ``pytest tests/test_qwt_analyticity.py``.
"""

from __future__ import annotations

import argparse
import itertools
import math

import numpy as np

SQRT2 = math.sqrt(2.0)


def halfband_z_roots(order: int) -> list[np.ndarray]:
    """Reciprocal z-root groups of the Daubechies halfband filter of a given order.

    Returns one group per independent choice. Each group is a pair of arrays: the
    roots to use if the group is kept, and the roots to use if it is flipped to
    the reciprocals. Conjugates travel together so the resulting filter stays real.
    """
    # P(y) = sum_k binom(order-1+k, k) y^k with y = sin^2(w/2).
    coefficients = [math.comb(order - 1 + k, k) for k in range(order)]
    y_roots = np.roots(coefficients[::-1])

    groups: list[tuple[np.ndarray, np.ndarray]] = []
    consumed = np.zeros(len(y_roots), dtype=bool)
    for index, y in enumerate(y_roots):
        if consumed[index]:
            continue
        partners = [index]
        if abs(y.imag) > 1e-9:
            distance = np.abs(y_roots - np.conj(y)) + np.where(consumed, 1e9, 0.0)
            distance[index] = 1e9
            partners.append(int(np.argmin(distance)))
        for position in partners:
            consumed[position] = True
        keep, flip = [], []
        for position in partners:
            value = y_roots[position]
            # z^2 - (2 - 4y) z + 1 = 0
            b = 2.0 - 4.0 * value
            discriminant = np.sqrt(complex(b * b - 4.0))
            first, second = (b + discriminant) / 2.0, (b - discriminant) / 2.0
            inner, outer = (first, second) if abs(first) < abs(second) else (second, first)
            keep.append(inner)
            flip.append(outer)
        groups.append((np.array(keep), np.array(flip)))
    return groups


def spectral_factors(order: int) -> list[np.ndarray]:
    """Every real orthonormal filter sharing the order-``order`` magnitude response."""
    groups = halfband_z_roots(order)
    at_minus_one = np.full(order, -1.0 + 0j)
    factors: list[np.ndarray] = []
    for choice in itertools.product((0, 1), repeat=len(groups)):
        selected = [at_minus_one]
        for group, flip in zip(groups, choice):
            selected.append(group[flip])
        polynomial = np.poly(np.concatenate(selected))
        if np.abs(polynomial.imag).max() > 1e-6:
            continue
        taps = polynomial.real[::-1].copy()
        norm = np.linalg.norm(taps)
        if norm < 1e-12:
            continue
        taps = taps / norm
        if taps.sum() < 0:
            taps = -taps
        if abs(taps.sum() - SQRT2) > 1e-6:
            continue
        if not any(np.allclose(taps, seen, atol=1e-9) for seen in factors):
            factors.append(taps)
    return factors


def quadrature_mirror(h0: np.ndarray) -> np.ndarray:
    return ((-1.0) ** np.arange(h0.size)) * h0[::-1]


def negative_frequency_energy(h1a, offset_a, h1b, offset_b, size: int = 2048) -> float:
    """Fraction of ``W_A + i W_B`` energy on negative frequencies; 0 is analytic."""
    wa, wb = np.zeros(size), np.zeros(size)
    wa[(np.arange(h1a.size) + offset_a) % size] = h1a
    wb[(np.arange(h1b.size) + offset_b) % size] = h1b
    combined = np.fft.fft(wa) + 1j * np.fft.fft(wb)
    half = size // 2
    positive = np.abs(combined[1:half]) ** 2
    negative = np.abs(combined[half + 1:]) ** 2
    return float(negative.sum() / (positive.sum() + negative.sum()))


def best_pair(order: int, offsets=range(-3, 4)):
    factors = spectral_factors(order)
    if not factors:
        return None
    highpass = [quadrature_mirror(taps) for taps in factors]
    best = None
    for a in range(len(factors)):
        for b in range(len(factors)):
            for offset in offsets:
                score = negative_frequency_energy(highpass[a], 0, highpass[b], offset)
                if best is None or score < best[0]:
                    best = (score, factors[a], factors[b], offset)
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", type=int, nargs="+", default=[4, 6, 7, 8])
    parser.add_argument("--select", type=int, default=None,
                        help="order to print coefficients for (default: the best scoring)")
    args = parser.parse_args()

    integer_shift_ceiling = negative_frequency_energy(
        quadrature_mirror(spectral_factors(4)[0]), 0,
        quadrature_mirror(spectral_factors(4)[0]), 1,
    )
    print(f"integer-shift ceiling (same filter, offset 1) : {integer_shift_ceiling:.4f}")
    print(f"single real wavelet (control)                 : 0.5000\n")

    print(f"{'order':>6} {'taps':>5} {'factors':>8} {'neg-freq':>10} {'offset':>7}")
    results = {}
    for order in args.orders:
        found = best_pair(order)
        if found is None:
            continue
        score, tree_a, tree_b, offset = found
        results[order] = found
        print(f"{order:>6} {tree_a.size:>5} {len(spectral_factors(order)):>8} "
              f"{score:>10.4f} {offset:>+7d}")

    if not results:
        print("no valid factorisation found")
        return 1
    chosen = args.select if args.select in results else min(results, key=lambda k: results[k][0])
    score, tree_a, tree_b, offset = results[chosen]
    print(f"\nselected order {chosen}: {tree_a.size} taps, offset B {offset:+d}, "
          f"neg-freq {score:.4f} "
          f"({integer_shift_ceiling / score:.2f}x better than the integer shift)")
    print("\nHILBERT_TREE_A = (")
    for value in tree_a:
        print(f"    {value!r},")
    print(")\nHILBERT_TREE_B = (")
    for value in tree_b:
        print(f"    {value!r},")
    print(f")\nHILBERT_OFFSETS = (0, {offset})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
