"""Independent reproducible random streams based on SHA-256."""

from __future__ import annotations

import hashlib

import numpy as np


def derive_seed(master_seed: int, *context: object) -> int:
    payload = "|".join(str(part) for part in (master_seed, *context)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") >> 1


def generator(master_seed: int, *context: object) -> np.random.Generator:
    return np.random.default_rng(derive_seed(master_seed, *context))

