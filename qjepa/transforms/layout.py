"""Static metadata required to invert a wavelet transform safely."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, TypeVar

import torch


@dataclass(frozen=True)
class TransformLayout:
    backend: str
    revision: str
    original_shape: tuple[int, ...]
    coefficient_shape: tuple[int, ...]
    levels: int
    boundary_mode: str
    scale_convention: str
    band_order: tuple[str, ...] = ()
    component_order: tuple[str, ...] = ()
    channel_order: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    def require(self, backend: str, revision: str) -> None:
        if self.backend != backend or self.revision != revision:
            raise ValueError(
                f"Incompatible transform layout {self.backend}@{self.revision}; "
                f"expected {backend}@{revision}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


F = TypeVar("F", bound=Callable)


def fp32_transform(method: F) -> F:
    """Keep fixed transforms in FP32 under AMP without detaching gradients."""
    import functools

    @functools.wraps(method)
    def wrapped(self, x: torch.Tensor, *args, **kwargs):
        promoted = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        with torch.autocast(x.device.type, enabled=False):
            return method(self, promoted, *args, **kwargs)

    return wrapped  # type: ignore[return-value]

