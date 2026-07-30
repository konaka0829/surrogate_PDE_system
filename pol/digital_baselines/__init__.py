"""Digital neural-operator baselines sharing validated data and metrics."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .protocol import DigitalBaselineSpec


def __getattr__(name: str) -> Any:
    if name == "DigitalBaselineSpec":
        from .protocol import DigitalBaselineSpec

        return DigitalBaselineSpec
    raise AttributeError(name)

__all__ = ["DigitalBaselineSpec"]
