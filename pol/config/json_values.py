from __future__ import annotations

import json
import math
from typing import Any


class NonFiniteJsonConstantError(ValueError):
    """Raised when a non-standard non-finite JSON constant is encountered."""


def _reject_nonfinite_constant(value: str) -> None:
    raise NonFiniteJsonConstantError(
        f"non-finite JSON constant {value!r} is not permitted"
    )


def strict_json_loads(source: str) -> Any:
    """Parse JSON while rejecting Python's NaN/Infinity extensions."""

    return json.loads(source, parse_constant=_reject_nonfinite_constant)


def ensure_finite_json_value(value: Any, *, path: str = "$") -> Any:
    """Reject non-finite floats anywhere in a JSON-compatible value."""

    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite numeric value")
        return value
    if isinstance(value, (str, int)):
        return value
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            ensure_finite_json_value(item, path=f"{path}[{index}]")
        return value
    if isinstance(value, dict):
        for key, item in value.items():
            ensure_finite_json_value(item, path=f"{path}[{key!r}]")
        return value
    return value


__all__ = [
    "NonFiniteJsonConstantError",
    "ensure_finite_json_value",
    "strict_json_loads",
]
