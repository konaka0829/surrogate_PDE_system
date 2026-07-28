"""Execution-device policy for the validated scientific artifact workflow."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


EXECUTION_DEVICE_POLICY = "cpu_only"
COMPUTE_DEVICE = "cpu"


def execution_device_policy() -> dict[str, str]:
    """Return the canonical, JSON-compatible execution-device contract."""
    return {
        "execution_device_policy": EXECUTION_DEVICE_POLICY,
        "compute_device": COMPUTE_DEVICE,
    }


def verify_execution_device_policy(
    payload: Mapping[str, Any],
    *,
    boundary: str,
) -> None:
    """Reject a missing or altered CPU-only policy copy at an artifact boundary."""
    for field, expected in execution_device_policy().items():
        actual = payload.get(field)
        if actual != expected:
            raise ValueError(
                f"{boundary} execution-device policy mismatch: "
                f"{field}={actual!r}; expected {expected!r}"
            )


def require_cpu_tensor(
    tensor: torch.Tensor,
    *,
    boundary: str,
    name: str,
) -> None:
    """Reject a tensor outside CPU at a public scientific-workflow boundary."""
    if tensor.device.type != COMPUTE_DEVICE:
        raise RuntimeError(
            f"{boundary} requires CPU tensor {name}; "
            f"found device={tensor.device}"
        )


def require_cpu_tensors(
    value: Any,
    *,
    boundary: str,
    name: str = "payload",
) -> None:
    """Recursively enforce CPU placement for tensors in an artifact payload."""
    if isinstance(value, torch.Tensor):
        require_cpu_tensor(value, boundary=boundary, name=name)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            require_cpu_tensors(
                item,
                boundary=boundary,
                name=f"{name}.{key}",
            )
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            require_cpu_tensors(
                item,
                boundary=boundary,
                name=f"{name}[{index}]",
            )


__all__ = [
    "COMPUTE_DEVICE",
    "EXECUTION_DEVICE_POLICY",
    "execution_device_policy",
    "require_cpu_tensor",
    "require_cpu_tensors",
    "verify_execution_device_policy",
]
