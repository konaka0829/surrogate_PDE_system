from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch

from .burgers import solve_burgers
from .heat import solve_heat_exact
from .reaction_diffusion import solve_reaction_diffusion


SystemEvolver = Callable[
    [torch.Tensor, Mapping[str, Any], float, float],
    tuple[torch.Tensor, dict[str, object]],
]


def _evolve_heat(
    values: torch.Tensor,
    system: Mapping[str, Any],
    time: float,
    domain_length: float,
) -> tuple[torch.Tensor, dict[str, object]]:
    out = solve_heat_exact(
        values,
        nu=float(system["nu"]),
        time=time,
        domain_length=domain_length,
    )
    return out, {
        "kind": "heat",
        "solver": "spectral_exact",
        "nu": float(system["nu"]),
        "time": time,
        "domain_length": float(domain_length),
        "dtype": str(values.dtype).removeprefix("torch."),
        "device": str(values.device),
    }


def _evolve_burgers(
    values: torch.Tensor,
    system: Mapping[str, Any],
    time: float,
    domain_length: float,
) -> tuple[torch.Tensor, dict[str, object]]:
    result = solve_burgers(
        values,
        nu=float(system["nu"]),
        time=time,
        dt=float(system["dt"]),
        fine_dt=(
            None if system.get("fine_dt") is None else float(system["fine_dt"])
        ),
        solver=str(system["solver"]),
        dealias=bool(system["dealias"]),
        domain_length=domain_length,
        advection_coefficient=float(system.get("advection_coefficient", 1.0)),
    )
    return result.values, result.metadata


def _evolve_reaction_diffusion(
    values: torch.Tensor,
    system: Mapping[str, Any],
    time: float,
    domain_length: float,
) -> tuple[torch.Tensor, dict[str, object]]:
    result = solve_reaction_diffusion(
        values,
        nu=float(system["nu"]),
        alpha=float(system.get("alpha", 1.0)),
        beta=float(system.get("beta", 1.0)),
        time=time,
        dt=float(system["dt"]),
        domain_length=domain_length,
        nonlinear_filter=str(system.get("nonlinear_filter", "two_thirds")),
    )
    return result.values, result.metadata


_SYSTEM_EVOLVERS: dict[str, SystemEvolver] = {
    "heat": _evolve_heat,
    "burgers": _evolve_burgers,
    "reaction_diffusion": _evolve_reaction_diffusion,
}


@torch.no_grad()
def evolve(
    values: torch.Tensor,
    evolution: Mapping[str, Any],
    *,
    domain_length: float,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Evolve a batch through the registered system adapter."""
    system = evolution["system"]
    if not isinstance(system, Mapping):
        raise TypeError("evolution.system must be a mapping")
    kind = str(system["kind"])
    try:
        handler = _SYSTEM_EVOLVERS[kind]
    except KeyError as exc:
        raise ValueError(f"unknown system kind: {kind}") from exc
    return handler(values, system, float(evolution["time"]), float(domain_length))


def system_metadata(evolution: Mapping[str, Any]) -> dict[str, Any]:
    """Return a canonical JSON-compatible description of an evolution."""
    return {
        "time": float(evolution["time"]),
        "system": dict(evolution["system"]),
    }
