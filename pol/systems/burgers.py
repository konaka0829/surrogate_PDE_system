from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from pol.numerics.burgers import simulate_burgers_split_step
from pol.numerics.etdrk4 import simulate_burgers_etdrk4


def normalize_solver_name(solver: str) -> str:
    if solver in {"etdrk4", "fourier_pseudospectral_etdrk4"}:
        return "etdrk4"
    if solver in {"split_step", "semi_implicit"}:
        return "split_step"
    raise ValueError(f"unsupported Burgers solver: {solver}")


def step_metadata(*, solver: str, dt: float, fine_dt: float | None) -> tuple[float, int]:
    if dt <= 0:
        raise ValueError("dt must be positive")
    normalized = normalize_solver_name(solver)
    if normalized == "etdrk4":
        return float(dt), 1
    if fine_dt is None or fine_dt <= 0:
        raise ValueError("split_step requires positive fine_dt")
    substeps = max(1, int(math.ceil(dt / fine_dt)))
    return float(dt) / substeps, substeps


@dataclass(frozen=True)
class BurgersResult:
    values: torch.Tensor
    metadata: dict[str, object]


@torch.no_grad()
def solve_burgers(
    u0: torch.Tensor,
    *,
    nu: float,
    time: float,
    dt: float,
    fine_dt: float | None,
    solver: str,
    dealias: bool,
    domain_length: float,
    advection_coefficient: float = 1.0,
) -> BurgersResult:
    """Solve periodic viscous Burgers to a terminal time.

    The neutral solver currently supports the standard coefficient-one
    advection term.  Keeping the coefficient explicit in the system spec makes
    unsupported physical changes fail early instead of being silently ignored.
    """
    if advection_coefficient != 1.0:
        raise ValueError("only advection_coefficient=1.0 is currently supported")
    if u0.ndim != 2 or not u0.dtype.is_floating_point:
        raise ValueError("u0 must be real with shape (batch, nx)")
    if nu <= 0 or time <= 0 or dt <= 0 or domain_length <= 0:
        raise ValueError("nu, time, dt, and domain_length must be positive")
    steps = int(round(time / dt))
    if abs(steps * dt - time) > 1e-10 * max(1.0, abs(time)):
        raise ValueError(f"time={time} must align with dt={dt}")
    normalized = normalize_solver_name(solver)
    effective, substeps = step_metadata(solver=solver, dt=dt, fine_dt=fine_dt)
    if normalized == "etdrk4":
        values = simulate_burgers_etdrk4(
            u0,
            nu=nu,
            T=time,
            dt=dt,
            dealias=dealias,
            domain_length=domain_length,
        )
    else:
        assert fine_dt is not None
        values = simulate_burgers_split_step(
            u0,
            dt=dt,
            Tr=time,
            obs_steps=[steps],
            nu=nu,
            fine_dt=fine_dt,
            dealias=dealias,
            domain_length=domain_length,
        )[-1]
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("Burgers solver produced NaN/Inf")
    metadata = {
        "kind": "burgers",
        "solver": normalized,
        "nu": float(nu),
        "time": float(time),
        "requested_dt": float(dt),
        "requested_fine_dt": None if fine_dt is None else float(fine_dt),
        "effective_inner_step": float(effective),
        "outer_steps": steps,
        "substeps_per_outer": substeps,
        "dealias": bool(dealias),
        "domain_length": float(domain_length),
        "dtype": str(u0.dtype).removeprefix("torch."),
        "device": str(u0.device),
    }
    return BurgersResult(values.detach(), metadata)
