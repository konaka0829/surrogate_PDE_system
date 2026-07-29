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


def step_metadata(
    *,
    solver: str,
    dt: float,
    fine_dt: float | None,
    final_time: float,
) -> dict[str, object]:
    """Return the canonical step condition actually used by a Burgers solve."""
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite")
    if not math.isfinite(final_time) or final_time <= 0:
        raise ValueError("final_time must be positive and finite")
    normalized = normalize_solver_name(solver)
    outer_step_count = int(round(final_time / dt))
    if (
        outer_step_count < 1
        or abs(outer_step_count * dt - final_time)
        > 1e-10 * max(1.0, abs(final_time))
    ):
        raise ValueError(
            f"final_time={final_time} must align with requested outer dt={dt}"
        )
    if normalized == "etdrk4":
        if fine_dt is not None:
            raise ValueError("ETDRK4 requires fine_dt=null")
        substeps_per_outer = 1
        effective_substep = float(dt)
    else:
        if (
            fine_dt is None
            or not math.isfinite(fine_dt)
            or fine_dt <= 0
        ):
            raise ValueError("split_step requires positive finite fine_dt")
        substeps_per_outer = max(1, int(math.ceil(dt / fine_dt)))
        effective_substep = float(dt) / substeps_per_outer
    return {
        "solver": normalized,
        "requested_outer_dt": float(dt),
        "requested_fine_dt": (
            None if fine_dt is None else float(fine_dt)
        ),
        "outer_step_count": outer_step_count,
        "effective_substep": effective_substep,
        "substeps_per_outer": substeps_per_outer,
    }


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
    steps = step_metadata(
        solver=solver,
        dt=dt,
        fine_dt=fine_dt,
        final_time=time,
    )
    normalized = str(steps["solver"])
    outer_step_count = int(steps["outer_step_count"])
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
            obs_steps=[outer_step_count],
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
        **steps,
        "dealias": bool(dealias),
        "domain_length": float(domain_length),
        "dtype": str(u0.dtype).removeprefix("torch."),
        "device": str(u0.device),
    }
    return BurgersResult(values.detach(), metadata)
