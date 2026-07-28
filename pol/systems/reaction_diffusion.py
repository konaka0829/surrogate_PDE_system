from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class ReactionDiffusionResult:
    values: torch.Tensor
    metadata: dict[str, object]


@torch.no_grad()
def solve_reaction_diffusion(
    u0: torch.Tensor,
    *,
    nu: float,
    alpha: float,
    beta: float,
    time: float,
    dt: float,
    domain_length: float,
    nonlinear_filter: str = "two_thirds",
) -> ReactionDiffusionResult:
    """Semi-implicit spectral Euler for ``r_t=nu*r_xx+alpha*r-beta*r^3``."""
    if u0.ndim != 2 or not u0.dtype.is_floating_point:
        raise ValueError("u0 must be real with shape (batch, nx)")
    if not all(
        math.isfinite(v)
        for v in (nu, alpha, beta, time, dt, domain_length)
    ):
        raise ValueError("reaction-diffusion parameters must be finite")
    if nu <= 0 or time <= 0 or dt <= 0 or domain_length <= 0:
        raise ValueError("nu, time, dt, and domain_length must be positive")
    if nonlinear_filter not in {"none", "two_thirds"}:
        raise ValueError("nonlinear_filter must be none or two_thirds")
    steps = int(round(time / dt))
    if abs(steps * dt - time) > 1e-10 * max(1.0, abs(time)):
        raise ValueError(f"time={time} must align with dt={dt}")
    nx = int(u0.shape[-1])
    k = 2 * math.pi * torch.fft.rfftfreq(
        nx,
        d=domain_length / nx,
        device=u0.device,
        dtype=u0.dtype,
    )
    denominator = 1.0 + dt * nu * k.square()
    mask = (torch.arange(k.numel(), device=u0.device) <= nx // 3).to(u0.dtype)
    values = u0.clone()
    for _ in range(steps):
        rhat = torch.fft.rfft(values, dim=-1)
        cubic_hat = torch.fft.rfft(values.pow(3), dim=-1)
        if nonlinear_filter == "two_thirds":
            cubic_hat = cubic_hat * mask
        values = torch.fft.irfft(
            (rhat + dt * alpha * rhat - dt * beta * cubic_hat) / denominator,
            n=nx,
            dim=-1,
        )
        if not bool(torch.isfinite(values).all()):
            raise FloatingPointError("reaction-diffusion solver produced NaN/Inf")
    metadata = {
        "kind": "reaction_diffusion",
        "solver": "semi_implicit_spectral_euler",
        "nu": float(nu),
        "alpha": float(alpha),
        "beta": float(beta),
        "time": float(time),
        "requested_dt": float(dt),
        "effective_inner_step": float(dt),
        "step_count": steps,
        "nonlinear_filter": nonlinear_filter,
        "domain_length": float(domain_length),
        "dtype": str(u0.dtype).removeprefix("torch."),
        "device": str(u0.device),
    }
    return ReactionDiffusionResult(values.detach(), metadata)
