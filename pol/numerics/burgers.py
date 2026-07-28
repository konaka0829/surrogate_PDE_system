from __future__ import annotations

import math
from typing import Iterable, Optional

import torch


def make_wavenumbers(
    s: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    domain_length: float = 1.0,
) -> torch.Tensor:
    if s <= 1:
        raise ValueError("s must be >= 2")
    if domain_length <= 0.0:
        raise ValueError("domain_length must be positive")
    dx = float(domain_length) / float(s)
    k = 2.0 * torch.pi * torch.fft.rfftfreq(s, d=dx, device=device)
    return k.to(dtype=dtype)


def make_dealias_mask(s: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if s <= 1:
        raise ValueError("s must be >= 2")
    cutoff = s // 3
    freqs = torch.arange(s // 2 + 1, device=device)
    return (freqs <= cutoff).to(dtype=dtype)


def _apply_dealias(
    u_hat: torch.Tensor,
    *,
    dealias: bool,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if not dealias:
        return u_hat
    if mask is None:
        s = (u_hat.shape[-1] - 1) * 2
        mask = make_dealias_mask(s, u_hat.device, u_hat.real.dtype)
    return u_hat * mask.to(device=u_hat.device, dtype=u_hat.real.dtype).unsqueeze(0)


def burgers_nonlinear_hat(
    u_hat: torch.Tensor,
    k: torch.Tensor,
    b: float = 1.0,
    dealias: bool = False,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    s = (u_hat.shape[-1] - 1) * 2
    u = torch.fft.irfft(u_hat, n=s, dim=-1, norm="forward")
    n_hat = -0.5j * b * k * torch.fft.rfft(u * u, dim=-1, norm="forward")
    return _apply_dealias(n_hat, dealias=dealias, mask=mask)


def burgers_split_step_outer(
    u_hat: torch.Tensor,
    dt: float,
    nu: float,
    k: torch.Tensor,
    fine_dt: float,
    b: float = 1.0,
    forcing_hat: Optional[torch.Tensor] = None,
    dealias: bool = False,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if nu < 0.0:
        raise ValueError("nu must be non-negative")

    if fine_dt > 0.0:
        n_sub = max(1, int(math.ceil(dt / fine_dt)))
    else:
        n_sub = 1
    h = dt / float(n_sub)

    heat = torch.exp(-nu * (k.pow(2)) * h).to(dtype=u_hat.real.dtype)
    out_hat = u_hat
    for _ in range(n_sub):
        out_hat = out_hat * heat
        n_hat = burgers_nonlinear_hat(out_hat, k, b=b, dealias=dealias, mask=mask)
        if forcing_hat is not None:
            n_hat = n_hat + forcing_hat
        out_hat = out_hat + h * n_hat
        out_hat = _apply_dealias(out_hat, dealias=dealias, mask=mask)
    return out_hat


@torch.no_grad()
def simulate_burgers_split_step(
    z0: torch.Tensor,
    dt: float,
    Tr: float,
    obs_steps: Iterable[int],
    nu: float,
    fine_dt: float,
    b: float = 1.0,
    forcing: Optional[torch.Tensor] = None,
    forcing_steps: Optional[tuple[int, int]] = None,
    dealias: bool = False,
    domain_length: float = 1.0,
) -> list[torch.Tensor]:
    if z0.ndim != 2:
        raise ValueError(f"z0 must have shape (B, s), got {tuple(z0.shape)}")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if Tr <= 0.0:
        raise ValueError("Tr must be positive")
    if nu < 0.0:
        raise ValueError("nu must be non-negative")

    obs_sorted = sorted(set(int(v) for v in obs_steps))
    if not obs_sorted:
        raise ValueError("obs_steps must be non-empty")
    if obs_sorted[0] < 1:
        raise ValueError("obs_steps must be >= 1")

    t_steps = int(round(Tr / dt))
    if t_steps < obs_sorted[-1]:
        raise ValueError(f"obs step {obs_sorted[-1]} exceeds total integration steps {t_steps}")

    s = z0.shape[-1]
    k = make_wavenumbers(s, z0.device, z0.dtype, domain_length=domain_length)
    mask = make_dealias_mask(s, z0.device, z0.dtype) if dealias else None

    forcing_hat: Optional[torch.Tensor] = None
    if forcing is not None:
        if forcing.shape != z0.shape:
            raise ValueError(f"forcing must have shape {tuple(z0.shape)}, got {tuple(forcing.shape)}")
        forcing = forcing.to(device=z0.device, dtype=z0.dtype)
        forcing_hat = torch.fft.rfft(forcing, dim=-1, norm="forward")
        forcing_hat = _apply_dealias(forcing_hat, dealias=dealias, mask=mask)

    if forcing_steps is not None:
        start_step, end_step = int(forcing_steps[0]), int(forcing_steps[1])
        if start_step < 1:
            raise ValueError("forcing_steps start_step must be >= 1")
    else:
        start_step, end_step = 1, 10**18

    observed: list[torch.Tensor] = []
    obs_ptr = 0
    max_obs = obs_sorted[-1]
    u_hat = torch.fft.rfft(z0, dim=-1, norm="forward")
    u_hat = _apply_dealias(u_hat, dealias=dealias, mask=mask)

    for step in range(1, max_obs + 1):
        active_forcing_hat = None
        if forcing_hat is not None and start_step <= step <= end_step:
            active_forcing_hat = forcing_hat
        u_hat = burgers_split_step_outer(
            u_hat,
            dt=dt,
            nu=nu,
            k=k,
            fine_dt=fine_dt,
            b=b,
            forcing_hat=active_forcing_hat,
            dealias=dealias,
            mask=mask,
        )
        while obs_ptr < len(obs_sorted) and step == obs_sorted[obs_ptr]:
            observed.append(torch.fft.irfft(u_hat, n=s, dim=-1, norm="forward").clone())
            obs_ptr += 1

    return observed
