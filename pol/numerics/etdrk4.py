from __future__ import annotations

import torch


def rfft_wavenumbers(nx: int, *, domain_length: float = 1.0, device=None, dtype=torch.float64) -> torch.Tensor:
    freq = torch.fft.rfftfreq(nx, d=float(domain_length) / float(nx), device=device)
    return (2.0 * torch.pi * freq).to(dtype=dtype)


def dealias_mask_2_3(nx: int, *, device=None) -> torch.Tensor:
    modes = torch.arange(nx // 2 + 1, device=device)
    return modes <= (nx // 3)


def _phi(z: torch.Tensor, order: int) -> torch.Tensor:
    small = torch.abs(z) < 1e-7
    if order == 1:
        out = (torch.exp(z) - 1.0) / z
        series = 1.0 + z / 2.0 + z * z / 6.0
    elif order == 2:
        out = (torch.exp(z) - 1.0 - z) / (z * z)
        series = 0.5 + z / 6.0 + z * z / 24.0
    elif order == 3:
        out = (torch.exp(z) - 1.0 - z - 0.5 * z * z) / (z * z * z)
        series = 1.0 / 6.0 + z / 24.0 + z * z / 120.0
    else:
        raise ValueError("order must be 1, 2, or 3")
    return torch.where(small, series, out)


def burgers_nonlinear_hat(u: torch.Tensor, k: torch.Tensor, *, b: float = 1.0, dealias: bool = True) -> torch.Tensor:
    nh = -0.5j * float(b) * k * torch.fft.rfft(u * u, dim=-1)
    if dealias:
        nh = nh * dealias_mask_2_3(u.shape[-1], device=u.device).to(nh.dtype)
    return nh


def cox_matthews_coefficients(L: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    z = float(dt) * L
    E = torch.exp(z)
    E_half = torch.exp(z / 2.0)
    Q = 0.5 * float(dt) * _phi(z / 2.0, 1)
    f1 = float(dt) * (_phi(z, 1) - 3.0 * _phi(z, 2) + 4.0 * _phi(z, 3))
    f2 = float(dt) * (_phi(z, 2) - 2.0 * _phi(z, 3))
    f3 = float(dt) * (-_phi(z, 2) + 4.0 * _phi(z, 3))
    return E, E_half, Q, f1, f2, f3


@torch.no_grad()
def cox_matthews_etdrk4_step(
    v: torch.Tensor,
    *,
    L: torch.Tensor,
    dt: float,
    nonlinear,
) -> torch.Tensor:
    E, E_half, Q, f1, f2, f3 = cox_matthews_coefficients(L, dt)
    Nv = nonlinear(v)
    a = E_half * v + Q * Nv
    Na = nonlinear(a)
    b = E_half * v + Q * Na
    Nb = nonlinear(b)
    c = E_half * a + Q * (2.0 * Nb - Nv)
    Nc = nonlinear(c)
    return E * v + f1 * Nv + 2.0 * f2 * (Na + Nb) + f3 * Nc


@torch.no_grad()
def simulate_burgers_etdrk4_trajectory(
    u0: torch.Tensor,
    *,
    nu: float,
    b: float = 1.0,
    T: float,
    dt: float,
    obs_steps: list[int] | tuple[int, ...] | None = None,
    domain_length: float = 1.0,
    dealias: bool = True,
) -> list[torch.Tensor]:
    if T <= 0 or dt <= 0:
        raise ValueError("T and dt must be positive")
    steps = int(round(T / dt))
    if abs(steps * dt - T) > 1e-10:
        raise ValueError("T must be aligned with dt")
    u = u0
    dtype = u.real.dtype
    k = rfft_wavenumbers(u.shape[-1], domain_length=domain_length, device=u.device, dtype=dtype)
    L = -float(nu) * k * k
    E, E_half, Q, f1, f2, f3 = cox_matthews_coefficients(L, dt)
    vh = torch.fft.rfft(u, dim=-1)
    obs_sorted = sorted(set(int(step) for step in (obs_steps if obs_steps is not None else [steps])))
    if not obs_sorted or obs_sorted[0] < 1:
        raise ValueError("obs_steps must be positive")
    if obs_sorted[-1] > steps:
        raise ValueError("obs_steps exceed total steps")
    observed: list[torch.Tensor] = []
    obs_ptr = 0
    for step in range(1, steps + 1):
        Nv = burgers_nonlinear_hat(torch.fft.irfft(vh, n=u.shape[-1], dim=-1), k, b=b, dealias=dealias)
        a = E_half * vh + Q * Nv
        Na = burgers_nonlinear_hat(torch.fft.irfft(a, n=u.shape[-1], dim=-1), k, b=b, dealias=dealias)
        bstage = E_half * vh + Q * Na
        Nb = burgers_nonlinear_hat(torch.fft.irfft(bstage, n=u.shape[-1], dim=-1), k, b=b, dealias=dealias)
        c = E_half * a + Q * (2.0 * Nb - Nv)
        Nc = burgers_nonlinear_hat(torch.fft.irfft(c, n=u.shape[-1], dim=-1), k, b=b, dealias=dealias)
        vh = E * vh + f1 * Nv + 2.0 * f2 * (Na + Nb) + f3 * Nc
        while obs_ptr < len(obs_sorted) and step == obs_sorted[obs_ptr]:
            out = torch.fft.irfft(vh, n=u.shape[-1], dim=-1)
            if not torch.isfinite(out).all():
                raise FloatingPointError("ETDRK4 Burgers produced NaN/Inf")
            observed.append(out.clone())
            obs_ptr += 1
    return observed


@torch.no_grad()
def simulate_burgers_etdrk4(
    u0: torch.Tensor,
    *,
    nu: float,
    b: float = 1.0,
    T: float,
    dt: float,
    domain_length: float = 1.0,
    dealias: bool = True,
) -> torch.Tensor:
    return simulate_burgers_etdrk4_trajectory(
        u0,
        nu=nu,
        b=b,
        T=T,
        dt=dt,
        obs_steps=None,
        domain_length=domain_length,
        dealias=dealias,
    )[-1]
