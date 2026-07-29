from __future__ import annotations

import math

import torch


def solve_heat_exact(
    values: torch.Tensor,
    *,
    nu: float,
    time: float,
    domain_length: float,
) -> torch.Tensor:
    """Exact periodic heat flow using the discrete Fourier representation."""
    if values.ndim < 1 or values.shape[-1] < 2:
        raise ValueError("values must have shape (..., nx) with nx >= 2")
    if values.dtype not in {torch.float32, torch.float64}:
        raise TypeError("values must use float32 or float64")
    if not torch.isfinite(values).all():
        raise ValueError("values must be finite")
    if not all(
        math.isfinite(float(value))
        for value in (nu, time, domain_length)
    ):
        raise ValueError("nu, time, and domain_length must be finite")
    if nu <= 0.0 or time < 0.0 or domain_length <= 0.0:
        raise ValueError("nu and domain_length must be positive; time nonnegative")
    nx = int(values.shape[-1])
    k = 2.0 * math.pi * torch.fft.rfftfreq(
        nx,
        d=float(domain_length) / float(nx),
        device=values.device,
        dtype=values.dtype,
    )
    multiplier = torch.exp(-float(nu) * float(time) * k.square())
    result = torch.fft.irfft(
        torch.fft.rfft(values, dim=-1) * multiplier,
        n=nx,
        dim=-1,
    ).to(values.dtype)
    if not torch.isfinite(result).all():
        raise RuntimeError("heat evolution produced non-finite values")
    return result


def heat_multiplier_vector(
    q: int,
    *,
    target_nu: float,
    target_time: float,
    surrogate_nu: float,
    surrogate_time: float,
    domain_length: float,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Ideal diagonal multiplier from surrogate heat coefficients to target ones."""
    if q <= 0 or q % 2 == 0:
        raise ValueError("q must be a positive odd integer")
    if min(target_nu, surrogate_nu, domain_length) <= 0:
        raise ValueError("diffusivities and domain_length must be positive")
    kmax = (q - 1) // 2
    values = [torch.ones((), dtype=dtype, device=device)]
    for k in range(1, kmax + 1):
        w2 = (2.0 * math.pi * float(k) / float(domain_length)) ** 2
        factor = math.exp(
            -(float(target_nu) * float(target_time)
              - float(surrogate_nu) * float(surrogate_time))
            * w2
        )
        value = torch.tensor(factor, dtype=dtype, device=device)
        values.extend((value, value))
    return torch.stack(values)
