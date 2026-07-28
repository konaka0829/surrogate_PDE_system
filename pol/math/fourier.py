from __future__ import annotations

import math

import torch

from .periodic import periodic_grid


def validate_real_fourier_dim(q: int, nx: int) -> int:
    """Validate ``q=2K+1`` and return ``K``."""
    if nx < 2:
        raise ValueError("nx must be >= 2")
    if q <= 0 or q % 2 == 0:
        raise ValueError("q must be a positive odd integer")
    kmax = (q - 1) // 2
    if kmax >= nx / 2:
        raise ValueError("q includes a Fourier mode not representable below Nyquist")
    return kmax


def real_fourier_analysis(
    values: torch.Tensor,
    q: int,
    *,
    domain_length: float,
) -> torch.Tensor:
    """Compute real L2-orthonormal Fourier coefficients ``(..., nx)->(..., q)``."""
    if values.ndim < 1:
        raise ValueError("values must have shape (..., nx)")
    if domain_length <= 0.0:
        raise ValueError("domain_length must be positive")
    nx = int(values.shape[-1])
    kmax = validate_real_fourier_dim(q, nx)
    x = periodic_grid(nx, domain_length, device=values.device, dtype=values.dtype)
    dx = float(domain_length) / float(nx)
    coeffs = [values.sum(dim=-1) * (dx / math.sqrt(float(domain_length)))]
    scale = math.sqrt(2.0 / float(domain_length)) * dx
    for k in range(1, kmax + 1):
        phase = (2.0 * torch.pi * float(k) / float(domain_length)) * x
        coeffs.append((values * torch.cos(phase)).sum(dim=-1) * scale)
        coeffs.append((values * torch.sin(phase)).sum(dim=-1) * scale)
    return torch.stack(coeffs, dim=-1)


def real_fourier_synthesis(
    coefficients: torch.Tensor,
    nx: int,
    *,
    domain_length: float,
) -> torch.Tensor:
    """Synthesize endpoint-free grid values ``(..., q)->(..., nx)``."""
    if coefficients.ndim < 1:
        raise ValueError("coefficients must have shape (..., q)")
    if domain_length <= 0.0:
        raise ValueError("domain_length must be positive")
    q = int(coefficients.shape[-1])
    kmax = validate_real_fourier_dim(q, nx)
    x = periodic_grid(
        nx, domain_length, device=coefficients.device, dtype=coefficients.dtype
    )
    out = coefficients[..., 0:1] * (1.0 / math.sqrt(float(domain_length)))
    scale = math.sqrt(2.0 / float(domain_length))
    for k in range(1, kmax + 1):
        phase = (2.0 * torch.pi * float(k) / float(domain_length)) * x
        out = out + scale * (
            coefficients[..., 2 * k - 1 : 2 * k] * torch.cos(phase)
            + coefficients[..., 2 * k : 2 * k + 1] * torch.sin(phase)
        )
    return out
