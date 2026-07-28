from __future__ import annotations

import math

import torch


GRF_SAMPLER_SEMANTICS = "physical-angular-wavenumber-2pi-m-over-L-v1"


def sample_gaussian_random_field_initial_conditions(
    num_samples: int,
    nx: int,
    *,
    domain_length: float,
    seed: int,
    gamma: float = 2.0,
    tau: float = 5.0,
    sigma: float = 25.0,
    mean: float = 0.0,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if nx <= 1:
        raise ValueError("nx must be >= 2")
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")
    if tau < 0.0:
        raise ValueError("tau must be non-negative")
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative")
    domain_length = float(domain_length)
    if not math.isfinite(domain_length) or domain_length <= 0.0:
        raise ValueError("domain_length must be finite and positive")

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    cycles_per_length = torch.fft.rfftfreq(
        nx,
        d=domain_length / float(nx),
        device=device,
    ).to(dtype=dtype)
    wavenumbers = 2.0 * torch.pi * cycles_per_length
    eigvals = (sigma**2) * torch.pow(wavenumbers.pow(2) + tau**2, -gamma)
    eigvals = eigvals.to(device=device, dtype=dtype)

    coeff_shape = (num_samples, cycles_per_length.shape[0])
    real_part = torch.zeros(coeff_shape, device=device, dtype=dtype)
    imag_part = torch.zeros(coeff_shape, device=device, dtype=dtype)

    # Match MATLAB GRF1.m periodic sampling:
    # - only positive modes k >= 1 are randomized
    # - the constant mode is deterministic and set by `mean`
    # - uu(t) is shifted to uu(t - 0.5), which multiplies Fourier mode k by (-1)^k
    real_part[:, 0] = float(mean)

    if nx % 2 == 0:
        nyquist_idx = coeff_shape[1] - 1
        nyquist_sign = -1.0 if (nx // 2) % 2 else 1.0
        nyquist_noise = torch.randn(num_samples, generator=gen, dtype=dtype, device="cpu").to(device=device)
        real_part[:, nyquist_idx] = nyquist_sign * torch.sqrt(2.0 * eigvals[nyquist_idx]) * nyquist_noise
        interior_end = nyquist_idx
    else:
        interior_end = coeff_shape[1]

    n_interior = interior_end - 1
    if n_interior > 0:
        signs = torch.where(
            (torch.arange(1, interior_end, device=device) % 2) == 0,
            torch.ones(interior_end - 1, device=device, dtype=dtype),
            -torch.ones(interior_end - 1, device=device, dtype=dtype),
        )
        std = torch.sqrt(eigvals[1:interior_end] / 2.0).unsqueeze(0)
        real_noise = torch.randn((num_samples, n_interior), generator=gen, dtype=dtype, device="cpu").to(device=device)
        imag_noise = torch.randn((num_samples, n_interior), generator=gen, dtype=dtype, device="cpu").to(device=device)
        real_part[:, 1:interior_end] = signs.unsqueeze(0) * std * real_noise
        imag_part[:, 1:interior_end] = signs.unsqueeze(0) * std * imag_noise

    complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    coeffs = torch.complex(real_part, imag_part).to(dtype=complex_dtype)
    return torch.fft.irfft(coeffs, n=nx, dim=-1, norm="forward")
