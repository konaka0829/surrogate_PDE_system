from __future__ import annotations

import torch


def periodic_grid(
    nx: int,
    domain_length: float,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return endpoint-free periodic nodes ``x_j = j L / nx``."""
    if nx < 2:
        raise ValueError("nx must be >= 2")
    if domain_length <= 0.0:
        raise ValueError("domain_length must be positive")
    return torch.arange(nx, device=device, dtype=dtype) * (
        float(domain_length) / float(nx)
    )


def _complex_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.float32:
        return torch.complex64
    if dtype == torch.float64:
        return torch.complex128
    raise TypeError("values must be float32 or float64")


def spectral_resample_periodic(
    values: torch.Tensor,
    n_out: int,
    *,
    domain_length: float,
) -> torch.Tensor:
    """Resample endpoint-free periodic data by Fourier coefficient transfer.

    The routine preserves all shared non-Nyquist integer modes.  It also
    handles even-grid Nyquist modes explicitly, so down/up-sampling does not
    silently duplicate or discard the Nyquist cosine coefficient.
    """
    if n_out < 2:
        raise ValueError("n_out must be >= 2")
    if domain_length <= 0.0:
        raise ValueError("domain_length must be positive")
    if values.ndim < 1:
        raise ValueError("values must have shape (..., n_in)")
    if values.shape[-1] < 2:
        raise ValueError("n_in must be >= 2")
    if not values.dtype.is_floating_point:
        raise TypeError("values must be real floating point")
    if values.shape[-1] == n_out:
        return values.clone()

    n_in = int(values.shape[-1])
    coeff_in = torch.fft.fft(values, dim=-1, norm="forward")
    coeff_out = torch.zeros(
        *values.shape[:-1],
        n_out,
        device=values.device,
        dtype=_complex_dtype(values.dtype),
    )

    in_non_nyq_max = (n_in - 1) // 2
    out_non_nyq_max = (n_out - 1) // 2
    shared = min(in_non_nyq_max, out_non_nyq_max)
    coeff_out[..., 0] = coeff_in[..., 0]
    for k in range(1, shared + 1):
        coeff_out[..., k] = coeff_in[..., k]
        coeff_out[..., -k] = coeff_in[..., -k]

    if n_in % 2 == 0:
        k = n_in // 2
        if k <= out_non_nyq_max:
            half = 0.5 * coeff_in[..., k]
            coeff_out[..., k] = coeff_out[..., k] + half
            coeff_out[..., -k] = coeff_out[..., -k] + half

    if n_out % 2 == 0:
        k = n_out // 2
        if k <= in_non_nyq_max:
            coeff_out[..., k] = coeff_in[..., k] + coeff_in[..., -k]
        elif n_in % 2 == 0 and k == n_in // 2:
            coeff_out[..., k] = coeff_in[..., k]

    return torch.fft.ifft(coeff_out, dim=-1, norm="forward").real.to(
        dtype=values.dtype
    )
