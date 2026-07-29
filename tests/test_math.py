from __future__ import annotations

import math

import pytest
import torch

from pol.learning.metrics import periodic_l2_norm
from pol.math.fourier import real_fourier_analysis, real_fourier_synthesis
from pol.math.periodic import periodic_grid, spectral_resample_periodic


def test_periodic_grid_is_endpoint_free() -> None:
    x = periodic_grid(8, 2.0)
    assert torch.equal(x, torch.arange(8, dtype=torch.float64) / 4.0)
    assert float(x[-1]) < 2.0


def test_spectral_resampling_is_identity_at_the_same_resolution() -> None:
    generator = torch.Generator().manual_seed(2024)
    values = torch.randn(3, 17, generator=generator, dtype=torch.float64)
    resampled = spectral_resample_periodic(values, 17, domain_length=2.5)
    assert torch.equal(resampled, values)
    assert resampled.data_ptr() != values.data_ptr()


def test_spectral_resampling_preserves_shared_modes_and_even_nyquist() -> None:
    for n_in, n_out, mode in (
        (31, 48, 5),
        (48, 31, 5),
        (32, 64, 16),
        (64, 32, 16),
    ):
        x_in = periodic_grid(n_in, 1.0)
        x_out = periodic_grid(n_out, 1.0)
        source = 0.3 + 0.7 * torch.cos(2 * torch.pi * mode * x_in)
        expected = 0.3 + 0.7 * torch.cos(2 * torch.pi * mode * x_out)
        actual = spectral_resample_periodic(source, n_out, domain_length=1.0)
        assert torch.allclose(actual, expected, atol=1e-11, rtol=1e-11)


def test_spectral_downsampling_discards_unrepresentable_high_modes() -> None:
    x = periodic_grid(64, 1.0)
    low = 0.2 + torch.cos(4 * torch.pi * x)
    high = low + 0.4 * torch.cos(2 * torch.pi * 20 * x)
    down = spectral_resample_periodic(torch.stack([low, high]), 16, domain_length=1.0)
    assert torch.allclose(down[0], down[1], atol=1e-11, rtol=1e-11)


def test_real_fourier_roundtrip_for_bandlimited_fields() -> None:
    generator = torch.Generator().manual_seed(4)
    coefficients = torch.randn(5, 17, generator=generator, dtype=torch.float64)
    field = real_fourier_synthesis(coefficients, 64, domain_length=1.0)
    recovered = real_fourier_analysis(field, 17, domain_length=1.0)
    assert torch.allclose(recovered, coefficients, atol=1e-11, rtol=1e-11)


def test_real_fourier_field_projection_is_idempotent() -> None:
    generator = torch.Generator().manual_seed(2048)
    field = torch.randn(4, 32, generator=generator, dtype=torch.float64)
    projected = real_fourier_synthesis(
        real_fourier_analysis(field, 9, domain_length=1.0),
        32,
        domain_length=1.0,
    )
    projected_twice = real_fourier_synthesis(
        real_fourier_analysis(projected, 9, domain_length=1.0),
        32,
        domain_length=1.0,
    )
    assert torch.allclose(projected_twice, projected, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize(
    ("nx", "domain_length", "constant", "dtype"),
    [
        (15, 1.0, -0.4, torch.float64),
        (16, 2.5, 0.3, torch.float64),
        (17, 1.7, 0.2, torch.float32),
        (18, 3.1, -0.6, torch.float32),
    ],
)
def test_periodic_l2_norm_matches_constant_field_analytic_norm(
    nx: int,
    domain_length: float,
    constant: float,
    dtype: torch.dtype,
) -> None:
    values = torch.full((2, nx), constant, dtype=dtype)
    actual = periodic_l2_norm(values, domain_length=domain_length)
    expected = torch.full(
        (2,),
        abs(constant) * math.sqrt(domain_length),
        dtype=dtype,
    )
    tolerance = 2e-6 if dtype == torch.float32 else 2e-14
    assert torch.allclose(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize(
    ("basis", "nx", "domain_length", "mode", "amplitude"),
    [
        ("cosine", 15, 2.5, 3, 0.8),
        ("sine", 16, 1.7, 3, -0.6),
        ("cosine", 17, 3.2, 4, -0.35),
        ("sine", 18, 2.2, 4, 0.45),
    ],
)
def test_periodic_l2_norm_matches_resolved_sine_cosine_analytic_norm(
    basis: str,
    nx: int,
    domain_length: float,
    mode: int,
    amplitude: float,
) -> None:
    x = periodic_grid(nx, domain_length)
    phase = 2.0 * torch.pi * mode * x / domain_length
    values = amplitude * (
        torch.cos(phase) if basis == "cosine" else torch.sin(phase)
    )
    actual = periodic_l2_norm(values, domain_length=domain_length)
    expected = abs(amplitude) * math.sqrt(domain_length / 2.0)
    assert float(actual) == pytest.approx(expected, abs=2e-14, rel=2e-14)


@pytest.mark.parametrize(
    ("nx", "domain_length", "dtype"),
    [
        (15, 2.5, torch.float64),
        (16, 1.7, torch.float64),
        (17, 3.2, torch.float32),
        (18, 2.2, torch.float32),
    ],
)
def test_periodic_l2_norm_matches_multimode_parseval_with_batch_axes(
    nx: int,
    domain_length: float,
    dtype: torch.dtype,
) -> None:
    coefficients = torch.tensor(
        [
            [[0.4, 0.7, -0.2, 0.1, 0.3], [-0.2, 0.1, 0.6, -0.4, 0.2]],
            [[0.3, -0.5, 0.2, 0.25, -0.1], [0.1, 0.2, 0.3, 0.4, 0.5]],
        ],
        dtype=dtype,
    )
    values = real_fourier_synthesis(
        coefficients,
        nx,
        domain_length=domain_length,
    )
    actual = periodic_l2_norm(values, domain_length=domain_length)
    expected = torch.linalg.vector_norm(coefficients, dim=-1)
    tolerance = 3e-6 if dtype == torch.float32 else 3e-14
    assert actual.shape == coefficients.shape[:-1]
    assert actual.dtype == dtype
    assert torch.allclose(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize(
    ("values", "domain_length", "error"),
    [
        (torch.tensor(1.0), 1.0, ValueError),
        (torch.empty((2, 0), dtype=torch.float64), 1.0, ValueError),
        (torch.ones((2, 4), dtype=torch.float64), 0.0, ValueError),
        (torch.ones((2, 4), dtype=torch.float64), -1.0, ValueError),
        (torch.ones((2, 4), dtype=torch.float64), float("nan"), ValueError),
        (torch.ones((2, 4), dtype=torch.float64), float("inf"), ValueError),
        (torch.ones((2, 4), dtype=torch.float64), True, ValueError),
        (torch.ones((2, 4), dtype=torch.int64), 1.0, TypeError),
    ],
)
def test_periodic_l2_norm_rejects_invalid_shape_domain_or_dtype(
    values: torch.Tensor,
    domain_length: float,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        periodic_l2_norm(values, domain_length=domain_length)
