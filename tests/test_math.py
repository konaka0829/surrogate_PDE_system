from __future__ import annotations

import torch

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
