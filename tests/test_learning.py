from __future__ import annotations

import torch

from pol.learning.direct import decode_point_observation_to_real_fourier
from pol.learning.metrics import fourier_prediction_metrics
from pol.learning.observations import observe_equispaced_periodic
from pol.learning.random_features import RandomFeatureMap
from pol.learning.ridge import fit_centered_affine_ridge
from pol.math.fourier import real_fourier_analysis, real_fourier_synthesis
from pol.math.periodic import periodic_grid, spectral_resample_periodic
from pol.systems.heat import solve_heat_exact


def test_fixed_decoder_recovers_observable_bandlimited_coefficients() -> None:
    generator = torch.Generator().manual_seed(8)
    coefficients = torch.randn(4, 9, generator=generator, dtype=torch.float64)
    field = real_fourier_synthesis(coefficients, 64, domain_length=1.0)
    features = observe_equispaced_periodic(field, 16, domain_length=1.0, l2_scale=True)
    decoded = decode_point_observation_to_real_fourier(features, 9, domain_length=1.0)
    assert torch.allclose(decoded, coefficients, atol=1e-11, rtol=1e-11)


def test_model1_is_consistent_for_matched_bandlimited_heat_dynamics() -> None:
    coefficients = torch.tensor(
        [[0.2, 1.0, -0.3, 0.4, 0.1, -0.2, 0.5, 0.25, -0.1]],
        dtype=torch.float64,
    )
    finite_input = real_fourier_synthesis(coefficients, 16, domain_length=1.0)
    target = solve_heat_exact(
        finite_input, nu=0.1, time=0.2, domain_length=1.0
    )
    feature_input = spectral_resample_periodic(
        finite_input, 32, domain_length=1.0
    )
    matched_surrogate = solve_heat_exact(
        feature_input, nu=0.1, time=0.2, domain_length=1.0
    )
    features = observe_equispaced_periodic(
        matched_surrogate, 16, domain_length=1.0, l2_scale=True
    )
    prediction = decode_point_observation_to_real_fourier(
        features, 9, domain_length=1.0
    )
    expected = real_fourier_analysis(target, 9, domain_length=1.0)
    assert torch.allclose(prediction, expected, atol=1e-11, rtol=1e-11)


def test_fixed_decoder_zero_pads_unobservable_output_modes() -> None:
    coefficients = torch.tensor([[1.0, 2.0, -1.0, 0.5, 0.25]], dtype=torch.float64)
    field = real_fourier_synthesis(coefficients, 32, domain_length=1.0)
    features = observe_equispaced_periodic(field, 4, domain_length=1.0, l2_scale=True)
    decoded = decode_point_observation_to_real_fourier(features, 7, domain_length=1.0)
    assert decoded.shape == (1, 7)
    assert torch.equal(decoded[:, 3:], torch.zeros_like(decoded[:, 3:]))


def test_centered_zero_ridge_recovers_affine_map() -> None:
    generator = torch.Generator().manual_seed(3)
    x = torch.randn(20, 4, generator=generator, dtype=torch.float64)
    W = torch.randn(3, 4, generator=generator, dtype=torch.float64)
    b = torch.randn(3, generator=generator, dtype=torch.float64)
    y = x @ W.T + b
    readout = fit_centered_affine_ridge(x, y, 0.0)
    assert readout.solver == "svd_minimum_norm"
    assert torch.allclose(readout(x), y, atol=1e-11, rtol=1e-11)


def test_random_feature_map_is_seed_deterministic_and_skip_connected() -> None:
    first = RandomFeatureMap.create(
        3,
        5,
        activation="tanh",
        seed=9,
        weight_scale=0.5,
        bias_scale=0.1,
        dtype=torch.float64,
        device="cpu",
    )
    second = RandomFeatureMap.create(
        3,
        5,
        activation="tanh",
        seed=9,
        weight_scale=0.5,
        bias_scale=0.1,
        dtype=torch.float64,
        device="cpu",
    )
    phi = torch.randn(2, 3, dtype=torch.float64)
    lifted = first(phi)
    assert lifted.shape == (2, 8)
    assert torch.equal(lifted[:, :3], phi)
    assert torch.equal(first.A, second.A) and torch.equal(first.c, second.c)


def test_reference_field_metric_is_distinct_from_finite_data_metric() -> None:
    x = periodic_grid(32, 1.0, dtype=torch.float64)
    reference = (
        torch.cos(2.0 * torch.pi * x)
        + 0.25 * torch.cos(10.0 * torch.pi * x)
    ).unsqueeze(0)
    data = spectral_resample_periodic(reference, 8, domain_length=1.0)
    coefficient_target = real_fourier_analysis(data, 3, domain_length=1.0)
    metrics = fourier_prediction_metrics(
        coefficient_target,
        coefficient_target,
        data,
        reference,
        n_data=8,
        n_reference=32,
        domain_length=1.0,
    )
    assert metrics["data_field_relative_l2_max"] < 1e-12
    assert metrics["field_relative_l2_mean"] > 0.1
