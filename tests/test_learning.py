from __future__ import annotations

import math

import pytest
import torch

from pol.learning.direct import (
    DIRECT_DECODER_POLICY,
    decode_point_observation_to_real_fourier,
    fixed_fourier_decoder_bandwidth,
)
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


@pytest.mark.parametrize(
    (
        "J",
        "q",
        "observable_q",
        "retained_q",
        "requested_max_mode",
        "observable_max_mode",
        "zero_filled_mode_count",
        "zero_filled_coefficient_count",
        "zero_fill_applied",
    ),
    [
        (5, 3, 5, 3, 1, 2, 0, 0, False),
        (5, 5, 5, 5, 2, 2, 0, 0, False),
        (5, 7, 5, 5, 3, 2, 1, 2, True),
        (4, 1, 3, 1, 0, 1, 0, 0, False),
        (4, 3, 3, 3, 1, 1, 0, 0, False),
        (4, 7, 3, 3, 3, 1, 2, 4, True),
    ],
)
def test_fixed_decoder_bandwidth_contract(
    J: int,
    q: int,
    observable_q: int,
    retained_q: int,
    requested_max_mode: int,
    observable_max_mode: int,
    zero_filled_mode_count: int,
    zero_filled_coefficient_count: int,
    zero_fill_applied: bool,
) -> None:
    diagnostic = fixed_fourier_decoder_bandwidth(J, q)
    assert diagnostic.observation_count == J
    assert diagnostic.requested_q == q
    assert diagnostic.observable_q == observable_q
    assert diagnostic.retained_q == retained_q
    assert diagnostic.requested_max_mode == requested_max_mode
    assert diagnostic.observable_max_mode == observable_max_mode
    assert diagnostic.zero_filled_mode_count == zero_filled_mode_count
    assert (
        diagnostic.zero_filled_coefficient_count
        == zero_filled_coefficient_count
    )
    assert diagnostic.zero_fill_applied is zero_fill_applied
    assert diagnostic.decoder_policy == DIRECT_DECODER_POLICY


@pytest.mark.parametrize(
    ("J", "q", "error"),
    [
        (1, 3, "J >= 2"),
        (4, 0, "positive"),
        (4, -1, "positive"),
        (4, 2, "odd"),
    ],
)
def test_fixed_decoder_bandwidth_rejects_invalid_inputs(
    J: int,
    q: int,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        fixed_fourier_decoder_bandwidth(J, q)


def test_fixed_decoder_entry_point_rejects_invalid_J_and_q() -> None:
    with pytest.raises(ValueError, match="J >= 2"):
        decode_point_observation_to_real_fourier(
            torch.zeros((1, 1), dtype=torch.float64),
            3,
            domain_length=1.0,
        )
    for q in (0, -1, 2):
        with pytest.raises(ValueError, match="positive|odd"):
            decode_point_observation_to_real_fourier(
                torch.zeros((1, 4), dtype=torch.float64),
                q,
                domain_length=1.0,
            )


def test_fixed_decoder_zero_fill_is_exactly_the_pre_diagnostic_tensor() -> None:
    features = torch.tensor(
        [[0.5, -1.25, 0.75, 2.0], [-0.25, 0.5, 1.5, -2.0]],
        dtype=torch.float64,
    )
    q = 7
    decoded = decode_point_observation_to_real_fourier(
        features,
        q,
        domain_length=2.0,
    )

    # This is the pre-P0-05 numerical implementation written out explicitly.
    raw = features * math.sqrt(4.0 / 2.0)
    observable_prefix = real_fourier_analysis(
        raw,
        3,
        domain_length=2.0,
    )
    legacy = torch.cat(
        [observable_prefix, torch.zeros((2, 4), dtype=torch.float64)],
        dim=-1,
    )
    assert torch.equal(decoded, legacy)


def test_fixed_decoder_recovers_observable_prefix_and_exactly_zero_fills() -> None:
    observable = torch.tensor(
        [[0.2, 1.0, -0.3]],
        dtype=torch.float64,
    )
    requested = torch.cat(
        [observable, torch.zeros((1, 4), dtype=torch.float64)],
        dim=-1,
    )
    field = real_fourier_synthesis(requested, 32, domain_length=1.0)
    features = observe_equispaced_periodic(
        field,
        4,
        domain_length=1.0,
        l2_scale=True,
    )
    decoded = decode_point_observation_to_real_fourier(
        features,
        7,
        domain_length=1.0,
    )
    assert torch.allclose(decoded[:, :3], observable, atol=1e-12, rtol=1e-12)
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
