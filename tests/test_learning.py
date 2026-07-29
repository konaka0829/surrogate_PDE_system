from __future__ import annotations

import math

import pytest
import torch

import pol.validation.model1_consistency as model1_consistency
from pol.config.models import InterfaceDimensionsSpec
from pol.learning.direct import (
    DIRECT_DECODER_POLICY,
    decode_point_observation_to_real_fourier,
    fixed_fourier_decoder_bandwidth,
)
from pol.learning.metrics import (
    fourier_prediction_metrics,
    fourier_representation_floor,
    samplewise_l2_errors,
)
from pol.learning.observations import observe_equispaced_periodic
from pol.learning.random_features import RandomFeatureMap
from pol.learning.ridge import fit_centered_affine_ridge
from pol.math.fourier import real_fourier_analysis, real_fourier_synthesis
from pol.math.periodic import periodic_grid, spectral_resample_periodic
from pol.systems.heat import solve_heat_exact
from pol.validation.model1_consistency import (
    matched_model1_cases,
    run_model1_consistency_case,
)
from pol.validation.quadrature import (
    FIELD_QUADRATURE_TOLERANCE,
    evaluate_orthonormal_trigonometric_field,
    run_field_quadrature_check,
)


@pytest.mark.parametrize(
    ("dtype", "nx", "domain_length", "constant", "modes"),
    [
        (torch.float64, 15, 1.0, 0.4, ()),
        (torch.float64, 16, 1.0, 0.0, ((3, 0.7, 0.0),)),
        (torch.float64, 15, 2.5, 0.0, ((2, 0.0, -0.5),)),
        (
            torch.float64,
            16,
            1.7,
            0.2,
            ((1, 0.7, -0.2), (3, -0.15, 0.4)),
        ),
        (torch.float32, 15, 2.0, -0.1, ((2, 0.4, 0.0),)),
        (torch.float32, 16, 2.2, 0.1, ((3, 0.0, 0.35),)),
    ],
)
def test_heat_exact_matches_independent_analytic_modes(
    dtype: torch.dtype,
    nx: int,
    domain_length: float,
    constant: float,
    modes: tuple[tuple[int, float, float], ...],
) -> None:
    nu = 0.07
    time = 0.13
    x = torch.arange(nx, dtype=dtype) * (domain_length / nx)
    initial = torch.full((nx,), constant, dtype=dtype)
    expected = torch.full((nx,), constant, dtype=dtype)
    for mode, cosine_amplitude, sine_amplitude in modes:
        angular_wavenumber = 2.0 * math.pi * mode / domain_length
        phase = angular_wavenumber * x
        component = (
            cosine_amplitude * torch.cos(phase)
            + sine_amplitude * torch.sin(phase)
        )
        initial = initial + component
        expected = expected + math.exp(
            -nu * angular_wavenumber**2 * time
        ) * component
    actual = solve_heat_exact(
        initial,
        nu=nu,
        time=time,
        domain_length=domain_length,
    )
    tolerance = 2e-5 if dtype == torch.float32 else 1e-11
    assert actual.shape == initial.shape
    assert actual.dtype == dtype
    assert actual.device == torch.device("cpu")
    assert torch.isfinite(actual).all()
    assert torch.allclose(actual, expected, atol=tolerance, rtol=tolerance)


def test_heat_exact_treats_even_nyquist_as_an_unpaired_cosine() -> None:
    nx = 16
    domain_length = 1.3
    nu = 0.05
    time = 0.08
    mode = nx // 2
    x = torch.arange(nx, dtype=torch.float64) * (domain_length / nx)
    nyquist = 0.3 * torch.cos(
        2.0 * math.pi * mode * x / domain_length
    )
    initial = 0.25 + nyquist
    multiplier = math.exp(
        -nu * (2.0 * math.pi * mode / domain_length) ** 2 * time
    )
    expected = 0.25 + multiplier * nyquist
    actual = solve_heat_exact(
        initial,
        nu=nu,
        time=time,
        domain_length=domain_length,
    )
    assert torch.allclose(actual, expected, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize(
    ("values", "nu", "time", "domain_length", "error"),
    [
        (torch.zeros(1), 0.1, 0.1, 1.0, "nx >= 2"),
        (torch.zeros(4, dtype=torch.int64), 0.1, 0.1, 1.0, "float32"),
        (
            torch.tensor([0.0, float("nan")]),
            0.1,
            0.1,
            1.0,
            "finite",
        ),
        (torch.zeros(4), float("nan"), 0.1, 1.0, "finite"),
        (torch.zeros(4), 0.1, float("inf"), 1.0, "finite"),
        (torch.zeros(4), 0.1, 0.1, float("nan"), "finite"),
        (torch.zeros(4), 0.0, 0.1, 1.0, "positive"),
        (torch.zeros(4), 0.1, -0.1, 1.0, "nonnegative"),
    ],
)
def test_heat_exact_rejects_malformed_inputs_and_parameters(
    values: torch.Tensor,
    nu: float,
    time: float,
    domain_length: float,
    error: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        solve_heat_exact(
            values,
            nu=nu,
            time=time,
            domain_length=domain_length,
        )


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


def _matched_model1_case(case_id: str):
    return next(
        case
        for case in matched_model1_cases()
        if case.case_id == case_id
    )


@pytest.mark.parametrize(
    "case_id",
    ["heat_same_resolution_odd", "heat_same_resolution_even"],
)
def test_matched_model1_heat_same_resolution_odd_and_even_pass(
    case_id: str,
) -> None:
    result = run_model1_consistency_case(
        _matched_model1_case(case_id),
        domain_length=1.0,
    )
    assert result["status"] == "pass"
    assert result["n_tar"] == result["n_sur"] == result["J"]
    assert result["q"] <= result["observable_q"]
    assert result["target_surrogate_conditions_equal"] is True
    assert result["matched_tolerance_satisfied"] is True
    assert result["coefficient_max_abs_error"] <= result["tolerance"]
    assert result["projected_field_relative_l2"] <= result["tolerance"]


def test_matched_model1_heat_different_resolution_and_information_isolation() -> None:
    result = run_model1_consistency_case(
        _matched_model1_case(
            "heat_different_resolution_information_isolation"
        ),
        domain_length=1.0,
    )
    isolation = result["information_isolation"]
    assert result["status"] == "pass"
    assert result["n_tar"] != result["n_sur"]
    assert result["matched_tolerance_satisfied"] is True
    assert isolation["status"] == "pass"
    assert isolation["reference_pair"]["status"] == "pass"
    assert isolation["finite_equality"]["status"] == "pass"
    assert isolation["feature_input_equality"]["status"] == "pass"
    assert isolation["prediction_equality"]["status"] == "pass"
    assert isolation["finite_equality"]["max_abs_difference"] <= (
        result["tolerance"]
    )


@pytest.mark.parametrize(
    "case_id",
    ["burgers_split_step_odd", "burgers_split_step_even"],
)
def test_matched_model1_burgers_split_step_odd_and_even_pass(
    case_id: str,
) -> None:
    result = run_model1_consistency_case(
        _matched_model1_case(case_id),
        domain_length=1.0,
    )
    assert result["status"] == "pass"
    assert result["target_solve_metadata"]["solver"] == "split_step"
    assert result["surrogate_solve_metadata"]["solver"] == "split_step"
    assert result["target_solve_metadata"]["requested_fine_dt"] == 0.005
    assert result["matched_tolerance_satisfied"] is True


def test_matched_model1_burgers_etdrk4_small_case_passes() -> None:
    result = run_model1_consistency_case(
        _matched_model1_case("burgers_etdrk4_small"),
        domain_length=1.0,
    )
    assert result["status"] == "pass"
    assert result["target_solve_metadata"]["solver"] == "etdrk4"
    assert result["target_solve_metadata"]["requested_fine_dt"] is None
    assert result["matched_tolerance_satisfied"] is True


def test_matched_model1_reaction_diffusion_passes_and_separates_field_floor() -> None:
    result = run_model1_consistency_case(
        _matched_model1_case("reaction_diffusion_matched"),
        domain_length=1.0,
    )
    assert result["status"] == "pass"
    assert result["target_solve_metadata"]["nonlinear_filter"] == "two_thirds"
    assert result["coefficient_relative_l2"] <= result["tolerance"]
    assert result["projected_field_relative_l2"] <= result["tolerance"]
    assert result["representation_floor_relative_l2"] > result["tolerance"]


def test_matched_model1_negative_time_control_detects_difference() -> None:
    result = run_model1_consistency_case(
        _matched_model1_case("heat_time_mismatch_control"),
        domain_length=1.0,
    )
    assert result["status"] == "pass"
    assert result["expected_status"] == "difference_detected"
    assert result["target_surrogate_conditions_equal"] is False
    assert result["matched_tolerance_satisfied"] is False
    assert result["coefficient_max_abs_error"] > result["tolerance"]


def test_matched_model1_uses_separate_solves_and_only_finite_feature_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _matched_model1_case("heat_same_resolution_odd")
    original_evolve = model1_consistency.evolve
    original_build = model1_consistency.build_feature_initial_state
    solve_inputs: list[tuple[int, int]] = []
    encoding_inputs: list[int] = []

    def tracked_evolve(values, evolution, *, domain_length):
        solve_inputs.append((int(values.shape[-1]), values.data_ptr()))
        return original_evolve(
            values,
            evolution,
            domain_length=domain_length,
        )

    def tracked_build(finite_inputs, *, n_sur, domain_length):
        encoding_inputs.append(int(finite_inputs.shape[-1]))
        return original_build(
            finite_inputs,
            n_sur=n_sur,
            domain_length=domain_length,
        )

    monkeypatch.setattr(model1_consistency, "evolve", tracked_evolve)
    monkeypatch.setattr(
        model1_consistency,
        "build_feature_initial_state",
        tracked_build,
    )
    result = run_model1_consistency_case(case, domain_length=1.0)

    assert result["status"] == "pass"
    assert encoding_inputs == [case.n_tar]
    assert [nx for nx, _ in solve_inputs] == [case.n_tar, case.n_sur]
    assert len(solve_inputs) == 2
    assert solve_inputs[0][1] != solve_inputs[1][1]
    assert result["independent_solve_outputs"]["status"] == "pass"


def test_matched_model1_exact_cases_stay_inside_observable_band() -> None:
    assert all(
        case.q <= fixed_fourier_decoder_bandwidth(
            case.J,
            case.q,
        ).observable_q
        for case in matched_model1_cases()
    )


def test_general_interface_keeps_q_greater_than_J_valid() -> None:
    dimensions = InterfaceDimensionsSpec(
        n_tar=15,
        n_sur=8,
        J=4,
        q=7,
    )
    assert dimensions.q > dimensions.J


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


def test_samplewise_l2_errors_match_known_parseval_values() -> None:
    domain_length = 2.3
    target_coefficients = torch.tensor(
        [
            [0.4, 0.7, -0.2, 0.1, 0.3],
            [-0.2, 0.1, 0.6, -0.4, 0.2],
        ],
        dtype=torch.float64,
    )
    prediction_coefficients = torch.tensor(
        [
            [0.3, 0.6, -0.1, 0.0, 0.2],
            [-0.1, 0.2, 0.4, -0.3, 0.0],
        ],
        dtype=torch.float64,
    )
    target = real_fourier_synthesis(
        target_coefficients,
        17,
        domain_length=domain_length,
    )
    prediction = real_fourier_synthesis(
        prediction_coefficients,
        17,
        domain_length=domain_length,
    )
    absolute, relative = samplewise_l2_errors(
        prediction,
        target,
        domain_length=domain_length,
    )
    expected_absolute = torch.linalg.vector_norm(
        prediction_coefficients - target_coefficients,
        dim=-1,
    )
    expected_relative = expected_absolute / torch.linalg.vector_norm(
        target_coefficients,
        dim=-1,
    )
    assert torch.allclose(
        absolute,
        expected_absolute,
        atol=3e-14,
        rtol=3e-14,
    )
    assert torch.allclose(
        relative,
        expected_relative,
        atol=3e-14,
        rtol=3e-14,
    )


def test_samplewise_l2_zero_target_uses_dtype_epsilon_clamp() -> None:
    domain_length = 2.5
    target = torch.zeros((2, 16), dtype=torch.float64)
    prediction = torch.stack(
        (
            torch.zeros(16, dtype=torch.float64),
            torch.ones(16, dtype=torch.float64),
        )
    )
    absolute, relative = samplewise_l2_errors(
        prediction,
        target,
        domain_length=domain_length,
    )
    expected_absolute = torch.tensor(
        [0.0, math.sqrt(domain_length)],
        dtype=torch.float64,
    )
    expected_relative = expected_absolute / torch.finfo(torch.float64).eps
    assert torch.equal(absolute, expected_absolute)
    assert torch.equal(relative, expected_relative)


def test_fourier_prediction_metrics_field_error_matches_coefficient_parseval() -> None:
    domain_length = 1.7
    target_coefficients = torch.tensor(
        [[0.4, 0.7, -0.2, 0.1, 0.3]],
        dtype=torch.float64,
    )
    prediction_coefficients = torch.tensor(
        [[0.3, 0.6, -0.1, 0.0, 0.2]],
        dtype=torch.float64,
    )
    data_target = real_fourier_synthesis(
        target_coefficients,
        16,
        domain_length=domain_length,
    )
    reference_target = real_fourier_synthesis(
        target_coefficients,
        17,
        domain_length=domain_length,
    )
    metrics = fourier_prediction_metrics(
        prediction_coefficients,
        target_coefficients,
        data_target,
        reference_target,
        n_data=16,
        n_reference=17,
        domain_length=domain_length,
    )
    expected_absolute = float(
        torch.linalg.vector_norm(
            prediction_coefficients - target_coefficients,
        )
    )
    expected_relative = expected_absolute / float(
        torch.linalg.vector_norm(target_coefficients)
    )
    assert metrics["field_absolute_l2_mean"] == pytest.approx(
        expected_absolute,
        abs=3e-14,
        rel=3e-14,
    )
    assert metrics["field_relative_l2_mean"] == pytest.approx(
        expected_relative,
        abs=3e-14,
        rel=3e-14,
    )
    assert metrics["data_field_absolute_l2_mean"] == pytest.approx(
        expected_absolute,
        abs=3e-14,
        rel=3e-14,
    )


def test_field_quadrature_convergence_keeps_field_data_and_floor_separate() -> None:
    check = run_field_quadrature_check(domain_length=2.3)
    convergence = check["convergence"]
    rows = convergence["rows"]

    assert check["status"] == "pass"
    assert convergence["candidate_n_ref"] == [8, 15, 16, 31, 32]
    assert rows[0]["resolved_for_squared_integrands"] is False
    assert rows[0]["status"] == "fail"
    assert rows[0]["characterization_status"] == "pass"
    assert all(row["status"] == "pass" for row in rows[1:])
    assert {row["grid_parity"] for row in rows} == {"odd", "even"}
    assert convergence["selected_n_ref"] == 15
    assert convergence["allowed_suffix_n_ref"] == [15, 16, 31, 32]
    assert convergence["finest_resolved_pair_status"] == "pass"
    assert all(
        row["metric_wrapper_consistency"]["status"] == "pass"
        for row in rows
    )
    assert check["metric_wrapper_consistency"][
        "data_space_invariance"
    ]["field_relative_l2"]["range"] == 0.0
    assert check["metric_wrapper_consistency"][
        "data_space_invariance"
    ]["representation_floor_relative_l2"]["range"] == 0.0
    assert rows[0]["absolute_discrepancy"] > FIELD_QUADRATURE_TOLERANCE
    assert rows[0]["representation_floor"][
        "relative_discrepancy"
    ] > FIELD_QUADRATURE_TOLERANCE
    assert all(
        row["absolute_discrepancy"] <= FIELD_QUADRATURE_TOLERANCE
        and row["representation_floor"]["relative_discrepancy"]
        <= FIELD_QUADRATURE_TOLERANCE
        for row in rows[1:]
    )


def test_representation_floor_reference_converges_while_data_is_fixed() -> None:
    domain_length = 2.3
    target_coefficients = torch.tensor(
        [[0.4, 0.7, -0.2, 0.0, 0.0, 0.15, 0.0]],
        dtype=torch.float64,
    )
    full_target_coefficients = torch.tensor(
        [
            [
                0.4,
                0.7,
                -0.2,
                0.0,
                0.0,
                0.15,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.5,
                0.25,
            ]
        ],
        dtype=torch.float64,
    )
    data_target = evaluate_orthonormal_trigonometric_field(
        full_target_coefficients,
        16,
        domain_length=domain_length,
    )
    values = []
    for n_ref in (8, 15, 16, 31, 32):
        reference_target = evaluate_orthonormal_trigonometric_field(
            full_target_coefficients,
            n_ref,
            domain_length=domain_length,
        )
        values.append(
            fourier_representation_floor(
                target_coefficients,
                data_target,
                reference_target,
                n_data=16,
                n_reference=n_ref,
                domain_length=domain_length,
            )
        )
    reference_values = [
        value["representation_floor_relative_l2_mean"]
        for value in values
    ]
    data_values = [
        value["data_representation_floor_relative_l2_mean"]
        for value in values
    ]
    assert abs(reference_values[0] - reference_values[-1]) > 0.1
    assert max(reference_values[1:]) - min(reference_values[1:]) < 1e-14
    assert max(data_values) - min(data_values) == 0.0
