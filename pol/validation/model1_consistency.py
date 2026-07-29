"""Matched-dynamics consistency checks for the fixed Model 1 pipeline.

The diagnostic in this module is deliberately independent of validation
profiles, datasets, and the filesystem.  Every surrogate initial state is
constructed from a finite ``n_tar`` field, and target and surrogate evolutions
are evaluated by separate calls through the registered system interface.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal, Mapping

import torch

from pol.data.finite import build_feature_initial_state
from pol.learning.direct import (
    decode_point_observation_to_real_fourier,
    fixed_fourier_decoder_bandwidth,
)
from pol.learning.metrics import samplewise_l2_errors
from pol.learning.observations import observe_equispaced_periodic
from pol.math.fourier import real_fourier_analysis, real_fourier_synthesis
from pol.math.periodic import periodic_grid, spectral_resample_periodic
from pol.runtime.hashing import stable_object_hash, tensor_sha256
from pol.systems.registry import evolve


MODEL1_CONSISTENCY_CHECK_SCHEMA_VERSION = (
    "pol-matched-model1-pipeline-check-v1"
)
MODEL1_CONSISTENCY_TOLERANCE = 1.0e-10

ExpectedStatus = Literal["match", "difference_detected"]


@dataclass(frozen=True)
class Model1ConsistencyCase:
    """Complete synthetic specification for one finite-boundary pipeline."""

    case_id: str
    system_kind: str
    n_ref_synthetic: int
    n_tar: int
    n_sur: int
    J: int
    q: int
    coefficients: tuple[float, ...]
    target_evolution: Mapping[str, Any]
    surrogate_evolution: Mapping[str, Any]
    expected_status: ExpectedStatus = "match"
    tolerance: float = MODEL1_CONSISTENCY_TOLERANCE
    information_isolation_high_mode: int | None = None


@dataclass(frozen=True)
class _PipelineState:
    finite_input: torch.Tensor
    feature_input: torch.Tensor
    target_field: torch.Tensor
    surrogate_field: torch.Tensor
    expected_coefficients: torch.Tensor
    prediction: torch.Tensor
    target_metadata: dict[str, object]
    surrogate_metadata: dict[str, object]
    independent_solve_outputs: bool


def _evolution(
    system: Mapping[str, Any],
    *,
    time: float,
) -> dict[str, Any]:
    return {
        "system": dict(system),
        "time": float(time),
    }


def matched_model1_cases() -> tuple[Model1ConsistencyCase, ...]:
    """Return the fixed, deterministic case suite."""

    heat = {"kind": "heat", "nu": 0.05}
    burgers_split = {
        "kind": "burgers",
        "nu": 0.05,
        "advection_coefficient": 1.0,
        "solver": "split_step",
        "dt": 0.01,
        "fine_dt": 0.005,
        "dealias": True,
    }
    burgers_etdrk4 = {
        "kind": "burgers",
        "nu": 0.05,
        "advection_coefficient": 1.0,
        "solver": "etdrk4",
        "dt": 0.01,
        "fine_dt": None,
        "dealias": True,
    }
    reaction_diffusion = {
        "kind": "reaction_diffusion",
        "nu": 0.05,
        "alpha": 1.0,
        "beta": 1.0,
        "solver": "semi_implicit_spectral_euler",
        "dt": 0.005,
        "nonlinear_filter": "two_thirds",
    }
    coefficients_q9 = (
        0.2,
        0.45,
        -0.15,
        0.2,
        0.1,
        -0.12,
        0.08,
        0.06,
        -0.04,
    )
    coefficients_q7 = coefficients_q9[:7]
    return (
        Model1ConsistencyCase(
            case_id="heat_same_resolution_odd",
            system_kind="heat",
            n_ref_synthetic=31,
            n_tar=15,
            n_sur=15,
            J=15,
            q=9,
            coefficients=coefficients_q9,
            target_evolution=_evolution(heat, time=0.04),
            surrogate_evolution=_evolution(heat, time=0.04),
        ),
        Model1ConsistencyCase(
            case_id="heat_same_resolution_even",
            system_kind="heat",
            n_ref_synthetic=32,
            n_tar=16,
            n_sur=16,
            J=16,
            q=9,
            coefficients=coefficients_q9,
            target_evolution=_evolution(heat, time=0.04),
            surrogate_evolution=_evolution(heat, time=0.04),
        ),
        Model1ConsistencyCase(
            case_id="heat_different_resolution_information_isolation",
            system_kind="heat",
            n_ref_synthetic=48,
            n_tar=15,
            n_sur=24,
            J=12,
            q=9,
            coefficients=coefficients_q9,
            target_evolution=_evolution(heat, time=0.04),
            surrogate_evolution=_evolution(heat, time=0.04),
            information_isolation_high_mode=11,
        ),
        Model1ConsistencyCase(
            case_id="burgers_split_step_odd",
            system_kind="burgers",
            n_ref_synthetic=31,
            n_tar=15,
            n_sur=15,
            J=15,
            q=7,
            coefficients=coefficients_q7,
            target_evolution=_evolution(burgers_split, time=0.02),
            surrogate_evolution=_evolution(burgers_split, time=0.02),
        ),
        Model1ConsistencyCase(
            case_id="burgers_split_step_even",
            system_kind="burgers",
            n_ref_synthetic=32,
            n_tar=16,
            n_sur=16,
            J=16,
            q=7,
            coefficients=coefficients_q7,
            target_evolution=_evolution(burgers_split, time=0.02),
            surrogate_evolution=_evolution(burgers_split, time=0.02),
        ),
        Model1ConsistencyCase(
            case_id="burgers_etdrk4_small",
            system_kind="burgers",
            n_ref_synthetic=31,
            n_tar=15,
            n_sur=15,
            J=15,
            q=7,
            coefficients=coefficients_q7,
            target_evolution=_evolution(burgers_etdrk4, time=0.02),
            surrogate_evolution=_evolution(burgers_etdrk4, time=0.02),
        ),
        Model1ConsistencyCase(
            case_id="reaction_diffusion_matched",
            system_kind="reaction_diffusion",
            n_ref_synthetic=32,
            n_tar=16,
            n_sur=16,
            J=16,
            q=7,
            coefficients=coefficients_q7,
            target_evolution=_evolution(reaction_diffusion, time=0.02),
            surrogate_evolution=_evolution(reaction_diffusion, time=0.02),
        ),
        Model1ConsistencyCase(
            case_id="heat_time_mismatch_control",
            system_kind="heat",
            n_ref_synthetic=32,
            n_tar=16,
            n_sur=16,
            J=16,
            q=9,
            coefficients=coefficients_q9,
            target_evolution=_evolution(heat, time=0.04),
            surrogate_evolution=_evolution(heat, time=0.06),
            expected_status="difference_detected",
        ),
    )


def _validate_case(case: Model1ConsistencyCase) -> None:
    if len(case.coefficients) != case.q:
        raise ValueError(f"{case.case_id}: coefficients must have length q")
    if case.q <= 0 or case.q % 2 == 0 or case.q > case.n_tar:
        raise ValueError(f"{case.case_id}: requires odd q with q <= n_tar")
    if case.J > case.n_sur:
        raise ValueError(f"{case.case_id}: requires J <= n_sur")
    observable_q = fixed_fourier_decoder_bandwidth(case.J, case.q).observable_q
    if case.q > observable_q:
        raise ValueError(
            f"{case.case_id}: exact-recovery cases require q <= observable_q"
        )
    if case.n_tar > case.n_ref_synthetic:
        raise ValueError(f"{case.case_id}: n_tar exceeds synthetic reference")
    if (
        not math.isfinite(case.tolerance)
        or case.tolerance < 0.0
    ):
        raise ValueError(f"{case.case_id}: tolerance must be finite and nonnegative")
    for role, evolution in (
        ("target", case.target_evolution),
        ("surrogate", case.surrogate_evolution),
    ):
        system = evolution.get("system")
        if (
            not isinstance(system, Mapping)
            or system.get("kind") != case.system_kind
        ):
            raise ValueError(
                f"{case.case_id}: {role} system kind does not match the case"
            )


def _synthetic_reference(
    case: Model1ConsistencyCase,
    *,
    domain_length: float,
) -> torch.Tensor:
    coefficients = torch.tensor(
        case.coefficients,
        dtype=torch.float64,
    ).unsqueeze(0)
    return real_fourier_synthesis(
        coefficients,
        case.n_ref_synthetic,
        domain_length=domain_length,
    )


def _condition_payload(
    evolution: Mapping[str, Any],
    *,
    domain_length: float,
) -> dict[str, Any]:
    return {
        "system": dict(evolution["system"]),
        "time": float(evolution["time"]),
        "domain_length": float(domain_length),
    }


def _execute_pipeline(
    case: Model1ConsistencyCase,
    reference: torch.Tensor,
    *,
    domain_length: float,
) -> _PipelineState:
    """Execute the full target and surrogate paths from one reference field."""

    finite_input = spectral_resample_periodic(
        reference,
        case.n_tar,
        domain_length=domain_length,
    )
    feature_input = build_feature_initial_state(
        finite_input,
        n_sur=case.n_sur,
        domain_length=domain_length,
    )

    # These are intentionally separate calls.  In the same-resolution cases,
    # build_feature_initial_state still returns a distinct tensor.
    target_field, target_metadata = evolve(
        finite_input.clone(),
        case.target_evolution,
        domain_length=domain_length,
    )
    surrogate_field, surrogate_metadata = evolve(
        feature_input.clone(),
        case.surrogate_evolution,
        domain_length=domain_length,
    )
    independent_solve_outputs = (
        target_field is not surrogate_field
        and target_field.data_ptr() != surrogate_field.data_ptr()
    )

    observations = observe_equispaced_periodic(
        surrogate_field,
        case.J,
        domain_length=domain_length,
        l2_scale=True,
    )
    prediction = decode_point_observation_to_real_fourier(
        observations,
        case.q,
        domain_length=domain_length,
    )
    expected_coefficients = real_fourier_analysis(
        target_field,
        case.q,
        domain_length=domain_length,
    )
    return _PipelineState(
        finite_input=finite_input,
        feature_input=feature_input,
        target_field=target_field,
        surrogate_field=surrogate_field,
        expected_coefficients=expected_coefficients,
        prediction=prediction,
        target_metadata=target_metadata,
        surrogate_metadata=surrogate_metadata,
        independent_solve_outputs=independent_solve_outputs,
    )


def _max_abs_difference(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max())


def _equal_within_tolerance(
    a: torch.Tensor,
    b: torch.Tensor,
    tolerance: float,
) -> bool:
    return bool(torch.allclose(a, b, atol=tolerance, rtol=tolerance))


def _information_isolation(
    case: Model1ConsistencyCase,
    reference: torch.Tensor,
    first: _PipelineState,
    *,
    domain_length: float,
) -> dict[str, Any]:
    high_mode = case.information_isolation_high_mode
    if high_mode is None:
        raise ValueError("information-isolation case has no discarded high mode")
    if high_mode < case.n_tar / 2 or high_mode >= case.n_ref_synthetic / 2:
        raise ValueError("information-isolation high mode is not discarded")
    x = periodic_grid(
        case.n_ref_synthetic,
        domain_length,
        dtype=reference.dtype,
        device=reference.device,
    )
    second_reference = reference + 0.375 * torch.cos(
        2.0 * torch.pi * float(high_mode) * x / float(domain_length)
    ).unsqueeze(0)
    second = _execute_pipeline(
        case,
        second_reference,
        domain_length=domain_length,
    )
    tolerance = case.tolerance
    reference_distinct = not _equal_within_tolerance(
        reference,
        second_reference,
        tolerance,
    )
    finite_equal = _equal_within_tolerance(
        first.finite_input,
        second.finite_input,
        tolerance,
    )
    feature_equal = _equal_within_tolerance(
        first.feature_input,
        second.feature_input,
        tolerance,
    )
    prediction_equal = _equal_within_tolerance(
        first.prediction,
        second.prediction,
        tolerance,
    )
    expected_equal = _equal_within_tolerance(
        first.expected_coefficients,
        second.expected_coefficients,
        tolerance,
    )
    passed = (
        reference_distinct
        and finite_equal
        and feature_equal
        and prediction_equal
        and expected_equal
        and second.independent_solve_outputs
    )
    return {
        "status": "pass" if passed else "fail",
        "discarded_high_mode": int(high_mode),
        "reference_pair": {
            "status": "pass" if reference_distinct else "fail",
            "first_hash": tensor_sha256(reference),
            "second_hash": tensor_sha256(second_reference),
            "max_abs_difference": _max_abs_difference(
                reference,
                second_reference,
            ),
        },
        "finite_equality": {
            "status": "pass" if finite_equal else "fail",
            "first_hash": tensor_sha256(first.finite_input),
            "second_hash": tensor_sha256(second.finite_input),
            "max_abs_difference": _max_abs_difference(
                first.finite_input,
                second.finite_input,
            ),
        },
        "feature_input_equality": {
            "status": "pass" if feature_equal else "fail",
            "first_hash": tensor_sha256(first.feature_input),
            "second_hash": tensor_sha256(second.feature_input),
            "max_abs_difference": _max_abs_difference(
                first.feature_input,
                second.feature_input,
            ),
        },
        "prediction_equality": {
            "status": "pass" if prediction_equal else "fail",
            "first_hash": tensor_sha256(first.prediction),
            "second_hash": tensor_sha256(second.prediction),
            "max_abs_difference": _max_abs_difference(
                first.prediction,
                second.prediction,
            ),
        },
        "target_coefficient_equality": {
            "status": "pass" if expected_equal else "fail",
            "max_abs_difference": _max_abs_difference(
                first.expected_coefficients,
                second.expected_coefficients,
            ),
        },
        "second_pipeline_independent_solve_outputs": {
            "status": (
                "pass" if second.independent_solve_outputs else "fail"
            ),
            "value": second.independent_solve_outputs,
        },
    }


def run_model1_consistency_case(
    case: Model1ConsistencyCase,
    *,
    domain_length: float,
) -> dict[str, Any]:
    """Run one complete synthetic Model 1 consistency case."""

    if (
        isinstance(domain_length, bool)
        or not math.isfinite(float(domain_length))
        or float(domain_length) <= 0.0
    ):
        raise ValueError("domain_length must be positive and finite")
    _validate_case(case)
    reference = _synthetic_reference(case, domain_length=domain_length)
    state = _execute_pipeline(case, reference, domain_length=domain_length)

    coefficient_error = state.prediction - state.expected_coefficients
    coefficient_max_abs_error = float(coefficient_error.abs().max())
    coefficient_denominator = torch.linalg.vector_norm(
        state.expected_coefficients,
        dim=-1,
    ).clamp_min(torch.finfo(state.expected_coefficients.dtype).eps)
    coefficient_relative_l2 = float(
        (
            torch.linalg.vector_norm(coefficient_error, dim=-1)
            / coefficient_denominator
        ).max()
    )

    prediction_projection = real_fourier_synthesis(
        state.prediction,
        case.n_tar,
        domain_length=domain_length,
    )
    target_projection = real_fourier_synthesis(
        state.expected_coefficients,
        case.n_tar,
        domain_length=domain_length,
    )
    projected_absolute, projected_relative = samplewise_l2_errors(
        prediction_projection,
        target_projection,
        domain_length=domain_length,
    )
    floor_absolute, floor_relative = samplewise_l2_errors(
        target_projection,
        state.target_field,
        domain_length=domain_length,
    )
    projected_field_absolute_l2 = float(projected_absolute.max())
    projected_field_relative_l2 = float(projected_relative.max())
    representation_floor_absolute_l2 = float(floor_absolute.max())
    representation_floor_relative_l2 = float(floor_relative.max())

    target_condition = _condition_payload(
        case.target_evolution,
        domain_length=domain_length,
    )
    surrogate_condition = _condition_payload(
        case.surrogate_evolution,
        domain_length=domain_length,
    )
    conditions_equal = target_condition == surrogate_condition
    matched_tolerance_satisfied = (
        coefficient_max_abs_error <= case.tolerance
        and coefficient_relative_l2 <= case.tolerance
        and projected_field_absolute_l2 <= case.tolerance
        and projected_field_relative_l2 <= case.tolerance
    )
    if case.expected_status == "match":
        passed = (
            conditions_equal
            and matched_tolerance_satisfied
            and state.independent_solve_outputs
        )
    else:
        passed = (
            not conditions_equal
            and not matched_tolerance_satisfied
            and state.independent_solve_outputs
        )

    information_isolation = None
    if case.information_isolation_high_mode is not None:
        information_isolation = _information_isolation(
            case,
            reference,
            state,
            domain_length=domain_length,
        )
        passed = passed and information_isolation["status"] == "pass"

    bandwidth = fixed_fourier_decoder_bandwidth(case.J, case.q)
    metrics = (
        coefficient_max_abs_error,
        coefficient_relative_l2,
        projected_field_absolute_l2,
        projected_field_relative_l2,
        representation_floor_absolute_l2,
        representation_floor_relative_l2,
    )
    if not all(math.isfinite(value) and value >= 0.0 for value in metrics):
        passed = False
    return {
        "case_id": case.case_id,
        "system_kind": case.system_kind,
        "domain_length": float(domain_length),
        "n_ref_synthetic": case.n_ref_synthetic,
        "n_tar": case.n_tar,
        "n_sur": case.n_sur,
        "J": case.J,
        "q": case.q,
        "observable_q": bandwidth.observable_q,
        "finite_input_hash": tensor_sha256(state.finite_input),
        "feature_input_hash": tensor_sha256(state.feature_input),
        "target_condition": target_condition,
        "surrogate_condition": surrogate_condition,
        "target_surrogate_conditions_equal": conditions_equal,
        "target_solve_metadata": state.target_metadata,
        "surrogate_solve_metadata": state.surrogate_metadata,
        "target_field_hash": tensor_sha256(state.target_field),
        "surrogate_field_hash": tensor_sha256(state.surrogate_field),
        "coefficient_max_abs_error": coefficient_max_abs_error,
        "coefficient_relative_l2": coefficient_relative_l2,
        "projected_field_absolute_l2": projected_field_absolute_l2,
        "projected_field_relative_l2": projected_field_relative_l2,
        "representation_floor_absolute_l2": representation_floor_absolute_l2,
        "representation_floor_relative_l2": representation_floor_relative_l2,
        "field_space_interpretation": (
            "consistency uses the q-coefficient projection; the separately "
            "reported representation floor compares that projection with "
            "the full finite target field"
        ),
        "independent_solve_outputs": {
            "status": (
                "pass" if state.independent_solve_outputs else "fail"
            ),
            "value": state.independent_solve_outputs,
        },
        "information_isolation": information_isolation,
        "tolerance": float(case.tolerance),
        "matched_tolerance_satisfied": matched_tolerance_satisfied,
        "expected_status": case.expected_status,
        "status": "pass" if passed else "fail",
    }


def run_matched_model1_pipeline_check(
    *,
    domain_length: float,
) -> dict[str, Any]:
    """Run the profile-independent matched Model 1 foundation suite."""

    cases = [
        run_model1_consistency_case(case, domain_length=domain_length)
        for case in matched_model1_cases()
    ]
    positive_cases = [
        case for case in cases if case["expected_status"] == "match"
    ]
    controls = [
        case
        for case in cases
        if case["expected_status"] == "difference_detected"
    ]
    information_cases = [
        case["information_isolation"]
        for case in cases
        if case["information_isolation"] is not None
    ]
    passed = (
        bool(positive_cases)
        and bool(controls)
        and all(case["status"] == "pass" for case in cases)
        and all(
            case["matched_tolerance_satisfied"] is True
            for case in positive_cases
        )
        and all(
            case["matched_tolerance_satisfied"] is False
            for case in controls
        )
        and all(
            information["status"] == "pass"
            for information in information_cases
        )
    )
    return {
        "schema_version": MODEL1_CONSISTENCY_CHECK_SCHEMA_VERSION,
        "status": "pass" if passed else "fail",
        "case_count": len(cases),
        "positive_case_count": len(positive_cases),
        "negative_control_case_count": len(controls),
        "cases_hash": stable_object_hash(cases),
        "case_ids": [case["case_id"] for case in cases],
        "positive_case_ids": [
            case["case_id"] for case in positive_cases
        ],
        "negative_control_case_ids": [
            case["case_id"] for case in controls
        ],
        "cases": cases,
    }


def validate_matched_model1_pipeline_check(
    check: Mapping[str, Any],
    *,
    domain_length: float,
) -> None:
    """Reject missing, legacy, or altered matched-pipeline evidence."""

    if check.get("schema_version") != MODEL1_CONSISTENCY_CHECK_SCHEMA_VERSION:
        raise ValueError("unsupported matched Model 1 pipeline check schema")
    expected = run_matched_model1_pipeline_check(
        domain_length=domain_length,
    )
    if stable_object_hash(dict(check)) != stable_object_hash(expected):
        raise ValueError(
            "matched Model 1 pipeline check is missing or inconsistent"
        )


def model1_foundation_summary(
    check: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the certificate-bound summary of detailed case evidence."""

    cases = check.get("cases")
    if not isinstance(cases, list):
        raise ValueError("matched Model 1 cases must be a list")
    summaries = [
        {
            "case_id": case["case_id"],
            "system_kind": case["system_kind"],
            "expected_status": case["expected_status"],
            "status": case["status"],
            "target_surrogate_conditions_equal": (
                case["target_surrogate_conditions_equal"]
            ),
            "coefficient_max_abs_error": (
                case["coefficient_max_abs_error"]
            ),
            "coefficient_relative_l2": case["coefficient_relative_l2"],
            "projected_field_absolute_l2": (
                case["projected_field_absolute_l2"]
            ),
            "projected_field_relative_l2": (
                case["projected_field_relative_l2"]
            ),
            "representation_floor_relative_l2": (
                case["representation_floor_relative_l2"]
            ),
            "tolerance": case["tolerance"],
            "information_isolation_status": (
                None
                if case["information_isolation"] is None
                else case["information_isolation"]["status"]
            ),
        }
        for case in cases
    ]
    return {
        "schema_version": MODEL1_CONSISTENCY_CHECK_SCHEMA_VERSION,
        "status": check["status"],
        "case_count": check["case_count"],
        "positive_case_count": check["positive_case_count"],
        "negative_control_case_count": check[
            "negative_control_case_count"
        ],
        "case_summaries": summaries,
        "case_summaries_hash": stable_object_hash(summaries),
        "detailed_cases_hash": check["cases_hash"],
        "check_hash": stable_object_hash(dict(check)),
    }


__all__ = [
    "MODEL1_CONSISTENCY_CHECK_SCHEMA_VERSION",
    "MODEL1_CONSISTENCY_TOLERANCE",
    "Model1ConsistencyCase",
    "matched_model1_cases",
    "model1_foundation_summary",
    "run_matched_model1_pipeline_check",
    "run_model1_consistency_case",
    "validate_matched_model1_pipeline_check",
]
