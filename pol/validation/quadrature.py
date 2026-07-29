"""Analytic characterization of periodic field-space quadrature.

This module is profile-independent and filesystem-free.  It keeps one fixed
continuous target and one fixed coefficient prediction while changing only
the endpoint-free reference grid used by the periodic trapezoidal rule.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch

from pol.learning.metrics import (
    fourier_prediction_metrics,
    fourier_representation_floor,
    periodic_l2_norm,
    samplewise_l2_errors,
)
from pol.math.fourier import real_fourier_synthesis
from pol.math.periodic import periodic_grid
from pol.runtime.hashing import stable_object_hash


FIELD_QUADRATURE_CHECK_SCHEMA_VERSION = "pol-field-quadrature-check-v1"
FIELD_QUADRATURE_TOLERANCE = 1.0e-11
FIELD_QUADRATURE_CANDIDATES = (8, 15, 16, 31, 32)

_PREDICTION_COEFFICIENTS = (
    0.35,
    0.65,
    -0.10,
    0.12,
    -0.08,
    0.05,
    0.04,
)
_TARGET_COEFFICIENTS = (
    0.40,
    0.70,
    -0.20,
    0.00,
    0.00,
    0.15,
    0.00,
    0.00,
    0.00,
    0.00,
    0.00,
    0.00,
    0.00,
    0.50,
    0.25,
)


def analytic_orthonormal_fourier_l2_norm(
    coefficients: Sequence[float] | torch.Tensor,
) -> float | torch.Tensor:
    """Return the continuous L2 norm under the real orthonormal convention."""

    if isinstance(coefficients, torch.Tensor):
        if coefficients.ndim < 1 or coefficients.shape[-1] < 1:
            raise ValueError("coefficients must have a nonempty coefficient axis")
        if not coefficients.dtype.is_floating_point:
            raise TypeError("coefficients must be real floating point")
        return torch.linalg.vector_norm(coefficients, dim=-1)
    values = tuple(float(value) for value in coefficients)
    if not values:
        raise ValueError("coefficients must not be empty")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("coefficients must be finite")
    return math.sqrt(math.fsum(value * value for value in values))


def evaluate_orthonormal_trigonometric_field(
    coefficients: Sequence[float] | torch.Tensor,
    nx: int,
    *,
    domain_length: float,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Evaluate an odd-length real-Fourier coefficient vector at ``nx`` nodes.

    Unlike ``real_fourier_synthesis``, this continuous evaluator intentionally
    allows a field whose highest mode is under-resolved by ``nx``.  It is used
    only to characterize quadrature aliasing, never to recover information.
    """

    if nx < 2:
        raise ValueError("nx must be >= 2")
    if (
        isinstance(domain_length, bool)
        or not math.isfinite(float(domain_length))
        or float(domain_length) <= 0.0
    ):
        raise ValueError("domain_length must be positive and finite")
    tensor = torch.as_tensor(coefficients, dtype=dtype, device="cpu")
    if tensor.ndim < 1 or tensor.shape[-1] < 1 or tensor.shape[-1] % 2 == 0:
        raise ValueError("coefficients must have a positive odd final dimension")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("coefficients must be finite")
    x = periodic_grid(
        nx,
        domain_length,
        dtype=dtype,
        device="cpu",
    )
    out = tensor[..., 0:1] / math.sqrt(float(domain_length))
    scale = math.sqrt(2.0 / float(domain_length))
    for mode in range(1, (int(tensor.shape[-1]) - 1) // 2 + 1):
        phase = (
            2.0
            * torch.pi
            * float(mode)
            * x
            / float(domain_length)
        )
        out = out + scale * (
            tensor[..., 2 * mode - 1 : 2 * mode] * torch.cos(phase)
            + tensor[..., 2 * mode : 2 * mode + 1] * torch.sin(phase)
        )
    return out


def _case_tolerance(dtype: torch.dtype) -> float:
    return 5.0e-6 if dtype == torch.float32 else 5.0e-12


def _norm_case(
    *,
    case_id: str,
    kind: str,
    values: torch.Tensor,
    analytic_norm: torch.Tensor,
    nx: int,
    domain_length: float,
    details: Mapping[str, Any],
) -> dict[str, Any]:
    quadrature = periodic_l2_norm(values, domain_length=domain_length)
    discrepancy = (quadrature - analytic_norm).abs()
    tolerance = _case_tolerance(values.dtype)
    passed = bool(torch.all(discrepancy <= tolerance))
    return {
        "case_id": case_id,
        "kind": kind,
        "nx": int(nx),
        "grid_parity": "odd" if nx % 2 else "even",
        "domain_length": float(domain_length),
        "dtype": str(values.dtype).removeprefix("torch."),
        "batch_shape": list(values.shape[:-1]),
        "analytic_norm": analytic_norm.detach().cpu().tolist(),
        "quadrature_norm": quadrature.detach().cpu().tolist(),
        "max_abs_discrepancy": float(discrepancy.max()),
        "tolerance": tolerance,
        "details": dict(details),
        "status": "pass" if passed else "fail",
    }


def _analytic_norm_cases(
    *,
    configured_domain_length: float,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for case_id, nx, domain_length, constant in (
        ("constant_odd_unit", 15, 1.0, -0.4),
        ("constant_even_nonunit", 16, 2.5, 0.3),
    ):
        values = torch.full((2, nx), constant, dtype=torch.float64)
        analytic = torch.full(
            (2,),
            abs(constant) * math.sqrt(domain_length),
            dtype=torch.float64,
        )
        cases.append(
            _norm_case(
                case_id=case_id,
                kind="constant",
                values=values,
                analytic_norm=analytic,
                nx=nx,
                domain_length=domain_length,
                details={"constant": constant},
            )
        )
    for case_id, basis, nx, domain_length, mode, amplitude in (
        ("cosine_odd_nonunit", "cosine", 17, 1.7, 3, 0.8),
        ("sine_even_nonunit", "sine", 18, 2.2, 4, -0.6),
    ):
        x = periodic_grid(nx, domain_length, dtype=torch.float64)
        phase = 2.0 * torch.pi * float(mode) * x / domain_length
        values = amplitude * (
            torch.cos(phase) if basis == "cosine" else torch.sin(phase)
        )
        analytic = torch.tensor(
            abs(amplitude) * math.sqrt(domain_length / 2.0),
            dtype=torch.float64,
        )
        cases.append(
            _norm_case(
                case_id=case_id,
                kind=basis,
                values=values,
                analytic_norm=analytic,
                nx=nx,
                domain_length=domain_length,
                details={"mode": mode, "amplitude": amplitude},
            )
        )
    parseval_specs = (
        (
            "parseval_odd_configured_domain_float64",
            15,
            float(configured_domain_length),
            torch.float64,
        ),
        (
            "parseval_even_nonunit_float32",
            16,
            1.7,
            torch.float32,
        ),
    )
    coefficients_base = torch.tensor(
        [
            [
                [0.4, 0.7, -0.2, 0.1, 0.3],
                [-0.2, 0.1, 0.6, -0.4, 0.2],
            ],
            [
                [0.3, -0.5, 0.2, 0.25, -0.1],
                [0.1, 0.2, 0.3, 0.4, 0.5],
            ],
        ],
        dtype=torch.float64,
    )
    for case_id, nx, domain_length, dtype in parseval_specs:
        coefficients = coefficients_base.to(dtype=dtype)
        values = real_fourier_synthesis(
            coefficients,
            nx,
            domain_length=domain_length,
        )
        analytic = analytic_orthonormal_fourier_l2_norm(coefficients)
        assert isinstance(analytic, torch.Tensor)
        cases.append(
            _norm_case(
                case_id=case_id,
                kind="orthonormal_multimode_parseval",
                values=values,
                analytic_norm=analytic,
                nx=nx,
                domain_length=domain_length,
                details={
                    "coefficient_dimension": int(coefficients.shape[-1]),
                    "parseval_formula": (
                        "continuous_norm_squared=sum_i coefficient_i^2"
                    ),
                },
            )
        )
    return cases


def _with_row_hash(row: dict[str, Any]) -> dict[str, Any]:
    return {
        **row,
        "row_hash": stable_object_hash(row),
    }


def _convergence_row(
    n_ref: int,
    *,
    domain_length: float,
    prediction_coefficients: torch.Tensor,
    target_projection_coefficients: torch.Tensor,
    target_coefficients: torch.Tensor,
    data_target: torch.Tensor,
    analytic_absolute_l2: float,
    analytic_relative_l2: float,
    analytic_floor_absolute_l2: float,
    analytic_floor_relative_l2: float,
) -> dict[str, Any]:
    target = evaluate_orthonormal_trigonometric_field(
        target_coefficients,
        n_ref,
        domain_length=domain_length,
    )
    prediction = real_fourier_synthesis(
        prediction_coefficients,
        n_ref,
        domain_length=domain_length,
    )
    direct_absolute, direct_relative = samplewise_l2_errors(
        prediction,
        target,
        domain_length=domain_length,
    )
    direct_absolute_value = float(direct_absolute.max())
    direct_relative_value = float(direct_relative.max())
    absolute_discrepancy = abs(
        direct_absolute_value - analytic_absolute_l2
    )
    relative_discrepancy = abs(
        direct_relative_value - analytic_relative_l2
    )

    wrapper = fourier_prediction_metrics(
        prediction_coefficients,
        target_projection_coefficients,
        data_target,
        target,
        n_data=int(data_target.shape[-1]),
        n_reference=n_ref,
        domain_length=domain_length,
    )
    wrapper_absolute = float(wrapper["field_absolute_l2_mean"])
    wrapper_relative = float(wrapper["field_relative_l2_mean"])
    wrapper_absolute_discrepancy = abs(
        wrapper_absolute - direct_absolute_value
    )
    wrapper_relative_discrepancy = abs(
        wrapper_relative - direct_relative_value
    )
    wrapper_status = (
        wrapper_absolute_discrepancy <= FIELD_QUADRATURE_TOLERANCE
        and wrapper_relative_discrepancy <= FIELD_QUADRATURE_TOLERANCE
    )

    target_projection = real_fourier_synthesis(
        target_projection_coefficients,
        n_ref,
        domain_length=domain_length,
    )
    floor_absolute, floor_relative = samplewise_l2_errors(
        target_projection,
        target,
        domain_length=domain_length,
    )
    floor_absolute_value = float(floor_absolute.max())
    floor_relative_value = float(floor_relative.max())
    floor_absolute_discrepancy = abs(
        floor_absolute_value - analytic_floor_absolute_l2
    )
    floor_relative_discrepancy = abs(
        floor_relative_value - analytic_floor_relative_l2
    )
    floor_wrapper = fourier_representation_floor(
        target_projection_coefficients,
        data_target,
        target,
        n_data=int(data_target.shape[-1]),
        n_reference=n_ref,
        domain_length=domain_length,
    )
    floor_wrapper_value = float(
        floor_wrapper["representation_floor_relative_l2_mean"]
    )
    floor_wrapper_discrepancy = abs(
        floor_wrapper_value - floor_relative_value
    )
    floor_wrapper_status = (
        floor_wrapper_discrepancy <= FIELD_QUADRATURE_TOLERANCE
    )

    analytic_agreement = (
        absolute_discrepancy <= FIELD_QUADRATURE_TOLERANCE
        and relative_discrepancy <= FIELD_QUADRATURE_TOLERANCE
    )
    floor_analytic_agreement = (
        floor_absolute_discrepancy <= FIELD_QUADRATURE_TOLERANCE
        and floor_relative_discrepancy <= FIELD_QUADRATURE_TOLERANCE
    )
    candidate_pass = (
        analytic_agreement
        and floor_analytic_agreement
        and wrapper_status
        and floor_wrapper_status
    )
    target_max_mode = (int(target_coefficients.shape[-1]) - 1) // 2
    resolved = n_ref > 2 * target_max_mode
    characterization_pass = candidate_pass is resolved
    return _with_row_hash(
        {
            "n_ref": n_ref,
            "grid_parity": "odd" if n_ref % 2 else "even",
            "target_max_mode": target_max_mode,
            "resolved_for_squared_integrands": resolved,
            "quadrature_absolute_l2": direct_absolute_value,
            "quadrature_relative_l2": direct_relative_value,
            "analytic_absolute_l2": analytic_absolute_l2,
            "analytic_relative_l2": analytic_relative_l2,
            "absolute_discrepancy": absolute_discrepancy,
            "relative_discrepancy": relative_discrepancy,
            "tolerance": FIELD_QUADRATURE_TOLERANCE,
            "analytic_agreement_status": (
                "pass" if analytic_agreement else "fail"
            ),
            "metric_wrapper_consistency": {
                "field_absolute_l2": wrapper_absolute,
                "field_relative_l2": wrapper_relative,
                "direct_absolute_discrepancy": (
                    wrapper_absolute_discrepancy
                ),
                "direct_relative_discrepancy": (
                    wrapper_relative_discrepancy
                ),
                "status": "pass" if wrapper_status else "fail",
            },
            "data_space_metrics": {
                "data_field_absolute_l2": float(
                    wrapper["data_field_absolute_l2_mean"]
                ),
                "data_field_relative_l2": float(
                    wrapper["data_field_relative_l2_mean"]
                ),
            },
            "representation_floor": {
                "quadrature_absolute_l2": floor_absolute_value,
                "quadrature_relative_l2": floor_relative_value,
                "analytic_absolute_l2": analytic_floor_absolute_l2,
                "analytic_relative_l2": analytic_floor_relative_l2,
                "absolute_discrepancy": floor_absolute_discrepancy,
                "relative_discrepancy": floor_relative_discrepancy,
                "analytic_agreement_status": (
                    "pass" if floor_analytic_agreement else "fail"
                ),
                "wrapper_relative_l2": floor_wrapper_value,
                "wrapper_direct_discrepancy": (
                    floor_wrapper_discrepancy
                ),
                "wrapper_status": (
                    "pass" if floor_wrapper_status else "fail"
                ),
                "data_relative_l2": float(
                    floor_wrapper[
                        "data_representation_floor_relative_l2_mean"
                    ]
                ),
            },
            "expected_status": "pass" if resolved else "fail",
            "characterization_status": (
                "pass" if characterization_pass else "fail"
            ),
            "status": "pass" if candidate_pass else "fail",
        }
    )


def _selected_stable_suffix(
    rows: Sequence[Mapping[str, Any]],
) -> int | None:
    if len(rows) < 2 or any(
        row.get("status") != "pass" for row in rows[-2:]
    ):
        return None
    for index in range(len(rows)):
        if all(row.get("status") == "pass" for row in rows[index:]):
            return index
    return None


def _range_payload(values: Sequence[float]) -> dict[str, float]:
    return {
        "minimum": min(values),
        "maximum": max(values),
        "range": max(values) - min(values),
    }


def run_field_quadrature_check(
    *,
    domain_length: float,
) -> dict[str, Any]:
    """Run analytic norm and reference-grid quadrature characterization."""

    if (
        isinstance(domain_length, bool)
        or not math.isfinite(float(domain_length))
        or float(domain_length) <= 0.0
    ):
        raise ValueError("domain_length must be positive and finite")
    analytic_cases = _analytic_norm_cases(
        configured_domain_length=float(domain_length),
    )

    prediction = torch.tensor(
        [_PREDICTION_COEFFICIENTS],
        dtype=torch.float64,
    )
    target_coefficients = torch.tensor(
        [_TARGET_COEFFICIENTS],
        dtype=torch.float64,
    )
    q = int(prediction.shape[-1])
    target_projection = target_coefficients[:, :q].clone()
    embedded_prediction = torch.zeros_like(target_coefficients)
    embedded_prediction[:, :q] = prediction
    error_coefficients = target_coefficients - embedded_prediction
    analytic_absolute_l2 = float(
        analytic_orthonormal_fourier_l2_norm(error_coefficients)
    )
    analytic_target_l2 = float(
        analytic_orthonormal_fourier_l2_norm(target_coefficients)
    )
    analytic_relative_l2 = analytic_absolute_l2 / analytic_target_l2
    floor_coefficients = target_coefficients.clone()
    floor_coefficients[:, :q] = 0.0
    analytic_floor_absolute_l2 = float(
        analytic_orthonormal_fourier_l2_norm(floor_coefficients)
    )
    analytic_floor_relative_l2 = (
        analytic_floor_absolute_l2 / analytic_target_l2
    )

    n_data = 16
    data_target = evaluate_orthonormal_trigonometric_field(
        target_coefficients,
        n_data,
        domain_length=domain_length,
    )
    rows = [
        _convergence_row(
            n_ref,
            domain_length=domain_length,
            prediction_coefficients=prediction,
            target_projection_coefficients=target_projection,
            target_coefficients=target_coefficients,
            data_target=data_target,
            analytic_absolute_l2=analytic_absolute_l2,
            analytic_relative_l2=analytic_relative_l2,
            analytic_floor_absolute_l2=analytic_floor_absolute_l2,
            analytic_floor_relative_l2=analytic_floor_relative_l2,
        )
        for n_ref in FIELD_QUADRATURE_CANDIDATES
    ]
    selected_index = _selected_stable_suffix(rows)
    allowed_indices = (
        []
        if selected_index is None
        else list(range(selected_index, len(rows)))
    )
    data_absolute_values = [
        float(row["data_space_metrics"]["data_field_absolute_l2"])
        for row in rows
    ]
    data_relative_values = [
        float(row["data_space_metrics"]["data_field_relative_l2"])
        for row in rows
    ]
    data_floor_values = [
        float(row["representation_floor"]["data_relative_l2"])
        for row in rows
    ]
    data_invariance = {
        "n_tar": n_data,
        "field_absolute_l2": _range_payload(data_absolute_values),
        "field_relative_l2": _range_payload(data_relative_values),
        "representation_floor_relative_l2": _range_payload(
            data_floor_values
        ),
    }
    data_invariance_status = all(
        payload["range"] <= FIELD_QUADRATURE_TOLERANCE
        for name, payload in data_invariance.items()
        if name != "n_tar"
    )
    data_invariance["status"] = (
        "pass" if data_invariance_status else "fail"
    )

    wrapper_status = all(
        row["metric_wrapper_consistency"]["status"] == "pass"
        for row in rows
    )
    floor_characterization_status = all(
        row["representation_floor"]["analytic_agreement_status"]
        == row["expected_status"]
        and row["representation_floor"]["wrapper_status"] == "pass"
        for row in rows
    )
    convergence_characterization_status = all(
        row["characterization_status"] == "pass" for row in rows
    )
    finest_pair_status = (
        len(rows) >= 2
        and all(row["status"] == "pass" for row in rows[-2:])
    )
    selection_status = (
        selected_index is not None
        and rows[selected_index]["resolved_for_squared_integrands"] is True
        and all(row["status"] == "pass" for row in rows[selected_index:])
    )
    analytic_status = all(case["status"] == "pass" for case in analytic_cases)
    passed = (
        analytic_status
        and convergence_characterization_status
        and finest_pair_status
        and selection_status
        and wrapper_status
        and floor_characterization_status
        and data_invariance_status
    )
    convergence = {
        "candidate_n_ref": list(FIELD_QUADRATURE_CANDIDATES),
        "candidate_order": "strictly_increasing",
        "n_tar": n_data,
        "prediction_q": q,
        "target_max_mode": (
            int(target_coefficients.shape[-1]) - 1
        )
        // 2,
        "prediction_coefficients": list(_PREDICTION_COEFFICIENTS),
        "target_coefficients": list(_TARGET_COEFFICIENTS),
        "analytic_target_l2": analytic_target_l2,
        "analytic_absolute_l2": analytic_absolute_l2,
        "analytic_relative_l2": analytic_relative_l2,
        "analytic_representation_floor_absolute_l2": (
            analytic_floor_absolute_l2
        ),
        "analytic_representation_floor_relative_l2": (
            analytic_floor_relative_l2
        ),
        "tolerance": FIELD_QUADRATURE_TOLERANCE,
        "rows": rows,
        "row_hashes": [row["row_hash"] for row in rows],
        "rows_hash": stable_object_hash(rows),
        "selection_policy": (
            "coarsest_complete_analytic_agreement_suffix_"
            "with_finest_pair_required"
        ),
        "selected_candidate_index": selected_index,
        "selected_n_ref": (
            None
            if selected_index is None
            else int(rows[selected_index]["n_ref"])
        ),
        "allowed_suffix_indices": allowed_indices,
        "allowed_suffix_n_ref": [
            int(rows[index]["n_ref"]) for index in allowed_indices
        ],
        "finest_resolved_pair": [
            int(rows[-2]["n_ref"]),
            int(rows[-1]["n_ref"]),
        ],
        "finest_resolved_pair_status": (
            "pass" if finest_pair_status else "fail"
        ),
        "under_resolved_candidate_status": rows[0]["status"],
        "status": "pass" if (
            convergence_characterization_status
            and finest_pair_status
            and selection_status
        ) else "fail",
    }
    return {
        "schema_version": FIELD_QUADRATURE_CHECK_SCHEMA_VERSION,
        "status": "pass" if passed else "fail",
        "norm_convention": {
            "name": "endpoint_free_periodic_trapezoidal_l2",
            "formula": "sqrt((L/n) * sum_j value_j^2)",
            "continuous_measure": "dx on [0,L)",
            "real_fourier_basis": "L2_orthonormal",
            "parseval_formula": (
                "continuous_norm_squared=sum_i coefficient_i^2"
            ),
        },
        "relative_error_denominator_policy": (
            "target_periodic_l2_norm_clamped_below_by_dtype_epsilon"
        ),
        "analytic_cases": analytic_cases,
        "analytic_cases_hash": stable_object_hash(analytic_cases),
        "convergence": convergence,
        "convergence_hash": stable_object_hash(convergence),
        "metric_wrapper_consistency": {
            "same_prediction_all_reference_grids": True,
            "same_continuous_target_all_reference_grids": True,
            "only_reference_quadrature_grid_changes": True,
            "direct_wrapper_status": (
                "pass" if wrapper_status else "fail"
            ),
            "data_space_invariance": data_invariance,
            "status": "pass" if (
                wrapper_status and data_invariance_status
            ) else "fail",
        },
        "representation_floor_consistency": {
            "reference_floor_characterization_status": (
                "pass" if floor_characterization_status else "fail"
            ),
            "data_floor_invariance_status": (
                "pass" if data_invariance_status else "fail"
            ),
            "status": "pass" if (
                floor_characterization_status and data_invariance_status
            ) else "fail",
        },
    }


def _validate_check_structure(check: Mapping[str, Any]) -> None:
    convergence = check.get("convergence")
    if not isinstance(convergence, Mapping):
        raise ValueError("field quadrature convergence block is missing")
    candidates = convergence.get("candidate_n_ref")
    if (
        not isinstance(candidates, list)
        or len(candidates) < 3
        or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in candidates
        )
        or candidates != sorted(set(candidates))
    ):
        raise ValueError(
            "field quadrature candidate grids must be strictly increasing"
        )
    rows = convergence.get("rows")
    if not isinstance(rows, list) or len(rows) != len(candidates):
        raise ValueError("field quadrature rows do not match candidate grids")
    for candidate, row in zip(candidates, rows, strict=True):
        if not isinstance(row, Mapping) or row.get("n_ref") != candidate:
            raise ValueError("field quadrature row/candidate mismatch")
        unhashed = {
            key: value for key, value in row.items() if key != "row_hash"
        }
        if row.get("row_hash") != stable_object_hash(unhashed):
            raise ValueError("field quadrature row hash mismatch")
    if convergence.get("rows_hash") != stable_object_hash(rows):
        raise ValueError("field quadrature rows hash mismatch")
    selected_index = _selected_stable_suffix(rows)
    if convergence.get("selected_candidate_index") != selected_index:
        raise ValueError("field quadrature selected candidate mismatch")
    allowed_indices = (
        []
        if selected_index is None
        else list(range(selected_index, len(rows)))
    )
    if (
        convergence.get("allowed_suffix_indices") != allowed_indices
        or convergence.get("allowed_suffix_n_ref")
        != [candidates[index] for index in allowed_indices]
    ):
        raise ValueError("field quadrature allowed suffix mismatch")


def validate_field_quadrature_check(
    check: Mapping[str, Any],
    *,
    domain_length: float,
) -> None:
    """Reject legacy, malformed, or altered field-quadrature evidence."""

    if check.get("schema_version") != FIELD_QUADRATURE_CHECK_SCHEMA_VERSION:
        raise ValueError("unsupported field quadrature check schema")
    _validate_check_structure(check)
    expected = run_field_quadrature_check(domain_length=domain_length)
    if stable_object_hash(dict(check)) != stable_object_hash(expected):
        raise ValueError("field quadrature check is missing or inconsistent")


def field_quadrature_foundation_summary(
    check: Mapping[str, Any],
) -> dict[str, Any]:
    """Return certificate-bound semantic summary and hashes."""

    convergence = check["convergence"]
    metric_wrapper = check["metric_wrapper_consistency"]
    representation_floor = check["representation_floor_consistency"]
    return {
        "schema_version": FIELD_QUADRATURE_CHECK_SCHEMA_VERSION,
        "status": check["status"],
        "norm_convention": dict(check["norm_convention"]),
        "relative_error_denominator_policy": check[
            "relative_error_denominator_policy"
        ],
        "analytic_cases_hash": check["analytic_cases_hash"],
        "convergence_hash": check["convergence_hash"],
        "selected_n_ref": convergence["selected_n_ref"],
        "allowed_suffix_n_ref": list(
            convergence["allowed_suffix_n_ref"]
        ),
        "tolerance": convergence["tolerance"],
        "metric_wrapper_consistency_status": metric_wrapper["status"],
        "data_space_invariance_status": metric_wrapper[
            "data_space_invariance"
        ]["status"],
        "representation_floor_consistency_status": representation_floor[
            "status"
        ],
        "check_hash": stable_object_hash(dict(check)),
    }


__all__ = [
    "FIELD_QUADRATURE_CANDIDATES",
    "FIELD_QUADRATURE_CHECK_SCHEMA_VERSION",
    "FIELD_QUADRATURE_TOLERANCE",
    "analytic_orthonormal_fourier_l2_norm",
    "evaluate_orthonormal_trigonometric_field",
    "field_quadrature_foundation_summary",
    "run_field_quadrature_check",
    "validate_field_quadrature_check",
]
