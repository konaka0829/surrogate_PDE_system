from __future__ import annotations

import math
from typing import Any

import torch

from pol.config.models import HeatAnalyticReferenceSpec, ValidationSpec
from pol.data.initial_conditions import InitialConditionArchive
from pol.learning.metrics import samplewise_l2_errors
from pol.math.fourier import real_fourier_analysis
from pol.math.periodic import periodic_grid, spectral_resample_periodic
from pol.runtime.device import require_cpu_tensor, require_cpu_tensors
from pol.runtime.hashing import stable_object_hash
from pol.systems.heat import solve_heat_exact
from .conditions import canonical_numerical_condition, make_convergence_row
from .reference_convergence import (
    checked_evolve,
    coarsest_stable_index,
    passes_reference_tolerances,
)


def _heat_analytic_case(
    spec: ValidationSpec,
    *,
    case_id: str,
    nx: int,
    domain_length: float,
    dtype: torch.dtype,
    basis: str,
    modes: tuple[tuple[int, float, float], ...],
    constant: float,
) -> dict[str, Any]:
    target = spec.target_reference
    if not isinstance(target, HeatAnalyticReferenceSpec):
        raise TypeError("heat analytic checks require heat_analytic")
    system = target.reference_evolution.system
    nu = float(system.nu)
    time = float(target.reference_evolution.time)
    x = periodic_grid(nx, domain_length, dtype=dtype, device="cpu")
    values = torch.full((nx,), constant, dtype=dtype)
    expected = torch.full((nx,), constant, dtype=dtype)
    mode_values: list[int] = []
    multipliers: list[float] = []
    for mode, cosine_amplitude, sine_amplitude in modes:
        angular_wavenumber = (
            2.0 * math.pi * float(mode) / float(domain_length)
        )
        phase = angular_wavenumber * x
        multiplier = math.exp(
            -nu * angular_wavenumber * angular_wavenumber * time
        )
        values = values + (
            cosine_amplitude * torch.cos(phase)
            + sine_amplitude * torch.sin(phase)
        )
        expected = expected + multiplier * (
            cosine_amplitude * torch.cos(phase)
            + sine_amplitude * torch.sin(phase)
        )
        mode_values.append(mode)
        multipliers.append(multiplier)
    actual = solve_heat_exact(
        values,
        nu=nu,
        time=time,
        domain_length=domain_length,
    )
    actual_coefficients = torch.fft.rfft(actual, dim=-1, norm="forward")
    expected_coefficients = torch.fft.rfft(
        expected,
        dim=-1,
        norm="forward",
    )
    max_abs_error = float((actual - expected).abs().max())
    max_coefficient_abs_error = float(
        (actual_coefficients - expected_coefficients).abs().max()
    )
    if dtype == torch.float32:
        atol = float(spec.algebraic_tolerances.float32_atol)
        rtol = float(spec.algebraic_tolerances.float32_rtol)
    else:
        atol = float(spec.algebraic_tolerances.float64_atol)
        rtol = float(spec.algebraic_tolerances.float64_rtol)
    tolerance = atol + rtol * float(expected.abs().max())
    value_pass = bool(
        torch.allclose(actual, expected, atol=atol, rtol=rtol)
    )
    coefficient_pass = bool(
        torch.allclose(
            actual_coefficients,
            expected_coefficients,
            atol=atol,
            rtol=rtol,
        )
    )
    shape_pass = actual.shape == values.shape
    dtype_pass = actual.dtype == dtype
    device_pass = actual.device == torch.device("cpu")
    finite_pass = bool(torch.isfinite(actual).all())
    passed = (
        value_pass
        and coefficient_pass
        and shape_pass
        and dtype_pass
        and device_pass
        and finite_pass
    )
    return {
        "case_id": case_id,
        "basis": basis,
        "nx": nx,
        "domain_length": float(domain_length),
        "dtype": str(dtype).removeprefix("torch."),
        "mode": (
            0
            if not mode_values
            else mode_values[0]
            if len(mode_values) == 1
            else mode_values
        ),
        "expected_multiplier": (
            1.0
            if not multipliers
            else multipliers[0]
            if len(multipliers) == 1
            else multipliers
        ),
        "max_abs_error": max_abs_error,
        "max_coefficient_abs_error": max_coefficient_abs_error,
        "tolerance": tolerance,
        "shape_status": "pass" if shape_pass else "fail",
        "dtype_status": "pass" if dtype_pass else "fail",
        "device_status": "pass" if device_pass else "fail",
        "finite_status": "pass" if finite_pass else "fail",
        "status": "pass" if passed else "fail",
    }


def run_heat_analytic_checks(spec: ValidationSpec) -> dict[str, Any]:
    cases = [
        _heat_analytic_case(
            spec,
            case_id="constant_odd_float64",
            nx=15,
            domain_length=1.0,
            dtype=torch.float64,
            basis="constant",
            modes=(),
            constant=0.375,
        ),
        _heat_analytic_case(
            spec,
            case_id="cosine_even_float64",
            nx=16,
            domain_length=1.0,
            dtype=torch.float64,
            basis="cosine",
            modes=((3, 0.7, 0.0),),
            constant=0.0,
        ),
        _heat_analytic_case(
            spec,
            case_id="sine_odd_nonunit_float64",
            nx=15,
            domain_length=2.5,
            dtype=torch.float64,
            basis="sine",
            modes=((2, 0.0, -0.55),),
            constant=0.0,
        ),
        _heat_analytic_case(
            spec,
            case_id="multimode_even_nonunit_float64",
            nx=16,
            domain_length=1.7,
            dtype=torch.float64,
            basis="constant_plus_sine_cosine",
            modes=((1, 0.7, -0.2), (3, -0.15, 0.4)),
            constant=0.2,
        ),
        _heat_analytic_case(
            spec,
            case_id="cosine_odd_nonunit_float32",
            nx=15,
            domain_length=2.0,
            dtype=torch.float32,
            basis="cosine",
            modes=((2, 0.4, 0.0),),
            constant=-0.1,
        ),
        _heat_analytic_case(
            spec,
            case_id="sine_even_nonunit_float32",
            nx=16,
            domain_length=2.2,
            dtype=torch.float32,
            basis="sine",
            modes=((3, 0.0, 0.35),),
            constant=0.1,
        ),
        _heat_analytic_case(
            spec,
            case_id="nyquist_cosine_unpaired",
            nx=16,
            domain_length=1.3,
            dtype=torch.float64,
            basis="nyquist_cosine_unpaired",
            modes=((8, 0.3, 0.0),),
            constant=0.25,
        ),
    ]
    return {
        "status": (
            "pass"
            if all(case["status"] == "pass" for case in cases)
            else "fail"
        ),
        "temporal_status": "analytic_exact",
        "cases": cases,
    }


def _heat_pair_metrics(
    coarse: torch.Tensor,
    fine: torch.Tensor,
    *,
    q: int,
    domain_length: float,
) -> dict[str, float]:
    coarse_common = spectral_resample_periodic(
        coarse,
        int(fine.shape[-1]),
        domain_length=domain_length,
    )
    _, relative = samplewise_l2_errors(
        coarse_common,
        fine,
        domain_length=domain_length,
    )
    coarse_coeff = real_fourier_analysis(
        coarse_common,
        q,
        domain_length=domain_length,
    )
    fine_coeff = real_fourier_analysis(
        fine,
        q,
        domain_length=domain_length,
    )
    denominator = torch.linalg.vector_norm(fine_coeff, dim=-1).clamp_min(
        torch.finfo(fine_coeff.dtype).eps
    )
    low_relative = torch.linalg.vector_norm(
        coarse_coeff - fine_coeff,
        dim=-1,
    ) / denominator
    return {
        "mean_relative_l2": float(relative.mean()),
        "max_relative_l2": float(relative.max()),
        "low_mode_relative_l2": float(low_relative.mean()),
    }


def _heat_reference_convergence(
    spec: ValidationSpec,
    archive: InitialConditionArchive,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    target = spec.target_reference
    if not isinstance(target, HeatAnalyticReferenceSpec):
        raise TypeError("heat reference convergence requires heat_analytic")
    require_cpu_tensors(
        archive.__dict__,
        boundary="validation heat reference-convergence input",
        name="archive",
    )
    L = float(spec.domain.length)
    ids = torch.tensor(
        target.calibration_sample_ids,
        dtype=torch.long,
        device=archive.values.device,
    )
    initial_master = archive.values.index_select(0, ids)
    nx_values = [int(value) for value in target.reference_nx_candidates]
    evolution = target.reference_evolution.model_dump(mode="json")
    condition = canonical_numerical_condition(
        "heat",
        target.reference_evolution.system.model_dump(mode="json"),
    )
    solutions: dict[int, torch.Tensor] = {}
    metadata: dict[str, Any] = {}
    for nx in nx_values:
        initial = spectral_resample_periodic(
            initial_master,
            nx,
            domain_length=L,
        )
        solution, solver_metadata = checked_evolve(
            initial,
            evolution,
            domain_length=L,
            stage="heat_spatial_reference_convergence",
            candidate_index=0,
            nx=nx,
        )
        require_cpu_tensor(
            solution,
            boundary="validation heat spatial reference-convergence solve",
            name=f"solution_nx_{nx}",
        )
        solutions[nx] = solution
        metadata[f"spatial_{nx}"] = solver_metadata

    rows: list[dict[str, Any]] = []
    spatial_rows: list[dict[str, Any]] = []
    for index, (coarse_nx, fine_nx) in enumerate(
        zip(nx_values[:-1], nx_values[1:])
    ):
        metrics = _heat_pair_metrics(
            solutions[coarse_nx],
            solutions[fine_nx],
            q=int(target.q_reference_check),
            domain_length=L,
        )
        row = make_convergence_row(
            check_kind="spatial",
            candidate_axis="reference_resolution",
            coarse_candidate_index=index,
            fine_candidate_index=index + 1,
            coarse_reference_candidate_index=index,
            fine_reference_candidate_index=index + 1,
            coarse_nx=coarse_nx,
            fine_nx=fine_nx,
            coarse_condition_index=0,
            fine_condition_index=0,
            coarse_condition=condition,
            fine_condition=condition,
            common_nx=fine_nx,
            metrics=metrics,
            status=(
                "pass"
                if passes_reference_tolerances(
                    metrics,
                    target.reference_tolerances,
                )
                else "fail"
            ),
        )
        spatial_rows.append(row)
        rows.append(row)
    spatial_index = coarsest_stable_index(spatial_rows)
    selected_nx = (
        None if spatial_index is None else nx_values[spatial_index]
    )
    finest_nx = nx_values[-1]
    finest_reference_index = len(nx_values) - 1
    joint_row: dict[str, Any] | None = None
    joint_status = "fail"
    if selected_nx is not None:
        joint_metrics = _heat_pair_metrics(
            solutions[selected_nx],
            solutions[finest_nx],
            q=int(target.q_reference_check),
            domain_length=L,
        )
        joint_status = (
            "pass"
            if passes_reference_tolerances(
                joint_metrics,
                target.reference_tolerances,
            )
            else "fail"
        )
        joint_row = make_convergence_row(
            check_kind="joint",
            candidate_axis="coupled",
            coarse_candidate_index=spatial_index,
            fine_candidate_index=finest_reference_index,
            coarse_reference_candidate_index=spatial_index,
            fine_reference_candidate_index=finest_reference_index,
            coarse_nx=selected_nx,
            fine_nx=finest_nx,
            coarse_condition_index=0,
            fine_condition_index=0,
            coarse_condition=condition,
            fine_condition=condition,
            common_nx=finest_nx,
            metrics=joint_metrics,
            status=joint_status,
        )
        rows.append(joint_row)
    return (
        {
            "status": (
                "pass"
                if selected_nx is not None and joint_status == "pass"
                else "fail"
            ),
            "spatial_status": (
                "pass" if selected_nx is not None else "fail"
            ),
            "temporal_status": "analytic_exact",
            "joint_status": joint_status,
            "selected_reference_nx": selected_nx,
            "selected_reference_candidate_index": spatial_index,
            "selected_numerical_condition": condition,
            "selected_numerical_condition_index": 0,
            "finest_reference_nx": finest_nx,
            "solver_metadata": metadata,
            "joint_row": joint_row,
            "rows": rows,
        },
        rows,
    )


def run_heat_reference_checks(
    spec: ValidationSpec,
    archive: InitialConditionArchive,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    analytic = run_heat_analytic_checks(spec)
    convergence, rows = _heat_reference_convergence(spec, archive)
    return {
        "heat_analytic": analytic,
        "reference_convergence": convergence,
    }, rows


def validate_heat_reference_checks(
    spec: ValidationSpec,
    checks: dict[str, Any],
) -> None:
    analytic = checks.get("heat_analytic")
    if not isinstance(analytic, dict) or stable_object_hash(
        analytic
    ) != stable_object_hash(run_heat_analytic_checks(spec)):
        raise ValueError(
            "validation heat analytic check is missing or inconsistent"
        )
    if "reaction_diffusion_characterization" in checks:
        raise ValueError(
            "heat validation must not contain reaction-diffusion checks"
        )
    if "cross_solver_validation" in checks:
        raise ValueError(
            "heat validation must not contain Burgers cross-solver evidence"
        )


def heat_contract_components(
    spec: ValidationSpec,
    convergence: dict[str, Any],
) -> dict[str, Any]:
    target = spec.target_reference
    if not isinstance(target, HeatAnalyticReferenceSpec):
        raise TypeError("heat contract requires heat_analytic")
    return {
        "conditions": [
            canonical_numerical_condition(
                "heat",
                target.reference_evolution.system.model_dump(mode="json"),
            )
        ],
        "refinement_proof": None,
        "condition_index": convergence.get(
            "selected_numerical_condition_index"
        ),
        "method_kind": "analytic_exact",
        "temporal_status": "analytic_exact",
    }
