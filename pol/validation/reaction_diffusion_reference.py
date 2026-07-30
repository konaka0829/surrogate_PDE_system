from __future__ import annotations

import math
from typing import Any

import torch

from pol.config.models import (
    ReactionDiffusionConvergenceReferenceSpec,
    ValidationSpec,
)
from pol.data.initial_conditions import InitialConditionArchive
from pol.math.periodic import periodic_grid
from pol.runtime.hashing import stable_object_hash
from pol.systems.reaction_diffusion import solve_reaction_diffusion
from .check_utils import algebraic_allclose
from .reference_convergence import (
    ValidationSolveFailure,
    run_time_refined_reference_convergence,
)


def _reaction_diffusion_actual(
    values: torch.Tensor,
    *,
    case_id: str,
    nu: float,
    alpha: float,
    beta: float,
    time: float,
    dt: float,
    domain_length: float,
    nonlinear_filter: str,
) -> torch.Tensor:
    try:
        result = solve_reaction_diffusion(
            values,
            nu=nu,
            alpha=alpha,
            beta=beta,
            time=time,
            dt=dt,
            domain_length=domain_length,
            nonlinear_filter=nonlinear_filter,
        )
    except FloatingPointError as exc:
        raise ValidationSolveFailure(
            {
                "status": "fail",
                "failure_kind": "nonfinite_reaction_diffusion_solve",
                "stage": "analytic_characterization",
                "case_id": case_id,
                "message": str(exc),
            }
        ) from exc
    if not bool(torch.isfinite(result.values).all()):
        raise ValidationSolveFailure(
            {
                "status": "fail",
                "failure_kind": "nonfinite_reaction_diffusion_solve",
                "stage": "analytic_characterization",
                "case_id": case_id,
                "message": (
                    "reaction-diffusion characterization produced NaN/Inf"
                ),
            }
        )
    return result.values


def _reaction_diffusion_constant_case(
    spec: ValidationSpec,
    *,
    case_id: str,
    value: float,
    nx: int,
    domain_length: float,
    nonlinear_filter: str,
    steps: int,
) -> dict[str, Any]:
    target = spec.target_reference
    if not isinstance(
        target,
        ReactionDiffusionConvergenceReferenceSpec,
    ):
        raise TypeError(
            "reaction-diffusion characterization requires "
            "reaction_diffusion_convergence"
        )
    system = target.reference_evolution.system
    dt = float(system.dt)
    alpha = float(system.alpha)
    beta = float(system.beta)
    scalar = float(value)
    for _ in range(steps):
        scalar = scalar + dt * alpha * scalar - dt * beta * scalar**3
    initial = torch.full(
        (2, nx),
        float(value),
        dtype=torch.float64,
        device="cpu",
    )
    actual = _reaction_diffusion_actual(
        initial,
        case_id=case_id,
        nu=float(system.nu),
        alpha=alpha,
        beta=beta,
        time=steps * dt,
        dt=dt,
        domain_length=domain_length,
        nonlinear_filter=nonlinear_filter,
    )
    expected = torch.full_like(actual, scalar)
    passed = algebraic_allclose(actual, expected, spec)
    return {
        "case_id": case_id,
        "characterization": "independent_scalar_recurrence",
        "initial_constant": float(value),
        "expected_final_constant": scalar,
        "nx": nx,
        "grid_parity": "even" if nx % 2 == 0 else "odd",
        "domain_length": float(domain_length),
        "dt": dt,
        "step_count": steps,
        "nonlinear_filter": nonlinear_filter,
        "max_abs_error": float((actual - expected).abs().max()),
        "finite_status": "pass",
        "status": "pass" if passed else "fail",
    }


def _reaction_diffusion_equilibrium_case(
    spec: ValidationSpec,
    *,
    sign: int,
    nonlinear_filter: str,
) -> dict[str, Any]:
    target = spec.target_reference
    if not isinstance(
        target,
        ReactionDiffusionConvergenceReferenceSpec,
    ):
        raise TypeError(
            "reaction-diffusion characterization requires "
            "reaction_diffusion_convergence"
        )
    system = target.reference_evolution.system
    alpha = float(system.alpha)
    beta = float(system.beta)
    case_id = (
        f"equilibrium_{'positive' if sign > 0 else 'negative'}_"
        f"{nonlinear_filter}"
    )
    if alpha <= 0.0 or beta <= 0.0:
        return {
            "case_id": case_id,
            "characterization": "nonzero_constant_equilibrium",
            "applicable": False,
            "reason": "requires alpha > 0 and beta > 0",
            "status": "not_applicable",
        }
    value = float(sign) * math.sqrt(alpha / beta)
    dt = float(system.dt)
    nx = 15 if sign > 0 else 16
    initial = torch.full((1, nx), value, dtype=torch.float64)
    actual = _reaction_diffusion_actual(
        initial,
        case_id=case_id,
        nu=float(system.nu),
        alpha=alpha,
        beta=beta,
        time=3 * dt,
        dt=dt,
        domain_length=2.3,
        nonlinear_filter=nonlinear_filter,
    )
    expected = torch.full_like(actual, value)
    passed = algebraic_allclose(actual, expected, spec)
    return {
        "case_id": case_id,
        "characterization": "nonzero_constant_equilibrium",
        "applicable": True,
        "equilibrium": value,
        "nx": nx,
        "domain_length": 2.3,
        "dt": dt,
        "step_count": 3,
        "nonlinear_filter": nonlinear_filter,
        "max_abs_error": float((actual - expected).abs().max()),
        "finite_status": "pass",
        "status": "pass" if passed else "fail",
    }


def _reaction_diffusion_linear_mode_case(
    spec: ValidationSpec,
    *,
    case_id: str,
    nx: int,
    domain_length: float,
    mode: int,
    basis: str,
    nonlinear_filter: str,
) -> dict[str, Any]:
    target = spec.target_reference
    if not isinstance(
        target,
        ReactionDiffusionConvergenceReferenceSpec,
    ):
        raise TypeError(
            "reaction-diffusion characterization requires "
            "reaction_diffusion_convergence"
        )
    system = target.reference_evolution.system
    dt = float(system.dt)
    nu = float(system.nu)
    alpha = float(system.alpha)
    x = periodic_grid(
        nx,
        domain_length,
        dtype=torch.float64,
        device="cpu",
    )
    angular_wavenumber = 2.0 * math.pi * mode / domain_length
    if basis == "cosine":
        initial_one = 0.4 * torch.cos(angular_wavenumber * x)
    elif basis == "sine":
        initial_one = -0.35 * torch.sin(angular_wavenumber * x)
    else:
        raise ValueError(f"unsupported linear-mode basis: {basis}")
    initial = initial_one.unsqueeze(0)
    multiplier = (1.0 + dt * alpha) / (
        1.0 + dt * nu * angular_wavenumber**2
    )
    expected = multiplier * initial
    actual = _reaction_diffusion_actual(
        initial,
        case_id=case_id,
        nu=nu,
        alpha=alpha,
        beta=0.0,
        time=dt,
        dt=dt,
        domain_length=domain_length,
        nonlinear_filter=nonlinear_filter,
    )
    passed = algebraic_allclose(actual, expected, spec)
    return {
        "case_id": case_id,
        "characterization": "beta_zero_linear_mode_one_step",
        "beta": 0.0,
        "basis": basis,
        "mode": mode,
        "physical_angular_wavenumber": angular_wavenumber,
        "expected_multiplier": multiplier,
        "nx": nx,
        "grid_parity": "even" if nx % 2 == 0 else "odd",
        "domain_length": float(domain_length),
        "dt": dt,
        "step_count": 1,
        "nonlinear_filter": nonlinear_filter,
        "max_abs_error": float((actual - expected).abs().max()),
        "finite_status": "pass",
        "status": "pass" if passed else "fail",
    }


def run_reaction_diffusion_characterization(
    spec: ValidationSpec,
) -> dict[str, Any]:
    target = spec.target_reference
    if not isinstance(
        target,
        ReactionDiffusionConvergenceReferenceSpec,
    ):
        raise TypeError(
            "reaction-diffusion characterization requires "
            "reaction_diffusion_convergence"
        )
    system = target.reference_evolution.system
    dt = float(system.dt)
    zero_initial = torch.zeros((2, 15), dtype=torch.float64)
    zero_actual = _reaction_diffusion_actual(
        zero_initial,
        case_id="zero_equilibrium",
        nu=float(system.nu),
        alpha=float(system.alpha),
        beta=float(system.beta),
        time=4 * dt,
        dt=dt,
        domain_length=1.9,
        nonlinear_filter="two_thirds",
    )
    zero_exact = bool(torch.equal(zero_actual, zero_initial))
    zero_case = {
        "case_id": "zero_equilibrium",
        "characterization": "zero_equilibrium",
        "nx": 15,
        "domain_length": 1.9,
        "dt": dt,
        "step_count": 4,
        "nonlinear_filter": "two_thirds",
        "exact_zero": zero_exact,
        "max_abs_error": float(zero_actual.abs().max()),
        "finite_status": "pass",
        "status": "pass" if zero_exact else "fail",
    }
    constant_cases = [
        _reaction_diffusion_constant_case(
            spec,
            case_id="positive_odd_none",
            value=0.25,
            nx=15,
            domain_length=2.5,
            nonlinear_filter="none",
            steps=4,
        ),
        _reaction_diffusion_constant_case(
            spec,
            case_id="negative_even_two_thirds",
            value=-0.4,
            nx=16,
            domain_length=1.7,
            nonlinear_filter="two_thirds",
            steps=5,
        ),
    ]
    equilibrium_cases = [
        _reaction_diffusion_equilibrium_case(
            spec,
            sign=1,
            nonlinear_filter="none",
        ),
        _reaction_diffusion_equilibrium_case(
            spec,
            sign=-1,
            nonlinear_filter="two_thirds",
        ),
    ]
    linear_mode_cases = [
        _reaction_diffusion_linear_mode_case(
            spec,
            case_id="linear_cosine_odd_nonunit_none",
            nx=15,
            domain_length=2.5,
            mode=2,
            basis="cosine",
            nonlinear_filter="none",
        ),
        _reaction_diffusion_linear_mode_case(
            spec,
            case_id="linear_sine_even_nonunit_two_thirds",
            nx=16,
            domain_length=1.7,
            mode=3,
            basis="sine",
            nonlinear_filter="two_thirds",
        ),
    ]
    required_statuses = [
        zero_case["status"],
        *(case["status"] for case in constant_cases),
        *(
            case["status"]
            for case in equilibrium_cases
            if case["status"] != "not_applicable"
        ),
        *(case["status"] for case in linear_mode_cases),
    ]
    return {
        "schema_version": (
            "pol-reaction-diffusion-characterization-v1"
        ),
        "expected_value_construction": (
            "independent_scalar_and_fourier_mode_algebra"
        ),
        "status": (
            "pass"
            if all(value == "pass" for value in required_statuses)
            else "fail"
        ),
        "zero_equilibrium": zero_case,
        "constant_scalar_recurrence": constant_cases,
        "nonzero_equilibria": equilibrium_cases,
        "beta_zero_linear_modes": linear_mode_cases,
    }


def run_reaction_diffusion_reference_checks(
    spec: ValidationSpec,
    archive: InitialConditionArchive,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    characterization = run_reaction_diffusion_characterization(spec)
    convergence, rows = run_time_refined_reference_convergence(
        spec,
        archive,
    )
    return {
        "reaction_diffusion_characterization": characterization,
        "reference_convergence": convergence,
    }, rows


def validate_reaction_diffusion_reference_checks(
    spec: ValidationSpec,
    checks: dict[str, Any],
) -> None:
    characterization = checks.get("reaction_diffusion_characterization")
    if not isinstance(characterization, dict) or stable_object_hash(
        characterization
    ) != stable_object_hash(
        run_reaction_diffusion_characterization(spec)
    ):
        raise ValueError(
            "validation reaction-diffusion characterization is missing "
            "or inconsistent"
        )
    if "heat_analytic" in checks:
        raise ValueError(
            "reaction-diffusion validation must not contain heat checks"
        )
    if "cross_solver_validation" in checks:
        raise ValueError(
            "reaction-diffusion validation must not contain cross-solver evidence"
        )
