from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from pol.config.models import (
    BurgersConvergenceReferenceSpec,
    BurgersTimeCandidateSpec,
    ReactionDiffusionConvergenceReferenceSpec,
    ReactionDiffusionTimeCandidateSpec,
    ReferenceToleranceSpec,
    ValidationSpec,
)
from pol.data.initial_conditions import InitialConditionArchive
from pol.learning.metrics import samplewise_l2_errors
from pol.math.fourier import real_fourier_analysis
from pol.math.periodic import spectral_resample_periodic
from pol.runtime.device import require_cpu_tensor, require_cpu_tensors
from pol.runtime.hashing import stable_object_hash
from pol.systems.registry import evolve
from .conditions import (
    burgers_refinement_proof,
    make_convergence_row,
    reaction_diffusion_refinement_proof,
)


@dataclass(frozen=True)
class TimeSequenceResult:
    refinement_proof: dict[str, Any]
    conditions: list[dict[str, Any]]
    solutions: list[torch.Tensor]
    runtime_metadata: list[dict[str, Any]]
    rows: list[dict[str, Any]]
    selected_index: int | None


class ValidationSolveFailure(RuntimeError):
    def __init__(self, diagnostic: dict[str, Any]) -> None:
        super().__init__(str(diagnostic.get("message", "validation solve failed")))
        self.diagnostic = diagnostic


TimeCandidateSpec = (
    BurgersTimeCandidateSpec | ReactionDiffusionTimeCandidateSpec
)


def _candidate_evolution(
    spec: ValidationSpec,
    candidate: TimeCandidateSpec,
) -> dict[str, Any]:
    target = spec.target_reference
    if not isinstance(
        target,
        (
            BurgersConvergenceReferenceSpec,
            ReactionDiffusionConvergenceReferenceSpec,
        ),
    ):
        raise TypeError(
            "candidate evolution requires a time-refined target reference"
        )
    base = target.reference_evolution.model_dump(mode="json")
    system = dict(base["system"])
    if isinstance(candidate, BurgersTimeCandidateSpec):
        system.update(
            {
                "solver": candidate.solver,
                "dt": candidate.dt,
                "fine_dt": candidate.fine_dt,
                "dealias": candidate.dealias,
            }
        )
    elif isinstance(candidate, ReactionDiffusionTimeCandidateSpec):
        system.update(
            {
                "solver": candidate.solver,
                "dt": candidate.dt,
                "nonlinear_filter": candidate.nonlinear_filter,
            }
        )
    else:
        raise TypeError(f"unsupported time candidate: {type(candidate).__name__}")
    return {"system": system, "time": base["time"]}


def _pair_metrics(
    coarse: torch.Tensor,
    fine: torch.Tensor,
    *,
    q: int,
    domain_length: float,
) -> dict[str, float]:
    fine_common = spectral_resample_periodic(
        fine, coarse.shape[-1], domain_length=domain_length
    )
    _, relative = samplewise_l2_errors(
        coarse, fine_common, domain_length=domain_length
    )
    coarse_coeff = real_fourier_analysis(coarse, q, domain_length=domain_length)
    fine_coeff = real_fourier_analysis(fine_common, q, domain_length=domain_length)
    denominator = torch.linalg.vector_norm(fine_coeff, dim=-1).clamp_min(
        torch.finfo(fine_coeff.dtype).eps
    )
    low_relative = torch.linalg.vector_norm(
        coarse_coeff - fine_coeff, dim=-1
    ) / denominator
    return {
        "mean_relative_l2": float(relative.mean()),
        "max_relative_l2": float(relative.max()),
        "low_mode_relative_l2": float(low_relative.mean()),
    }


def passes_reference_tolerances(
    metrics: dict[str, float],
    tolerances: ReferenceToleranceSpec,
) -> bool:
    return (
        metrics["mean_relative_l2"] <= tolerances.mean_relative_l2
        and metrics["max_relative_l2"] <= tolerances.max_relative_l2
        and metrics["low_mode_relative_l2"]
        <= tolerances.low_mode_relative_l2
    )


def coarsest_stable_index(rows: list[dict[str, Any]]) -> int | None:
    # A candidate is accepted only if every refinement after it is also within
    # tolerance.  This prevents selecting an accidentally good nonmonotone pair.
    for index in range(len(rows)):
        if all(row["status"] == "pass" for row in rows[index:]):
            return index
    return None


def target_time_refinement_proof(
    target: (
        BurgersConvergenceReferenceSpec
        | ReactionDiffusionConvergenceReferenceSpec
    ),
    candidates: list[TimeCandidateSpec],
) -> dict[str, Any]:
    values = [
        candidate.model_dump(mode="json")
        for candidate in candidates
    ]
    evolution_time = float(target.reference_evolution.time)
    if isinstance(target, BurgersConvergenceReferenceSpec):
        return burgers_refinement_proof(
            values,
            evolution_time=evolution_time,
        )
    return reaction_diffusion_refinement_proof(
        values,
        evolution_time=evolution_time,
    )


def _verify_solver_metadata(
    system_kind: str,
    metadata: dict[str, Any],
    condition: dict[str, Any],
) -> None:
    if system_kind == "burgers":
        actual = {
            name: metadata.get(name)
            for name in (
                "solver",
                "requested_outer_dt",
                "requested_fine_dt",
                "outer_step_count",
                "effective_substep",
                "substeps_per_outer",
                "dealias",
            )
        }
    elif system_kind == "reaction_diffusion":
        actual = {
            "solver": metadata.get("solver"),
            "dt": metadata.get("requested_dt"),
            "nonlinear_filter": metadata.get("nonlinear_filter"),
        }
    else:
        raise ValueError(
            f"unsupported runtime numerical condition: {system_kind}"
        )
    if stable_object_hash(actual) != stable_object_hash(condition):
        raise ValueError(
            f"{system_kind} runtime metadata disagrees with the canonical "
            "numerical condition"
        )


def checked_evolve(
    initial: torch.Tensor,
    evolution: dict[str, Any],
    *,
    domain_length: float,
    stage: str,
    candidate_index: int,
    nx: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    system = evolution.get("system")
    system_kind = (
        str(system.get("kind"))
        if isinstance(system, dict)
        else "unknown"
    )
    try:
        solution, metadata = evolve(
            initial,
            evolution,
            domain_length=domain_length,
        )
    except FloatingPointError as exc:
        raise ValidationSolveFailure(
            {
                "status": "fail",
                "failure_kind": "nonfinite_solver_state",
                "stage": stage,
                "system_kind": system_kind,
                "candidate_index": candidate_index,
                "nx": nx,
                "numerical_condition": (
                    dict(system) if isinstance(system, dict) else system
                ),
                "message": str(exc),
            }
        ) from exc
    if not bool(torch.isfinite(solution).all()):
        raise ValidationSolveFailure(
            {
                "status": "fail",
                "failure_kind": "nonfinite_solver_state",
                "stage": stage,
                "system_kind": system_kind,
                "candidate_index": candidate_index,
                "nx": nx,
                "numerical_condition": (
                    dict(system) if isinstance(system, dict) else system
                ),
                "message": "solver returned a state containing NaN/Inf",
            }
        )
    return solution, metadata


def run_time_sequence_convergence(
    spec: ValidationSpec,
    *,
    initial: torch.Tensor,
    candidates: list[TimeCandidateSpec],
    nx: int,
    reference_candidate_index: int,
    tolerances: ReferenceToleranceSpec,
    boundary: str,
) -> TimeSequenceResult:
    """Solve and score one already-validated fixed-method sequence."""
    target = spec.target_reference
    if not isinstance(
        target,
        (
            BurgersConvergenceReferenceSpec,
            ReactionDiffusionConvergenceReferenceSpec,
        ),
    ):
        raise TypeError(
            "time convergence requires a time-refined target reference"
        )
    proof = target_time_refinement_proof(
        target,
        candidates,
    )
    conditions = proof["ordered_candidates"]
    system_kind = target.reference_evolution.system.kind
    solutions: list[torch.Tensor] = []
    metadata_rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        solution, metadata = checked_evolve(
            initial,
            _candidate_evolution(spec, candidate),
            domain_length=float(spec.domain.length),
            stage=boundary,
            candidate_index=index,
            nx=nx,
        )
        require_cpu_tensor(
            solution,
            boundary=boundary,
            name=f"solution_candidate_{index}",
        )
        _verify_solver_metadata(
            system_kind,
            metadata,
            conditions[index],
        )
        solutions.append(solution)
        metadata_rows.append(metadata)
    rows: list[dict[str, Any]] = []
    for index in range(len(candidates) - 1):
        metrics = _pair_metrics(
            solutions[index],
            solutions[index + 1],
            q=int(target.q_reference_check),
            domain_length=float(spec.domain.length),
        )
        rows.append(
            make_convergence_row(
                check_kind="temporal",
                candidate_axis="numerical_condition",
                coarse_candidate_index=index,
                fine_candidate_index=index + 1,
                coarse_reference_candidate_index=(
                    reference_candidate_index
                ),
                fine_reference_candidate_index=(
                    reference_candidate_index
                ),
                coarse_nx=nx,
                fine_nx=nx,
                coarse_condition_index=index,
                fine_condition_index=index + 1,
                coarse_condition=conditions[index],
                fine_condition=conditions[index + 1],
                common_nx=nx,
                metrics=metrics,
                status=(
                    "pass"
                    if passes_reference_tolerances(metrics, tolerances)
                    else "fail"
                ),
            )
        )
    return TimeSequenceResult(
        refinement_proof=proof,
        conditions=conditions,
        solutions=solutions,
        runtime_metadata=metadata_rows,
        rows=rows,
        selected_index=coarsest_stable_index(rows),
    )


def run_time_refined_reference_convergence(
    spec: ValidationSpec,
    archive: InitialConditionArchive,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    target = spec.target_reference
    if not isinstance(
        target,
        (
            BurgersConvergenceReferenceSpec,
            ReactionDiffusionConvergenceReferenceSpec,
        ),
    ):
        raise TypeError(
            "time-refined reference convergence requires Burgers or "
            "reaction-diffusion"
        )
    require_cpu_tensors(
        archive.__dict__,
        boundary="validation reference-convergence input",
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
    candidates = list(target.time_candidates)
    refinement_proof = target_time_refinement_proof(
        target,
        candidates,
    )
    conditions = refinement_proof["ordered_candidates"]
    system_kind = target.reference_evolution.system.kind
    finest_candidate = candidates[-1]
    finest_condition_index = len(conditions) - 1
    finest_condition = conditions[finest_condition_index]
    spatial_solutions: dict[int, torch.Tensor] = {}
    metadata: dict[str, Any] = {}
    for reference_index, nx in enumerate(nx_values):
        initial = spectral_resample_periodic(initial_master, nx, domain_length=L)
        solution, meta = checked_evolve(
            initial,
            _candidate_evolution(spec, finest_candidate),
            domain_length=L,
            stage="spatial_reference_convergence",
            candidate_index=finest_condition_index,
            nx=nx,
        )
        require_cpu_tensor(
            solution,
            boundary="validation spatial reference-convergence solve",
            name=f"solution_nx_{nx}",
        )
        _verify_solver_metadata(system_kind, meta, finest_condition)
        spatial_solutions[nx] = solution
        metadata[f"spatial_{reference_index}"] = meta
    rows: list[dict[str, Any]] = []
    spatial_rows: list[dict[str, Any]] = []
    for index, (coarse_nx, fine_nx) in enumerate(
        zip(nx_values[:-1], nx_values[1:])
    ):
        metrics = _pair_metrics(
            spatial_solutions[coarse_nx],
            spatial_solutions[fine_nx],
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
            coarse_condition_index=finest_condition_index,
            fine_condition_index=finest_condition_index,
            coarse_condition=finest_condition,
            fine_condition=finest_condition,
            common_nx=coarse_nx,
            metrics=metrics,
            status=(
                "pass"
                if passes_reference_tolerances(metrics, target.reference_tolerances)
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
    initial_finest = spectral_resample_periodic(
        initial_master,
        finest_nx,
        domain_length=L,
    )
    temporal = run_time_sequence_convergence(
        spec,
        initial=initial_finest,
        candidates=candidates,
        nx=finest_nx,
        reference_candidate_index=finest_reference_index,
        tolerances=target.reference_tolerances,
        boundary="validation temporal reference-convergence solve",
    )
    if stable_object_hash(temporal.conditions) != stable_object_hash(
        conditions
    ):
        raise ValueError(
            "primary time convergence conditions changed during reuse"
        )
    temporal_solutions = temporal.solutions
    temporal_metadata = temporal.runtime_metadata
    temporal_rows = temporal.rows
    rows.extend(temporal_rows)
    selected_time_index = temporal.selected_index

    joint_status = "fail"
    joint_row: dict[str, Any] | None = None
    if selected_nx is not None and selected_time_index is not None:
        selected_initial = spectral_resample_periodic(
            initial_master, selected_nx, domain_length=L
        )
        selected_solution, selected_meta = checked_evolve(
            selected_initial,
            _candidate_evolution(spec, candidates[selected_time_index]),
            domain_length=L,
            stage="joint_reference_convergence",
            candidate_index=selected_time_index,
            nx=selected_nx,
        )
        require_cpu_tensor(
            selected_solution,
            boundary="validation joint reference-convergence solve",
            name="selected_solution",
        )
        _verify_solver_metadata(
            system_kind,
            selected_meta,
            conditions[selected_time_index],
        )
        joint_metrics = _pair_metrics(
            selected_solution,
            temporal_solutions[-1],
            q=int(target.q_reference_check),
            domain_length=L,
        )
        joint_status = (
            "pass"
            if passes_reference_tolerances(joint_metrics, target.reference_tolerances)
            else "fail"
        )
        joint_row = make_convergence_row(
            check_kind="joint",
            candidate_axis="coupled",
            coarse_candidate_index=selected_time_index,
            fine_candidate_index=finest_condition_index,
            coarse_reference_candidate_index=spatial_index,
            fine_reference_candidate_index=finest_reference_index,
            coarse_nx=selected_nx,
            fine_nx=finest_nx,
            coarse_condition_index=selected_time_index,
            fine_condition_index=finest_condition_index,
            coarse_condition=conditions[selected_time_index],
            fine_condition=finest_condition,
            common_nx=selected_nx,
            metrics=joint_metrics,
            status=joint_status,
        )
        rows.append(joint_row)
        metadata["joint_selected"] = selected_meta

    result = {
        "status": "pass"
        if selected_nx is not None
        and selected_time_index is not None
        and joint_status == "pass"
        else "fail",
        "spatial_status": "pass" if selected_nx is not None else "fail",
        "temporal_status": "pass" if selected_time_index is not None else "fail",
        "joint_status": joint_status,
        "selected_reference_nx": selected_nx,
        "selected_reference_candidate_index": spatial_index,
        "selected_time_candidate_index": selected_time_index,
        "selected_time_candidate": (
            None
            if selected_time_index is None
            else conditions[selected_time_index]
        ),
        "finest_reference_nx": finest_nx,
        "finest_time_candidate": finest_condition,
        "candidate_refinement_proof": refinement_proof,
        "solver_metadata": {
            **metadata,
            "temporal": temporal_metadata,
        },
        "joint_row": joint_row,
        "rows": rows,
    }
    return result, rows


def time_refined_contract_components(
    spec: ValidationSpec,
    convergence: dict[str, Any],
) -> dict[str, Any]:
    target = spec.target_reference
    if not isinstance(
        target,
        (
            BurgersConvergenceReferenceSpec,
            ReactionDiffusionConvergenceReferenceSpec,
        ),
    ):
        raise TypeError(
            "time-refined contract requires Burgers or reaction-diffusion"
        )
    refinement_proof = target_time_refinement_proof(
        target,
        list(target.time_candidates),
    )
    return {
        "conditions": refinement_proof["ordered_candidates"],
        "refinement_proof": refinement_proof,
        "condition_index": convergence.get(
            "selected_time_candidate_index"
        ),
        "method_kind": "candidate_refinement",
        "temporal_status": (
            "converged"
            if convergence.get("temporal_status") == "pass"
            else "failed"
        ),
    }
