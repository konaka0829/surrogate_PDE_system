from __future__ import annotations

from typing import Any

import torch

from pol.config.models import (
    BurgersConvergenceReferenceSpec,
    EnabledBurgersCrossSolverValidationSpec,
    ValidationSpec,
)
from pol.data.initial_conditions import InitialConditionArchive
from pol.learning.metrics import symmetric_field_discrepancy
from pol.math.periodic import spectral_resample_periodic
from pol.runtime.device import require_cpu_tensors
from pol.runtime.hashing import stable_object_hash
from .conditions import (
    CROSS_SOLVER_CHECK_SCHEMA_VERSION,
    CROSS_SOLVER_METRIC_DEFINITION,
    burgers_refinement_proof,
    canonical_invariant_parameters,
    cross_solver_discrepancy_evidence_hash,
    validate_cross_solver_validation_block,
)
from .reference_convergence import (
    TimeSequenceResult,
    passes_reference_tolerances,
    run_time_refined_reference_convergence,
    run_time_sequence_convergence,
)


def _cross_solver_self_evidence(
    result: TimeSequenceResult,
) -> dict[str, Any]:
    finest_index = len(result.conditions) - 1
    selected_index = result.selected_index
    return {
        "status": "pass" if selected_index is not None else "fail",
        "ordered_candidates": result.conditions,
        "candidate_refinement_proof": result.refinement_proof,
        "rows": result.rows,
        "rows_hash": stable_object_hash(result.rows),
        "pairwise_row_hashes": [
            row["row_hash"] for row in result.rows
        ],
        "selected_candidate_index": selected_index,
        "selected_condition": (
            None
            if selected_index is None
            else result.conditions[selected_index]
        ),
        "finest_candidate_index": finest_index,
        "finest_condition": result.conditions[finest_index],
        "runtime_solver_metadata": result.runtime_metadata,
    }


def _validate_cross_solver_check_against_spec(
    spec: ValidationSpec,
    block: dict[str, Any],
) -> None:
    target = spec.target_reference
    if not isinstance(target, BurgersConvergenceReferenceSpec):
        raise ValueError("cross-solver evidence requires a Burgers spec")
    diagnostic = target.cross_solver_validation
    if not isinstance(
        diagnostic,
        EnabledBurgersCrossSolverValidationSpec,
    ):
        raise ValueError("cross-solver evidence is disabled in the spec")
    system = target.reference_evolution.system
    expected_context = {
        "system_kind": "burgers",
        "invariant_parameters": canonical_invariant_parameters(
            "burgers",
            system.model_dump(mode="json"),
        ),
        "evolution_time": float(target.reference_evolution.time),
        "domain_length": float(spec.domain.length),
        "dtype": spec.samples.dtype,
        "dealias": diagnostic.context.dealias,
        "common_nx": int(target.reference_nx_candidates[-1]),
        "reference_candidate_index": (
            len(target.reference_nx_candidates) - 1
        ),
        "sample_ids": [
            int(value) for value in target.calibration_sample_ids
        ],
    }
    if stable_object_hash(block.get("context")) != stable_object_hash(
        expected_context
    ):
        raise ValueError(
            "cross-solver validation context disagrees with the resolved spec"
        )
    if stable_object_hash(block.get("tolerances")) != stable_object_hash(
        diagnostic.tolerances.model_dump(mode="json")
    ):
        raise ValueError(
            "cross-solver tolerances disagree with the resolved spec"
        )
    self_convergence = block.get("self_convergence")
    if not isinstance(self_convergence, dict):
        raise ValueError("cross-solver self-convergence evidence is missing")
    for family in ("split_step", "etdrk4"):
        expected_proof = burgers_refinement_proof(
            [
                candidate.model_dump(mode="json")
                for candidate in getattr(
                    diagnostic.solvers,
                    family,
                ).candidates
            ],
            evolution_time=float(target.reference_evolution.time),
        )
        evidence = self_convergence.get(family)
        if (
            not isinstance(evidence, dict)
            or stable_object_hash(evidence.get("ordered_candidates"))
            != stable_object_hash(expected_proof["ordered_candidates"])
            or stable_object_hash(
                evidence.get("candidate_refinement_proof")
            )
            != stable_object_hash(expected_proof)
        ):
            raise ValueError(
                f"cross-solver {family} candidates disagree with the "
                "resolved spec"
            )


def _burgers_cross_solver_validation(
    spec: ValidationSpec,
    archive: InitialConditionArchive,
) -> dict[str, Any]:
    """Build supporting, symmetric split-step/ETDRK4 evidence."""
    target = spec.target_reference
    if not isinstance(target, BurgersConvergenceReferenceSpec):
        raise TypeError("cross-solver validation requires burgers_convergence")
    diagnostic = target.cross_solver_validation
    if not isinstance(
        diagnostic,
        EnabledBurgersCrossSolverValidationSpec,
    ):
        raise TypeError("cross-solver validation is not enabled")
    require_cpu_tensors(
        archive.__dict__,
        boundary="cross-solver validation input",
        name="archive",
    )
    domain_length = float(spec.domain.length)
    sample_ids = [
        int(value) for value in target.calibration_sample_ids
    ]
    ids = torch.tensor(
        sample_ids,
        dtype=torch.long,
        device=archive.values.device,
    )
    finest_nx = int(target.reference_nx_candidates[-1])
    reference_index = len(target.reference_nx_candidates) - 1
    initial_master = archive.values.index_select(0, ids)
    initial_finest = spectral_resample_periodic(
        initial_master,
        finest_nx,
        domain_length=domain_length,
    )

    sequence_results: dict[str, TimeSequenceResult] = {}
    for family in ("split_step", "etdrk4"):
        family_spec = getattr(diagnostic.solvers, family)
        sequence_results[family] = run_time_sequence_convergence(
            spec,
            initial=initial_finest,
            candidates=list(family_spec.candidates),
            nx=finest_nx,
            reference_candidate_index=reference_index,
            tolerances=diagnostic.tolerances,
            boundary=f"cross-solver {family} self-convergence solve",
        )
    self_convergence = {
        family: _cross_solver_self_evidence(sequence_results[family])
        for family in ("split_step", "etdrk4")
    }
    finest_conditions = {
        family: sequence_results[family].conditions[-1]
        for family in ("split_step", "etdrk4")
    }
    self_pass = all(
        evidence["status"] == "pass"
        for evidence in self_convergence.values()
    )
    if self_pass:
        discrepancy_metrics = symmetric_field_discrepancy(
            sequence_results["split_step"].solutions[-1],
            sequence_results["etdrk4"].solutions[-1],
            q=int(target.q_reference_check),
            domain_length=domain_length,
        )
        discrepancy_status = (
            "pass"
            if passes_reference_tolerances(discrepancy_metrics, diagnostic.tolerances)
            else "fail"
        )
        not_evaluated_reason = None
    else:
        discrepancy_metrics = None
        discrepancy_status = "not_evaluated"
        not_evaluated_reason = (
            "self_convergence_must_pass_before_cross_comparison"
        )
    system = target.reference_evolution.system
    context = {
        "system_kind": "burgers",
        "invariant_parameters": canonical_invariant_parameters(
            "burgers",
            system.model_dump(mode="json"),
        ),
        "evolution_time": float(target.reference_evolution.time),
        "domain_length": domain_length,
        "dtype": spec.samples.dtype,
        "dealias": diagnostic.context.dealias,
        "common_nx": finest_nx,
        "reference_candidate_index": reference_index,
        "sample_ids": sample_ids,
    }
    block = {
        "schema_version": CROSS_SOLVER_CHECK_SCHEMA_VERSION,
        "enabled": True,
        "status": (
            "pass"
            if self_pass and discrepancy_status == "pass"
            else "fail"
        ),
        "role": "supporting_evidence_not_primary_allowed_refinement",
        "context": context,
        "tolerances": diagnostic.tolerances.model_dump(mode="json"),
        "self_convergence": self_convergence,
        "finest_conditions": finest_conditions,
        "discrepancy_definition": CROSS_SOLVER_METRIC_DEFINITION,
        "discrepancy_metrics": discrepancy_metrics,
        "discrepancy_status": discrepancy_status,
        "discrepancy_not_evaluated_reason": not_evaluated_reason,
        "discrepancy_evidence_hash": (
            cross_solver_discrepancy_evidence_hash(
                finest_conditions=finest_conditions,
                common_nx=finest_nx,
                sample_ids=sample_ids,
                metrics=discrepancy_metrics,
                not_evaluated_reason=not_evaluated_reason,
            )
        ),
    }
    validate_cross_solver_validation_block(block)
    _validate_cross_solver_check_against_spec(spec, block)
    return block


def run_burgers_reference_checks(
    spec: ValidationSpec,
    archive: InitialConditionArchive,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    convergence, rows = run_time_refined_reference_convergence(
        spec,
        archive,
    )
    checks: dict[str, Any] = {
        "reference_convergence": convergence,
    }
    target = spec.target_reference
    if not isinstance(target, BurgersConvergenceReferenceSpec):
        raise TypeError("Burgers checks require burgers_convergence")
    if target.cross_solver_validation.enabled:
        checks["cross_solver_validation"] = (
            _burgers_cross_solver_validation(spec, archive)
        )
    return checks, rows


def validate_burgers_reference_checks(
    spec: ValidationSpec,
    checks: dict[str, Any],
) -> None:
    target = spec.target_reference
    if not isinstance(target, BurgersConvergenceReferenceSpec):
        raise TypeError("Burgers checks require burgers_convergence")
    if "heat_analytic" in checks:
        raise ValueError("Burgers validation must not contain heat checks")
    if "reaction_diffusion_characterization" in checks:
        raise ValueError(
            "Burgers validation must not contain reaction-diffusion checks"
        )
    if target.cross_solver_validation.enabled:
        block = checks.get("cross_solver_validation")
        if not isinstance(block, dict):
            raise ValueError(
                "enabled cross-solver validation evidence is missing"
            )
        validate_cross_solver_validation_block(block)
        _validate_cross_solver_check_against_spec(spec, block)
    elif "cross_solver_validation" in checks:
        raise ValueError(
            "disabled cross-solver validation must not contain evidence"
        )
