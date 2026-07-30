from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pol.artifacts import ArtifactRef, ArtifactStore
from pol.config.models import ValidationSpec
from pol.data.initial_conditions import GRF_SAMPLER_SEMANTICS, generate_grf_archive
from pol.runtime.device import execution_device_policy, require_cpu_tensors
from pol.runtime.environment import numerical_environment_fingerprint
from .certificates import (
    assemble_validation_certificate,
    load_validation_certificate,
)
from .conditions import (
    CONVERGENCE_CSV_SCHEMA_VERSION,
    CROSS_SOLVER_CHECK_SCHEMA_VERSION,
)
from .contracts import (
    calibration_provenance,
    master_payload,
    validate_master_payload_against_spec,
)
from .foundation_checks import run_foundation_checks
from .model1_consistency import MODEL1_CONSISTENCY_CHECK_SCHEMA_VERSION
from .publication import (
    publish_solve_failure,
    publish_validation_check_failure,
    publish_validation_success,
)
from .quadrature import FIELD_QUADRATURE_CHECK_SCHEMA_VERSION
from .reference_convergence import ValidationSolveFailure
from .target_checks import run_target_checks


@dataclass(frozen=True)
class ValidationOutcome:
    reference: ArtifactRef
    certificate: dict[str, Any]


def _scientific_identity(spec: ValidationSpec) -> dict[str, Any]:
    payload = spec.model_dump(mode="json")
    payload.pop("artifact_root", None)
    return {
        "schema_version": "pol-validation-identity-v12",
        "reference_convergence_csv_schema_version": (
            CONVERGENCE_CSV_SCHEMA_VERSION
        ),
        "cross_solver_check_schema_version": (
            CROSS_SOLVER_CHECK_SCHEMA_VERSION
        ),
        "matched_model1_pipeline_check_schema_version": (
            MODEL1_CONSISTENCY_CHECK_SCHEMA_VERSION
        ),
        "field_quadrature_check_schema_version": (
            FIELD_QUADRATURE_CHECK_SCHEMA_VERSION
        ),
        **execution_device_policy(),
        "grf_sampler_semantics": GRF_SAMPLER_SEMANTICS,
        "calibration_provenance": calibration_provenance(spec),
        "environment": numerical_environment_fingerprint(),
        "spec": payload,
    }


def validation_reference(spec: ValidationSpec) -> ArtifactRef:
    identity = _scientific_identity(spec)
    return ArtifactStore(spec.artifact_root).reference("validations", identity)


def ensure_validation(spec: ValidationSpec, *, force: bool = False) -> ValidationOutcome:
    ref = validation_reference(spec)
    store = ArtifactStore(spec.artifact_root)
    if store.exists(ref) and not force:
        return ValidationOutcome(ref, load_validation_certificate(ref.path))
    return run_validation(spec, force=force)


def run_validation(spec: ValidationSpec, *, force: bool = False) -> ValidationOutcome:
    calibration_provenance(spec)
    max_nx = max(
        int(value)
        for value in spec.target_reference.reference_nx_candidates
    )
    ic = spec.samples.initial_condition
    archive = generate_grf_archive(
        total_samples=spec.samples.total_samples,
        nx=max_nx,
        seed=spec.samples.seed,
        gamma=ic.gamma,
        tau=ic.tau,
        sigma=ic.sigma,
        mean=ic.mean,
        domain_length=spec.domain.length,
        dtype=spec.samples.dtype,
        device=spec.samples.device,
    )
    require_cpu_tensors(
        archive.__dict__,
        boundary="validation workflow input",
        name="initial_conditions",
    )

    identity = _scientific_identity(spec)
    reference = ArtifactStore(spec.artifact_root).reference(
        "validations",
        identity,
    )
    master = master_payload(archive)
    validate_master_payload_against_spec(master, spec)

    try:
        checks = run_foundation_checks(spec, archive)
        target_checks, convergence_rows = run_target_checks(spec, archive)
        checks.update(target_checks)
    except ValidationSolveFailure as exc:
        failure_ref = publish_solve_failure(
            spec,
            identity=identity,
            master_payload=master,
            diagnostic=exc.diagnostic,
            force=force,
        )
        raise RuntimeError(
            "validation failed because a numerical solve produced a "
            f"non-finite state; diagnostics: {failure_ref.path}"
        ) from exc

    certificate = assemble_validation_certificate(
        spec,
        artifact_id=reference.artifact_id,
        checks=checks,
        master_payload=master,
    )
    statuses = certificate["checks"]
    if certificate["status"] != "pass":
        failure_ref = publish_validation_check_failure(
            spec,
            identity=identity,
            checks=checks,
            certificate=certificate,
            convergence_rows=convergence_rows,
            master_payload=master,
            force=force,
        )
        raise RuntimeError(
            f"validation failed ({statuses}); diagnostics: {failure_ref.path}"
        )

    publish_validation_success(
        spec,
        reference=reference,
        identity=identity,
        checks=checks,
        certificate=certificate,
        convergence_rows=convergence_rows,
        master_payload=master,
        force=force,
    )
    return ValidationOutcome(
        reference,
        load_validation_certificate(reference.path),
    )
