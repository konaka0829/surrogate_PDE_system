from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

from pol.artifacts import ArtifactRef, ArtifactStore, verify_artifact
from pol.config.models import ValidationSpec
from pol.runtime.io import atomic_torch_save, write_csv, write_strict_json
from pol.runtime.hashing import stable_object_hash
from .conditions import CONVERGENCE_ROW_FIELDS


def publish_solve_failure(
    spec: ValidationSpec,
    *,
    identity: dict[str, Any],
    master_payload: dict[str, Any],
    diagnostic: dict[str, Any],
    force: bool,
) -> ArtifactRef:
    store = ArtifactStore(spec.artifact_root)
    diagnostic_hash = stable_object_hash(diagnostic)
    failure_identity = {
        **identity,
        "failure_artifact_schema_version": "pol-validation-failure-v1",
        "failure_diagnostic_hash": diagnostic_hash,
    }
    failure_ref = store.reference(
        "validation_failures",
        failure_identity,
    )
    failure = {
        "schema_version": "pol-validation-failure-v1",
        "status": "fail",
        "name": spec.name,
        "profile": spec.profile,
        "diagnostic": diagnostic,
        "diagnostic_hash": diagnostic_hash,
    }

    def writer(root: Path) -> Iterable[str]:
        write_strict_json(root / "resolved_spec.json", identity["spec"])
        write_strict_json(root / "failure.json", failure)
        atomic_torch_save(
            root / "master_initial_conditions.pt",
            master_payload,
        )
        return (
            "resolved_spec.json",
            "failure.json",
            "master_initial_conditions.pt",
        )

    store.publish(
        failure_ref,
        identity=failure_identity,
        writer=writer,
        force=force,
    )
    verify_artifact(failure_ref.path)
    return failure_ref


def _validation_writer(
    *,
    identity: dict[str, Any],
    checks: dict[str, Any],
    certificate: dict[str, Any],
    convergence_rows: list[dict[str, Any]],
    master_payload: dict[str, Any],
) -> Callable[[Path], Iterable[str]]:
    def writer(root: Path) -> Iterable[str]:
        write_strict_json(root / "resolved_spec.json", identity["spec"])
        write_strict_json(root / "checks.json", checks)
        write_strict_json(root / "certificate.json", certificate)
        write_csv(
            root / "reference_convergence.csv",
            convergence_rows,
            fieldnames=CONVERGENCE_ROW_FIELDS,
        )
        atomic_torch_save(
            root / "master_initial_conditions.pt",
            master_payload,
        )
        return (
            "resolved_spec.json",
            "checks.json",
            "certificate.json",
            "reference_convergence.csv",
            "master_initial_conditions.pt",
        )

    return writer


def publish_validation_success(
    spec: ValidationSpec,
    *,
    reference: ArtifactRef,
    identity: dict[str, Any],
    checks: dict[str, Any],
    certificate: dict[str, Any],
    convergence_rows: list[dict[str, Any]],
    master_payload: dict[str, Any],
    force: bool,
) -> ArtifactRef:
    store = ArtifactStore(spec.artifact_root)
    store.publish(
        reference,
        identity=identity,
        writer=_validation_writer(
            identity=identity,
            checks=checks,
            certificate=certificate,
            convergence_rows=convergence_rows,
            master_payload=master_payload,
        ),
        force=force,
    )
    return reference


def publish_validation_check_failure(
    spec: ValidationSpec,
    *,
    identity: dict[str, Any],
    checks: dict[str, Any],
    certificate: dict[str, Any],
    convergence_rows: list[dict[str, Any]],
    master_payload: dict[str, Any],
    force: bool,
) -> ArtifactRef:
    store = ArtifactStore(spec.artifact_root)
    failure_ref = store.reference("validation_failures", identity)
    store.publish(
        failure_ref,
        identity=identity,
        writer=_validation_writer(
            identity=identity,
            checks=checks,
            certificate=certificate,
            convergence_rows=convergence_rows,
            master_payload=master_payload,
        ),
        force=force,
    )
    return failure_ref
