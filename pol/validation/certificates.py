from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch

from pol.artifacts import verify_artifact
from pol.config.models import ValidationSpec
from pol.data.initial_conditions import GRF_SAMPLER_SEMANTICS
from pol.runtime.device import (
    execution_device_policy,
    require_cpu_tensors,
    verify_execution_device_policy,
)
from pol.runtime.hashing import stable_object_hash
from .conditions import (
    CONVERGENCE_CSV_SCHEMA_VERSION,
    CONVERGENCE_ROW_FIELDS,
    CONVERGENCE_ROW_SCHEMA_VERSION,
    CROSS_SOLVER_CHECK_SCHEMA_VERSION,
    validate_cross_solver_validation_block,
)
from .contracts import (
    calibration_provenance,
    foundation_contract,
    target_reference_contract,
    validate_master_payload_against_spec,
    validate_target_reference_contract_payload,
)
from .foundation_checks import validate_foundation_checks
from .model1_consistency import MODEL1_CONSISTENCY_CHECK_SCHEMA_VERSION
from .quadrature import FIELD_QUADRATURE_CHECK_SCHEMA_VERSION
from .target_checks import validate_target_checks


def _validate_convergence_csv(
    path: Path,
    expected_rows: list[dict[str, Any]],
) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if reader.fieldnames != list(CONVERGENCE_ROW_FIELDS):
        raise ValueError("reference convergence CSV schema/header mismatch")
    expected = [
        {
            name: "" if row[name] is None else str(row[name])
            for name in CONVERGENCE_ROW_FIELDS
        }
        for row in expected_rows
    ]
    if rows != expected:
        raise ValueError(
            "reference convergence CSV rows disagree with certificate evidence"
        )


def load_validation_certificate(path: Path | str) -> dict[str, Any]:
    root = Path(path).resolve()
    manifest = verify_artifact(root)
    if manifest.get("kind") != "validations":
        raise ValueError("artifact is not a passing validation artifact")
    certificate = json.loads((root / "certificate.json").read_text(encoding="utf-8"))
    if not isinstance(certificate, dict):
        raise ValueError("validation certificate payload must be an object")
    if certificate.get("schema_version") != "pol-validation-certificate-v12":
        raise ValueError(
            "unsupported validation certificate schema; Phase 2-05B requires "
            "pol-validation-certificate-v12"
        )
    identity = manifest.get("identity")
    if not isinstance(identity, dict) or identity.get("schema_version") != (
        "pol-validation-identity-v12"
    ):
        raise ValueError("unsupported legacy validation artifact identity")
    if identity.get("reference_convergence_csv_schema_version") != (
        CONVERGENCE_CSV_SCHEMA_VERSION
    ):
        raise ValueError("unsupported reference convergence CSV schema")
    if identity.get("cross_solver_check_schema_version") != (
        CROSS_SOLVER_CHECK_SCHEMA_VERSION
    ):
        raise ValueError("unsupported cross-solver check schema")
    if identity.get("matched_model1_pipeline_check_schema_version") != (
        MODEL1_CONSISTENCY_CHECK_SCHEMA_VERSION
    ):
        raise ValueError("unsupported matched Model 1 pipeline check schema")
    if identity.get("field_quadrature_check_schema_version") != (
        FIELD_QUADRATURE_CHECK_SCHEMA_VERSION
    ):
        raise ValueError("unsupported field quadrature check schema")
    verify_execution_device_policy(
        identity,
        boundary="validation artifact identity",
    )
    environment = identity.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("validation artifact numerical environment is missing")
    verify_execution_device_policy(
        environment,
        boundary="validation numerical environment",
    )
    verify_execution_device_policy(
        certificate,
        boundary="validation certificate",
    )
    if identity.get("grf_sampler_semantics") != GRF_SAMPLER_SEMANTICS:
        raise ValueError("validation artifact has unsupported GRF sampler semantics")
    resolved = json.loads((root / "resolved_spec.json").read_text(encoding="utf-8"))
    if not isinstance(resolved, dict):
        raise ValueError("validation resolved spec must be an object")
    if stable_object_hash(resolved) != stable_object_hash(identity.get("spec")):
        raise ValueError("validation resolved spec does not match artifact identity")
    try:
        spec = ValidationSpec.model_validate(resolved)
    except ValueError as exc:
        raise ValueError("validation artifact contains an invalid resolved spec") from exc
    checks = json.loads((root / "checks.json").read_text(encoding="utf-8"))
    if not isinstance(checks, dict):
        raise ValueError("validation checks payload must be an object")
    validate_foundation_checks(spec, checks)
    validate_target_checks(spec, checks)
    master = torch.load(
        root / "master_initial_conditions.pt",
        map_location="cpu",
        weights_only=True,
    )
    require_cpu_tensors(
        master,
        boundary="master initial-condition archive load",
        name="master",
    )
    validate_master_payload_against_spec(master, spec)
    expected_calibration = calibration_provenance(
        spec,
        sample_ids=master["sample_ids"],
    )
    if stable_object_hash(identity.get("calibration_provenance")) != (
        stable_object_hash(expected_calibration)
    ):
        raise ValueError(
            "validation identity calibration provenance does not match "
            "the deterministic split"
        )
    try:
        expected = assemble_validation_certificate(
            spec,
            artifact_id=str(manifest.get("artifact_id")),
            checks=checks,
            master_payload=master,
        )
    except (AttributeError, KeyError, TypeError) as exc:
        raise ValueError(
            "validation checks cannot reconstruct the required certificate contract"
        ) from exc
    validate_target_reference_contract_payload(
        expected["target_reference_contract"]
    )
    expected_rows = expected["target_reference_contract"][
        "convergence_evidence"
    ]["rows"]
    _validate_convergence_csv(
        root / "reference_convergence.csv",
        expected_rows,
    )
    if stable_object_hash(certificate) != stable_object_hash(expected):
        raise ValueError(
            "validation certificate contract is missing, contradictory, or "
            "inconsistent with the resolved spec/checks/master archive"
        )
    if certificate.get("artifact_id") != manifest.get("artifact_id"):
        raise ValueError("validation certificate identity mismatch")
    if certificate.get("status") != "pass":
        raise ValueError("validation certificate is not passing")
    check_statuses = certificate.get("checks")
    if not isinstance(check_statuses, dict) or not check_statuses or any(
        value != "pass" for value in check_statuses.values()
    ):
        raise ValueError("validation certificate contains a non-passing check")
    return certificate


def assemble_validation_certificate(
    spec: ValidationSpec,
    *,
    artifact_id: str,
    checks: dict[str, Any],
    master_payload: dict[str, Any],
) -> dict[str, Any]:
    cross_value = checks.get("cross_solver_validation")
    if cross_value is None:
        cross_solver_validation: dict[str, Any] | None = None
    elif isinstance(cross_value, dict):
        validate_cross_solver_validation_block(cross_value)
        cross_solver_validation = cross_value
    else:
        raise ValueError(
            "cross-solver validation evidence must be an object"
        )
    statuses = {name: value["status"] for name, value in checks.items()}
    overall = "pass" if all(value == "pass" for value in statuses.values()) else "fail"
    foundation = foundation_contract(
        spec, checks=checks, master_payload=master_payload
    )
    target_reference = target_reference_contract(
        spec, checks["reference_convergence"]
    )
    if overall == "pass":
        validate_target_reference_contract_payload(target_reference)
    return {
        "schema_version": "pol-validation-certificate-v12",
        "reference_convergence_csv_schema_version": (
            CONVERGENCE_CSV_SCHEMA_VERSION
        ),
        "reference_convergence_row_schema_version": (
            CONVERGENCE_ROW_SCHEMA_VERSION
        ),
        "reference_convergence_rows_hash": target_reference[
            "convergence_evidence"
        ]["rows_hash"],
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
        "status": overall,
        "name": spec.name,
        "profile": spec.profile,
        "artifact_id": artifact_id,
        "checks": statuses,
        "foundation_contract": foundation,
        "foundation_contract_hash": stable_object_hash(foundation),
        "target_reference_contract": target_reference,
        "target_reference_contract_hash": stable_object_hash(target_reference),
        "cross_solver_validation": cross_solver_validation,
        "cross_solver_validation_hash": (
            None
            if cross_solver_validation is None
            else stable_object_hash(cross_solver_validation)
        ),
    }
