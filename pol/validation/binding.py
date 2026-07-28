"""Pure validation-certificate to dataset-condition binding proofs."""
from __future__ import annotations

import json
import math
from typing import Any, Mapping

from pol.config.models import DatasetSpec, ValidationSpec
from pol.numerics.initial_conditions import GRF_SAMPLER_SEMANTICS
from pol.runtime.device import (
    execution_device_policy,
    verify_execution_device_policy,
)
from pol.runtime.hashing import stable_object_hash


class DatasetBindingError(ValueError):
    """Raised when a requested dataset condition is outside its binding."""


def _encoded(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _failure(
    *,
    path: str,
    allowed: Any,
    requested: Any,
    binding_kind: str,
) -> DatasetBindingError:
    return DatasetBindingError(
        "dataset validation binding failed: "
        f"field_path={path}; "
        f"certificate_or_allowed_value={_encoded(allowed)}; "
        f"requested_dataset_value={_encoded(requested)}; "
        f"binding_kind={binding_kind}"
    )


def _canonical_equal(left: Any, right: Any) -> bool:
    return _encoded(left) == _encoded(right)


def _canonical_index(values: list[Any], requested: Any) -> int | None:
    requested_encoded = _encoded(requested)
    for index, value in enumerate(values):
        if _encoded(value) == requested_encoded:
            return index
    return None


def _exact_check(
    checks: list[dict[str, Any]],
    *,
    path: str,
    allowed: Any,
    requested: Any,
    binding_kind: str,
) -> None:
    if not _canonical_equal(requested, allowed):
        raise _failure(
            path=path,
            allowed=allowed,
            requested=requested,
            binding_kind=binding_kind,
        )
    checks.append(
        {
            "field_path": path,
            "comparison": "canonical_exact",
            "certificate_or_allowed_value": allowed,
            "requested_dataset_value": requested,
            "status": "pass",
        }
    )


def _foundation_checks(
    certificate: Mapping[str, Any],
    validation_spec: ValidationSpec,
    *,
    binding_kind: str,
) -> list[dict[str, Any]]:
    if certificate.get("status") != "pass":
        raise _failure(
            path="$.certificate.status",
            allowed="pass",
            requested=certificate.get("status"),
            binding_kind=binding_kind,
        )
    foundation = certificate.get("foundation_contract")
    if not isinstance(foundation, Mapping):
        raise _failure(
            path="$.certificate.foundation_contract",
            allowed="object",
            requested=foundation,
            binding_kind=binding_kind,
        )
    try:
        verify_execution_device_policy(
            certificate,
            boundary="dataset binding validation certificate",
        )
        verify_execution_device_policy(
            foundation,
            boundary="dataset binding foundation contract",
        )
    except ValueError as exc:
        raise DatasetBindingError(str(exc)) from exc
    checks: list[dict[str, Any]] = []
    samples = foundation.get("samples")
    if not isinstance(samples, Mapping):
        raise _failure(
            path="$.certificate.foundation_contract.samples",
            allowed="object",
            requested=samples,
            binding_kind=binding_kind,
        )
    requested_samples = validation_spec.samples
    master = foundation.get("master_initial_conditions")
    if not isinstance(master, Mapping):
        raise _failure(
            path="$.certificate.foundation_contract.master_initial_conditions",
            allowed="object",
            requested=master,
            binding_kind=binding_kind,
        )
    master_metadata = master.get("metadata")
    if not isinstance(master_metadata, Mapping):
        raise _failure(
            path="$.certificate.foundation_contract.master_initial_conditions.metadata",
            allowed="object",
            requested=master_metadata,
            binding_kind=binding_kind,
        )
    exact_values = (
        (
            "$.foundation_contract.domain_length",
            foundation.get("domain_length"),
            float(validation_spec.domain.length),
        ),
        (
            "$.foundation_contract.grf_sampler_domain_length",
            foundation.get("grf_sampler_domain_length"),
            float(validation_spec.domain.length),
        ),
        (
            "$.foundation_contract.grf_sampler_semantics",
            foundation.get("grf_sampler_semantics"),
            GRF_SAMPLER_SEMANTICS,
        ),
        (
            "$.foundation_contract.master_initial_conditions.domain_length",
            master.get("domain_length"),
            float(validation_spec.domain.length),
        ),
        (
            "$.foundation_contract.master_initial_conditions.metadata.domain_length",
            master_metadata.get("domain_length"),
            float(validation_spec.domain.length),
        ),
        (
            "$.foundation_contract.master_initial_conditions.metadata.sampler_semantics",
            master_metadata.get("sampler_semantics"),
            GRF_SAMPLER_SEMANTICS,
        ),
        (
            "$.foundation_contract.dtype",
            foundation.get("dtype"),
            requested_samples.dtype,
        ),
        (
            "$.foundation_contract.samples.total_samples",
            samples.get("total_samples"),
            int(requested_samples.total_samples),
        ),
        (
            "$.foundation_contract.samples.n_train",
            samples.get("n_train"),
            int(requested_samples.n_train),
        ),
        (
            "$.foundation_contract.samples.n_validation",
            samples.get("n_validation"),
            int(requested_samples.n_validation),
        ),
        (
            "$.foundation_contract.samples.n_test",
            samples.get("n_test"),
            int(requested_samples.n_test),
        ),
        (
            "$.foundation_contract.samples.seed",
            samples.get("seed"),
            int(requested_samples.seed),
        ),
        (
            "$.foundation_contract.samples.device",
            samples.get("device"),
            "cpu",
        ),
        (
            "$.foundation_contract.samples.preprocessing",
            samples.get("preprocessing"),
            requested_samples.preprocessing,
        ),
        (
            "$.foundation_contract.initial_condition",
            foundation.get("initial_condition"),
            requested_samples.initial_condition.model_dump(mode="json"),
        ),
    )
    for path, allowed, requested in exact_values:
        _exact_check(
            checks,
            path=path,
            allowed=allowed,
            requested=requested,
            binding_kind=binding_kind,
        )
    general = foundation.get("general_foundation_checks")
    if not isinstance(general, Mapping):
        raise _failure(
            path="$.foundation_contract.general_foundation_checks",
            allowed="passing checks object",
            requested=general,
            binding_kind=binding_kind,
        )
    _exact_check(
        checks,
        path="$.foundation_contract.general_foundation_checks.status",
        allowed="pass",
        requested=general.get("status"),
        binding_kind=binding_kind,
    )
    check_statuses = general.get("checks")
    if not isinstance(check_statuses, Mapping) or not check_statuses:
        raise _failure(
            path="$.foundation_contract.general_foundation_checks.checks",
            allowed="nonempty all-pass object",
            requested=check_statuses,
            binding_kind=binding_kind,
        )
    for name in sorted(check_statuses):
        _exact_check(
            checks,
            path=f"$.foundation_contract.general_foundation_checks.checks.{name}",
            allowed="pass",
            requested=check_statuses[name],
            binding_kind=binding_kind,
        )
    return checks


def _proof_with_hash(payload: dict[str, Any]) -> dict[str, Any]:
    proof = dict(payload)
    proof["proof_hash"] = stable_object_hash(proof)
    return proof


def _dataset_condition(
    dataset_spec: DatasetSpec, validation_spec: ValidationSpec
) -> dict[str, Any]:
    return {
        "reference_nx": int(dataset_spec.reference_nx),
        "target": dataset_spec.target.model_dump(mode="json"),
        "dtype": validation_spec.samples.dtype,
        "domain_length": float(validation_spec.domain.length),
        **execution_device_policy(),
    }


def _evaluate_validated_reference(
    certificate: Mapping[str, Any],
    validation_spec: ValidationSpec,
    dataset_spec: DatasetSpec,
) -> dict[str, Any]:
    binding_kind = "validated_reference"
    foundation_checks = _foundation_checks(
        certificate, validation_spec, binding_kind=binding_kind
    )
    foundation = certificate["foundation_contract"]
    contract = certificate.get("target_reference_contract")
    if not isinstance(contract, Mapping):
        raise _failure(
            path="$.certificate.target_reference_contract",
            allowed="object",
            requested=contract,
            binding_kind=binding_kind,
        )
    checks: list[dict[str, Any]] = []
    target = dataset_spec.target.model_dump(mode="json")
    target_system = target["system"]
    exact_values = (
        (
            "$.dataset.target.system.kind",
            contract.get("system_kind"),
            target_system.get("kind"),
        ),
        (
            "$.dataset.target.time",
            contract.get("evolution_time"),
            target.get("time"),
        ),
        (
            "$.dataset.dtype",
            contract.get("dtype"),
            validation_spec.samples.dtype,
        ),
        (
            "$.dataset.domain_length",
            contract.get("domain_length"),
            float(validation_spec.domain.length),
        ),
    )
    for path, allowed, requested in exact_values:
        _exact_check(
            checks,
            path=path,
            allowed=allowed,
            requested=requested,
            binding_kind=binding_kind,
        )
    invariants = contract.get("invariant_parameters")
    if not isinstance(invariants, Mapping):
        raise _failure(
            path="$.certificate.target_reference_contract.invariant_parameters",
            allowed="object",
            requested=invariants,
            binding_kind=binding_kind,
        )
    for name in sorted(invariants):
        _exact_check(
            checks,
            path=f"$.dataset.target.system.{name}",
            allowed=invariants[name],
            requested=target_system.get(name),
            binding_kind=binding_kind,
        )

    relation = contract.get("allowed_refinement_relation")
    reference = contract.get("reference_resolution")
    time_discretization = contract.get("time_discretization")
    if not isinstance(relation, Mapping):
        raise _failure(
            path="$.certificate.target_reference_contract.allowed_refinement_relation",
            allowed="machine-readable object",
            requested=relation,
            binding_kind=binding_kind,
        )
    if not isinstance(reference, Mapping) or not isinstance(
        time_discretization, Mapping
    ):
        raise _failure(
            path="$.certificate.target_reference_contract",
            allowed="reference_resolution and time_discretization objects",
            requested={
                "reference_resolution": reference,
                "time_discretization": time_discretization,
            },
            binding_kind=binding_kind,
        )
    allowed_nx = relation.get("reference_nx_allowed_values")
    if not isinstance(allowed_nx, list):
        raise _failure(
            path="$.certificate.target_reference_contract.allowed_refinement_relation.reference_nx_allowed_values",
            allowed="list",
            requested=allowed_nx,
            binding_kind=binding_kind,
        )
    requested_nx = int(dataset_spec.reference_nx)
    allowed_nx_index = _canonical_index(allowed_nx, requested_nx)
    if allowed_nx_index is None:
        raise _failure(
            path="$.dataset.reference_nx",
            allowed=allowed_nx,
            requested=requested_nx,
            binding_kind=binding_kind,
        )
    reference_candidates = reference.get("candidates")
    matched_reference_index = (
        _canonical_index(reference_candidates, requested_nx)
        if isinstance(reference_candidates, list)
        else None
    )
    checks.append(
        {
            "field_path": "$.dataset.reference_nx",
            "comparison": "exact_member_of_validated_candidate_suffix",
            "certificate_or_allowed_value": allowed_nx,
            "requested_dataset_value": requested_nx,
            "status": "pass",
        }
    )

    requested_time_candidate = {
        "dt": target_system.get("dt"),
        "fine_dt": target_system.get("fine_dt"),
        "solver": target_system.get("solver"),
        "dealias": target_system.get("dealias"),
    }
    allowed_time = relation.get("time_candidate_allowed_values")
    if not isinstance(allowed_time, list):
        raise _failure(
            path="$.certificate.target_reference_contract.allowed_refinement_relation.time_candidate_allowed_values",
            allowed="list",
            requested=allowed_time,
            binding_kind=binding_kind,
        )
    for field in ("solver", "dealias", "dt", "fine_dt"):
        allowed_field_values: list[Any] = []
        for candidate in allowed_time:
            if (
                isinstance(candidate, Mapping)
                and _canonical_index(
                    allowed_field_values, candidate.get(field)
                )
                is None
            ):
                allowed_field_values.append(candidate.get(field))
        requested_value = requested_time_candidate[field]
        if _canonical_index(allowed_field_values, requested_value) is None:
            raise _failure(
                path=f"$.dataset.target.system.{field}",
                allowed=allowed_field_values,
                requested=requested_value,
                binding_kind=binding_kind,
            )
    if _canonical_index(allowed_time, requested_time_candidate) is None:
        raise _failure(
            path="$.dataset.target.time_candidate",
            allowed=allowed_time,
            requested=requested_time_candidate,
            binding_kind=binding_kind,
        )
    time_candidates = time_discretization.get("candidates")
    matched_time_index = (
        _canonical_index(time_candidates, requested_time_candidate)
        if isinstance(time_candidates, list)
        else None
    )
    checks.append(
        {
            "field_path": "$.dataset.target.time_candidate",
            "comparison": "canonical_exact_member_of_validated_candidate_suffix",
            "certificate_or_allowed_value": allowed_time,
            "requested_dataset_value": requested_time_candidate,
            "status": "pass",
        }
    )
    return _proof_with_hash(
        {
            "schema_version": "pol-dataset-binding-proof-v3",
            **execution_device_policy(),
            "binding_kind": binding_kind,
            "status": "pass",
            "target_reference_validation_status": "validated",
            "certificate_artifact_id": certificate.get("artifact_id"),
            "grf_sampler_domain_length": foundation[
                "grf_sampler_domain_length"
            ],
            "grf_sampler_semantics": foundation["grf_sampler_semantics"],
            "foundation_contract_hash": stable_object_hash(foundation),
            "foundation_checks": foundation_checks,
            "validated_condition": dict(contract),
            "allowed_refinement_relation": dict(relation),
            "dataset_condition": _dataset_condition(dataset_spec, validation_spec),
            "matched_reference_candidate_index": matched_reference_index,
            "matched_time_candidate_index": matched_time_index,
            "per_field_checks": checks,
        }
    )


def _evaluate_foundation_only(
    certificate: Mapping[str, Any],
    validation_spec: ValidationSpec,
    dataset_spec: DatasetSpec,
) -> dict[str, Any]:
    binding_kind = "foundation_only"
    checks = _foundation_checks(
        certificate, validation_spec, binding_kind=binding_kind
    )
    foundation = certificate["foundation_contract"]
    master = foundation.get("master_initial_conditions")
    if not isinstance(master, Mapping):
        raise _failure(
            path="$.certificate.foundation_contract.master_initial_conditions",
            allowed="object",
            requested=master,
            binding_kind=binding_kind,
        )
    master_nx = master.get("nx")
    requested_nx = int(dataset_spec.reference_nx)
    if not isinstance(master_nx, int) or requested_nx > master_nx:
        raise _failure(
            path="$.dataset.reference_nx",
            allowed={"maximum_master_archive_nx": master_nx},
            requested=requested_nx,
            binding_kind=binding_kind,
        )
    checks.append(
        {
            "field_path": "$.dataset.reference_nx",
            "comparison": "master_archive_capacity_only",
            "certificate_or_allowed_value": {
                "maximum_master_archive_nx": master_nx
            },
            "requested_dataset_value": requested_nx,
            "status": "pass",
        }
    )
    reason = dataset_spec.binding.reason
    return _proof_with_hash(
        {
            "schema_version": "pol-dataset-binding-proof-v3",
            **execution_device_policy(),
            "binding_kind": binding_kind,
            "status": "pass",
            "target_reference_validation_status": "not_claimed",
            "certificate_artifact_id": certificate.get("artifact_id"),
            "grf_sampler_domain_length": foundation[
                "grf_sampler_domain_length"
            ],
            "grf_sampler_semantics": foundation["grf_sampler_semantics"],
            "foundation_contract_hash": stable_object_hash(foundation),
            "foundation_contract": dict(foundation),
            "foundation_checks": checks,
            "dataset_condition": _dataset_condition(dataset_spec, validation_spec),
            "reason": reason,
        }
    )


def evaluate_dataset_binding(
    certificate: Mapping[str, Any],
    validation_spec: ValidationSpec,
    dataset_spec: DatasetSpec,
) -> dict[str, Any]:
    """Return a deterministic passing proof or raise an actionable failure."""
    kind = dataset_spec.binding.kind
    if kind == "validated_reference":
        return _evaluate_validated_reference(
            certificate, validation_spec, dataset_spec
        )
    if kind == "foundation_only":
        return _evaluate_foundation_only(
            certificate, validation_spec, dataset_spec
        )
    raise DatasetBindingError(f"unsupported dataset binding kind: {kind}")


def verify_binding_proof(proof: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a stored proof's self-hash and non-upgradeable status semantics."""
    copied = dict(proof)
    proof_hash = copied.pop("proof_hash", None)
    if not isinstance(proof_hash, str) or stable_object_hash(copied) != proof_hash:
        raise ValueError("dataset binding proof hash mismatch")
    if copied.get("schema_version") != "pol-dataset-binding-proof-v3":
        raise ValueError("unsupported dataset binding proof schema")
    verify_execution_device_policy(
        copied,
        boundary="dataset binding proof",
    )
    if copied.get("status") != "pass":
        raise ValueError("dataset binding proof is not passing")
    kind = copied.get("binding_kind")
    target_status = copied.get("target_reference_validation_status")
    sampler_domain_length = copied.get("grf_sampler_domain_length")
    sampler_semantics = copied.get("grf_sampler_semantics")
    dataset_condition = copied.get("dataset_condition")
    if sampler_semantics != GRF_SAMPLER_SEMANTICS:
        raise ValueError("dataset binding proof has unsupported GRF sampler semantics")
    if (
        isinstance(sampler_domain_length, bool)
        or not isinstance(sampler_domain_length, (int, float))
        or not math.isfinite(float(sampler_domain_length))
        or float(sampler_domain_length) <= 0.0
    ):
        raise ValueError("dataset binding proof has invalid GRF sampler domain")
    if not isinstance(dataset_condition, Mapping) or not _canonical_equal(
        sampler_domain_length, dataset_condition.get("domain_length")
    ):
        raise ValueError("dataset binding proof GRF sampler domain mismatch")
    verify_execution_device_policy(
        dataset_condition,
        boundary="dataset binding proof condition",
    )
    if kind == "validated_reference":
        if target_status != "validated":
            raise ValueError(
                "validated_reference proof has inconsistent target validation status"
            )
        required = (
            "validated_condition",
            "allowed_refinement_relation",
            "matched_reference_candidate_index",
            "matched_time_candidate_index",
            "per_field_checks",
        )
        expected_keys = {
            "schema_version",
            "execution_device_policy",
            "compute_device",
            "binding_kind",
            "status",
            "target_reference_validation_status",
            "certificate_artifact_id",
            "grf_sampler_domain_length",
            "grf_sampler_semantics",
            "foundation_contract_hash",
            "foundation_checks",
            "validated_condition",
            "allowed_refinement_relation",
            "dataset_condition",
            "matched_reference_candidate_index",
            "matched_time_candidate_index",
            "per_field_checks",
            "proof_hash",
        }
        validated_condition = copied.get("validated_condition")
        if not isinstance(validated_condition, Mapping) or not _canonical_equal(
            sampler_domain_length, validated_condition.get("domain_length")
        ):
            raise ValueError(
                "validated-reference proof GRF sampler domain mismatch"
            )
    elif kind == "foundation_only":
        if target_status != "not_claimed":
            raise ValueError(
                "foundation_only proof must keep target validation status not_claimed"
            )
        reason = copied.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("foundation_only proof has no nonempty reason")
        required = ("foundation_contract", "foundation_checks", "dataset_condition")
        expected_keys = {
            "schema_version",
            "execution_device_policy",
            "compute_device",
            "binding_kind",
            "status",
            "target_reference_validation_status",
            "certificate_artifact_id",
            "grf_sampler_domain_length",
            "grf_sampler_semantics",
            "foundation_contract_hash",
            "foundation_contract",
            "foundation_checks",
            "dataset_condition",
            "reason",
            "proof_hash",
        }
        foundation = copied.get("foundation_contract")
        master = (
            foundation.get("master_initial_conditions")
            if isinstance(foundation, Mapping)
            else None
        )
        master_metadata = (
            master.get("metadata") if isinstance(master, Mapping) else None
        )
        domain_copies = (
            foundation.get("domain_length")
            if isinstance(foundation, Mapping)
            else None,
            foundation.get("grf_sampler_domain_length")
            if isinstance(foundation, Mapping)
            else None,
            master.get("domain_length") if isinstance(master, Mapping) else None,
            master_metadata.get("domain_length")
            if isinstance(master_metadata, Mapping)
            else None,
        )
        semantics_copies = (
            foundation.get("grf_sampler_semantics")
            if isinstance(foundation, Mapping)
            else None,
            master_metadata.get("sampler_semantics")
            if isinstance(master_metadata, Mapping)
            else None,
        )
        if not isinstance(foundation, Mapping) or stable_object_hash(
            dict(foundation)
        ) != copied.get("foundation_contract_hash"):
            raise ValueError("foundation-only proof foundation contract mismatch")
        verify_execution_device_policy(
            foundation,
            boundary="foundation-only dataset binding proof",
        )
        if any(
            not _canonical_equal(value, sampler_domain_length)
            for value in domain_copies
        ):
            raise ValueError(
                "foundation-only proof GRF sampler domain mismatch"
            )
        if any(value != GRF_SAMPLER_SEMANTICS for value in semantics_copies):
            raise ValueError(
                "foundation-only proof GRF sampler semantics mismatch"
            )
    else:
        raise ValueError("unsupported dataset binding proof kind")
    if any(name not in copied for name in required):
        raise ValueError("dataset binding proof is missing required fields")
    if set(proof) != expected_keys:
        raise ValueError("dataset binding proof has unknown or missing fields")
    return dict(proof)


__all__ = [
    "DatasetBindingError",
    "evaluate_dataset_binding",
    "verify_binding_proof",
]
