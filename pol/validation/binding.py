"""Pure validation-certificate to dataset-condition binding proofs."""
from __future__ import annotations

import json
import math
from typing import Any, Mapping

import torch

from pol.config.models import DatasetSpec, ValidationSpec
from pol.data.splits import (
    build_data_split,
    calibration_split_provenance,
    split_contract,
)
from pol.numerics.initial_conditions import GRF_SAMPLER_SEMANTICS
from pol.runtime.device import (
    execution_device_policy,
    verify_execution_device_policy,
)
from pol.runtime.hashing import stable_object_hash
from .conditions import (
    canonical_invariant_parameters,
    canonical_numerical_condition,
    validate_target_reference_contract,
)


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


def _split_provenance(
    validation_spec: ValidationSpec,
) -> tuple[dict[str, object], dict[str, int | str]]:
    samples = validation_spec.samples
    split = build_data_split(
        total_samples=int(samples.total_samples),
        n_train=int(samples.n_train),
        n_validation=int(samples.n_validation),
        n_test=int(samples.n_test),
        seed=int(samples.seed),
    )
    sample_ids = torch.arange(split.total_samples, dtype=torch.long)
    calibration = calibration_split_provenance(
        split,
        validation_spec.target_reference.calibration_sample_ids,
        sample_ids=sample_ids,
    )
    return calibration, split_contract(split, sample_ids=sample_ids)


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
    if certificate.get("schema_version") != "pol-validation-certificate-v12":
        raise _failure(
            path="$.certificate.schema_version",
            allowed="pol-validation-certificate-v12",
            requested=certificate.get("schema_version"),
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
    if foundation.get("schema_version") != (
        "pol-validation-foundation-contract-v8"
    ):
        raise _failure(
            path="$.foundation_contract.schema_version",
            allowed="pol-validation-foundation-contract-v8",
            requested=foundation.get("schema_version"),
            binding_kind=binding_kind,
        )
    if certificate.get("foundation_contract_hash") != stable_object_hash(
        dict(foundation)
    ):
        raise _failure(
            path="$.certificate.foundation_contract_hash",
            allowed=stable_object_hash(dict(foundation)),
            requested=certificate.get("foundation_contract_hash"),
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
    expected_calibration, _ = _split_provenance(validation_spec)
    _exact_check(
        checks,
        path="$.foundation_contract.calibration_provenance",
        allowed=foundation.get("calibration_provenance"),
        requested=expected_calibration,
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
    _, split = _split_provenance(validation_spec)
    return {
        "reference_nx": int(dataset_spec.reference_nx),
        "target": dataset_spec.target.model_dump(mode="json"),
        "dtype": validation_spec.samples.dtype,
        "domain_length": float(validation_spec.domain.length),
        "split": split,
        **execution_device_policy(),
    }


def _validated_condition_binding(
    contract: Mapping[str, Any],
    dataset_condition: Mapping[str, Any],
    *,
    binding_kind: str,
) -> tuple[list[dict[str, Any]], int, int]:
    try:
        validate_target_reference_contract(contract)
    except ValueError as exc:
        raise _failure(
            path="$.certificate.target_reference_contract",
            allowed="self-consistent pol-target-reference-contract-v4",
            requested=dict(contract),
            binding_kind=binding_kind,
        ) from exc
    target = dataset_condition.get("target")
    if not isinstance(target, Mapping):
        raise _failure(
            path="$.dataset.target",
            allowed="canonical evolution object",
            requested=target,
            binding_kind=binding_kind,
        )
    target_system = target.get("system")
    if not isinstance(target_system, Mapping):
        raise _failure(
            path="$.dataset.target.system",
            allowed="canonical system object",
            requested=target_system,
            binding_kind=binding_kind,
        )
    checks: list[dict[str, Any]] = []
    system_kind = contract["system_kind"]
    exact_values = (
        (
            "$.dataset.target.system.kind",
            system_kind,
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
            dataset_condition.get("dtype"),
        ),
        (
            "$.dataset.domain_length",
            contract.get("domain_length"),
            dataset_condition.get("domain_length"),
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

    requested_invariants = canonical_invariant_parameters(
        str(system_kind),
        target_system,
    )
    invariants = contract["invariant_parameters"]
    for name in sorted(invariants):
        _exact_check(
            checks,
            path=f"$.dataset.target.system.{name}",
            allowed=invariants[name],
            requested=requested_invariants.get(name),
            binding_kind=binding_kind,
        )

    relation = contract["allowed_refinement_relation"]
    reference = contract["reference_resolution"]
    allowed_nx = relation["reference_nx_allowed_values"]
    requested_nx = dataset_condition.get("reference_nx")
    allowed_nx_index = _canonical_index(allowed_nx, requested_nx)
    if allowed_nx_index is None:
        raise _failure(
            path="$.dataset.reference_nx",
            allowed=allowed_nx,
            requested=requested_nx,
            binding_kind=binding_kind,
        )
    reference_candidates = reference["candidates"]
    matched_reference_index = _canonical_index(
        reference_candidates,
        requested_nx,
    )
    if matched_reference_index is None:
        raise _failure(
            path="$.dataset.reference_nx",
            allowed=reference_candidates,
            requested=requested_nx,
            binding_kind=binding_kind,
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

    requested_condition = canonical_numerical_condition(
        str(system_kind),
        target_system,
        evolution_time=float(contract["evolution_time"]),
    )
    allowed_conditions = relation[
        "numerical_condition_allowed_values"
    ]
    for field in requested_condition:
        allowed_field_values: list[Any] = []
        for candidate in allowed_conditions:
            if (
                isinstance(candidate, Mapping)
                and _canonical_index(
                    allowed_field_values,
                    candidate.get(field),
                )
                is None
            ):
                allowed_field_values.append(candidate.get(field))
        requested_value = requested_condition[field]
        if _canonical_index(allowed_field_values, requested_value) is None:
            field_path = {
                "requested_outer_dt": "dt",
                "requested_fine_dt": "fine_dt",
            }.get(field, field)
            if field in {
                "outer_step_count",
                "effective_substep",
                "substeps_per_outer",
            }:
                path = "$.dataset.target.numerical_condition"
            else:
                path = (
                    "$.dataset.target.system.solver_condition"
                    if system_kind == "heat" and field == "solver"
                    else f"$.dataset.target.system.{field_path}"
                )
            raise _failure(
                path=path,
                allowed=allowed_field_values,
                requested=requested_value,
                binding_kind=binding_kind,
            )
    if _canonical_index(allowed_conditions, requested_condition) is None:
        raise _failure(
            path="$.dataset.target.numerical_condition",
            allowed=allowed_conditions,
            requested=requested_condition,
            binding_kind=binding_kind,
        )
    method = contract["numerical_method_validation"]
    matched_condition_index = _canonical_index(
        method["candidates"],
        requested_condition,
    )
    if matched_condition_index is None:
        raise _failure(
            path="$.dataset.target.numerical_condition",
            allowed=method["candidates"],
            requested=requested_condition,
            binding_kind=binding_kind,
        )
    checks.append(
        {
            "field_path": "$.dataset.target.numerical_condition",
            "comparison": (
                "canonical_exact_member_of_validated_candidate_suffix"
            ),
            "certificate_or_allowed_value": allowed_conditions,
            "requested_dataset_value": requested_condition,
            "status": "pass",
        }
    )
    return checks, matched_reference_index, matched_condition_index


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
    if certificate.get("target_reference_contract_hash") != (
        stable_object_hash(dict(contract))
    ):
        raise _failure(
            path="$.certificate.target_reference_contract_hash",
            allowed=stable_object_hash(dict(contract)),
            requested=certificate.get("target_reference_contract_hash"),
            binding_kind=binding_kind,
        )
    dataset_condition = _dataset_condition(dataset_spec, validation_spec)
    (
        checks,
        matched_reference_index,
        matched_condition_index,
    ) = _validated_condition_binding(
        contract,
        dataset_condition,
        binding_kind=binding_kind,
    )
    relation = contract["allowed_refinement_relation"]
    return _proof_with_hash(
        {
            "schema_version": "pol-dataset-binding-proof-v7",
            **execution_device_policy(),
            "binding_kind": binding_kind,
            "status": "pass",
            "target_reference_validation_status": "validated",
            "certificate_artifact_id": certificate.get("artifact_id"),
            "grf_sampler_domain_length": foundation[
                "grf_sampler_domain_length"
            ],
            "grf_sampler_semantics": foundation["grf_sampler_semantics"],
            "calibration_provenance": dict(
                foundation["calibration_provenance"]
            ),
            "foundation_contract_hash": stable_object_hash(foundation),
            "foundation_checks": foundation_checks,
            "validated_condition": dict(contract),
            "allowed_refinement_relation": dict(relation),
            "dataset_condition": dataset_condition,
            "matched_reference_candidate_index": matched_reference_index,
            "matched_numerical_condition_index": matched_condition_index,
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
            "schema_version": "pol-dataset-binding-proof-v7",
            **execution_device_policy(),
            "binding_kind": binding_kind,
            "status": "pass",
            "target_reference_validation_status": "not_claimed",
            "certificate_artifact_id": certificate.get("artifact_id"),
            "grf_sampler_domain_length": foundation[
                "grf_sampler_domain_length"
            ],
            "grf_sampler_semantics": foundation["grf_sampler_semantics"],
            "calibration_provenance": dict(
                foundation["calibration_provenance"]
            ),
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
    if copied.get("schema_version") != "pol-dataset-binding-proof-v7":
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
    calibration = copied.get("calibration_provenance")
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
    if not isinstance(calibration, Mapping):
        raise ValueError("dataset binding proof has no calibration provenance")
    split_condition = dataset_condition.get("split")
    if not isinstance(split_condition, Mapping):
        raise ValueError("dataset binding proof has no split condition")
    try:
        calibration_ids = calibration.get("calibration_sample_ids")
        if not isinstance(calibration_ids, list):
            raise ValueError("calibration_sample_ids must be a list")
        total_samples = calibration.get("total_samples")
        n_train = calibration.get("n_train")
        n_validation = calibration.get("n_validation")
        n_test = calibration.get("n_test")
        seed = calibration.get("seed")
        split = build_data_split(
            total_samples=total_samples,
            n_train=n_train,
            n_validation=n_validation,
            n_test=n_test,
            seed=seed,
        )
        sample_ids = torch.arange(split.total_samples, dtype=torch.long)
        expected_calibration = calibration_split_provenance(
            split,
            calibration_ids,
            sample_ids=sample_ids,
        )
        expected_split = split_contract(split, sample_ids=sample_ids)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "dataset binding proof has invalid calibration split provenance"
        ) from exc
    if not _canonical_equal(dict(calibration), expected_calibration):
        raise ValueError("dataset binding proof calibration provenance mismatch")
    if not _canonical_equal(dict(split_condition), expected_split):
        raise ValueError("dataset binding proof split condition mismatch")
    if kind == "validated_reference":
        if target_status != "validated":
            raise ValueError(
                "validated_reference proof has inconsistent target validation status"
            )
        required = (
            "validated_condition",
            "allowed_refinement_relation",
            "matched_reference_candidate_index",
            "matched_numerical_condition_index",
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
            "calibration_provenance",
            "foundation_contract_hash",
            "foundation_checks",
            "validated_condition",
            "allowed_refinement_relation",
            "dataset_condition",
            "matched_reference_candidate_index",
            "matched_numerical_condition_index",
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
        try:
            (
                expected_checks,
                expected_reference_index,
                expected_condition_index,
            ) = _validated_condition_binding(
                validated_condition,
                dataset_condition,
                binding_kind="validated_reference",
            )
        except DatasetBindingError as exc:
            raise ValueError(
                "validated-reference proof condition binding mismatch"
            ) from exc
        if not _canonical_equal(
            copied.get("allowed_refinement_relation"),
            validated_condition.get("allowed_refinement_relation"),
        ):
            raise ValueError(
                "validated-reference proof refinement relation mismatch"
            )
        if (
            copied.get("matched_reference_candidate_index")
            != expected_reference_index
            or copied.get("matched_numerical_condition_index")
            != expected_condition_index
        ):
            raise ValueError(
                "validated-reference proof matched candidate index mismatch"
            )
        if not _canonical_equal(
            copied.get("per_field_checks"),
            expected_checks,
        ):
            raise ValueError(
                "validated-reference proof per-field checks mismatch"
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
            "calibration_provenance",
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
        if not _canonical_equal(
            foundation.get("calibration_provenance"),
            calibration,
        ):
            raise ValueError(
                "foundation-only proof calibration provenance mismatch"
            )
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
