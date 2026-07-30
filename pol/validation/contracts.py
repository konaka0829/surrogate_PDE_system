from __future__ import annotations

from typing import Any

import torch

from pol.config.models import ValidationSpec
from pol.data.initial_conditions import GRF_SAMPLER_SEMANTICS, InitialConditionArchive
from pol.data.splits import build_data_split, calibration_split_provenance
from pol.runtime.device import (
    execution_device_policy,
    require_cpu_tensors,
    verify_execution_device_policy,
)
from pol.runtime.hashing import stable_object_hash, tensor_sha256
from .conditions import (
    CONVERGENCE_CSV_SCHEMA_VERSION,
    CONVERGENCE_ROW_SCHEMA_VERSION,
    canonical_invariant_parameters,
    reference_refinement_proof,
    validate_target_reference_contract,
)
from .model1_consistency import model1_foundation_summary
from .quadrature import field_quadrature_foundation_summary
from .target_checks import target_contract_components


def calibration_provenance(
    spec: ValidationSpec,
    *,
    sample_ids: torch.Tensor | None = None,
) -> dict[str, object]:
    samples = spec.samples
    split = build_data_split(
        total_samples=int(samples.total_samples),
        n_train=int(samples.n_train),
        n_validation=int(samples.n_validation),
        n_test=int(samples.n_test),
        seed=int(samples.seed),
    )
    if sample_ids is None:
        sample_ids = torch.arange(split.total_samples, dtype=torch.long)
    return calibration_split_provenance(
        split,
        spec.target_reference.calibration_sample_ids,
        sample_ids=sample_ids,
    )


def master_payload(archive: InitialConditionArchive) -> dict[str, Any]:
    require_cpu_tensors(
        archive.__dict__,
        boundary="master initial-condition archive publication",
        name="archive",
    )
    return {
        "schema_version": "pol-initial-condition-archive-v4",
        **execution_device_policy(),
        "sample_ids": archive.sample_ids.detach().cpu(),
        "values": archive.values.detach().cpu(),
        "fourier": archive.fourier.detach().cpu(),
        "nx": archive.nx,
        "domain_length": archive.domain_length,
        "seed": archive.seed,
        "metadata": archive.metadata,
    }


def validate_master_payload_against_spec(
    payload: dict[str, Any], spec: ValidationSpec
) -> None:
    if not isinstance(payload, dict):
        raise ValueError("initial-condition archive payload must be an object")
    if payload.get("schema_version") != "pol-initial-condition-archive-v4":
        raise ValueError(
            "unsupported initial-condition archive schema; P0-04 requires v4"
        )
    verify_execution_device_policy(
        payload,
        boundary="master initial-condition archive",
    )
    require_cpu_tensors(
        payload,
        boundary="master initial-condition archive verification",
        name="master",
    )
    sample_ids = payload.get("sample_ids")
    values = payload.get("values")
    fourier = payload.get("fourier")
    if not all(isinstance(value, torch.Tensor) for value in (sample_ids, values, fourier)):
        raise ValueError("initial-condition archive tensors are missing")
    total = int(spec.samples.total_samples)
    nx = max(
        int(value)
        for value in spec.target_reference.reference_nx_candidates
    )
    if sample_ids.shape != (total,) or sample_ids.dtype != torch.long:
        raise ValueError("initial-condition archive sample_ids are inconsistent")
    if not torch.equal(sample_ids, torch.arange(total, dtype=torch.long)):
        raise ValueError("initial-condition archive sample_ids are not canonical")
    if values.shape != (total, nx) or fourier.shape != (total, nx // 2 + 1):
        raise ValueError("initial-condition archive tensor shapes are inconsistent")
    expected_dtype = torch.float64 if spec.samples.dtype == "float64" else torch.float32
    if values.dtype != expected_dtype:
        raise ValueError("initial-condition archive dtype mismatch")
    if not torch.isfinite(values).all() or not torch.isfinite(fourier).all():
        raise ValueError("initial-condition archive tensors must be finite")
    if stable_object_hash(payload.get("nx")) != stable_object_hash(nx):
        raise ValueError("initial-condition archive resolution mismatch")
    if stable_object_hash(payload.get("domain_length")) != stable_object_hash(
        float(spec.domain.length)
    ):
        raise ValueError("initial-condition archive domain mismatch")
    if stable_object_hash(payload.get("seed")) != stable_object_hash(
        int(spec.samples.seed)
    ):
        raise ValueError("initial-condition archive seed mismatch")
    metadata = payload.get("metadata")
    expected_ic = spec.samples.initial_condition.model_dump(mode="json")
    if not isinstance(metadata, dict):
        raise ValueError("initial-condition archive metadata is missing")
    for name in ("kind", "gamma", "tau", "sigma", "mean"):
        if stable_object_hash(metadata.get(name)) != stable_object_hash(
            expected_ic.get(name)
        ):
            raise ValueError(
                f"initial-condition archive specification mismatch: {name}"
            )
    if metadata.get("dtype") != spec.samples.dtype:
        raise ValueError("initial-condition archive metadata dtype mismatch")
    if metadata.get("device") != "cpu":
        raise ValueError("initial-condition archive metadata device must be cpu")
    verify_execution_device_policy(
        metadata,
        boundary="master initial-condition archive metadata",
    )
    if stable_object_hash(metadata.get("domain_length")) != stable_object_hash(
        float(spec.domain.length)
    ):
        raise ValueError("initial-condition archive sampler domain mismatch")
    if metadata.get("sampler_semantics") != GRF_SAMPLER_SEMANTICS:
        raise ValueError("initial-condition archive sampler semantics mismatch")


def _master_archive_binding(payload: dict[str, Any]) -> dict[str, Any]:
    tensor_hashes = {
        name: tensor_sha256(payload[name])
        for name in ("sample_ids", "values", "fourier")
    }
    identity = {
        "schema_version": "pol-master-initial-condition-binding-v3",
        **execution_device_policy(),
        "archive_schema_version": payload["schema_version"],
        "nx": int(payload["nx"]),
        "total_samples": int(payload["sample_ids"].numel()),
        "domain_length": float(payload["domain_length"]),
        "seed": int(payload["seed"]),
        "metadata": dict(payload["metadata"]),
        "tensor_hashes": tensor_hashes,
    }
    return {
        **identity,
        "archive_identity_hash": stable_object_hash(identity),
    }


def foundation_contract(
    spec: ValidationSpec,
    *,
    checks: dict[str, Any],
    master_payload: dict[str, Any],
) -> dict[str, Any]:
    general_names = (
        "periodic_resampling",
        "real_fourier_projector",
        "finite_input_interface",
        "fixed_decoder",
        "matched_model1_pipeline",
        "field_quadrature",
    )
    statuses = {
        name: checks[name]["status"]
        for name in general_names
    }
    samples = spec.samples
    return {
        "schema_version": "pol-validation-foundation-contract-v8",
        **execution_device_policy(),
        "domain_length": float(spec.domain.length),
        "grf_sampler_domain_length": float(
            master_payload["metadata"]["domain_length"]
        ),
        "grf_sampler_semantics": master_payload["metadata"]["sampler_semantics"],
        "dtype": samples.dtype,
        "samples": {
            "total_samples": int(samples.total_samples),
            "n_train": int(samples.n_train),
            "n_validation": int(samples.n_validation),
            "n_test": int(samples.n_test),
            "seed": int(samples.seed),
            "device": samples.device,
            "preprocessing": samples.preprocessing,
        },
        "initial_condition": samples.initial_condition.model_dump(mode="json"),
        "calibration_provenance": calibration_provenance(
            spec,
            sample_ids=master_payload["sample_ids"],
        ),
        "master_initial_conditions": _master_archive_binding(master_payload),
        "general_foundation_checks": {
            "status": (
                "pass"
                if all(value == "pass" for value in statuses.values())
                else "fail"
            ),
            "checks": statuses,
        },
        "fixed_decoder_bandwidth_contract": checks["fixed_decoder"][
            "zero_fill_characterization"
        ],
        "matched_model1_pipeline_contract": model1_foundation_summary(
            checks["matched_model1_pipeline"]
        ),
        "field_quadrature_contract": field_quadrature_foundation_summary(
            checks["field_quadrature"]
        ),
    }


def target_reference_contract(
    spec: ValidationSpec,
    convergence: dict[str, Any],
) -> dict[str, Any]:
    target = spec.target_reference
    nx_candidates = [
        int(value) for value in target.reference_nx_candidates
    ]
    reference_index = convergence.get("selected_reference_candidate_index")
    reference_allowed_indices = (
        []
        if not isinstance(reference_index, int)
        else list(range(reference_index, len(nx_candidates)))
    )
    system = target.reference_evolution.system
    system_values = system.model_dump(mode="json")
    evolution_time = float(target.reference_evolution.time)
    components = target_contract_components(spec, convergence)
    conditions = components["conditions"]
    refinement_proof = components["refinement_proof"]
    condition_index = components["condition_index"]
    method_kind = components["method_kind"]
    temporal_status = components["temporal_status"]
    condition_allowed_indices = (
        []
        if not isinstance(condition_index, int)
        else list(range(condition_index, len(conditions)))
    )
    finest_reference_index = len(nx_candidates) - 1
    finest_condition_index = len(conditions) - 1
    convergence_rows = convergence.get("rows")
    if not isinstance(convergence_rows, list):
        raise ValueError("reference convergence did not return canonical rows")
    pairwise_row_hashes = [
        row["row_hash"]
        for row in convergence_rows
        if row.get("check_kind") in {"spatial", "temporal"}
    ]
    joint_rows = [
        row
        for row in convergence_rows
        if row.get("check_kind") == "joint"
    ]
    contract = {
        "schema_version": "pol-target-reference-contract-v4",
        "system_kind": system.kind,
        "invariant_parameters": canonical_invariant_parameters(
            system.kind,
            system_values,
        ),
        "evolution_time": evolution_time,
        "dtype": spec.samples.dtype,
        "domain_length": float(spec.domain.length),
        "reference_resolution": {
            "selected_value": convergence.get("selected_reference_nx"),
            "selected_candidate_index": reference_index,
            "finest_value": nx_candidates[finest_reference_index],
            "finest_candidate_index": finest_reference_index,
            "candidates": nx_candidates,
            "candidate_refinement_proof": reference_refinement_proof(
                nx_candidates
            ),
        },
        "numerical_method_validation": {
            "kind": method_kind,
            "selected_condition": (
                None
                if not isinstance(condition_index, int)
                else conditions[condition_index]
            ),
            "selected_candidate_index": condition_index,
            "finest_condition": conditions[finest_condition_index],
            "finest_candidate_index": finest_condition_index,
            "candidates": conditions,
            "candidate_refinement_proof": refinement_proof,
            "temporal_status": temporal_status,
        },
        "convergence_evidence": {
            "schema_version": CONVERGENCE_CSV_SCHEMA_VERSION,
            "row_schema_version": CONVERGENCE_ROW_SCHEMA_VERSION,
            "row_semantics": (
                "adjacent_candidate_pairs_plus_selected_vs_finest"
            ),
            "blank_field_semantics": {
                "requested_fine_dt": (
                    "null/blank exactly for ETDRK4, reaction-diffusion, "
                    "and analytic heat conditions"
                ),
                "burgers_step_metadata": (
                    "null/blank for analytic heat and reaction-diffusion "
                    "conditions"
                ),
                "reaction_diffusion_method_fields": (
                    "null/blank for analytic heat and Burgers conditions"
                ),
            },
            "tolerances": target.reference_tolerances.model_dump(
                mode="json"
            ),
            "rows": convergence_rows,
            "rows_hash": stable_object_hash(convergence_rows),
            "pairwise_row_hashes": pairwise_row_hashes,
            "joint_row_hash": (
                None if not joint_rows else joint_rows[0]["row_hash"]
            ),
            "observed_order_diagnostics": [],
        },
        "selection_policy": target.selection_policy,
        "allowed_refinement_relation": {
            "kind": "validated_candidate_suffix_exact_membership",
            "reference_nx_allowed_indices": reference_allowed_indices,
            "reference_nx_allowed_values": [
                nx_candidates[index] for index in reference_allowed_indices
            ],
            "numerical_condition_allowed_indices": (
                condition_allowed_indices
            ),
            "numerical_condition_allowed_values": [
                conditions[index] for index in condition_allowed_indices
            ],
        },
    }
    return contract


def validate_target_reference_contract_payload(
    contract: dict[str, Any],
) -> None:
    validate_target_reference_contract(contract)
