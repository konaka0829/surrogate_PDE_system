"""Transactional lifecycle for a validated digital neural-operator baseline."""
from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable, Mapping

import torch

from pol.config.loader import load_dataset_spec, load_study_spec
from pol.data.dataset import ReferenceDataset, ensure_dataset
from pol.runtime.artifacts import RunTransaction, exact_artifact_tree, manifest_records
from pol.runtime.device import (
    execution_device_policy,
    require_cpu_tensors,
    verify_execution_device_policy,
)
from pol.runtime.environment import numerical_environment_fingerprint
from pol.runtime.hashing import stable_object_hash, tensor_sha256
from pol.runtime.io import (
    atomic_torch_save,
    file_sha256,
    write_csv,
    write_strict_json,
)
from pol.study.selection_source import (
    FrozenSourceBoundary,
    VerifiedCompletedRun,
    read_frozen_source_boundary,
    resolve_preflight_completed_run,
    resolve_verified_completed_run,
)

from .datasets import (
    build_selection_views,
    build_test_view,
    train_normalization,
)
from .evaluation import (
    TrainingOutcome,
    load_fno_checkpoint,
    predict_coefficients,
    prediction_metrics,
    representation_floor,
    state_dict_content_hash,
    summarize_seed_metrics,
    train_one_seed,
)
from .protocol import (
    DigitalBaselineSpec,
    FNO1dCandidateSpec,
    semantic_digital_baseline_spec,
)
from .parameter_accounting import (
    PARAMETER_COUNT_DEFINITION_VERSION,
    PARAMETER_COUNT_SCOPE,
    PARAMETER_SCALAR_POLICY,
    fno_parameter_counts,
    physical_parameter_counts,
)


DIGITAL_IDENTITY_SCHEMA = "pol-digital-baseline-identity-v4"
DIGITAL_MANIFEST_SCHEMA = "pol-digital-baseline-run-manifest-v4"
DIGITAL_SELECTION_SCHEMA = "pol-digital-selection-record-v3"
DIGITAL_CHECKPOINT_SCHEMA = "pol-digital-frozen-checkpoints-v3"
DIGITAL_PLAN_SCHEMA = "pol-digital-frozen-evaluation-plan-v3"
DIGITAL_SUMMARY_SCHEMA = "pol-digital-baseline-summary-v4"

_RUN_FILES = (
    "dataset_reference.json",
    "events.json",
    "evaluation_training_history.csv",
    "fairness_comparison.csv",
    "frozen_checkpoints.pt",
    "frozen_evaluation_plan.json",
    "physical_source_reference.json",
    "prediction_ensemble_metrics.csv",
    "resolved_spec.json",
    "run_summary.json",
    "selection_record.json",
    "selection_training_history.csv",
    "test_metrics.csv",
    "test_seed_metrics.csv",
    "training_compute.json",
)


@dataclass(frozen=True)
class DigitalBaselineRun:
    path: Path
    run_id: str
    summary: dict[str, Any]
    reused: bool


@dataclass(frozen=True)
class _PhysicalSourcePreflight:
    spec: Any
    completed: VerifiedCompletedRun
    boundary: FrozenSourceBoundary
    validation_rows: tuple[dict[str, str], ...]
    coordinates: tuple[dict[str, str], ...]
    reference: dict[str, Any]


@dataclass(frozen=True)
class _PhysicalTestRows:
    rows: tuple[dict[str, str], ...]
    file_sha256: str


@dataclass(frozen=True)
class _PhysicalSourceComparison:
    preflight: _PhysicalSourcePreflight
    completed: VerifiedCompletedRun
    test_rows: tuple[dict[str, str], ...]
    validation_rows: tuple[dict[str, str], ...]
    reference: dict[str, Any]


def _read_csv(path: Path) -> tuple[dict[str, str], ...]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or unsafe comparison table: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return tuple(dict(row) for row in csv.DictReader(handle))


def _read_physical_test_rows(path: Path) -> tuple[dict[str, str], ...]:
    """The single auditable parser for physical-source test metrics."""
    return _read_csv(path)


def _physical_coordinates(
    spec: DigitalBaselineSpec,
    *,
    boundary: FrozenSourceBoundary,
    validation_rows: tuple[dict[str, str], ...],
) -> tuple[dict[str, str], ...]:
    cases = boundary.selection.get("cases")
    plan_cases = boundary.plan.get("cases")
    models = boundary.archive.get("models")
    if (
        not isinstance(cases, Mapping)
        or not isinstance(plan_cases, Mapping)
        or not isinstance(models, Mapping)
    ):
        raise ValueError("physical source has an invalid frozen selection boundary")
    coordinates: list[dict[str, str]] = []
    for declared in spec.physical_comparison.rows:
        case_matches = [
            (str(case_id), case)
            for case_id, case in cases.items()
            if isinstance(case, Mapping)
            and case.get("variant_id") == declared.variant_id
        ]
        if len(case_matches) != 1:
            raise ValueError(
                "predeclared physical comparison coordinate must match exactly "
                f"one selected case: {declared.variant_id}/{declared.readout_id}"
            )
        case_id, case = case_matches[0]
        selected = case.get("selected_by_readout")
        candidate_id = (
            selected.get(declared.readout_id)
            if isinstance(selected, Mapping)
            else None
        )
        if not isinstance(candidate_id, str):
            raise ValueError(
                "predeclared physical comparison readout was not selected: "
                f"{declared.variant_id}/{declared.readout_id}"
            )
        plan_case = plan_cases.get(case_id)
        if (
            not isinstance(plan_case, Mapping)
            or plan_case.get("selected_by_readout") != selected
        ):
            raise ValueError(
                "physical comparison coordinate disagrees with frozen plan"
            )
        frozen_matches = [
            (str(model_key), model)
            for model_key, model in models.items()
            if isinstance(model, Mapping)
            and model.get("case_id") == case_id
            and model.get("variant_id") == declared.variant_id
            and model.get("readout_id") == declared.readout_id
            and model.get("candidate_id") == candidate_id
        ]
        if len(frozen_matches) != 1:
            raise ValueError(
                "physical comparison coordinate has no unique frozen model"
            )
        validation_matches = [
            row
            for row in validation_rows
            if row.get("case_id") == case_id
            and row.get("candidate_id") == candidate_id
            and row.get("readout_id") == declared.readout_id
        ]
        if len(validation_matches) != 1:
            raise ValueError(
                "physical comparison coordinate has no unique validation row"
            )
        coordinates.append(
            {
                "row_id": declared.id,
                "variant_id": declared.variant_id,
                "case_id": case_id,
                "readout_id": declared.readout_id,
                "candidate_id": candidate_id,
                "frozen_model_key": frozen_matches[0][0],
            }
        )
    return tuple(coordinates)


def _physical_source_preflight(
    spec: DigitalBaselineSpec,
    *,
    repo_root: Path,
) -> _PhysicalSourcePreflight:
    """Verify source bytes and non-test frozen state before digital training."""
    source_spec = load_study_spec(
        spec.physical_comparison.source_study_spec,
        repo_root=repo_root,
    )
    if source_spec.profile != spec.profile:
        raise ValueError("digital and physical comparison profiles must match")
    completed = resolve_preflight_completed_run(
        source_spec,
        repo_root=repo_root,
    )
    boundary = read_frozen_source_boundary(completed)
    validation_rows = _read_csv(completed.path / "validation_trials.csv")
    coordinates = _physical_coordinates(
        spec,
        boundary=boundary,
        validation_rows=validation_rows,
    )
    reference = {
        "schema_version": "pol-digital-physical-source-preflight-v2",
        "study": source_spec.name,
        "profile": source_spec.profile,
        "run_hash": completed.run_hash,
        "scientific_identity_hash": completed.scientific_identity_hash,
        "preflight_manifest_sha256": file_sha256(
            completed.path / "manifest.json"
        ),
        "preflight_manifest_records_hash": stable_object_hash(
            json.loads(
                (completed.path / "manifest.json").read_text(encoding="utf-8")
            )["files"]
        ),
        "selection_record_sha256": file_sha256(
            completed.path / "selection_record.json"
        ),
        "selection_record_hash": boundary.selection_hash,
        "frozen_evaluation_plan_sha256": file_sha256(
            completed.path / "frozen_evaluation_plan.json"
        ),
        "frozen_plan_hash": boundary.plan_hash,
        "frozen_model_archive_sha256": boundary.archive_hash,
        "validation_trials_sha256": file_sha256(
            completed.path / "validation_trials.csv"
        ),
        "dataset_artifact_id": completed.dataset.artifact_id,
        "dataset_split_hash": completed.dataset.split_hash,
        "validated_coordinates": list(coordinates),
        "source_verification": (
            "exact_manifest_bytes_and_non_test_frozen_boundary_before_training"
        ),
        "test_metrics_parsed": False,
        "source_filesystem_mutation": False,
    }
    return _PhysicalSourcePreflight(
        spec=source_spec,
        completed=completed,
        boundary=boundary,
        validation_rows=validation_rows,
        coordinates=coordinates,
        reference=reference,
    )


def _physical_test_rows_after_freeze(
    physical: _PhysicalSourcePreflight,
) -> _PhysicalTestRows:
    current_manifest_hash = file_sha256(
        physical.completed.path / "manifest.json"
    )
    if (
        current_manifest_hash
        != physical.reference["preflight_manifest_sha256"]
    ):
        raise ValueError(
            "physical source manifest changed after digital preflight"
        )
    path = physical.completed.path / "test_metrics.csv"
    test_rows = _read_physical_test_rows(path)
    for coordinate in physical.coordinates:
        matches = [
            row
            for row in test_rows
            if row.get("variant_id") == coordinate["variant_id"]
            and row.get("case_id") == coordinate["case_id"]
            and row.get("candidate_id") == coordinate["candidate_id"]
            and row.get("readout_id") == coordinate["readout_id"]
        ]
        if len(matches) != 1:
            raise ValueError(
                "frozen physical comparison coordinate must match exactly one "
                "primary test row after digital freeze"
            )
    return _PhysicalTestRows(
        rows=test_rows,
        file_sha256=file_sha256(path),
    )


def _fully_verified_physical_source(
    physical: _PhysicalSourcePreflight,
    test_data: _PhysicalTestRows,
    *,
    repo_root: Path,
    digital_plan_hash: str,
) -> _PhysicalSourceComparison:
    completed = resolve_verified_completed_run(
        physical.spec,
        repo_root=repo_root,
    )
    if (
        completed.path != physical.completed.path
        or completed.run_hash != physical.completed.run_hash
        or completed.identity != physical.completed.identity
    ):
        raise ValueError("physical source identity changed after digital freeze")
    full_manifest_hash = file_sha256(completed.path / "manifest.json")
    preflight_manifest_hash = physical.reference[
        "preflight_manifest_sha256"
    ]
    if full_manifest_hash != preflight_manifest_hash:
        raise ValueError(
            "physical source manifest changed before full verification"
        )
    if file_sha256(completed.path / "test_metrics.csv") != test_data.file_sha256:
        raise ValueError(
            "physical source test metrics changed during digital evaluation"
        )
    postfreeze = {
        "test_metrics_sha256": test_data.file_sha256,
        "test_parse_event": "physical_source_test_rows_parsed",
        "test_metrics_parsed_after_digital_plan_hash": digital_plan_hash,
        "fully_verified_source_manifest_sha256": full_manifest_hash,
        "source_manifest_unchanged": True,
        "full_source_verification": "complete_before_fairness_publication",
        "full_verification_evidence_hash": stable_object_hash(
            {
                "run_hash": completed.run_hash,
                "manifest_sha256": full_manifest_hash,
                "test_metrics_sha256": test_data.file_sha256,
                "digital_plan_hash": digital_plan_hash,
            }
        ),
    }
    reference = {
        "schema_version": "pol-digital-physical-source-reference-v3",
        "preflight": dict(physical.reference),
        "postfreeze": postfreeze,
    }
    return _PhysicalSourceComparison(
        preflight=physical,
        completed=completed,
        test_rows=test_data.rows,
        validation_rows=physical.validation_rows,
        reference=reference,
    )


def _dataset_reference(dataset: ReferenceDataset) -> dict[str, Any]:
    return {
        "schema_version": "pol-digital-dataset-reference-v1",
        **execution_device_policy(),
        "artifact_id": dataset.artifact_id,
        "split_hash": dataset.split_hash,
        "reference_nx": int(dataset.reference_nx),
        "domain_length": float(dataset.domain_length),
        "dtype": dataset.dtype_name,
        "train_count": int(dataset.train_ids.numel()),
        "validation_count": int(dataset.validation_ids.numel()),
        "test_count": int(dataset.test_ids.numel()),
        "binding_kind": dataset.binding_kind,
        "binding_status": dataset.binding_status,
        "target_reference_validation_status": (
            dataset.target_reference_validation_status
        ),
        "binding_proof_hash": dataset.binding_proof_hash,
    }


def _identity(
    spec: DigitalBaselineSpec,
    *,
    dataset: ReferenceDataset,
    physical: _PhysicalSourcePreflight,
) -> dict[str, Any]:
    return {
        "schema_version": DIGITAL_IDENTITY_SCHEMA,
        **execution_device_policy(),
        "environment": numerical_environment_fingerprint(),
        "digital_baseline": semantic_digital_baseline_spec(spec),
        "dataset": _dataset_reference(dataset),
        "physical_source": dict(physical.reference),
    }


def _normalization_hash(normalization: Mapping[str, object]) -> str:
    def hash_tensor(value: object) -> str:
        if not isinstance(value, torch.Tensor):
            raise ValueError("digital normalization entries must be tensors")
        return tensor_sha256(value.reshape(1) if value.ndim == 0 else value)

    return stable_object_hash(
        {
            "schema_version": normalization["schema_version"],
            "kind": normalization["kind"],
            "epsilon": float(normalization["epsilon"]),
            "input_mean_sha256": hash_tensor(normalization["input_mean"]),
            "input_scale_sha256": hash_tensor(normalization["input_scale"]),
            "target_mean_sha256": hash_tensor(normalization["target_mean"]),
            "target_scale_sha256": hash_tensor(normalization["target_scale"]),
            "input_clamped_count": int(normalization["input_clamped_count"]),
            "target_clamped_count": int(normalization["target_clamped_count"]),
        }
    )


def _lifting_input_channels(spec: DigitalBaselineSpec) -> int:
    return 1 if spec.model.coordinate_channel == "none" else 3


def _candidate(
    spec: DigitalBaselineSpec,
    candidate_id: str,
) -> FNO1dCandidateSpec:
    for candidate in spec.model.candidates:
        if candidate.id == candidate_id:
            return candidate
    raise ValueError(f"unknown frozen FNO candidate: {candidate_id}")


def _selection_record(
    spec: DigitalBaselineSpec,
    *,
    dataset: ReferenceDataset,
    candidate_summaries: list[dict[str, Any]],
    selected_candidate_id: str,
    history_sha256: str,
) -> dict[str, Any]:
    record = {
        "schema_version": DIGITAL_SELECTION_SCHEMA,
        "selection_kind": "validation_only_architecture_and_checkpoint",
        "dataset_artifact_id": dataset.artifact_id,
        "dataset_split_hash": dataset.split_hash,
        "n_tar": int(spec.input.n_tar),
        "q": int(spec.output.q),
        "coordinate_channel": spec.model.coordinate_channel,
        "lifting_input_channels": _lifting_input_channels(spec),
        "domain_length": float(dataset.domain_length),
        "selection_metric": spec.training.checkpoint_metric,
        "candidate_tie_break": spec.training.candidate_tie_break,
        "candidate_tie_tolerance": float(
            spec.training.candidate_tie_tolerance
        ),
        "checkpoint_tie_tolerance": float(
            spec.training.checkpoint_tie_tolerance
        ),
        "selection_seeds": list(spec.training.selection_seeds),
        "candidate_summaries": candidate_summaries,
        "selected_candidate_id": selected_candidate_id,
        "selection_training_history_sha256": history_sha256,
        "validation_data_used": True,
        "test_data_used": False,
        "evaluation_seeds_used_for_candidate_selection": False,
    }
    if any(
        key.startswith("test_") and key != "test_data_used"
        for key in record
    ):
        raise ValueError("selection record contains a test-derived field")
    return record


def _checkpoint_archive(
    spec: DigitalBaselineSpec,
    *,
    selection_hash: str,
    selected_candidate_id: str,
    outcomes: list[TrainingOutcome],
    normalization: Mapping[str, object],
    domain_length: float,
) -> dict[str, Any]:
    models = []
    for outcome in outcomes:
        models.append(
            {
                "seed": int(outcome.seed),
                "seed_role": outcome.seed_role,
                "candidate_id": outcome.candidate_id,
                "best_epoch": int(outcome.best_epoch),
                "validation_metrics": dict(outcome.best_validation_metrics),
                "parameter_count": int(outcome.parameter_count),
                "coordinate_channel": spec.model.coordinate_channel,
                "lifting_input_channels": _lifting_input_channels(spec),
                "state_dict_hash": outcome.state_dict_hash,
                "state_dict": outcome.state_dict,
            }
        )
    archive = {
        "schema_version": DIGITAL_CHECKPOINT_SCHEMA,
        **execution_device_policy(),
        "selection_record_hash": selection_hash,
        "selected_candidate_id": selected_candidate_id,
        "evaluation_seeds": list(spec.training.evaluation_seeds),
        "coordinate_channel": spec.model.coordinate_channel,
        "lifting_input_channels": _lifting_input_channels(spec),
        "domain_length": float(domain_length),
        "normalization_hash": _normalization_hash(normalization),
        "normalization": dict(normalization),
        "models": models,
    }
    require_cpu_tensors(
        archive,
        boundary="digital frozen checkpoint publication",
        name="archive",
    )
    return archive


def _frozen_plan(
    spec: DigitalBaselineSpec,
    *,
    dataset: ReferenceDataset,
    physical: _PhysicalSourcePreflight,
    selection_hash: str,
    selected_candidate_id: str,
    archive_sha256: str,
    archive: Mapping[str, Any],
) -> dict[str, Any]:
    unsigned = {
        "schema_version": DIGITAL_PLAN_SCHEMA,
        **execution_device_policy(),
        "selection_record_hash": selection_hash,
        "selected_candidate_id": selected_candidate_id,
        "frozen_checkpoints_file": "frozen_checkpoints.pt",
        "frozen_checkpoints_sha256": archive_sha256,
        "frozen_checkpoint_content_hashes": [
            model["state_dict_hash"] for model in archive["models"]
        ],
        "normalization_hash": archive["normalization_hash"],
        "evaluation_seeds": list(spec.training.evaluation_seeds),
        "dataset_artifact_id": dataset.artifact_id,
        "dataset_split_hash": dataset.split_hash,
        "n_ref": int(dataset.reference_nx),
        "n_tar": int(spec.input.n_tar),
        "q": int(spec.output.q),
        "coordinate_channel": spec.model.coordinate_channel,
        "lifting_input_channels": _lifting_input_channels(spec),
        "domain_length": float(dataset.domain_length),
        "test_sample_count": int(dataset.test_ids.numel()),
        "primary_result": spec.reporting.primary_result,
        "confidence_level": float(spec.reporting.confidence_level),
        "confidence_interval_method": (
            spec.reporting.confidence_interval_method
        ),
        "prediction_ensemble": spec.reporting.prediction_ensemble,
        "physical_source_run_hash": physical.completed.run_hash,
        "physical_source_preflight_manifest_sha256": physical.reference[
            "preflight_manifest_sha256"
        ],
        "physical_source_selection_record_hash": (
            physical.boundary.selection_hash
        ),
        "physical_source_frozen_plan_hash": physical.boundary.plan_hash,
        "test_data_used": False,
    }
    return {
        **unsigned,
        "plan_content_hash": stable_object_hash(unsigned),
    }


def _read_frozen_boundary(
    root: Path,
    spec: DigitalBaselineSpec,
    *,
    domain_length: float,
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    selection = json.loads(
        (root / "selection_record.json").read_text(encoding="utf-8")
    )
    if selection.get("schema_version") != DIGITAL_SELECTION_SCHEMA:
        raise ValueError("unsupported digital selection record")
    if selection.get("test_data_used") is not False:
        raise ValueError("digital selection record is not test isolated")
    expected_channels = _lifting_input_channels(spec)
    if (
        selection.get("coordinate_channel")
        != spec.model.coordinate_channel
        or selection.get("lifting_input_channels") != expected_channels
        or selection.get("domain_length") != float(domain_length)
    ):
        raise ValueError(
            "digital selection coordinate architecture mismatch"
        )
    candidate_summaries = selection.get("candidate_summaries")
    if (
        not isinstance(candidate_summaries, list)
        or any(
            not isinstance(summary, Mapping)
            or summary.get("coordinate_channel")
            != spec.model.coordinate_channel
            or summary.get("lifting_input_channels") != expected_channels
            for summary in candidate_summaries
        )
    ):
        raise ValueError(
            "digital candidate coordinate architecture mismatch"
        )
    selection_hash = stable_object_hash(selection)
    if selection.get("selection_training_history_sha256") != file_sha256(
        root / "selection_training_history.csv"
    ):
        raise ValueError("digital selection history hash mismatch")

    plan = json.loads(
        (root / "frozen_evaluation_plan.json").read_text(encoding="utf-8")
    )
    if plan.get("schema_version") != DIGITAL_PLAN_SCHEMA:
        raise ValueError("unsupported digital frozen evaluation plan")
    unsigned_plan = dict(plan)
    stored_plan_hash = unsigned_plan.pop("plan_content_hash", None)
    if (
        not isinstance(stored_plan_hash, str)
        or stable_object_hash(unsigned_plan) != stored_plan_hash
    ):
        raise ValueError("digital frozen evaluation plan content hash mismatch")
    if (
        plan.get("selection_record_hash") != selection_hash
        or plan.get("test_data_used") is not False
    ):
        raise ValueError("digital frozen plan does not match selection")
    if (
        plan.get("coordinate_channel") != spec.model.coordinate_channel
        or plan.get("lifting_input_channels") != expected_channels
        or plan.get("domain_length") != float(domain_length)
    ):
        raise ValueError("digital frozen plan coordinate architecture mismatch")

    archive_path = root / "frozen_checkpoints.pt"
    if file_sha256(archive_path) != plan.get("frozen_checkpoints_sha256"):
        raise ValueError("digital frozen checkpoint archive byte hash mismatch")
    archive = torch.load(
        archive_path,
        map_location="cpu",
        weights_only=True,
    )
    if (
        not isinstance(archive, dict)
        or archive.get("schema_version") != DIGITAL_CHECKPOINT_SCHEMA
    ):
        raise ValueError("unsupported digital frozen checkpoint archive")
    verify_execution_device_policy(
        archive,
        boundary="digital frozen checkpoint archive",
    )
    require_cpu_tensors(
        archive,
        boundary="digital frozen checkpoint read-back",
        name="archive",
    )
    if archive.get("selection_record_hash") != selection_hash:
        raise ValueError("digital checkpoint selection hash mismatch")
    if (
        archive.get("coordinate_channel") != spec.model.coordinate_channel
        or archive.get("lifting_input_channels") != expected_channels
        or archive.get("domain_length") != float(domain_length)
    ):
        raise ValueError(
            "digital checkpoint coordinate architecture mismatch"
        )
    if archive.get("selected_candidate_id") != selection.get(
        "selected_candidate_id"
    ):
        raise ValueError("digital checkpoint candidate mismatch")
    if archive.get("normalization_hash") != _normalization_hash(
        archive["normalization"]
    ):
        raise ValueError("digital checkpoint normalization hash mismatch")
    models = archive.get("models")
    if not isinstance(models, list):
        raise ValueError("digital checkpoint models must be a list")
    seeds = []
    for model in models:
        if not isinstance(model, dict):
            raise ValueError("digital frozen model record must be an object")
        state = model.get("state_dict")
        if not isinstance(state, dict):
            raise ValueError("digital frozen model has no state_dict")
        if (
            model.get("coordinate_channel") != spec.model.coordinate_channel
            or model.get("lifting_input_channels") != expected_channels
        ):
            raise ValueError(
                "digital frozen model coordinate architecture mismatch"
            )
        if state_dict_content_hash(state) != model.get("state_dict_hash"):
            raise ValueError("frozen FNO checkpoint content hash mismatch")
        lifting_weight = state.get("lifting.weight")
        if (
            not isinstance(lifting_weight, torch.Tensor)
            or lifting_weight.ndim != 2
            or int(lifting_weight.shape[1]) != expected_channels
        ):
            raise ValueError("FNO lifting input-channel mismatch")
        candidate = _candidate(spec, str(model.get("candidate_id")))
        load_fno_checkpoint(
            candidate,
            n_tar=int(spec.input.n_tar),
            dtype=lifting_weight.dtype,
            state_dict=state,
            expected_hash=str(model["state_dict_hash"]),
            coordinate_channel=spec.model.coordinate_channel,
            domain_length=domain_length,
        )
        seeds.append(int(model["seed"]))
    if seeds != list(spec.training.evaluation_seeds):
        raise ValueError("digital frozen evaluation seeds mismatch")
    if plan.get("frozen_checkpoint_content_hashes") != [
        model["state_dict_hash"] for model in models
    ]:
        raise ValueError("digital frozen plan checkpoint hashes mismatch")
    return selection, selection_hash, plan, archive


def _select_candidate(
    spec: DigitalBaselineSpec,
    outcomes: list[TrainingOutcome],
) -> tuple[str, list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    selected_id: str | None = None
    selected_value = math.inf
    metric = spec.training.checkpoint_metric
    for candidate in spec.model.candidates:
        members = [
            outcome for outcome in outcomes
            if outcome.candidate_id == candidate.id
        ]
        if [member.seed for member in members] != list(
            spec.training.selection_seeds
        ):
            raise RuntimeError("candidate selection seed coverage is incomplete")
        statistics = summarize_seed_metrics(
            [{metric: member.best_validation_metrics[metric]} for member in members]
        )
        value = float(statistics[metric])
        summary = {
            "candidate_id": candidate.id,
            "modes": int(candidate.modes),
            "width": int(candidate.width),
            "depth": int(candidate.depth),
            "coordinate_channel": spec.model.coordinate_channel,
            "lifting_input_channels": _lifting_input_channels(spec),
            "parameter_count": int(members[0].parameter_count),
            "selection_seed_count": len(members),
            "selection_seed_values": [
                float(member.best_validation_metrics[metric])
                for member in members
            ],
            "selection_seed_best_epochs": [
                int(member.best_epoch) for member in members
            ],
            **statistics,
        }
        summaries.append(summary)
        if value < selected_value - float(
            spec.training.candidate_tie_tolerance
        ):
            selected_value = value
            selected_id = candidate.id
    if selected_id is None:
        raise RuntimeError("digital baseline selected no architecture candidate")
    for summary in summaries:
        summary["selected"] = summary["candidate_id"] == selected_id
    return selected_id, summaries


def _history_rows(outcomes: Iterable[TrainingOutcome]) -> list[dict[str, Any]]:
    return [
        dict(row)
        for outcome in outcomes
        for row in outcome.history
    ]


def _write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(path, rows, fieldnames=list(rows[0]))


def _test_tables(
    spec: DigitalBaselineSpec,
    *,
    dataset: ReferenceDataset,
    test_view: Any,
    archive: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], list[torch.Tensor]]:
    candidate = _candidate(spec, str(archive["selected_candidate_id"]))
    floor = {
        f"test_{key}": value
        for key, value in representation_floor(
            test_view,
            domain_length=dataset.domain_length,
        ).items()
    }
    seed_rows: list[dict[str, Any]] = []
    seed_metric_values: list[dict[str, float]] = []
    predictions: list[torch.Tensor] = []
    for model_record in archive["models"]:
        model = load_fno_checkpoint(
            candidate,
            n_tar=int(spec.input.n_tar),
            dtype=test_view.inputs.dtype,
            state_dict=model_record["state_dict"],
            expected_hash=str(model_record["state_dict_hash"]),
            coordinate_channel=spec.model.coordinate_channel,
            domain_length=dataset.domain_length,
        )
        prediction = predict_coefficients(
            model,
            test_view.inputs,
            archive["normalization"],
            q=int(spec.output.q),
            domain_length=dataset.domain_length,
            batch_size=int(spec.training.batch_size),
        )
        predictions.append(prediction)
        metrics = {
            f"test_{key}": value
            for key, value in prediction_metrics(
                prediction,
                test_view,
                domain_length=dataset.domain_length,
            ).items()
        }
        seed_metric_values.append({**metrics, **floor})
        seed_rows.append(
            {
                "schema_version": "pol-digital-test-seed-row-v2",
                "test_result_kind": "independent_training_seed_realization",
                "model_kind": "fno1d",
                "candidate_id": candidate.id,
                "seed": int(model_record["seed"]),
                "checkpoint_epoch": int(model_record["best_epoch"]),
                "checkpoint_hash": model_record["state_dict_hash"],
                "parameter_count": int(model_record["parameter_count"]),
                "coordinate_channel": spec.model.coordinate_channel,
                "lifting_input_channels": _lifting_input_channels(spec),
                "n_ref": int(dataset.reference_nx),
                "n_tar": int(spec.input.n_tar),
                "q": int(spec.output.q),
                **metrics,
                **floor,
            }
        )
    statistics = summarize_seed_metrics(seed_metric_values)
    primary = {
        "schema_version": "pol-digital-test-summary-row-v2",
        "test_result_kind": "independent_training_seed_metric_summary",
        "model_kind": "fno1d",
        "candidate_id": candidate.id,
        "parameter_count": int(archive["models"][0]["parameter_count"]),
        "coordinate_channel": spec.model.coordinate_channel,
        "lifting_input_channels": _lifting_input_channels(spec),
        "n_ref": int(dataset.reference_nx),
        "n_tar": int(spec.input.n_tar),
        "q": int(spec.output.q),
        "test_seed_count": len(seed_rows),
        "test_seed_std_ddof": 1,
        "test_confidence_level": 0.95,
        "test_confidence_interval_method": "student_t",
        "prediction_ensemble_in_primary": False,
        **statistics,
    }
    ensemble_prediction = torch.stack(predictions, dim=0).mean(dim=0)
    ensemble_metrics = {
        f"test_ensemble_{key}": value
        for key, value in prediction_metrics(
            ensemble_prediction,
            test_view,
            domain_length=dataset.domain_length,
        ).items()
    }
    ensemble = {
        "schema_version": "pol-digital-test-ensemble-row-v2",
        "test_result_kind": "prediction_ensemble",
        "model_kind": "fno1d",
        "candidate_id": candidate.id,
        "coordinate_channel": spec.model.coordinate_channel,
        "lifting_input_channels": _lifting_input_channels(spec),
        "ensemble_member_count": len(seed_rows),
        "ensemble_member_seeds_hash": stable_object_hash(
            list(spec.training.evaluation_seeds)
        ),
        "ensemble_member_checkpoint_hash": stable_object_hash(
            [row["checkpoint_hash"] for row in seed_rows]
        ),
        "n_ref": int(dataset.reference_nx),
        "n_tar": int(spec.input.n_tar),
        "q": int(spec.output.q),
        **ensemble_metrics,
        **{
            key.replace("test_", "test_ensemble_", 1): value
            for key, value in floor.items()
        },
    }
    return seed_rows, primary, ensemble, predictions


def _float(row: Mapping[str, str], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"physical comparison row has invalid {key}") from exc
    if not math.isfinite(value):
        raise ValueError(f"physical comparison row has non-finite {key}")
    return value


def _physical_count_fields(
    physical: _PhysicalSourceComparison,
    coordinate: Mapping[str, str],
    row: Mapping[str, str],
) -> dict[str, Any]:
    models = physical.preflight.boundary.archive.get("models")
    model_key = coordinate["frozen_model_key"]
    entry = models.get(model_key) if isinstance(models, Mapping) else None
    model = entry.get("model") if isinstance(entry, Mapping) else None
    if not isinstance(model, Mapping):
        raise ValueError("physical fairness coordinate has no frozen model")
    if row.get("readout_kind") != model.get("kind"):
        raise ValueError(
            "physical fairness row readout kind disagrees with frozen model"
        )
    if model.get("kind") == "random_feature_ridge" and int(
        row.get("selected_random_feature_width", -1)
    ) != int(model.get("width", -2)):
        raise ValueError(
            "physical fairness row width disagrees with frozen model"
        )
    return physical_parameter_counts(
        model,
        observation_count=int(row["J"]),
        q=int(row["q"]),
    )


def _fairness_rows(
    spec: DigitalBaselineSpec,
    *,
    dataset: ReferenceDataset,
    physical: _PhysicalSourceComparison,
    selection: Mapping[str, Any],
    primary: Mapping[str, Any],
    digital_counts: Mapping[str, Any],
    compute: Mapping[str, Any],
    run_id: str,
) -> list[dict[str, Any]]:
    common = {
        "dataset_artifact_id": dataset.artifact_id,
        "dataset_split_hash": dataset.split_hash,
        "n_ref": int(dataset.reference_nx),
        "n_tar": int(spec.input.n_tar),
        "q": int(spec.output.q),
        "input_dimension": int(spec.input.n_tar),
        "output_dimension": int(spec.output.q),
        "digital_coordinate_channel": None,
        "digital_lifting_input_channels": None,
        "energy_measurement_status": "not_measured",
        "energy_comparison_allowed": False,
        "wall_clock_comparison_allowed": False,
        "wall_clock_energy_policy": (
            spec.reporting.wall_clock_energy_comparison
        ),
        "field_metric": "test_field_relative_l2_mean",
        "data_metric": "test_data_field_relative_l2_mean",
    }
    rows: list[dict[str, Any]] = []
    for declared in spec.physical_comparison.rows:
        coordinate = next(
            coordinate
            for coordinate in physical.preflight.coordinates
            if coordinate["row_id"] == declared.id
        )
        test_matches = [
            row
            for row in physical.test_rows
            if row.get("variant_id") == declared.variant_id
            and row.get("readout_id") == declared.readout_id
        ]
        test_row = test_matches[0]
        validation_matches = [
            row
            for row in physical.validation_rows
            if row.get("case_id") == test_row.get("case_id")
            and row.get("candidate_id") == test_row.get("candidate_id")
            and row.get("readout_id") == declared.readout_id
        ]
        if len(validation_matches) != 1:
            raise ValueError(
                "physical fairness row has no unique validation-selection row"
            )
        validation_row = validation_matches[0]
        if (
            int(test_row["n_tar"]) != int(spec.input.n_tar)
            or int(test_row["q"]) != int(spec.output.q)
        ):
            raise ValueError(
                "physical fairness row does not match digital n_tar/q"
            )
        seed_count = int(test_row.get("test_seed_count") or 1)
        rows.append(
            {
                "schema_version": "pol-digital-fairness-row-v3",
                "row_id": declared.id,
                "label": declared.label,
                "model_family": "physical_feature_readout",
                "model_kind": test_row["readout_kind"],
                "variant_id": declared.variant_id,
                "readout_id": declared.readout_id,
                "inference_path": (
                    "finite_n_tar_to_physical_feature_generator_to_J_"
                    "observations_to_frozen_readout_to_q_coefficients"
                ),
                "source_run_hash": physical.completed.run_hash,
                "source_condition_hash": test_row[
                    "feature_system_condition_hash"
                ],
                **common,
                **_physical_count_fields(
                    physical,
                    coordinate,
                    test_row,
                ),
                "training_compute_metadata_status": (
                    "not_recorded_under_common_digital_training_protocol"
                ),
                "training_wall_time_seconds": None,
                "training_process_time_seconds": None,
                "optimizer_step_count": None,
                "validation_selection_metric": (
                    spec.training.checkpoint_metric
                ),
                "validation_selection_value": _float(
                    validation_row,
                    spec.training.checkpoint_metric,
                ),
                "test_seed_count": seed_count,
                "test_seed_std_ddof": (
                    1 if test_row["readout_kind"] == "random_feature_ridge" else None
                ),
                "test_confidence_interval_method": (
                    test_row.get("test_confidence_interval_method")
                    or "not_applicable_deterministic_model"
                ),
                "test_field_relative_l2_mean": _float(
                    test_row,
                    "test_field_relative_l2_mean",
                ),
                "test_field_relative_l2_mean_seed_std": (
                    _float(
                        test_row,
                        "test_field_relative_l2_mean_seed_std",
                    )
                    if test_row["readout_kind"] == "random_feature_ridge"
                    else None
                ),
                "test_field_relative_l2_mean_seed_ci95_low": (
                    _float(
                        test_row,
                        "test_field_relative_l2_mean_seed_ci95_low",
                    )
                    if test_row["readout_kind"] == "random_feature_ridge"
                    else None
                ),
                "test_field_relative_l2_mean_seed_ci95_high": (
                    _float(
                        test_row,
                        "test_field_relative_l2_mean_seed_ci95_high",
                    )
                    if test_row["readout_kind"] == "random_feature_ridge"
                    else None
                ),
                "test_data_field_relative_l2_mean": _float(
                    test_row,
                    "test_data_field_relative_l2_mean",
                ),
                "test_field_representation_floor_relative_l2_mean": _float(
                    test_row,
                    "test_representation_floor_relative_l2_mean",
                ),
                "test_data_representation_floor_relative_l2_mean": _float(
                    test_row,
                    "test_data_representation_floor_relative_l2_mean",
                ),
                "prediction_ensemble_in_primary": False,
            }
        )
    selected = next(
        summary
        for summary in selection["candidate_summaries"]
        if summary["selected"]
    )
    rows.append(
        {
            "schema_version": "pol-digital-fairness-row-v3",
            "row_id": "fno1d",
            "label": "FNO1d",
            "model_family": "digital_neural_operator",
            "model_kind": "fno1d",
            "variant_id": "digital_fno1d",
            "readout_id": "not_applicable",
            "inference_path": (
                "finite_n_tar_to_fno1d_to_q_real_fourier_coefficients"
            ),
            "source_run_hash": run_id,
            "source_condition_hash": stable_object_hash(
                {
                    "candidate_id": selected["candidate_id"],
                    "modes": selected["modes"],
                    "width": selected["width"],
                    "depth": selected["depth"],
                    "coordinate_channel": selected["coordinate_channel"],
                    "lifting_input_channels": selected[
                        "lifting_input_channels"
                    ],
                }
            ),
            **common,
            "digital_coordinate_channel": spec.model.coordinate_channel,
            "digital_lifting_input_channels": _lifting_input_channels(spec),
            **dict(digital_counts),
            "training_compute_metadata_status": "measured_current_cpu_process",
            "training_wall_time_seconds": compute["total_training_wall_time_seconds"],
            "training_process_time_seconds": compute[
                "total_training_process_time_seconds"
            ],
            "optimizer_step_count": compute["optimizer_step_count"],
            "validation_selection_metric": spec.training.checkpoint_metric,
            "validation_selection_value": float(
                selected[spec.training.checkpoint_metric]
            ),
            "test_seed_count": int(primary["test_seed_count"]),
            "test_seed_std_ddof": int(primary["test_seed_std_ddof"]),
            "test_confidence_interval_method": primary[
                "test_confidence_interval_method"
            ],
            "test_field_relative_l2_mean": float(
                primary["test_field_relative_l2_mean"]
            ),
            "test_field_relative_l2_mean_seed_std": float(
                primary["test_field_relative_l2_mean_seed_std"]
            ),
            "test_field_relative_l2_mean_seed_ci95_low": float(
                primary["test_field_relative_l2_mean_seed_ci95_low"]
            ),
            "test_field_relative_l2_mean_seed_ci95_high": float(
                primary["test_field_relative_l2_mean_seed_ci95_high"]
            ),
            "test_data_field_relative_l2_mean": float(
                primary["test_data_field_relative_l2_mean"]
            ),
            "test_field_representation_floor_relative_l2_mean": float(
                primary["test_representation_floor_relative_l2_mean"]
            ),
            "test_data_representation_floor_relative_l2_mean": float(
                primary[
                    "test_data_representation_floor_relative_l2_mean"
                ]
            ),
            "prediction_ensemble_in_primary": False,
        }
    )
    return rows


def _write_manifest(root: Path, identity: Mapping[str, Any]) -> None:
    records = manifest_records(root, _RUN_FILES)
    write_strict_json(
        root / "manifest.json",
        {
            "schema_version": DIGITAL_MANIFEST_SCHEMA,
            "run_id": stable_object_hash(dict(identity)),
            "identity": dict(identity),
            "files": records,
        },
    )


def _build_run(
    staging: Path,
    spec: DigitalBaselineSpec,
    *,
    dataset: ReferenceDataset,
    physical: _PhysicalSourcePreflight,
    identity: Mapping[str, Any],
    repo_root: Path,
) -> None:
    source_manifest_hash = physical.reference[
        "preflight_manifest_sha256"
    ]
    source_selection_hash = physical.boundary.selection_hash
    source_plan_hash = physical.boundary.plan_hash
    events: list[dict[str, Any]] = [
        {
            "event": "physical_source_preflight_verified",
            "source_run_hash": physical.completed.run_hash,
            "source_manifest_sha256": source_manifest_hash,
            "source_selection_record_hash": source_selection_hash,
            "source_frozen_plan_hash": source_plan_hash,
            "source_test_metrics_parsed": False,
        }
    ]
    write_strict_json(
        staging / "resolved_spec.json",
        spec.model_dump(mode="json"),
    )
    write_strict_json(
        staging / "dataset_reference.json",
        _dataset_reference(dataset),
    )
    selection_views = build_selection_views(
        dataset,
        n_tar=int(spec.input.n_tar),
        q=int(spec.output.q),
    )
    normalization = train_normalization(
        selection_views.train,
        epsilon=float(spec.normalization.epsilon),
    )
    selection_outcomes: list[TrainingOutcome] = []
    for candidate in spec.model.candidates:
        for seed in spec.training.selection_seeds:
            selection_outcomes.append(
                train_one_seed(
                    candidate,
                    spec.training,
                    seed=int(seed),
                    seed_role="candidate_selection",
                    train_view=selection_views.train,
                    validation_view=selection_views.validation,
                    normalization=normalization,
                    domain_length=dataset.domain_length,
                    coordinate_channel=spec.model.coordinate_channel,
                )
            )
    selection_history = _history_rows(selection_outcomes)
    _write_history(
        staging / "selection_training_history.csv",
        selection_history,
    )
    selected_candidate_id, candidate_summaries = _select_candidate(
        spec,
        selection_outcomes,
    )
    selection = _selection_record(
        spec,
        dataset=dataset,
        candidate_summaries=candidate_summaries,
        selected_candidate_id=selected_candidate_id,
        history_sha256=file_sha256(
            staging / "selection_training_history.csv"
        ),
    )
    write_strict_json(staging / "selection_record.json", selection)
    loaded_selection = json.loads(
        (staging / "selection_record.json").read_text(encoding="utf-8")
    )
    if loaded_selection != selection:
        raise ValueError("digital selection record read-back mismatch")
    selection_hash = stable_object_hash(loaded_selection)
    events.append(
        {
            "event": "selection_complete",
            "selection_record_hash": selection_hash,
            "source_manifest_sha256": source_manifest_hash,
            "source_selection_record_hash": source_selection_hash,
            "source_frozen_plan_hash": source_plan_hash,
        }
    )

    selected_candidate = _candidate(spec, selected_candidate_id)
    evaluation_outcomes = [
        train_one_seed(
            selected_candidate,
            spec.training,
            seed=int(seed),
            seed_role="independent_evaluation_model",
            train_view=selection_views.train,
            validation_view=selection_views.validation,
            normalization=normalization,
            domain_length=dataset.domain_length,
            coordinate_channel=spec.model.coordinate_channel,
        )
        for seed in spec.training.evaluation_seeds
    ]
    evaluation_history = _history_rows(evaluation_outcomes)
    _write_history(
        staging / "evaluation_training_history.csv",
        evaluation_history,
    )
    archive = _checkpoint_archive(
        spec,
        selection_hash=selection_hash,
        selected_candidate_id=selected_candidate_id,
        outcomes=evaluation_outcomes,
        normalization=normalization,
        domain_length=dataset.domain_length,
    )
    atomic_torch_save(staging / "frozen_checkpoints.pt", archive)
    archive_sha256 = file_sha256(staging / "frozen_checkpoints.pt")
    plan = _frozen_plan(
        spec,
        dataset=dataset,
        physical=physical,
        selection_hash=selection_hash,
        selected_candidate_id=selected_candidate_id,
        archive_sha256=archive_sha256,
        archive=archive,
    )
    write_strict_json(staging / "frozen_evaluation_plan.json", plan)
    events.append(
        {
            "event": "freeze_written",
            "selection_record_hash": selection_hash,
            "plan_content_hash": plan["plan_content_hash"],
            "source_manifest_sha256": source_manifest_hash,
            "source_selection_record_hash": source_selection_hash,
            "source_frozen_plan_hash": source_plan_hash,
        }
    )
    persisted_selection, persisted_selection_hash, persisted_plan, loaded_archive = (
        _read_frozen_boundary(
            staging,
            spec,
            domain_length=dataset.domain_length,
        )
    )
    digital_counts = fno_parameter_counts(loaded_archive)
    events.append(
        {
            "event": "freeze_read_back",
            "selection_record_hash": persisted_selection_hash,
            "plan_content_hash": persisted_plan["plan_content_hash"],
            "source_manifest_sha256": source_manifest_hash,
            "source_selection_record_hash": source_selection_hash,
            "source_frozen_plan_hash": source_plan_hash,
        }
    )

    events.append(
        {
            "event": "first_test_tensor_request",
            "plan_content_hash": persisted_plan["plan_content_hash"],
            "selection_record_hash": persisted_selection_hash,
            "source_manifest_sha256": source_manifest_hash,
            "source_selection_record_hash": source_selection_hash,
            "source_frozen_plan_hash": source_plan_hash,
        }
    )
    test_view = build_test_view(
        dataset,
        n_tar=int(spec.input.n_tar),
        q=int(spec.output.q),
    )
    physical_test_data = _physical_test_rows_after_freeze(physical)
    events.append(
        {
            "event": "physical_source_test_rows_parsed",
            "plan_content_hash": persisted_plan["plan_content_hash"],
            "selection_record_hash": persisted_selection_hash,
            "source_manifest_sha256": source_manifest_hash,
            "source_test_metrics_sha256": physical_test_data.file_sha256,
            "source_selection_record_hash": source_selection_hash,
            "source_frozen_plan_hash": source_plan_hash,
        }
    )
    seed_rows, primary, ensemble, _ = _test_tables(
        spec,
        dataset=dataset,
        test_view=test_view,
        archive=loaded_archive,
    )
    if int(primary["parameter_count"]) != int(
        digital_counts["total_stored_parameter_count"]
    ):
        raise ValueError(
            "digital primary parameter count disagrees with frozen state_dict"
        )
    events.append(
        {
            "event": "first_test_metric",
            "plan_content_hash": persisted_plan["plan_content_hash"],
            "selection_record_hash": persisted_selection_hash,
            "source_manifest_sha256": source_manifest_hash,
            "source_test_metrics_sha256": physical_test_data.file_sha256,
            "source_selection_record_hash": source_selection_hash,
            "source_frozen_plan_hash": source_plan_hash,
        }
    )
    write_csv(
        staging / "test_seed_metrics.csv",
        seed_rows,
        fieldnames=list(seed_rows[0]),
    )
    write_csv(
        staging / "test_metrics.csv",
        [primary],
        fieldnames=list(primary),
    )
    write_csv(
        staging / "prediction_ensemble_metrics.csv",
        [ensemble],
        fieldnames=list(ensemble),
    )
    all_outcomes = [*selection_outcomes, *evaluation_outcomes]
    optimizer_steps = (
        len(all_outcomes)
        * int(spec.training.epochs)
        * math.ceil(
            int(dataset.train_ids.numel()) / int(spec.training.batch_size)
        )
    )
    compute = {
        "schema_version": "pol-digital-training-compute-v1",
        **execution_device_policy(),
        "optimizer": spec.training.optimizer.model_dump(mode="json"),
        "coordinate_channel": spec.model.coordinate_channel,
        "lifting_input_channels": _lifting_input_channels(spec),
        "domain_length": float(dataset.domain_length),
        "candidate_count": len(spec.model.candidates),
        "selection_training_model_count": len(selection_outcomes),
        "evaluation_training_model_count": len(evaluation_outcomes),
        "epochs_per_model": int(spec.training.epochs),
        "batch_size": int(spec.training.batch_size),
        "optimizer_step_count": optimizer_steps,
        "parameter_count_definition_version": (
            PARAMETER_COUNT_DEFINITION_VERSION
        ),
        "parameter_count_scope": PARAMETER_COUNT_SCOPE,
        "parameter_scalar_policy": PARAMETER_SCALAR_POLICY,
        "trainable_parameter_count": digital_counts[
            "trainable_parameter_count"
        ],
        "fixed_random_parameter_count": digital_counts[
            "fixed_random_parameter_count"
        ],
        "total_stored_parameter_count": digital_counts[
            "total_stored_parameter_count"
        ],
        "per_model": [
            {
                "candidate_id": outcome.candidate_id,
                "seed": int(outcome.seed),
                "seed_role": outcome.seed_role,
                "best_epoch": int(outcome.best_epoch),
                "wall_time_seconds": float(outcome.wall_time_seconds),
                "process_time_seconds": float(outcome.process_time_seconds),
            }
            for outcome in all_outcomes
        ],
        "total_training_wall_time_seconds": math.fsum(
            outcome.wall_time_seconds for outcome in all_outcomes
        ),
        "total_training_process_time_seconds": math.fsum(
            outcome.process_time_seconds for outcome in all_outcomes
        ),
        "wall_time_scope": "training_calls_only_current_cpu_process",
        "energy_measurement_status": "not_measured",
        "energy_comparison_allowed": False,
        "wall_clock_comparison_allowed": False,
        "comparison_reason": (
            "physical source was not measured under the same training and "
            "inference timing protocol"
        ),
    }
    write_strict_json(staging / "training_compute.json", compute)
    comparison_data = _PhysicalSourceComparison(
        preflight=physical,
        completed=physical.completed,
        test_rows=physical_test_data.rows,
        validation_rows=physical.validation_rows,
        reference={},
    )
    fairness = _fairness_rows(
        spec,
        dataset=dataset,
        physical=comparison_data,
        selection=persisted_selection,
        primary=primary,
        digital_counts=digital_counts,
        compute=compute,
        run_id=stable_object_hash(dict(identity)),
    )
    events.append(
        {
            "event": "fairness_comparison_computed",
            "plan_content_hash": persisted_plan["plan_content_hash"],
            "selection_record_hash": persisted_selection_hash,
            "source_manifest_sha256": source_manifest_hash,
            "source_test_metrics_sha256": physical_test_data.file_sha256,
            "source_selection_record_hash": source_selection_hash,
            "source_frozen_plan_hash": source_plan_hash,
        }
    )
    verified_physical = _fully_verified_physical_source(
        physical,
        physical_test_data,
        repo_root=repo_root,
        digital_plan_hash=persisted_plan["plan_content_hash"],
    )
    events.append(
        {
            "event": "physical_source_full_verified",
            "plan_content_hash": persisted_plan["plan_content_hash"],
            "selection_record_hash": persisted_selection_hash,
            "source_manifest_sha256": source_manifest_hash,
            "source_test_metrics_sha256": physical_test_data.file_sha256,
            "source_selection_record_hash": source_selection_hash,
            "source_frozen_plan_hash": source_plan_hash,
            "source_full_verification_evidence_hash": (
                verified_physical.reference["postfreeze"][
                    "full_verification_evidence_hash"
                ]
            ),
        }
    )
    write_strict_json(
        staging / "physical_source_reference.json",
        verified_physical.reference,
    )
    write_csv(
        staging / "fairness_comparison.csv",
        fairness,
        fieldnames=list(fairness[0]),
    )
    events.append(
        {
            "event": "fairness_comparison_written",
            "plan_content_hash": persisted_plan["plan_content_hash"],
            "selection_record_hash": persisted_selection_hash,
            "source_manifest_sha256": source_manifest_hash,
            "source_test_metrics_sha256": physical_test_data.file_sha256,
            "source_selection_record_hash": source_selection_hash,
            "source_frozen_plan_hash": source_plan_hash,
            "source_full_verification_evidence_hash": (
                verified_physical.reference["postfreeze"][
                    "full_verification_evidence_hash"
                ]
            ),
        }
    )

    summary = {
        "schema_version": DIGITAL_SUMMARY_SCHEMA,
        "status": "complete",
        **execution_device_policy(),
        "name": spec.name,
        "profile": spec.profile,
        "run_id": stable_object_hash(dict(identity)),
        "model_kind": "fno1d",
        "coordinate_channel": spec.model.coordinate_channel,
        "lifting_input_channels": _lifting_input_channels(spec),
        "domain_length": float(dataset.domain_length),
        "dataset_artifact_id": dataset.artifact_id,
        "dataset_split_hash": dataset.split_hash,
        "n_ref": int(dataset.reference_nx),
        "n_tar": int(spec.input.n_tar),
        "q": int(spec.output.q),
        "candidate_count": len(spec.model.candidates),
        "selected_candidate_id": selected_candidate_id,
        "parameter_count_scope": PARAMETER_COUNT_SCOPE,
        "parameter_scalar_policy": PARAMETER_SCALAR_POLICY,
        "parameter_count_definition_version": (
            PARAMETER_COUNT_DEFINITION_VERSION
        ),
        "model_family_parameter_count_policy": {
            "physical_feature_readout": (
                "frozen_tensor_shapes_cross_checked_against_readout_formula"
            ),
            "digital_neural_operator": (
                "frozen_state_dict_real_scalars_cross_checked_against_training"
            ),
            "physical_dynamics_included": False,
            "hardware_components_included": False,
        },
        "primary_model_parameter_counts": dict(digital_counts),
        "selection_seed_count": len(spec.training.selection_seeds),
        "evaluation_seed_count": len(spec.training.evaluation_seeds),
        "selection_record_hash": persisted_selection_hash,
        "frozen_plan_hash": persisted_plan["plan_content_hash"],
        "frozen_checkpoint_archive_sha256": archive_sha256,
        "physical_source_run_hash": physical.completed.run_hash,
        "physical_source_manifest_sha256": source_manifest_hash,
        "physical_source_preflight_verified_before_training": True,
        "physical_source_test_parsed_after_freeze": True,
        "physical_source_fully_verified_before_fairness_publication": True,
        "physical_source_test_metrics_sha256": physical_test_data.file_sha256,
        "freeze_verified_before_test": True,
        "primary_test_result_kind": primary["test_result_kind"],
        "prediction_ensemble_separate": True,
        "fairness_row_count": len(fairness),
        "main_profile_executed": spec.profile == "main",
    }
    write_strict_json(staging / "run_summary.json", summary)
    events.append(
        {
            "event": "numerical_run_complete",
            "run_id": summary["run_id"],
            "plan_content_hash": persisted_plan["plan_content_hash"],
            "selection_record_hash": persisted_selection_hash,
            "source_manifest_sha256": source_manifest_hash,
            "source_test_metrics_sha256": physical_test_data.file_sha256,
            "source_selection_record_hash": source_selection_hash,
            "source_frozen_plan_hash": source_plan_hash,
        }
    )
    write_strict_json(staging / "events.json", events)
    _write_manifest(staging, identity)


def run_digital_baseline(
    spec: DigitalBaselineSpec,
    *,
    repo_root: Path,
    force: bool = False,
) -> DigitalBaselineRun:
    """Run/reuse one digital baseline without entering physical StudyRunner."""
    physical = _physical_source_preflight(spec, repo_root=repo_root)
    dataset_spec = load_dataset_spec(spec.dataset_spec, repo_root=repo_root)
    dataset = ensure_dataset(dataset_spec, repo_root=repo_root, force=False)
    if (
        dataset.artifact_id != physical.completed.dataset.artifact_id
        or dataset.split_hash != physical.completed.dataset.split_hash
    ):
        raise ValueError(
            "digital and physical baselines must share the exact dataset and split"
        )
    if (
        dataset.validation_ids.numel() == 0
        or dataset.test_ids.numel() == 0
    ):
        raise ValueError("digital baseline requires validation and test samples")
    identity = _identity(spec, dataset=dataset, physical=physical)
    run_id = stable_object_hash(identity)
    final_dir = (
        spec.output_root / spec.name / f"{spec.profile}-{run_id[:12]}"
    )
    if final_dir.is_dir() and not force:
        manifest = verify_digital_baseline_run(final_dir)
        summary = json.loads(
            (final_dir / "run_summary.json").read_text(encoding="utf-8")
        )
        if manifest["identity"] != identity:
            raise ValueError("existing digital baseline identity mismatch")
        return DigitalBaselineRun(
            path=final_dir,
            run_id=run_id,
            summary=summary,
            reused=True,
        )

    torch.set_num_threads(int(spec.execution.torch_threads))
    transaction = RunTransaction(final_dir)
    staging = transaction.begin()
    try:
        _build_run(
            staging,
            spec,
            dataset=dataset,
            physical=physical,
            identity=identity,
            repo_root=repo_root,
        )
        transaction.publish(
            lambda root: verify_digital_baseline_run(root)
        )
    except BaseException:
        transaction.cleanup()
        raise
    summary = json.loads(
        (final_dir / "run_summary.json").read_text(encoding="utf-8")
    )
    return DigitalBaselineRun(
        path=final_dir,
        run_id=run_id,
        summary=summary,
        reused=False,
    )


def verify_digital_baseline_run(path: Path | str) -> dict[str, Any]:
    root = Path(path).resolve()
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"not a safe digital baseline run directory: {root}")
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("digital baseline run has no regular manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != DIGITAL_MANIFEST_SCHEMA:
        raise ValueError("unsupported digital baseline manifest")
    identity = manifest.get("identity")
    run_id = manifest.get("run_id")
    if (
        not isinstance(identity, dict)
        or not isinstance(run_id, str)
        or stable_object_hash(identity) != run_id
    ):
        raise ValueError("digital baseline identity hash mismatch")
    verify_execution_device_policy(
        identity,
        boundary="digital baseline identity",
    )
    records = manifest.get("files")
    if not isinstance(records, list):
        raise ValueError("digital baseline manifest files must be a list")
    if records != manifest_records(root, _RUN_FILES):
        raise ValueError("digital baseline bytes differ from manifest")
    exact_artifact_tree(root, [*_RUN_FILES, "manifest.json"])

    resolved = json.loads(
        (root / "resolved_spec.json").read_text(encoding="utf-8")
    )
    spec = DigitalBaselineSpec.model_validate(resolved)
    if semantic_digital_baseline_spec(spec) != identity.get(
        "digital_baseline"
    ):
        raise ValueError("digital resolved specification disagrees with identity")
    selection, selection_hash, plan, archive = _read_frozen_boundary(
        root,
        spec,
        domain_length=float(identity["dataset"]["domain_length"]),
    )
    dataset_reference = json.loads(
        (root / "dataset_reference.json").read_text(encoding="utf-8")
    )
    source_reference = json.loads(
        (root / "physical_source_reference.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        not isinstance(source_reference, dict)
        or not isinstance(source_reference.get("preflight"), dict)
        or not isinstance(source_reference.get("postfreeze"), dict)
    ):
        raise ValueError("digital physical source reference is invalid")
    summary = json.loads(
        (root / "run_summary.json").read_text(encoding="utf-8")
    )
    compute = json.loads(
        (root / "training_compute.json").read_text(encoding="utf-8")
    )
    verify_execution_device_policy(
        dataset_reference,
        boundary="digital dataset reference",
    )
    verify_execution_device_policy(
        summary,
        boundary="digital run summary",
    )
    verify_execution_device_policy(
        compute,
        boundary="digital training compute",
    )
    if (
        identity.get("dataset") != dataset_reference
        or source_reference.get("schema_version")
        != "pol-digital-physical-source-reference-v3"
        or identity.get("physical_source")
        != source_reference.get("preflight")
    ):
        raise ValueError("digital run references do not match identity")
    source_preflight_reference = source_reference["preflight"]
    if (
        plan.get("physical_source_run_hash")
        != source_preflight_reference.get("run_hash")
        or plan.get("physical_source_preflight_manifest_sha256")
        != source_preflight_reference.get("preflight_manifest_sha256")
        or plan.get("physical_source_selection_record_hash")
        != source_preflight_reference.get("selection_record_hash")
        or plan.get("physical_source_frozen_plan_hash")
        != source_preflight_reference.get("frozen_plan_hash")
    ):
        raise ValueError(
            "digital frozen plan disagrees with physical-source preflight"
        )
    if (
        summary.get("run_id") != run_id
        or summary.get("selection_record_hash") != selection_hash
        or summary.get("frozen_plan_hash") != plan.get("plan_content_hash")
        or summary.get("selected_candidate_id")
        != selection.get("selected_candidate_id")
        or summary.get("freeze_verified_before_test") is not True
        or summary.get("prediction_ensemble_separate") is not True
        or summary.get("coordinate_channel")
        != spec.model.coordinate_channel
        or summary.get("lifting_input_channels")
        != _lifting_input_channels(spec)
        or summary.get("domain_length")
        != float(dataset_reference["domain_length"])
        or summary.get(
            "physical_source_preflight_verified_before_training"
        )
        is not True
        or summary.get("physical_source_test_parsed_after_freeze") is not True
        or summary.get(
            "physical_source_fully_verified_before_fairness_publication"
        )
        is not True
    ):
        raise ValueError("digital run summary disagrees with frozen records")
    if summary.get("frozen_checkpoint_archive_sha256") != file_sha256(
        root / "frozen_checkpoints.pt"
    ):
        raise ValueError("digital summary checkpoint byte hash mismatch")
    if int(summary.get("evaluation_seed_count", -1)) != len(
        archive["models"]
    ):
        raise ValueError("digital evaluation seed count mismatch")
    digital_counts = fno_parameter_counts(archive)
    expected_compute_counts = {
        "coordinate_channel": spec.model.coordinate_channel,
        "lifting_input_channels": _lifting_input_channels(spec),
        "domain_length": float(dataset_reference["domain_length"]),
        "parameter_count_definition_version": (
            PARAMETER_COUNT_DEFINITION_VERSION
        ),
        "parameter_count_scope": PARAMETER_COUNT_SCOPE,
        "parameter_scalar_policy": PARAMETER_SCALAR_POLICY,
        "trainable_parameter_count": digital_counts[
            "trainable_parameter_count"
        ],
        "fixed_random_parameter_count": digital_counts[
            "fixed_random_parameter_count"
        ],
        "total_stored_parameter_count": digital_counts[
            "total_stored_parameter_count"
        ],
    }
    if any(
        compute.get(key) != value
        for key, value in expected_compute_counts.items()
    ):
        raise ValueError("digital training compute parameter-count mismatch")
    if (
        summary.get("parameter_count_scope") != PARAMETER_COUNT_SCOPE
        or summary.get("parameter_scalar_policy") != PARAMETER_SCALAR_POLICY
        or summary.get("parameter_count_definition_version")
        != PARAMETER_COUNT_DEFINITION_VERSION
        or summary.get("primary_model_parameter_counts") != digital_counts
        or summary.get("model_family_parameter_count_policy")
        != {
            "physical_feature_readout": (
                "frozen_tensor_shapes_cross_checked_against_readout_formula"
            ),
            "digital_neural_operator": (
                "frozen_state_dict_real_scalars_cross_checked_against_training"
            ),
            "physical_dynamics_included": False,
            "hardware_components_included": False,
        }
    ):
        raise ValueError("digital summary parameter-count policy mismatch")

    seed_rows = _read_csv(root / "test_seed_metrics.csv")
    primary_rows = _read_csv(root / "test_metrics.csv")
    ensemble_rows = _read_csv(root / "prediction_ensemble_metrics.csv")
    fairness_rows = _read_csv(root / "fairness_comparison.csv")
    if (
        len(seed_rows) != len(spec.training.evaluation_seeds)
        or len(primary_rows) != 1
        or len(ensemble_rows) != 1
        or len(fairness_rows) != len(spec.physical_comparison.rows) + 1
    ):
        raise ValueError("digital result table row count mismatch")
    for row in [*seed_rows, *primary_rows, *ensemble_rows]:
        if (
            row.get("coordinate_channel")
            != spec.model.coordinate_channel
            or int(row.get("lifting_input_channels", -1))
            != _lifting_input_channels(spec)
        ):
            raise ValueError(
                "digital result row coordinate architecture mismatch"
            )
    if primary_rows[0].get("test_result_kind") != (
        "independent_training_seed_metric_summary"
    ):
        raise ValueError("digital primary result is not independent-seed statistics")
    seed_metric_fields = [
        key
        for key in seed_rows[0]
        if key.startswith("test_") and key != "test_result_kind"
    ]
    recomputed = summarize_seed_metrics(
        [
            {key: _float(row, key) for key in seed_metric_fields}
            for row in seed_rows
        ]
    )
    for key, value in recomputed.items():
        if _float(primary_rows[0], key) != value:
            raise ValueError(
                "digital primary seed statistics disagree with per-seed rows"
            )
    if ensemble_rows[0].get("test_result_kind") != "prediction_ensemble":
        raise ValueError("digital ensemble result is not separately labeled")
    digital_fairness = [
        row for row in fairness_rows if row.get("row_id") == "fno1d"
    ]
    if (
        len(digital_fairness) != 1
        or digital_fairness[0].get("prediction_ensemble_in_primary")
        != "False"
    ):
        raise ValueError("digital fairness table conflates primary and ensemble")

    physical_preflight = _physical_source_preflight(
        spec,
        repo_root=Path.cwd().resolve(),
    )
    if physical_preflight.reference != source_reference.get("preflight"):
        raise ValueError(
            "digital physical source preflight no longer matches its reference"
        )
    physical_test_data = _physical_test_rows_after_freeze(
        physical_preflight
    )
    physical = _fully_verified_physical_source(
        physical_preflight,
        physical_test_data,
        repo_root=Path.cwd().resolve(),
        digital_plan_hash=str(plan["plan_content_hash"]),
    )
    if physical.reference != source_reference:
        raise ValueError(
            "digital physical source full verification no longer matches "
            "its reference"
        )
    expected_fairness = _fairness_rows(
        spec,
        dataset=physical.completed.dataset,
        physical=physical,
        selection=selection,
        primary=primary_rows[0],
        digital_counts=digital_counts,
        compute=compute,
        run_id=run_id,
    )
    expected_csv_rows = [
        {
            key: "" if value is None else str(value)
            for key, value in row.items()
        }
        for row in expected_fairness
    ]
    if list(fairness_rows) != expected_csv_rows:
        for index, (actual, expected) in enumerate(
            zip(fairness_rows, expected_csv_rows)
        ):
            if set(actual) != set(expected):
                raise ValueError(
                    "digital fairness table field set disagrees with "
                    f"verified sources at row={index}: "
                    f"extra={sorted(set(actual) - set(expected))}, "
                    f"missing={sorted(set(expected) - set(actual))}"
                )
            for key in expected:
                if actual.get(key) != expected.get(key):
                    raise ValueError(
                        "digital fairness table disagrees with verified "
                        f"sources at row={index}, field={key}"
                    )
        raise ValueError("digital fairness table disagrees with verified sources")

    events = json.loads((root / "events.json").read_text(encoding="utf-8"))
    if not isinstance(events, list):
        raise ValueError("digital event log must be a list")
    names = [
        event.get("event") for event in events
        if isinstance(event, Mapping)
    ]
    required = [
        "physical_source_preflight_verified",
        "selection_complete",
        "freeze_written",
        "freeze_read_back",
        "first_test_tensor_request",
        "physical_source_test_rows_parsed",
        "first_test_metric",
        "fairness_comparison_computed",
        "physical_source_full_verified",
        "fairness_comparison_written",
        "numerical_run_complete",
    ]
    if any(names.count(name) != 1 for name in required):
        raise ValueError("digital event log is incomplete")
    if not all(
        names.index(left) < names.index(right)
        for left, right in zip(required, required[1:])
    ):
        raise ValueError("digital freeze/test event order is invalid")
    by_name = {
        str(event["event"]): event
        for event in events
        if isinstance(event, Mapping) and "event" in event
    }
    source_preflight = source_reference["preflight"]
    source_postfreeze = source_reference["postfreeze"]
    for name in required:
        event = by_name[name]
        if (
            event.get("source_manifest_sha256")
            != source_preflight["preflight_manifest_sha256"]
            or event.get("source_selection_record_hash")
            != source_preflight["selection_record_hash"]
            or event.get("source_frozen_plan_hash")
            != source_preflight["frozen_plan_hash"]
        ):
            raise ValueError(
                "digital event has the wrong physical-source preflight binding"
            )
    for name in required[1:]:
        event = by_name[name]
        if event.get("selection_record_hash") != selection_hash:
            raise ValueError(
                "digital event has the wrong selection-record binding"
            )
    for name in required[2:]:
        event = by_name[name]
        if event.get("plan_content_hash") != plan["plan_content_hash"]:
            raise ValueError(
                "digital event has the wrong frozen-plan binding"
            )
    for name in required[5:]:
        event = by_name[name]
        if (
            event.get("source_test_metrics_sha256")
            != source_postfreeze["test_metrics_sha256"]
        ):
            raise ValueError(
                "digital event has the wrong physical test-table binding"
            )
    for name in (
        "physical_source_full_verified",
        "fairness_comparison_written",
    ):
        if by_name[name].get(
            "source_full_verification_evidence_hash"
        ) != source_postfreeze["full_verification_evidence_hash"]:
            raise ValueError(
                "digital event has the wrong full-source verification binding"
            )
    return manifest


__all__ = [
    "DigitalBaselineRun",
    "run_digital_baseline",
    "verify_digital_baseline_run",
]
