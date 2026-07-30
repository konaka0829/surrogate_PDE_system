from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import torch

from pol.config.models import StudySpec, TrialSpec
from pol.learning.direct import (
    DIRECT_DECODER_DIAGNOSTIC_FIELDS,
    has_fixed_fourier_decoder_diagnostic,
    verify_fixed_fourier_decoder_diagnostic,
)
from pol.runtime.device import (
    execution_device_policy,
    require_cpu_tensors,
    verify_execution_device_policy,
)
from pol.runtime.hashing import stable_object_hash, tensor_sha256
from pol.runtime.io import atomic_torch_save, file_sha256, write_strict_json
from .cases import StudyCase
from .evaluation import CandidateEvaluation, trial_parameters

if TYPE_CHECKING:
    from .search import SearchOutcome


@dataclass(frozen=True)
class FreezePreparation:
    selection_cases: dict[str, Any]
    selection_record: dict[str, Any]
    selection_hash: str
    frozen_archive: dict[str, Any]


@dataclass(frozen=True)
class PersistedFreeze:
    selection_hash: str
    frozen_plan_hash: str
    selection: dict[str, Any]
    plan: dict[str, Any]
    archive: dict[str, Any]
    direct_diagnostic_count: int
    direct_zero_fill_count: int
    events: tuple[dict[str, Any], ...]


def test_evaluation_contract() -> dict[str, Any]:
    return {
        "schema_version": "pol-test-evaluation-contract-v3",
        "random_feature_primary": "independent_seed_metric_summary",
        "random_feature_seed_result": "independent_seed_realization",
        "random_feature_ensemble_result": "prediction_ensemble",
        "seed_standard_deviation_ddof": 1,
        "confidence_level": 0.95,
        "confidence_interval_method": "student_t",
        "descriptive_quantiles": [0.25, 0.5, 0.75],
        "quantile_method": "linear",
        "quantiles_are_uncertainty_interval": False,
        "evaluation_seed_validation_used_for_selection": False,
        "random_map_identity": "content_hash_of_seed_A_c_and_map_contract",
        "frozen_member_identity": "content_hash_of_map_and_fitted_readout",
        "prediction_ensemble_member_binding": (
            "ordered_seed_and_frozen_member_hashes"
        ),
        "training_subset_policy": "canonical_train_order_prefix_v1",
        "training_subset_selection_boundary": (
            "all_subset_models_frozen_before_any_test_access"
        ),
    }


def assert_selection_record_safe(record: Mapping[str, Any]) -> None:
    forbidden_exact = {
        "test_ids",
        "test_targets",
        "test_target_hash",
        "test_metrics",
        "targets_reference",
        "full_target_hash",
    }

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                normalized = str(key).lower()
                if (
                    normalized in forbidden_exact
                    or normalized.startswith("test_metric")
                ):
                    raise ValueError(
                        f"selection record contains test binding at {path}.{key}"
                    )
                visit(item, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")

    visit(record, "$")


def verify_selection_source_provenance_bindings(
    *,
    selection: Mapping[str, Any],
    plan: Mapping[str, Any],
    archive: Mapping[str, Any],
) -> None:
    provenance = selection.get("selection_source_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("selection-source provenance must be an object")
    if plan.get("selection_source_provenance") != provenance:
        raise ValueError(
            "frozen plan selection-source provenance mismatch"
        )
    if archive.get("selection_source_provenance") != provenance:
        raise ValueError(
            "frozen model archive selection-source provenance mismatch"
        )


def _model_key(case_id: str, candidate_id: str, readout_id: str) -> str:
    return f"{case_id}/{candidate_id}/{readout_id}"


def _decoder_diagnostic_fields(values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field: values[field]
        for field in DIRECT_DECODER_DIAGNOSTIC_FIELDS
        if field in values and values.get(field) not in ("", None)
    }


def verify_no_decoder_diagnostic(
    values: Mapping[str, Any],
    *,
    boundary: str,
) -> None:
    if has_fixed_fourier_decoder_diagnostic(values):
        raise ValueError(f"{boundary} has false direct-decoder diagnostic fields")


def verify_frozen_decoder_bindings(
    models: Mapping[str, Any],
    *,
    selection: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> tuple[int, int]:
    """Bind validation-time diagnostics to frozen direct models before test use."""
    selection_cases = selection.get("cases")
    plan_cases = plan.get("cases")
    if not isinstance(selection_cases, Mapping) or not isinstance(
        plan_cases, Mapping
    ):
        raise ValueError("decoder binding requires selection and frozen-plan cases")
    direct_count = 0
    zero_fill_count = 0
    for entry in models.values():
        if not isinstance(entry, Mapping):
            raise ValueError("frozen model entry is not an object")
        try:
            case_id = str(entry["case_id"])
            readout_id = str(entry["readout_id"])
            trial = TrialSpec.model_validate(entry["trial"])
            model = entry["model"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "frozen model entry cannot bind decoder diagnostics"
            ) from exc
        if not isinstance(model, Mapping):
            raise ValueError("frozen model payload is not an object")
        selection_case = selection_cases.get(case_id)
        plan_case = plan_cases.get(case_id)
        if not isinstance(selection_case, Mapping) or not isinstance(
            plan_case, Mapping
        ):
            raise ValueError("frozen decoder binding references an unknown case")
        inner_by_readout = selection_case.get("inner_selections")
        plan_by_readout = plan_case.get("decoder_diagnostics_by_readout")
        if not isinstance(inner_by_readout, Mapping) or not isinstance(
            plan_by_readout, Mapping
        ):
            raise ValueError("decoder binding records are missing")
        inner = inner_by_readout.get(readout_id)
        if not isinstance(inner, Mapping):
            raise ValueError("selection inner record is missing for frozen readout")
        J = int(trial.feature.observation.J)
        q = int(trial.output.q)
        if model.get("kind") == "direct_fourier_decoder":
            direct_count += 1
            model_diagnostic = verify_fixed_fourier_decoder_diagnostic(
                model,
                observation_count=J,
                requested_q=q,
                boundary="frozen direct model",
            )
            verify_fixed_fourier_decoder_diagnostic(
                inner,
                observation_count=J,
                requested_q=q,
                boundary="direct readout inner selection",
            )
            plan_diagnostic = plan_by_readout.get(readout_id)
            if not isinstance(plan_diagnostic, Mapping):
                raise ValueError(
                    "frozen plan is missing direct-decoder diagnostics"
                )
            verify_fixed_fourier_decoder_diagnostic(
                plan_diagnostic,
                observation_count=J,
                requested_q=q,
                boundary="frozen plan direct readout",
            )
            if model_diagnostic.zero_fill_applied:
                zero_fill_count += 1
        else:
            verify_no_decoder_diagnostic(
                model,
                boundary="non-direct frozen model",
            )
            verify_no_decoder_diagnostic(
                inner,
                boundary="non-direct inner selection",
            )
            if readout_id in plan_by_readout:
                raise ValueError(
                    "frozen plan has false direct-decoder diagnostics for "
                    "a non-direct readout"
                )
    return direct_count, zero_fill_count


def verify_representative_feature_bindings(
    models: Mapping[str, Any],
    *,
    selection: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> None:
    selection_cases = selection.get("cases")
    plan_cases = plan.get("cases")
    if not isinstance(selection_cases, Mapping) or not isinstance(
        plan_cases,
        Mapping,
    ):
        raise ValueError("representative feature binding requires cases")
    entries = {
        (
            entry.get("case_id"),
            entry.get("readout_id"),
            entry.get("candidate_id"),
        ): entry
        for entry in models.values()
        if isinstance(entry, Mapping)
    }
    for case_id, selection_case in selection_cases.items():
        if not isinstance(selection_case, Mapping):
            raise ValueError("representative selection case is invalid")
        readout_id = selection_case.get("representative_readout")
        candidate_id = selection_case.get("representative_candidate_id")
        entry = entries.get((case_id, readout_id, candidate_id))
        condition = selection_case.get("representative_feature_condition")
        if not isinstance(entry, Mapping) or not isinstance(condition, Mapping):
            raise ValueError("representative frozen model binding is missing")
        trial = TrialSpec.model_validate(entry["trial"])
        expected = {
            "selection_split": "validation",
            "selection_metric": selection["selection_metric"],
            "selection_metric_value": selection_case["validation_metrics"][
                readout_id
            ],
            "representative_readout": readout_id,
            "candidate_id": candidate_id,
            "finite_input": trial.input.model_dump(mode="json"),
            "feature": trial.feature.model_dump(mode="json"),
            "output": trial.output.model_dump(mode="json"),
            "row_parameters": trial_parameters(trial),
        }
        if "training_subset" in entry:
            expected["training_subset"] = entry["training_subset"]
        if condition != expected:
            raise ValueError(
                "representative feature condition differs from frozen trial"
            )
        plan_case = plan_cases.get(case_id)
        if (
            not isinstance(plan_case, Mapping)
            or plan_case.get("representative_feature_condition") != condition
        ):
            raise ValueError(
                "frozen plan representative feature binding mismatch"
            )


def build_selection_cases(
    *,
    spec: StudySpec,
    cases: list[StudyCase],
    outcomes: Mapping[str, SearchOutcome],
    evaluations: Mapping[tuple[str, str], CandidateEvaluation],
) -> dict[str, Any]:
    selection_cases: dict[str, Any] = {}
    for case in cases:
        outcome = outcomes[case.case_id]
        representative_id = outcome.selected_by_readout[
            spec.selection.representative_readout
        ]
        representative_evaluation = evaluations[
            (case.case_id, representative_id)
        ]
        if any(
            candidate_id not in outcome.candidate_order
            for candidate_id in outcome.selected_by_readout.values()
        ):
            raise ValueError("selected candidate is absent from candidate order")
        if outcome.search_kind == "grid":
            if outcome.planned_cartesian_cell_count != len(
                outcome.grid_cells
            ):
                raise ValueError(
                    "grid search did not preserve every declared Cartesian cell"
                )
            if (
                len(outcome.evaluations) + len(outcome.skipped)
                != outcome.planned_cartesian_cell_count
            ):
                raise ValueError(
                    "evaluated and skipped grid cells do not cover the plan"
                )
        representative_condition = {
            "selection_split": "validation",
            "selection_metric": spec.selection.metric,
            "selection_metric_value": representative_evaluation.rows[
                spec.selection.representative_readout
            ][spec.selection.metric],
            "representative_readout": spec.selection.representative_readout,
            "candidate_id": representative_id,
            "finite_input": representative_evaluation.trial.input.model_dump(
                mode="json"
            ),
            "feature": representative_evaluation.trial.feature.model_dump(
                mode="json"
            ),
            "output": representative_evaluation.trial.output.model_dump(
                mode="json"
            ),
            "row_parameters": trial_parameters(
                representative_evaluation.trial
            ),
            "training_subset": dict(
                representative_evaluation.training_subset
            ),
        }
        selection_cases[case.case_id] = {
            "variant_id": case.variant_id,
            "global_values": case.global_values,
            "search_kind": outcome.search_kind,
            "declared_candidate_count": outcome.declared_candidate_count,
            "planned_cartesian_cell_count": (
                outcome.planned_cartesian_cell_count
            ),
            "evaluated_candidate_count": len(outcome.evaluations),
            "skipped_candidate_count": len(outcome.skipped),
            "candidate_order": list(outcome.candidate_order),
            "selection_order_by_readout": {
                readout_id: list(candidate_ids)
                for readout_id, candidate_ids in (
                    outcome.selection_order_by_readout.items()
                )
            },
            "grid_cells": list(outcome.grid_cells),
            "skipped_candidates": list(outcome.skipped),
            "selected_by_readout": outcome.selected_by_readout,
            "representative_readout": spec.selection.representative_readout,
            "representative_candidate_id": representative_id,
            "representative_feature_condition": representative_condition,
            "inner_selections": {
                readout_id: evaluations[(case.case_id, candidate_id)]
                .inner_selections[readout_id]
                for readout_id, candidate_id in outcome.selected_by_readout.items()
            },
            "validation_metrics": {
                readout_id: evaluations[(case.case_id, candidate_id)].rows[
                    readout_id
                ][spec.selection.metric]
                for readout_id, candidate_id in outcome.selected_by_readout.items()
            },
            "training_subsets_by_readout": {
                readout_id: dict(
                    evaluations[
                        (case.case_id, candidate_id)
                    ].training_subset
                )
                for readout_id, candidate_id in outcome.selected_by_readout.items()
            },
        }
    return selection_cases


def prepare_freeze(
    *,
    spec: StudySpec,
    dataset: Any,
    cases: list[StudyCase],
    outcomes: Mapping[str, SearchOutcome],
    evaluations: Mapping[tuple[str, str], CandidateEvaluation],
    convergence_statuses: Mapping[str, str],
    selection_source_provenance: Mapping[str, Mapping[str, Any]],
) -> FreezePreparation:
    selection_cases = build_selection_cases(
        spec=spec,
        cases=cases,
        outcomes=outcomes,
        evaluations=evaluations,
    )
    selection_record = {
        "schema_version": "pol-selection-record-v8",
        **execution_device_policy(),
        "study": spec.name,
        "profile": spec.profile,
        "dataset_artifact_id": dataset.artifact_id,
        "dataset_binding_kind": dataset.binding_kind,
        "dataset_binding_status": dataset.binding_status,
        "dataset_target_reference_validation_status": (
            dataset.target_reference_validation_status
        ),
        "dataset_binding_proof_hash": dataset.binding_proof_hash,
        "split_binding": {
            "split_hash": dataset.split_hash,
            "train_ids_hash": tensor_sha256(dataset.train_ids),
            "validation_ids_hash": tensor_sha256(dataset.validation_ids),
        },
        "selection_metric": spec.selection.metric,
        "selection_source_provenance": {
            key: dict(value)
            for key, value in selection_source_provenance.items()
        },
        "cases": selection_cases,
        "convergence": dict(convergence_statuses),
        "test_data_used": False,
    }
    assert_selection_record_safe(selection_record)
    selection_hash = stable_object_hash(selection_record)

    frozen_models: dict[str, Any] = {}
    for case in cases:
        outcome = outcomes[case.case_id]
        for readout_id, candidate_id in outcome.selected_by_readout.items():
            evaluation = evaluations[(case.case_id, candidate_id)]
            key = _model_key(case.case_id, candidate_id, readout_id)
            frozen_models[key] = {
                "case_id": case.case_id,
                "variant_id": case.variant_id,
                "candidate_id": candidate_id,
                "readout_id": readout_id,
                "trial": evaluation.trial.model_dump(mode="python"),
                "training_subset": dict(evaluation.training_subset),
                "model": evaluation.frozen_models[readout_id],
            }
    frozen_archive = {
        "schema_version": "pol-frozen-model-archive-v9",
        **execution_device_policy(),
        "selection_record_hash": selection_hash,
        "selection_source_provenance": {
            key: dict(value)
            for key, value in selection_source_provenance.items()
        },
        "models": frozen_models,
    }
    require_cpu_tensors(
        frozen_archive,
        boundary="frozen model archive publication",
        name="archive",
    )
    return FreezePreparation(
        selection_cases=selection_cases,
        selection_record=selection_record,
        selection_hash=selection_hash,
        frozen_archive=frozen_archive,
    )


def persist_and_read_back_freeze(
    staging: Path,
    *,
    preparation: FreezePreparation,
    spec: StudySpec,
    dataset: Any,
    convergence_statuses: Mapping[str, str],
    selection_source_provenance: Mapping[str, Mapping[str, Any]],
) -> PersistedFreeze:
    selection_hash = preparation.selection_hash
    write_strict_json(
        staging / "selection_record.json",
        preparation.selection_record,
    )
    events = [
        {
            "event": "selection_complete",
            "selection_record_hash": selection_hash,
        },
        {
            "event": "convergence_complete",
            "status": dict(convergence_statuses),
        },
    ]
    atomic_torch_save(staging / "frozen_models.pt", preparation.frozen_archive)
    model_file_hash = file_sha256(staging / "frozen_models.pt")
    frozen_plan = {
        "schema_version": "pol-frozen-evaluation-plan-v9",
        **execution_device_policy(),
        "study": spec.name,
        "dataset_artifact_id": dataset.artifact_id,
        "dataset_split_hash": dataset.split_hash,
        "dataset_binding_kind": dataset.binding_kind,
        "dataset_binding_status": dataset.binding_status,
        "dataset_target_reference_validation_status": (
            dataset.target_reference_validation_status
        ),
        "dataset_binding_proof_hash": dataset.binding_proof_hash,
        "selection_record_hash": selection_hash,
        "selection_source_provenance": {
            key: dict(value)
            for key, value in selection_source_provenance.items()
        },
        "frozen_models_file": "frozen_models.pt",
        "frozen_models_sha256": model_file_hash,
        "cases": {
            case_id: {
                "selected_by_readout": value["selected_by_readout"],
                "search_kind": value["search_kind"],
                "declared_candidate_count": value[
                    "declared_candidate_count"
                ],
                "planned_cartesian_cell_count": value[
                    "planned_cartesian_cell_count"
                ],
                "evaluated_candidate_count": value[
                    "evaluated_candidate_count"
                ],
                "skipped_candidate_count": value[
                    "skipped_candidate_count"
                ],
                "candidate_order": value["candidate_order"],
                "selection_order_by_readout": value[
                    "selection_order_by_readout"
                ],
                "representative_readout": value[
                    "representative_readout"
                ],
                "representative_candidate_id": value[
                    "representative_candidate_id"
                ],
                "representative_feature_condition": value[
                    "representative_feature_condition"
                ],
                "decoder_diagnostics_by_readout": {
                    readout_id: _decoder_diagnostic_fields(inner)
                    for readout_id, inner in value["inner_selections"].items()
                    if has_fixed_fourier_decoder_diagnostic(inner)
                },
                "training_subsets_by_readout": value[
                    "training_subsets_by_readout"
                ],
            }
            for case_id, value in preparation.selection_cases.items()
        },
        "test_evaluation_contract": test_evaluation_contract(),
        "test_data_used": False,
    }
    frozen_plan_hash = stable_object_hash(frozen_plan)
    frozen_plan["plan_content_hash"] = frozen_plan_hash
    write_strict_json(staging / "frozen_evaluation_plan.json", frozen_plan)
    events.append(
        {
            "event": "freeze_written",
            "plan_content_hash": frozen_plan_hash,
        }
    )

    loaded_selection = json.loads(
        (staging / "selection_record.json").read_text(encoding="utf-8")
    )
    if stable_object_hash(loaded_selection) != selection_hash:
        raise ValueError("selection record read-back hash mismatch")
    loaded_plan = json.loads(
        (staging / "frozen_evaluation_plan.json").read_text(encoding="utf-8")
    )
    stored_plan_hash = loaded_plan.pop("plan_content_hash", None)
    recomputed_plan_hash = stable_object_hash(loaded_plan)
    loaded_plan["plan_content_hash"] = stored_plan_hash
    if (
        stored_plan_hash != frozen_plan_hash
        or recomputed_plan_hash != frozen_plan_hash
    ):
        raise ValueError("frozen plan read-back hash mismatch")
    if loaded_plan.get("dataset_artifact_id") != dataset.artifact_id:
        raise ValueError("frozen plan dataset binding mismatch")
    verify_execution_device_policy(
        loaded_plan,
        boundary="frozen evaluation plan read-back",
    )
    if loaded_plan.get("dataset_split_hash") != dataset.split_hash:
        raise ValueError("frozen plan split binding mismatch")
    if loaded_plan.get("dataset_binding_kind") != dataset.binding_kind:
        raise ValueError("frozen plan dataset binding-kind mismatch")
    if loaded_plan.get("dataset_binding_status") != dataset.binding_status:
        raise ValueError("frozen plan dataset binding-status mismatch")
    if (
        loaded_plan.get("dataset_target_reference_validation_status")
        != dataset.target_reference_validation_status
    ):
        raise ValueError("frozen plan dataset target-validation status mismatch")
    if (
        loaded_plan.get("dataset_binding_proof_hash")
        != dataset.binding_proof_hash
    ):
        raise ValueError("frozen plan dataset binding-proof hash mismatch")
    if loaded_plan.get("selection_record_hash") != selection_hash:
        raise ValueError("frozen plan selection binding mismatch")
    loaded_selection_cases = loaded_selection.get("cases")
    loaded_plan_cases = loaded_plan.get("cases")
    if not isinstance(loaded_selection_cases, Mapping) or not isinstance(
        loaded_plan_cases,
        Mapping,
    ):
        raise ValueError("selection/frozen plan cases are missing")
    binding_fields = (
        "selected_by_readout",
        "search_kind",
        "declared_candidate_count",
        "planned_cartesian_cell_count",
        "evaluated_candidate_count",
        "skipped_candidate_count",
        "candidate_order",
        "selection_order_by_readout",
        "representative_readout",
        "representative_candidate_id",
        "representative_feature_condition",
        "training_subsets_by_readout",
    )
    if set(loaded_selection_cases) != set(loaded_plan_cases):
        raise ValueError("selection/frozen plan case sets differ")
    for case_id, selection_case in loaded_selection_cases.items():
        plan_case = loaded_plan_cases[case_id]
        if any(
            selection_case.get(field) != plan_case.get(field)
            for field in binding_fields
        ):
            raise ValueError(
                "frozen plan representative/search binding mismatch"
            )
    if loaded_plan.get("test_evaluation_contract") != test_evaluation_contract():
        raise ValueError("frozen plan test-evaluation contract mismatch")
    if (
        file_sha256(staging / loaded_plan["frozen_models_file"])
        != loaded_plan["frozen_models_sha256"]
    ):
        raise ValueError("frozen model archive read-back hash mismatch")
    loaded_archive = torch.load(
        staging / loaded_plan["frozen_models_file"],
        map_location="cpu",
        weights_only=True,
    )
    if loaded_archive.get("schema_version") != "pol-frozen-model-archive-v9":
        raise ValueError("unsupported frozen model archive schema")
    verify_execution_device_policy(
        loaded_archive,
        boundary="frozen model archive read-back",
    )
    require_cpu_tensors(
        loaded_archive,
        boundary="frozen model archive read-back",
        name="archive",
    )
    if loaded_archive.get("selection_record_hash") != selection_hash:
        raise ValueError("frozen model archive selection binding mismatch")
    verify_selection_source_provenance_bindings(
        selection=loaded_selection,
        plan=loaded_plan,
        archive=loaded_archive,
    )
    verify_representative_feature_bindings(
        loaded_archive["models"],
        selection=loaded_selection,
        plan=loaded_plan,
    )
    direct_diagnostic_count, direct_zero_fill_count = (
        verify_frozen_decoder_bindings(
            loaded_archive["models"],
            selection=loaded_selection,
            plan=loaded_plan,
        )
    )
    events.append(
        {
            "event": "freeze_read_back",
            "plan_content_hash": frozen_plan_hash,
        }
    )
    return PersistedFreeze(
        selection_hash=selection_hash,
        frozen_plan_hash=frozen_plan_hash,
        selection=loaded_selection,
        plan=loaded_plan,
        archive=loaded_archive,
        direct_diagnostic_count=direct_diagnostic_count,
        direct_zero_fill_count=direct_zero_fill_count,
        events=tuple(events),
    )
