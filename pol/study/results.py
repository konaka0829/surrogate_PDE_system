from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from pol.runtime.artifacts import manifest_records
from pol.runtime.device import execution_device_policy
from pol.runtime.hashing import stable_object_hash
from pol.runtime.io import write_csv, write_strict_json


RESULT_ROW_SCHEMA_VERSION = "pol-study-result-row-v3"
SELECTION_SOURCE_RESULT_FIELDS = (
    "selected_condition_source_kind",
    "selected_condition_source_marker",
    "selected_condition_source_provenance_hash",
    "selected_condition_source_study_run_hash",
    "selected_condition_source_study_scientific_identity_hash",
    "selected_condition_source_selection_record_hash",
    "selected_condition_source_frozen_plan_hash",
    "selected_condition_source_frozen_model_archive_hash",
    "selected_condition_source_candidate_id",
    "selected_condition_source_feature_condition_hash",
    "selected_condition_source_feature_system_hash",
)


@dataclass(frozen=True)
class ReporterInputs:
    validation_rows: list[dict[str, Any]]
    test_rows: list[dict[str, Any]]
    random_seed_rows: list[dict[str, Any]]
    multiplier_rows: list[dict[str, Any]]
    noise_rows: list[dict[str, Any]]
    skipped_rows: list[dict[str, Any]]


@dataclass(frozen=True)
class BoundTestRows:
    primary_row: dict[str, Any]
    seed_rows: tuple[dict[str, Any], ...]
    ensemble_row: dict[str, Any] | None


def row_fields(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    fields: set[str] = set()
    for row in rows:
        fields.update(row)
    preferred = [
        "case_id",
        "variant_id",
        "candidate_id",
        "readout_id",
        "readout_kind",
        "result_row_schema_version",
        "selected",
        "test_result_kind",
        "seed",
        "test_seed_count",
        "ensemble_member_count",
        "feature_system",
        "feature_system_condition_hash",
        "feature_nu",
        "feature_time",
        "n_tar",
        "n_sur",
        "J",
        "q",
        "mode_index",
        "coefficient_index",
        "coefficient_kind",
        "diffusion_condition",
    ]
    return [field for field in preferred if field in fields] + sorted(
        fields - set(preferred)
    )


def selection_source_result_fields(
    provenance: Mapping[str, Any] | None,
    *,
    feature_family: str | None = None,
) -> dict[str, Any]:
    if provenance is None:
        if feature_family == "static_input":
            marker = {
                "kind": "explicit_static_input",
                "evolution": None,
            }
            return {
                "selected_condition_source_kind": "explicit_static_input",
                "selected_condition_source_marker": stable_object_hash(marker),
            }
        return {}
    imported = provenance.get("resolved_imported_feature_condition")
    if not isinstance(imported, Mapping):
        raise ValueError(
            "selection-source provenance has no imported feature condition"
        )
    system = imported.get("feature.evolution.system")
    if system is not None and not isinstance(system, Mapping):
        raise ValueError(
            "selection-source imported feature system is not an object"
        )
    fields = {
        "selected_condition_source_kind": "completed_study_selection",
        "selected_condition_source_marker": "",
        "selected_condition_source_provenance_hash": stable_object_hash(
            dict(provenance)
        ),
        "selected_condition_source_study_run_hash": provenance[
            "source_study_run_hash"
        ],
        "selected_condition_source_study_scientific_identity_hash": (
            provenance["source_study_scientific_identity_hash"]
        ),
        "selected_condition_source_selection_record_hash": provenance[
            "source_selection_record_hash"
        ],
        "selected_condition_source_frozen_plan_hash": provenance[
            "source_frozen_plan_hash"
        ],
        "selected_condition_source_frozen_model_archive_hash": provenance[
            "source_frozen_model_archive_hash"
        ],
        "selected_condition_source_candidate_id": provenance[
            "source_candidate_id"
        ],
        "selected_condition_source_feature_condition_hash": (
            stable_object_hash(dict(imported))
        ),
    }
    if system is not None:
        fields["selected_condition_source_feature_system_hash"] = (
            stable_object_hash(dict(system))
        )
    return fields


def validation_result_rows(
    *,
    case: Any,
    outcome: Any,
    selection_source_provenance: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidate_indices = {
        candidate_id: index
        for index, candidate_id in enumerate(outcome.candidate_order)
    }
    grid_indices = {
        cell["candidate_id"]: cell["cell_index"]
        for cell in outcome.grid_cells
        if cell["candidate_id"] is not None
    }
    for evaluation in outcome.evaluations:
        for readout_id, base_row in evaluation.rows.items():
            rows.append(
                {
                    "result_row_schema_version": RESULT_ROW_SCHEMA_VERSION,
                    "case_id": case.case_id,
                    "variant_id": case.variant_id,
                    "variant_display_name": case.variant_display_name,
                    "global_values": json.dumps(
                        case.global_values,
                        sort_keys=True,
                    ),
                    "search_stages": ";".join(
                        outcome.stages_by_candidate.get(
                            evaluation.candidate_id,
                            (),
                        )
                    ),
                    "search_kind": outcome.search_kind,
                    "search_candidate_index": candidate_indices[
                        evaluation.candidate_id
                    ],
                    "grid_cell_index": grid_indices.get(
                        evaluation.candidate_id
                    ),
                    "planned_cartesian_cell_count": (
                        outcome.planned_cartesian_cell_count
                    ),
                    "selected": (
                        outcome.selected_by_readout.get(readout_id)
                        == evaluation.candidate_id
                    ),
                    "selection_feature_cache_id": evaluation.feature_cache_id,
                    **selection_source_result_fields(
                        selection_source_provenance,
                        feature_family=base_row.get("feature_family"),
                    ),
                    **base_row,
                }
            )
    return rows


def dataset_reference_payload(dataset: Any) -> dict[str, Any]:
    return {
        "schema_version": "pol-study-dataset-reference-v3",
        **execution_device_policy(),
        "artifact_id": dataset.artifact_id,
        "split_hash": dataset.split_hash,
        "validation_artifact_id": dataset.validation_artifact_id,
        "dataset_binding_kind": dataset.binding_kind,
        "dataset_binding_status": dataset.binding_status,
        "dataset_target_reference_validation_status": (
            dataset.target_reference_validation_status
        ),
        "dataset_binding_proof_hash": dataset.binding_proof_hash,
        "binding_proof": dataset.binding_proof,
    }


def write_pre_freeze_results(
    staging: Path,
    *,
    resolved_study: Mapping[str, Any],
    dataset: Any,
    validation_rows: list[dict[str, Any]],
    convergence_rows: list[dict[str, Any]],
) -> None:
    write_strict_json(staging / "resolved_study.json", resolved_study)
    write_strict_json(
        staging / "dataset_reference.json",
        dataset_reference_payload(dataset),
    )
    write_csv(
        staging / "validation_trials.csv",
        validation_rows,
        fieldnames=row_fields(validation_rows),
    )
    write_csv(
        staging / "convergence.csv",
        convergence_rows,
        fieldnames=row_fields(convergence_rows),
    )


def write_skipped_trials(
    staging: Path,
    skipped: list[dict[str, Any]],
) -> None:
    write_strict_json(staging / "skipped_trials.json", skipped)


def bind_test_evaluation(
    entry: Mapping[str, Any],
    evaluated: Any,
    *,
    selection_hash: str,
    frozen_plan_hash: str,
    selection_source_provenance: Mapping[str, Any] | None = None,
) -> BoundTestRows:
    binding = {
        "result_row_schema_version": RESULT_ROW_SCHEMA_VERSION,
        "case_id": entry["case_id"],
        "variant_id": entry["variant_id"],
        "selected": True,
        "selection_record_hash": selection_hash,
        "frozen_plan_hash": frozen_plan_hash,
        **selection_source_result_fields(selection_source_provenance),
    }
    binding.update(
        selection_source_result_fields(
            selection_source_provenance,
            feature_family=evaluated.primary_row.get("feature_family"),
        )
    )
    return BoundTestRows(
        primary_row={**binding, **evaluated.primary_row},
        seed_rows=tuple(
            {**binding, **seed_row} for seed_row in evaluated.seed_rows
        ),
        ensemble_row=(
            None
            if evaluated.ensemble_row is None
            else {**binding, **evaluated.ensemble_row}
        ),
    )


def write_test_tables(
    staging: Path,
    *,
    test_rows: list[dict[str, Any]],
    random_seed_rows: list[dict[str, Any]],
    random_ensemble_rows: list[dict[str, Any]],
) -> None:
    write_csv(
        staging / "test_metrics.csv",
        test_rows,
        fieldnames=row_fields(test_rows),
    )
    write_csv(
        staging / "random_feature_seed_metrics.csv",
        random_seed_rows,
        fieldnames=row_fields(random_seed_rows),
    )
    write_csv(
        staging / "random_feature_ensemble_metrics.csv",
        random_ensemble_rows,
        fieldnames=row_fields(random_ensemble_rows),
    )


def build_selected_comparison_rows(
    *,
    validation_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join validation-selected rows to their primary frozen test results."""
    selected_validation = {
        _comparison_key(row): row
        for row in validation_rows
        if (
            row.get("selected") is True
            or str(row.get("selected")).lower() == "true"
        )
    }
    primary_test = {_comparison_key(row): row for row in test_rows}
    if set(selected_validation) != set(primary_test):
        raise ValueError(
            "selected validation and primary test bindings do not match"
        )
    rows: list[dict[str, Any]] = []
    for key in sorted(selected_validation):
        validation = selected_validation[key]
        test = primary_test[key]
        merged = dict(validation)
        for field, value in test.items():
            if field in merged and merged[field] != value:
                if field.startswith("validation_"):
                    continue
                raise ValueError(
                    f"comparison-table binding mismatch for {field}"
                )
            merged[field] = value
        rows.append(merged)
    return rows


def _comparison_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["case_id"]),
        str(row["readout_id"]),
        str(row["candidate_id"]),
    )


def write_selected_comparison_table(
    staging: Path,
    rows: list[dict[str, Any]],
) -> None:
    write_csv(
        staging / "selected_comparison.csv",
        rows,
        fieldnames=row_fields(rows),
    )


def write_diagnostic_tables(
    staging: Path,
    *,
    multiplier_rows: list[dict[str, Any]],
    multiplier_summary_rows: list[dict[str, Any]],
    stability_model_rows: list[dict[str, Any]],
    stability_repeat_rows: list[dict[str, Any]],
    stability_summary_rows: list[dict[str, Any]],
    stability_ensemble_repeat_rows: list[dict[str, Any]],
    stability_ensemble_summary_rows: list[dict[str, Any]],
) -> None:
    if multiplier_rows:
        write_csv(
            staging / "heat_multiplier.csv",
            multiplier_rows,
            fieldnames=row_fields(multiplier_rows),
        )
    if multiplier_summary_rows:
        write_csv(
            staging / "heat_multiplier_summary.csv",
            multiplier_summary_rows,
            fieldnames=row_fields(multiplier_summary_rows),
        )
    for filename, rows in (
        ("readout_stability_models.csv", stability_model_rows),
        ("readout_stability_noise_repeats.csv", stability_repeat_rows),
        ("readout_stability_noise_summary.csv", stability_summary_rows),
        (
            "readout_stability_noise_ensemble_repeats.csv",
            stability_ensemble_repeat_rows,
        ),
        (
            "readout_stability_noise_ensemble_summary.csv",
            stability_ensemble_summary_rows,
        ),
    ):
        if not rows:
            continue
        write_csv(
            staging / filename,
            rows,
            fieldnames=row_fields(rows),
        )


def build_run_summary(
    *,
    spec: Any,
    run_hash: str,
    dataset: Any,
    case_count: int,
    validation_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    random_seed_rows: list[dict[str, Any]],
    random_ensemble_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    multiplier_rows: list[dict[str, Any]],
    multiplier_summary_rows: list[dict[str, Any]],
    noise_rows: list[dict[str, Any]],
    stability_model_rows: list[dict[str, Any]],
    stability_repeat_rows: list[dict[str, Any]],
    stability_ensemble_repeat_rows: list[dict[str, Any]],
    stability_ensemble_summary_rows: list[dict[str, Any]],
    direct_diagnostic_count: int,
    direct_zero_fill_count: int,
    selection_hash: str,
    frozen_plan_hash: str,
    convergence_statuses: Mapping[str, str],
    cache_stats: Mapping[str, Any],
    skipped_trial_count: int,
    planned_cartesian_cell_count: int,
    evaluated_cartesian_cell_count: int,
    skipped_cartesian_cell_count: int,
    created_figures: list[str],
    selection_source_provenance: Mapping[str, Mapping[str, Any]],
    prediction_capture_entry_count: int,
    prediction_capture_content_hash: str | None,
) -> dict[str, Any]:
    planned_global_axis_combination_count = 1
    for axis in spec.global_axes:
        planned_global_axis_combination_count *= len(axis.values)
    planned_global_axis_case_count = (
        planned_global_axis_combination_count * len(spec.variants)
    )
    return {
        "schema_version": "pol-study-run-summary-v14",
        **execution_device_policy(),
        "status": "pass",
        "study": spec.name,
        "profile": spec.profile,
        "run_hash": run_hash,
        "dataset_artifact_id": dataset.artifact_id,
        "dataset_binding_kind": dataset.binding_kind,
        "dataset_binding_status": dataset.binding_status,
        "dataset_target_reference_validation_status": (
            dataset.target_reference_validation_status
        ),
        "dataset_binding_proof_hash": dataset.binding_proof_hash,
        "case_count": case_count,
        "planned_global_axis_combination_count": (
            planned_global_axis_combination_count
        ),
        "planned_global_axis_case_count": planned_global_axis_case_count,
        "evaluated_global_axis_case_count": case_count,
        "skipped_global_axis_case_count": (
            planned_global_axis_case_count - case_count
        ),
        "validation_row_count": len(validation_rows),
        "primary_test_row_count": len(test_rows),
        "random_feature_seed_row_count": len(random_seed_rows),
        "random_feature_ensemble_row_count": len(random_ensemble_rows),
        "selected_comparison_row_count": len(comparison_rows),
        "heat_multiplier_coefficient_row_count": len(multiplier_rows),
        "heat_multiplier_summary_row_count": len(multiplier_summary_rows),
        "readout_stability_model_row_count": len(stability_model_rows),
        "readout_stability_repeat_row_count": len(stability_repeat_rows),
        "readout_stability_summary_row_count": len(noise_rows),
        "readout_stability_ensemble_repeat_row_count": len(
            stability_ensemble_repeat_rows
        ),
        "readout_stability_ensemble_summary_row_count": len(
            stability_ensemble_summary_rows
        ),
        "prediction_capture_status": (
            "complete"
            if spec.prediction_capture is not None
            else "not_configured"
        ),
        "prediction_capture_file": (
            "prediction_capture.pt"
            if spec.prediction_capture is not None
            else None
        ),
        "prediction_capture_entry_count": prediction_capture_entry_count,
        "prediction_capture_content_hash": prediction_capture_content_hash,
        "prediction_capture_spectrum_storage": (
            "predeclared_samples_plus_all_test_per_coefficient_aggregates"
            if spec.prediction_capture is not None
            else None
        ),
        "direct_decoder_diagnostic_count": direct_diagnostic_count,
        "direct_decoder_zero_fill_count": direct_zero_fill_count,
        "direct_decoder_zero_fill_applied": direct_zero_fill_count > 0,
        "selection_record_hash": selection_hash,
        "frozen_plan_hash": frozen_plan_hash,
        "selection_source_binding_count": len(
            selection_source_provenance
        ),
        "selection_source_provenance_hash": stable_object_hash(
            {
                key: dict(value)
                for key, value in selection_source_provenance.items()
            }
        ),
        "convergence": dict(convergence_statuses),
        "cache": dict(cache_stats),
        "skipped_trial_count": skipped_trial_count,
        "planned_cartesian_cell_count": planned_cartesian_cell_count,
        "evaluated_cartesian_cell_count": evaluated_cartesian_cell_count,
        "skipped_cartesian_cell_count": skipped_cartesian_cell_count,
        "figures": created_figures,
        "numerical_publication_status": "complete_verified_before_reporting",
        "report_status": (
            "numerical_complete_report_not_generated"
            if spec.execution.generate_plots
            else "not_requested"
        ),
        "report_source": None,
    }


def write_run_manifest(
    root: Path,
    *,
    identity: Mapping[str, Any],
    schema_version: str = "pol-study-run-manifest-v14",
) -> None:
    names = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    write_strict_json(
        root / "manifest.json",
        {
            "schema_version": schema_version,
            "identity": dict(identity),
            "files": manifest_records(root, names),
        },
    )


def write_completion_records(
    staging: Path,
    *,
    events: list[dict[str, Any]],
    summary: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> None:
    write_strict_json(staging / "events.json", events)
    write_strict_json(staging / "run_summary.json", summary)
    write_run_manifest(staging, identity=identity)


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_reporter_inputs(root: Path) -> ReporterInputs:
    skipped_path = root / "skipped_trials.json"
    skipped_rows: list[dict[str, Any]] = []
    if skipped_path.is_file():
        raw_skipped = json.loads(skipped_path.read_text(encoding="utf-8"))
        if not isinstance(raw_skipped, list) or not all(
            isinstance(item, dict) for item in raw_skipped
        ):
            raise ValueError("skipped-trial reporter input must be a list")
        skipped_rows = [dict(item) for item in raw_skipped]
    return ReporterInputs(
        validation_rows=load_rows(root / "validation_trials.csv"),
        test_rows=load_rows(root / "test_metrics.csv"),
        random_seed_rows=load_rows(root / "random_feature_seed_metrics.csv"),
        multiplier_rows=load_rows(root / "heat_multiplier.csv"),
        noise_rows=load_rows(root / "readout_stability_noise_summary.csv"),
        skipped_rows=skipped_rows,
    )
