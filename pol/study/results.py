from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from pol.runtime.artifacts import manifest_records
from pol.runtime.device import execution_device_policy
from pol.runtime.io import write_csv, write_strict_json


@dataclass(frozen=True)
class ReporterInputs:
    validation_rows: list[dict[str, Any]]
    test_rows: list[dict[str, Any]]
    noise_rows: list[dict[str, Any]]


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
        "selected",
        "test_result_kind",
        "seed",
        "test_seed_count",
        "ensemble_member_count",
        "feature_system",
        "feature_nu",
        "feature_time",
        "n_tar",
        "n_sur",
        "J",
        "q",
    ]
    return [field for field in preferred if field in fields] + sorted(
        fields - set(preferred)
    )


def validation_result_rows(
    *,
    case: Any,
    outcome: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for evaluation in outcome.evaluations:
        for readout_id, base_row in evaluation.rows.items():
            rows.append(
                {
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
                    "selected": (
                        outcome.selected_by_readout.get(readout_id)
                        == evaluation.candidate_id
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
) -> BoundTestRows:
    binding = {
        "case_id": entry["case_id"],
        "variant_id": entry["variant_id"],
        "selected": True,
        "selection_record_hash": selection_hash,
        "frozen_plan_hash": frozen_plan_hash,
    }
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


def write_diagnostic_tables(
    staging: Path,
    *,
    multiplier_rows: list[dict[str, Any]],
    noise_rows: list[dict[str, Any]],
) -> None:
    write_csv(
        staging / "heat_multiplier.csv",
        multiplier_rows,
        fieldnames=row_fields(multiplier_rows),
    )
    write_csv(
        staging / "noise_robustness.csv",
        noise_rows,
        fieldnames=row_fields(noise_rows),
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
    direct_diagnostic_count: int,
    direct_zero_fill_count: int,
    selection_hash: str,
    frozen_plan_hash: str,
    convergence_statuses: Mapping[str, str],
    cache_stats: Mapping[str, Any],
    skipped_trial_count: int,
    created_figures: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": "pol-study-run-summary-v5",
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
        "validation_row_count": len(validation_rows),
        "primary_test_row_count": len(test_rows),
        "random_feature_seed_row_count": len(random_seed_rows),
        "random_feature_ensemble_row_count": len(random_ensemble_rows),
        "direct_decoder_diagnostic_count": direct_diagnostic_count,
        "direct_decoder_zero_fill_count": direct_zero_fill_count,
        "direct_decoder_zero_fill_applied": direct_zero_fill_count > 0,
        "selection_record_hash": selection_hash,
        "frozen_plan_hash": frozen_plan_hash,
        "convergence": dict(convergence_statuses),
        "cache": dict(cache_stats),
        "skipped_trial_count": skipped_trial_count,
        "figures": created_figures,
    }


def write_run_manifest(
    root: Path,
    *,
    identity: Mapping[str, Any],
) -> None:
    names = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    write_strict_json(
        root / "manifest.json",
        {
            "schema_version": "pol-study-run-manifest-v5",
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
    return ReporterInputs(
        validation_rows=load_rows(root / "validation_trials.csv"),
        test_rows=load_rows(root / "test_metrics.csv"),
        noise_rows=load_rows(root / "noise_robustness.csv"),
    )
