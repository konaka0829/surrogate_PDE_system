from __future__ import annotations

from importlib import import_module
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from pol.config.models import TrialSpec
from pol.learning.direct import verify_fixed_fourier_decoder_diagnostic
from pol.runtime.artifacts import manifest_records
from pol.runtime.device import require_cpu_tensors, verify_execution_device_policy
from pol.runtime.hashing import stable_object_hash
from pol.runtime.io import file_sha256
from .evaluation import summarize_independent_seed_metrics
from .protocol import (
    assert_selection_record_safe,
    test_evaluation_contract,
    verify_frozen_decoder_bindings,
    verify_no_decoder_diagnostic,
)
from .results import load_rows


def _row_binding(row: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    return (
        row.get("case_id"),
        row.get("readout_id"),
        row.get("candidate_id"),
    )


def _has_csv_value(row: Mapping[str, Any], key: str) -> bool:
    return key in row and row.get(key) not in ("", None)


def _require_close(actual: Any, expected: float, *, label: str) -> None:
    try:
        value = float(actual)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric") from exc
    if not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError(f"{label} does not match the per-seed metrics")


def _verify_study_semantics(root: Path, manifest: Mapping[str, Any]) -> None:
    identity = manifest.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("study-run identity must be an object")
    if identity.get("schema_version") != "pol-study-run-identity-v5":
        raise ValueError("unsupported legacy study-run identity")
    verify_execution_device_policy(
        identity,
        boundary="study-run identity",
    )
    environment = identity.get("environment")
    if not isinstance(environment, Mapping):
        raise ValueError("study-run numerical environment is missing")
    verify_execution_device_policy(
        environment,
        boundary="study-run numerical environment",
    )
    run_hash = stable_object_hash(dict(identity))
    resolved_study = json.loads(
        (root / "resolved_study.json").read_text(encoding="utf-8")
    )
    if resolved_study != identity.get("study"):
        raise ValueError("resolved study does not match manifest identity")

    summary = json.loads((root / "run_summary.json").read_text(encoding="utf-8"))
    if summary.get("schema_version") != "pol-study-run-summary-v5":
        raise ValueError("unsupported study-run summary schema")
    verify_execution_device_policy(
        summary,
        boundary="study-run summary",
    )
    if summary.get("run_hash") != run_hash:
        raise ValueError("study-run summary hash does not match manifest identity")
    if summary.get("study") != resolved_study.get("name"):
        raise ValueError("study-run summary name mismatch")
    if summary.get("profile") != resolved_study.get("profile"):
        raise ValueError("study-run summary profile mismatch")

    selection = json.loads(
        (root / "selection_record.json").read_text(encoding="utf-8")
    )
    if selection.get("schema_version") != "pol-selection-record-v5":
        raise ValueError("unsupported selection-record schema")
    verify_execution_device_policy(
        selection,
        boundary="selection record",
    )
    assert_selection_record_safe(selection)
    if selection.get("test_data_used") is not False:
        raise ValueError("selection record is not test-isolated")
    selection_hash = stable_object_hash(selection)
    if summary.get("selection_record_hash") != selection_hash:
        raise ValueError("selection-record hash mismatch")

    plan = json.loads(
        (root / "frozen_evaluation_plan.json").read_text(encoding="utf-8")
    )
    if plan.get("schema_version") != "pol-frozen-evaluation-plan-v5":
        raise ValueError("unsupported frozen evaluation plan schema")
    verify_execution_device_policy(
        plan,
        boundary="frozen evaluation plan",
    )
    stored_plan_hash = plan.pop("plan_content_hash", None)
    computed_plan_hash = stable_object_hash(plan)
    plan["plan_content_hash"] = stored_plan_hash
    if stored_plan_hash != computed_plan_hash:
        raise ValueError("frozen evaluation plan content hash mismatch")
    if plan.get("test_data_used") is not False:
        raise ValueError("frozen evaluation plan is not test-isolated")
    if plan.get("selection_record_hash") != selection_hash:
        raise ValueError("frozen plan selection binding mismatch")
    if summary.get("frozen_plan_hash") != stored_plan_hash:
        raise ValueError("run summary frozen-plan hash mismatch")
    if plan.get("test_evaluation_contract") != test_evaluation_contract():
        raise ValueError("unsupported frozen test-evaluation contract")

    dataset_reference = json.loads(
        (root / "dataset_reference.json").read_text(encoding="utf-8")
    )
    if not isinstance(dataset_reference, Mapping):
        raise ValueError("study dataset reference must be an object")
    if (
        dataset_reference.get("schema_version")
        != "pol-study-dataset-reference-v3"
    ):
        raise ValueError("unsupported study dataset-reference schema")
    verify_execution_device_policy(
        dataset_reference,
        boundary="study dataset reference",
    )
    dataset_binding_proof = dataset_reference.get("binding_proof")
    if not isinstance(dataset_binding_proof, Mapping):
        raise ValueError("study dataset reference has no binding proof")
    # Load the data package first because its public package initialization
    # establishes the existing data/validation binding import order.
    import_module("pol.data.dataset")
    from pol.validation.binding import verify_binding_proof

    verify_binding_proof(dataset_binding_proof)
    expected_dataset_binding = {
        "dataset_binding_kind": dataset_binding_proof["binding_kind"],
        "dataset_binding_status": dataset_binding_proof["status"],
        "dataset_target_reference_validation_status": dataset_binding_proof[
            "target_reference_validation_status"
        ],
        "dataset_binding_proof_hash": dataset_binding_proof["proof_hash"],
    }
    for source_name, source in (
        ("identity", identity),
        ("selection record", selection),
        ("frozen plan", plan),
        ("run summary", summary),
        ("dataset reference", dataset_reference),
    ):
        for field, expected_value in expected_dataset_binding.items():
            if source.get(field) != expected_value:
                raise ValueError(
                    f"{source_name} dataset validation binding mismatch: {field}"
                )
        verify_execution_device_policy(
            source,
            boundary=source_name,
        )
    if identity.get("dataset_artifact_id") != dataset_reference.get(
        "artifact_id"
    ):
        raise ValueError("manifest dataset binding mismatch")
    if identity.get("dataset_split_hash") != dataset_reference.get("split_hash"):
        raise ValueError("manifest split binding mismatch")
    if plan.get("dataset_artifact_id") != dataset_reference.get("artifact_id"):
        raise ValueError("frozen plan dataset binding mismatch")
    if plan.get("dataset_split_hash") != dataset_reference.get("split_hash"):
        raise ValueError("frozen plan split binding mismatch")
    if summary.get("dataset_artifact_id") != dataset_reference.get(
        "artifact_id"
    ):
        raise ValueError("run summary dataset binding mismatch")
    if dataset_reference.get("validation_artifact_id") != (
        dataset_binding_proof.get("certificate_artifact_id")
    ):
        raise ValueError("study dataset-reference certificate binding mismatch")

    model_name = plan.get("frozen_models_file")
    if not isinstance(model_name, str) or Path(model_name).name != model_name:
        raise ValueError("unsafe frozen model filename")
    model_path = root / model_name
    if file_sha256(model_path) != plan.get("frozen_models_sha256"):
        raise ValueError("frozen model archive hash mismatch")
    archive = torch.load(model_path, map_location="cpu", weights_only=True)
    if archive.get("schema_version") != "pol-frozen-model-archive-v5":
        raise ValueError("unsupported frozen model archive schema")
    verify_execution_device_policy(
        archive,
        boundary="frozen model archive",
    )
    require_cpu_tensors(
        archive,
        boundary="frozen model archive load",
        name="archive",
    )
    if archive.get("selection_record_hash") != selection_hash:
        raise ValueError("frozen model archive selection binding mismatch")
    models = archive.get("models")
    if not isinstance(models, Mapping):
        raise ValueError("frozen model archive models must be an object")
    expected = {
        (case_id, readout_id, candidate_id)
        for case_id, case in plan.get("cases", {}).items()
        for readout_id, candidate_id in case.get(
            "selected_by_readout", {}
        ).items()
    }
    actual = {
        (
            entry.get("case_id"),
            entry.get("readout_id"),
            entry.get("candidate_id"),
        )
        for entry in models.values()
        if isinstance(entry, Mapping)
    }
    if actual != expected or len(actual) != len(models):
        raise ValueError("frozen model archive does not match selected candidates")
    entry_by_binding = {
        (
            entry["case_id"],
            entry["readout_id"],
            entry["candidate_id"],
        ): entry
        for entry in models.values()
    }
    model_by_binding = {
        binding: entry["model"]
        for binding, entry in entry_by_binding.items()
    }
    direct_diagnostic_count, direct_zero_fill_count = (
        verify_frozen_decoder_bindings(
            models,
            selection=selection,
            plan=plan,
        )
    )

    validation_rows = load_rows(root / "validation_trials.csv")
    test_rows = load_rows(root / "test_metrics.csv")
    seed_rows = load_rows(root / "random_feature_seed_metrics.csv")
    ensemble_rows = load_rows(root / "random_feature_ensemble_metrics.csv")
    if summary.get("validation_row_count") != len(validation_rows):
        raise ValueError("run summary validation-row count mismatch")
    if summary.get("primary_test_row_count") != len(test_rows):
        raise ValueError("run summary primary-test-row count mismatch")
    if summary.get("random_feature_seed_row_count") != len(seed_rows):
        raise ValueError("run summary random-feature-seed-row count mismatch")
    if summary.get("random_feature_ensemble_row_count") != len(
        ensemble_rows
    ):
        raise ValueError("run summary random-feature-ensemble-row count mismatch")
    if (
        summary.get("direct_decoder_diagnostic_count")
        != direct_diagnostic_count
    ):
        raise ValueError("run summary direct-decoder diagnostic count mismatch")
    if (
        summary.get("direct_decoder_zero_fill_count")
        != direct_zero_fill_count
    ):
        raise ValueError("run summary direct-decoder zero-fill count mismatch")
    if summary.get("direct_decoder_zero_fill_applied") is not (
        direct_zero_fill_count > 0
    ):
        raise ValueError("run summary direct-decoder zero-fill flag mismatch")
    for row in validation_rows:
        if row.get("readout_kind") == "direct_fourier_decoder":
            try:
                row_J = int(row["J"])
                row_q = int(row["q"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "direct validation row has invalid J/q"
                ) from exc
            frozen_entry = entry_by_binding.get(_row_binding(row))
            if frozen_entry is None:
                J, q = row_J, row_q
            else:
                frozen_trial = TrialSpec.model_validate(frozen_entry["trial"])
                J = int(frozen_trial.feature.observation.J)
                q = int(frozen_trial.output.q)
                if (row_J, row_q) != (J, q):
                    raise ValueError(
                        "selected direct validation row J/q does not match "
                        "the frozen trial"
                    )
            verify_fixed_fourier_decoder_diagnostic(
                row,
                observation_count=J,
                requested_q=q,
                boundary="direct validation row",
            )
        else:
            verify_no_decoder_diagnostic(
                row,
                boundary="non-direct validation row",
            )
    selected_validation = {
        (row.get("case_id"), row.get("readout_id"), row.get("candidate_id"))
        for row in validation_rows
        if str(row.get("selected", "")).lower() == "true"
    }
    if selected_validation != expected:
        raise ValueError(
            "validation selected rows do not match frozen candidates"
        )
    actual_test_rows = {
        (row.get("case_id"), row.get("readout_id"), row.get("candidate_id"))
        for row in test_rows
    }
    if len(test_rows) != len(expected) or actual_test_rows != expected:
        raise ValueError("test rows do not match frozen candidates")

    def verify_test_binding(
        row: Mapping[str, Any],
        *,
        table: str,
    ) -> None:
        if str(row.get("selected", "")).lower() != "true":
            raise ValueError(f"{table} row is not marked as selected")
        if row.get("selection_record_hash") != selection_hash:
            raise ValueError(f"{table} row selection binding mismatch")
        if row.get("frozen_plan_hash") != stored_plan_hash:
            raise ValueError(f"{table} row frozen-plan binding mismatch")

    seed_rows_by_binding: dict[
        tuple[Any, Any, Any],
        list[dict[str, Any]],
    ] = {}
    for row in seed_rows:
        verify_test_binding(row, table="random-feature seed")
        verify_no_decoder_diagnostic(
            row,
            boundary="random-feature seed row",
        )
        seed_rows_by_binding.setdefault(_row_binding(row), []).append(row)
    ensemble_rows_by_binding: dict[
        tuple[Any, Any, Any],
        list[dict[str, Any]],
    ] = {}
    for row in ensemble_rows:
        verify_test_binding(row, table="random-feature ensemble")
        verify_no_decoder_diagnostic(
            row,
            boundary="random-feature ensemble row",
        )
        ensemble_rows_by_binding.setdefault(_row_binding(row), []).append(row)

    random_bindings = {
        binding
        for binding, model in model_by_binding.items()
        if isinstance(model, Mapping)
        and model.get("kind") == "random_feature_ridge"
    }
    if set(seed_rows_by_binding) != random_bindings:
        raise ValueError(
            "per-seed rows do not match frozen random-feature models"
        )
    if set(ensemble_rows_by_binding) != random_bindings:
        raise ValueError(
            "ensemble rows do not match frozen random-feature models"
        )

    seed_summary_suffixes = (
        "_seed_mean",
        "_seed_std",
        "_seed_ci95_low",
        "_seed_ci95_high",
    )
    seed_metadata_fields = (
        "test_seed_count",
        "test_seed_std_ddof",
        "test_confidence_level",
        "test_confidence_interval_method",
    )
    for row in test_rows:
        verify_test_binding(row, table="primary test")
        binding = _row_binding(row)
        model = model_by_binding[binding]
        if not isinstance(model, Mapping):
            raise ValueError("frozen model entry is not an object")
        if model.get("kind") == "direct_fourier_decoder":
            try:
                row_J = int(row["J"])
                row_q = int(row["q"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("direct test row has invalid J/q") from exc
            frozen_trial = TrialSpec.model_validate(
                entry_by_binding[binding]["trial"]
            )
            J = int(frozen_trial.feature.observation.J)
            q = int(frozen_trial.output.q)
            if (row_J, row_q) != (J, q):
                raise ValueError(
                    "direct test row J/q does not match the frozen trial"
                )
            verify_fixed_fourier_decoder_diagnostic(
                row,
                observation_count=J,
                requested_q=q,
                boundary="direct test row",
            )
        else:
            verify_no_decoder_diagnostic(
                row,
                boundary="non-direct primary test row",
            )
        if model.get("kind") != "random_feature_ridge":
            if row.get("test_result_kind") != "single_model":
                raise ValueError(
                    "deterministic primary row has the wrong result kind"
                )
            if any(
                _has_csv_value(row, key) for key in seed_metadata_fields
            ):
                raise ValueError(
                    "single-model primary row has false seed uncertainty"
                )
            if any(
                _has_csv_value(row, key)
                for key in row
                if key.endswith(seed_summary_suffixes)
            ):
                raise ValueError(
                    "single-model primary row has false seed summary"
                )
            continue

        if (
            row.get("test_result_kind")
            != "independent_seed_metric_summary"
        ):
            raise ValueError(
                "random-feature primary row has the wrong result kind"
            )
        members = model.get("members")
        if not isinstance(members, list) or len(members) < 2:
            raise ValueError(
                "frozen random-feature model has too few members"
            )
        member_seeds = [int(member["seed"]) for member in members]
        if len(member_seeds) != len(set(member_seeds)):
            raise ValueError(
                "frozen random-feature member seeds are not unique"
            )
        matching_seed_rows = seed_rows_by_binding[binding]
        try:
            row_seeds = [
                int(seed_row["seed"]) for seed_row in matching_seed_rows
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("per-seed row has an invalid seed") from exc
        if len(matching_seed_rows) != len(member_seeds):
            raise ValueError(
                "per-seed row count does not match frozen members"
            )
        if (
            len(row_seeds) != len(set(row_seeds))
            or set(row_seeds) != set(member_seeds)
        ):
            raise ValueError(
                "per-seed IDs do not match frozen member seeds"
            )
        if int(row.get("test_seed_count", -1)) != len(member_seeds):
            raise ValueError(
                "primary seed count does not match frozen members"
            )
        if int(row.get("test_seed_std_ddof", -1)) != 1:
            raise ValueError(
                "primary seed standard-deviation ddof is not one"
            )
        _require_close(
            row.get("test_confidence_level"),
            0.95,
            label="primary confidence level",
        )
        if row.get("test_confidence_interval_method") != "student_t":
            raise ValueError(
                "primary confidence interval method is not Student-t"
            )
        if any(
            seed_row.get("test_result_kind")
            != "independent_seed_realization"
            for seed_row in matching_seed_rows
        ):
            raise ValueError("per-seed row has the wrong result kind")

        first_seed_row = matching_seed_rows[0]
        metric_keys = tuple(
            sorted(
                key
                for key, value in first_seed_row.items()
                if key.startswith("test_")
                and key != "test_result_kind"
                and value not in ("", None)
            )
        )
        if not metric_keys:
            raise ValueError("per-seed row has no test metrics")
        metric_items: list[dict[str, float]] = []
        for seed_row in matching_seed_rows:
            active_keys = tuple(
                sorted(
                    key
                    for key, value in seed_row.items()
                    if key.startswith("test_")
                    and key != "test_result_kind"
                    and value not in ("", None)
                )
            )
            if active_keys != metric_keys:
                raise ValueError(
                    "per-seed rows have inconsistent metric fields"
                )
            try:
                metric_items.append(
                    {key: float(seed_row[key]) for key in metric_keys}
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "per-seed metric is not numeric"
                ) from exc
        expected_summary = summarize_independent_seed_metrics(metric_items)
        for key, expected_value in expected_summary.items():
            _require_close(
                row.get(key),
                expected_value,
                label=f"primary metric {key}",
            )
        for key in metric_keys:
            _require_close(
                row.get(key),
                float(row[f"{key}_seed_mean"]),
                label=f"canonical metric {key}",
            )

        matching_ensemble_rows = ensemble_rows_by_binding[binding]
        if len(matching_ensemble_rows) != 1:
            raise ValueError(
                "random-feature model must have exactly one ensemble row"
            )
        ensemble_row = matching_ensemble_rows[0]
        if ensemble_row.get("test_result_kind") != "prediction_ensemble":
            raise ValueError("ensemble row has the wrong result kind")
        if int(ensemble_row.get("ensemble_member_count", -1)) != len(
            member_seeds
        ):
            raise ValueError(
                "ensemble member count does not match frozen members"
            )
        expected_ensemble_keys = {
            key.replace("test_", "test_ensemble_", 1)
            for key in metric_keys
            if "representation_floor" not in key
        }
        actual_ensemble_keys = {
            key
            for key, value in ensemble_row.items()
            if key.startswith("test_ensemble_") and value not in ("", None)
        }
        if actual_ensemble_keys != expected_ensemble_keys:
            raise ValueError(
                "ensemble metric fields do not match prediction metrics"
            )
        try:
            for key in actual_ensemble_keys:
                float(ensemble_row[key])
        except (TypeError, ValueError) as exc:
            raise ValueError("ensemble metric is not numeric") from exc

    events = json.loads((root / "events.json").read_text(encoding="utf-8"))
    names = [
        item.get("event") for item in events if isinstance(item, Mapping)
    ]
    required = (
        "selection_complete",
        "convergence_complete",
        "freeze_written",
        "freeze_read_back",
        "first_test_state_solve",
        "first_test_metric",
    )
    if any(name not in names for name in required):
        raise ValueError("study-run event log is incomplete")
    if not (
        names.index("selection_complete")
        < names.index("freeze_written")
        < names.index("freeze_read_back")
        < names.index("first_test_state_solve")
        <= names.index("first_test_metric")
    ):
        raise ValueError(
            "study-run event order violates the freeze/test boundary"
        )
    for event in events:
        if not isinstance(event, Mapping):
            raise ValueError("study-run event must be an object")
        if event.get("event") in {
            "freeze_written",
            "freeze_read_back",
            "first_test_state_solve",
            "first_test_metric",
        } and event.get("plan_content_hash") != stored_plan_hash:
            raise ValueError(
                "study-run event has the wrong frozen-plan binding"
            )


def verify_study_run(path: Path | str) -> dict[str, Any]:
    root = Path(path).resolve()
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"not a safe study run directory: {root}")
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("study run has no manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "pol-study-run-manifest-v5":
        raise ValueError("unsupported study-run manifest")
    expected_records = manifest.get("files")
    if not isinstance(expected_records, list):
        raise ValueError("study-run files must be a list")
    actual_names: list[str] = []
    for path_item in root.rglob("*"):
        if path_item.is_symlink():
            raise ValueError(f"study run contains a symlink: {path_item}")
        if path_item.is_file() and path_item.name != "manifest.json":
            actual_names.append(path_item.relative_to(root).as_posix())
    expected_names = [
        record["relative_path"] for record in expected_records
    ]
    if sorted(actual_names) != sorted(expected_names):
        raise ValueError("study-run file tree differs from manifest")
    if manifest_records(root, expected_names) != expected_records:
        raise ValueError("study-run bytes differ from manifest")
    _verify_study_semantics(root, manifest)
    return manifest
