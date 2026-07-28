from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
import math
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping

import torch

from pol.config.loader import load_dataset_spec
from pol.config.models import (
    ConvergenceSpec,
    HeatMultiplierDiagnosticSpec,
    NoiseDiagnosticSpec,
    StudySpec,
    TrialSpec,
)
from pol.data.dataset import ensure_dataset
from pol.learning.direct import (
    DIRECT_DECODER_DIAGNOSTIC_FIELDS,
    has_fixed_fourier_decoder_diagnostic,
    verify_fixed_fourier_decoder_diagnostic,
)
from pol.plotting.reporters import generate_reporters
from pol.runtime.artifacts import RunTransaction, manifest_records
from pol.runtime.device import (
    execution_device_policy,
    require_cpu_tensors,
    verify_execution_device_policy,
)
from pol.runtime.environment import numerical_environment_fingerprint
from pol.runtime.hashing import stable_object_hash, tensor_sha256
from pol.runtime.io import atomic_torch_save, file_sha256, write_csv, write_strict_json
from pol.validation.binding import verify_binding_proof
from .cache import FeatureStateCache
from .convergence import check_convergence
from .diagnostics import heat_multiplier_rows, noise_robustness_rows
from .overrides import apply_trial_overrides
from .search import SearchOutcome, run_search
from .trial import (
    CandidateEvaluation,
    TrialEngine,
    summarize_independent_seed_metrics,
)


@dataclass(frozen=True)
class StudyRunResult:
    path: Path
    reused: bool
    summary: dict[str, Any]


@dataclass(frozen=True)
class _Case:
    case_id: str
    variant_id: str
    variant_display_name: str
    global_values: dict[str, Any]
    trial: TrialSpec
    search: Any


def _scientific_spec(spec: StudySpec) -> dict[str, Any]:
    payload = spec.model_dump(mode="json")
    payload.pop("output_root", None)
    payload.pop("artifact_root", None)
    payload.pop("dataset_spec", None)
    return payload


def _axis_combinations(spec: StudySpec) -> list[dict[str, Any]]:
    if not spec.global_axes:
        return [{}]
    paths = [axis.path for axis in spec.global_axes]
    return [
        dict(zip(paths, values, strict=True))
        for values in itertools.product(*(axis.values for axis in spec.global_axes))
    ]


def _build_cases(spec: StudySpec) -> tuple[list[_Case], list[dict[str, Any]]]:
    cases: list[_Case] = []
    skipped: list[dict[str, Any]] = []
    for global_index, global_values in enumerate(_axis_combinations(spec)):
        try:
            global_trial = apply_trial_overrides(spec.base_trial, global_values)
        except ValueError as exc:
            if spec.execution.invalid_trial_policy == "error":
                raise
            skipped.append(
                {
                    "scope": "global",
                    "global_values": global_values,
                    "reason": str(exc),
                }
            )
            continue
        for variant in spec.variants:
            try:
                trial = apply_trial_overrides(global_trial, variant.overrides)
            except ValueError as exc:
                if spec.execution.invalid_trial_policy == "error":
                    raise
                skipped.append(
                    {
                        "scope": "variant",
                        "variant_id": variant.id,
                        "global_values": global_values,
                        "reason": str(exc),
                    }
                )
                continue
            suffix = "" if not global_values else f"-{global_index:03d}"
            cases.append(
                _Case(
                    case_id=f"{variant.id}{suffix}",
                    variant_id=variant.id,
                    variant_display_name=variant.display_name or variant.id,
                    global_values=dict(global_values),
                    trial=trial,
                    search=variant.search,
                )
            )
    if not cases:
        raise ValueError("study expansion produced no valid cases")
    return cases, skipped


def plan_study(spec: StudySpec) -> dict[str, Any]:
    cases, skipped = _build_cases(spec)
    planned: list[dict[str, Any]] = []
    for case in cases:
        search = case.search
        if search.kind == "static":
            candidate_upper_bound = 1
        elif search.kind == "grid":
            candidate_upper_bound = 1
            for axis in search.axes:
                candidate_upper_bound *= len(axis.values)
        else:
            per_readout = sum(len(axis.values) for axis in search.axes) * (
                search.rounds + 1
            )
            candidate_upper_bound = per_readout * len(case.trial.readouts)
        planned.append(
            {
                "case_id": case.case_id,
                "variant_id": case.variant_id,
                "global_values": case.global_values,
                "search_kind": search.kind,
                "candidate_upper_bound": candidate_upper_bound,
                "readout_ids": [readout.id for readout in case.trial.readouts],
            }
        )
    return {
        "schema_version": "pol-study-plan-v1",
        "study": spec.name,
        "profile": spec.profile,
        "case_count": len(cases),
        "cases": planned,
        "skipped": skipped,
        "filesystem_mutation": False,
    }


def _row_fields(rows: Iterable[Mapping[str, Any]]) -> list[str]:
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
    return [field for field in preferred if field in fields] + sorted(fields - set(preferred))


def _model_key(case_id: str, candidate_id: str, readout_id: str) -> str:
    return f"{case_id}/{candidate_id}/{readout_id}"


def _assert_selection_record_safe(record: Mapping[str, Any]) -> None:
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
                if normalized in forbidden_exact or normalized.startswith("test_metric"):
                    raise ValueError(f"selection record contains test binding at {path}.{key}")
                visit(item, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")

    visit(record, "$")


def _run_manifest(root: Path, *, identity: Mapping[str, Any]) -> None:
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


def _test_evaluation_contract() -> dict[str, Any]:
    return {
        "schema_version": "pol-test-evaluation-contract-v1",
        "random_feature_primary": "independent_seed_metric_summary",
        "random_feature_seed_result": "independent_seed_realization",
        "random_feature_ensemble_result": "prediction_ensemble",
        "seed_standard_deviation_ddof": 1,
        "confidence_level": 0.95,
        "confidence_interval_method": "student_t",
    }


def _row_binding(row: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    return (
        row.get("case_id"),
        row.get("readout_id"),
        row.get("candidate_id"),
    )


def _has_csv_value(row: Mapping[str, Any], key: str) -> bool:
    return key in row and row.get(key) not in ("", None)


def _decoder_diagnostic_fields(values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field: values[field]
        for field in DIRECT_DECODER_DIAGNOSTIC_FIELDS
        if field in values and values.get(field) not in ("", None)
    }


def _verify_no_decoder_diagnostic(
    values: Mapping[str, Any],
    *,
    boundary: str,
) -> None:
    if has_fixed_fourier_decoder_diagnostic(values):
        raise ValueError(
            f"{boundary} has false direct-decoder diagnostic fields"
        )


def _verify_frozen_decoder_bindings(
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
            _verify_no_decoder_diagnostic(
                model,
                boundary="non-direct frozen model",
            )
            _verify_no_decoder_diagnostic(
                inner,
                boundary="non-direct inner selection",
            )
            if readout_id in plan_by_readout:
                raise ValueError(
                    "frozen plan has false direct-decoder diagnostics for "
                    "a non-direct readout"
                )
    return direct_count, zero_fill_count


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
    _assert_selection_record_safe(selection)
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
    if plan.get("test_evaluation_contract") != _test_evaluation_contract():
        raise ValueError("unsupported frozen test-evaluation contract")

    dataset_reference = json.loads(
        (root / "dataset_reference.json").read_text(encoding="utf-8")
    )
    if not isinstance(dataset_reference, Mapping):
        raise ValueError("study dataset reference must be an object")
    if dataset_reference.get("schema_version") != "pol-study-dataset-reference-v3":
        raise ValueError("unsupported study dataset-reference schema")
    verify_execution_device_policy(
        dataset_reference,
        boundary="study dataset reference",
    )
    dataset_binding_proof = dataset_reference.get("binding_proof")
    if not isinstance(dataset_binding_proof, Mapping):
        raise ValueError("study dataset reference has no binding proof")
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
    if identity.get("dataset_artifact_id") != dataset_reference.get("artifact_id"):
        raise ValueError("manifest dataset binding mismatch")
    if identity.get("dataset_split_hash") != dataset_reference.get("split_hash"):
        raise ValueError("manifest split binding mismatch")
    if plan.get("dataset_artifact_id") != dataset_reference.get("artifact_id"):
        raise ValueError("frozen plan dataset binding mismatch")
    if plan.get("dataset_split_hash") != dataset_reference.get("split_hash"):
        raise ValueError("frozen plan split binding mismatch")
    if summary.get("dataset_artifact_id") != dataset_reference.get("artifact_id"):
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
        for readout_id, candidate_id in case.get("selected_by_readout", {}).items()
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
        _verify_frozen_decoder_bindings(
            models,
            selection=selection,
            plan=plan,
        )
    )

    validation_rows = _load_rows(root / "validation_trials.csv")
    test_rows = _load_rows(root / "test_metrics.csv")
    seed_rows = _load_rows(root / "random_feature_seed_metrics.csv")
    ensemble_rows = _load_rows(root / "random_feature_ensemble_metrics.csv")
    if summary.get("validation_row_count") != len(validation_rows):
        raise ValueError("run summary validation-row count mismatch")
    if summary.get("primary_test_row_count") != len(test_rows):
        raise ValueError("run summary primary-test-row count mismatch")
    if summary.get("random_feature_seed_row_count") != len(seed_rows):
        raise ValueError("run summary random-feature-seed-row count mismatch")
    if summary.get("random_feature_ensemble_row_count") != len(ensemble_rows):
        raise ValueError("run summary random-feature-ensemble-row count mismatch")
    if summary.get("direct_decoder_diagnostic_count") != direct_diagnostic_count:
        raise ValueError("run summary direct-decoder diagnostic count mismatch")
    if summary.get("direct_decoder_zero_fill_count") != direct_zero_fill_count:
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
            _verify_no_decoder_diagnostic(
                row,
                boundary="non-direct validation row",
            )
    selected_validation = {
        (row.get("case_id"), row.get("readout_id"), row.get("candidate_id"))
        for row in validation_rows
        if str(row.get("selected", "")).lower() == "true"
    }
    if selected_validation != expected:
        raise ValueError("validation selected rows do not match frozen candidates")
    actual_test_rows = {
        (row.get("case_id"), row.get("readout_id"), row.get("candidate_id"))
        for row in test_rows
    }
    if len(test_rows) != len(expected) or actual_test_rows != expected:
        raise ValueError("test rows do not match frozen candidates")

    def verify_test_binding(row: Mapping[str, Any], *, table: str) -> None:
        if str(row.get("selected", "")).lower() != "true":
            raise ValueError(f"{table} row is not marked as selected")
        if row.get("selection_record_hash") != selection_hash:
            raise ValueError(f"{table} row selection binding mismatch")
        if row.get("frozen_plan_hash") != stored_plan_hash:
            raise ValueError(f"{table} row frozen-plan binding mismatch")

    seed_rows_by_binding: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = {}
    for row in seed_rows:
        verify_test_binding(row, table="random-feature seed")
        _verify_no_decoder_diagnostic(
            row,
            boundary="random-feature seed row",
        )
        seed_rows_by_binding.setdefault(_row_binding(row), []).append(row)
    ensemble_rows_by_binding: dict[
        tuple[Any, Any, Any], list[dict[str, Any]]
    ] = {}
    for row in ensemble_rows:
        verify_test_binding(row, table="random-feature ensemble")
        _verify_no_decoder_diagnostic(
            row,
            boundary="random-feature ensemble row",
        )
        ensemble_rows_by_binding.setdefault(_row_binding(row), []).append(row)

    random_bindings = {
        binding
        for binding, model in model_by_binding.items()
        if isinstance(model, Mapping) and model.get("kind") == "random_feature_ridge"
    }
    if set(seed_rows_by_binding) != random_bindings:
        raise ValueError("per-seed rows do not match frozen random-feature models")
    if set(ensemble_rows_by_binding) != random_bindings:
        raise ValueError("ensemble rows do not match frozen random-feature models")

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
                raise ValueError("direct test row J/q does not match the frozen trial")
            verify_fixed_fourier_decoder_diagnostic(
                row,
                observation_count=J,
                requested_q=q,
                boundary="direct test row",
            )
        else:
            _verify_no_decoder_diagnostic(
                row,
                boundary="non-direct primary test row",
            )
        if model.get("kind") != "random_feature_ridge":
            if row.get("test_result_kind") != "single_model":
                raise ValueError("deterministic primary row has the wrong result kind")
            if any(_has_csv_value(row, key) for key in seed_metadata_fields):
                raise ValueError("single-model primary row has false seed uncertainty")
            if any(
                _has_csv_value(row, key)
                for key in row
                if key.endswith(seed_summary_suffixes)
            ):
                raise ValueError("single-model primary row has false seed summary")
            continue

        if row.get("test_result_kind") != "independent_seed_metric_summary":
            raise ValueError("random-feature primary row has the wrong result kind")
        members = model.get("members")
        if not isinstance(members, list) or len(members) < 2:
            raise ValueError("frozen random-feature model has too few members")
        member_seeds = [int(member["seed"]) for member in members]
        if len(member_seeds) != len(set(member_seeds)):
            raise ValueError("frozen random-feature member seeds are not unique")
        matching_seed_rows = seed_rows_by_binding[binding]
        try:
            row_seeds = [int(seed_row["seed"]) for seed_row in matching_seed_rows]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("per-seed row has an invalid seed") from exc
        if len(matching_seed_rows) != len(member_seeds):
            raise ValueError("per-seed row count does not match frozen members")
        if len(row_seeds) != len(set(row_seeds)) or set(row_seeds) != set(member_seeds):
            raise ValueError("per-seed IDs do not match frozen member seeds")
        if int(row.get("test_seed_count", -1)) != len(member_seeds):
            raise ValueError("primary seed count does not match frozen members")
        if int(row.get("test_seed_std_ddof", -1)) != 1:
            raise ValueError("primary seed standard-deviation ddof is not one")
        _require_close(
            row.get("test_confidence_level"),
            0.95,
            label="primary confidence level",
        )
        if row.get("test_confidence_interval_method") != "student_t":
            raise ValueError("primary confidence interval method is not Student-t")
        if any(
            seed_row.get("test_result_kind") != "independent_seed_realization"
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
                raise ValueError("per-seed rows have inconsistent metric fields")
            try:
                metric_items.append(
                    {key: float(seed_row[key]) for key in metric_keys}
                )
            except (TypeError, ValueError) as exc:
                raise ValueError("per-seed metric is not numeric") from exc
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
            raise ValueError("random-feature model must have exactly one ensemble row")
        ensemble_row = matching_ensemble_rows[0]
        if ensemble_row.get("test_result_kind") != "prediction_ensemble":
            raise ValueError("ensemble row has the wrong result kind")
        if int(ensemble_row.get("ensemble_member_count", -1)) != len(member_seeds):
            raise ValueError("ensemble member count does not match frozen members")
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
            raise ValueError("ensemble metric fields do not match prediction metrics")
        try:
            for key in actual_ensemble_keys:
                float(ensemble_row[key])
        except (TypeError, ValueError) as exc:
            raise ValueError("ensemble metric is not numeric") from exc

    events = json.loads((root / "events.json").read_text(encoding="utf-8"))
    names = [item.get("event") for item in events if isinstance(item, Mapping)]
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
        raise ValueError("study-run event order violates the freeze/test boundary")
    for event in events:
        if not isinstance(event, Mapping):
            raise ValueError("study-run event must be an object")
        if event.get("event") in {
            "freeze_written",
            "freeze_read_back",
            "first_test_state_solve",
            "first_test_metric",
        } and event.get("plan_content_hash") != stored_plan_hash:
            raise ValueError("study-run event has the wrong frozen-plan binding")


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
    expected_names = [record["relative_path"] for record in expected_records]
    if sorted(actual_names) != sorted(expected_names):
        raise ValueError("study-run file tree differs from manifest")
    if manifest_records(root, expected_names) != expected_records:
        raise ValueError("study-run bytes differ from manifest")
    _verify_study_semantics(root, manifest)
    return manifest


def _with_extended_convergence(
    original: ConvergenceSpec,
    rerun: int,
) -> ConvergenceSpec:
    values = list(original.n_sur_candidates)
    for _ in range(rerun):
        values.append(values[-1] * 2)
    return original.model_copy(update={"n_sur_candidates": tuple(values)})


def _evaluate_convergence(
    *,
    spec: StudySpec,
    dataset,
    cache: FeatureStateCache,
    case_id: str,
    selected: CandidateEvaluation,
    model: Mapping[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    if spec.convergence is None:
        return "not_requested", []
    final_rows: list[dict[str, Any]] = []
    for rerun in range(spec.convergence.max_auto_reruns + 1):
        active = _with_extended_convergence(spec.convergence, rerun)
        outcome = check_convergence(
            dataset=dataset,
            cache=cache,
            trial=selected.trial,
            model=model,
            spec=active,
        )
        final_rows = [
            {"case_id": case_id, "rerun": rerun, **row} for row in outcome.rows
        ]
        if outcome.status == "pass":
            return "pass", final_rows
    return "fail", final_rows


def _load_rows(path: Path) -> list[dict[str, Any]]:
    import csv

    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def regenerate_plots(spec: StudySpec, run_dir: Path) -> list[str]:
    """Regenerate figures without ever invalidating the verified source run."""
    existing_manifest = verify_study_run(run_dir)
    identity = existing_manifest["identity"]
    if identity.get("study") != _scientific_spec(spec):
        raise ValueError(
            "plot specification does not match the verified study-run identity"
        )
    transaction = RunTransaction(run_dir)
    staging = transaction.begin()
    try:
        shutil.copytree(run_dir, staging, dirs_exist_ok=True)
        validation_rows = _load_rows(staging / "validation_trials.csv")
        test_rows = _load_rows(staging / "test_metrics.csv")
        noise_rows = _load_rows(staging / "noise_robustness.csv")
        figures = staging / "figures"
        if figures.exists():
            shutil.rmtree(figures)
        created = generate_reporters(
            spec.reporters,
            validation_rows=validation_rows,
            test_rows=test_rows,
            noise_rows=noise_rows,
            output_dir=figures,
        )
        summary_path = staging / "run_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["figures"] = created
        write_strict_json(summary_path, summary)
        _run_manifest(staging, identity=identity)
        transaction.publish(lambda root: verify_study_run(root))
    except BaseException:
        transaction.cleanup()
        raise
    return created


def run_study(
    spec: StudySpec,
    *,
    repo_root: Path,
    force: bool = False,
    plots_only: bool = False,
) -> StudyRunResult:
    if spec.execution.torch_threads is not None:
        torch.set_num_threads(int(spec.execution.torch_threads))
    dataset_spec = load_dataset_spec(spec.dataset_spec, repo_root=repo_root)
    dataset = ensure_dataset(dataset_spec, repo_root=repo_root, force=False)
    verify_execution_device_policy(
        dataset.__dict__,
        boundary="study dataset",
    )
    require_cpu_tensors(
        {
            "sample_ids": dataset.sample_ids,
            "inputs_reference": dataset.inputs_reference,
            "targets_reference": dataset.targets_reference,
            "train_ids": dataset.train_ids,
            "validation_ids": dataset.validation_ids,
            "test_ids": dataset.test_ids,
        },
        boundary="study dataset",
        name="dataset",
    )
    if dataset.validation_ids.numel() == 0 or dataset.test_ids.numel() == 0:
        raise ValueError(
            "operator-learning studies require nonempty validation and test splits"
        )
    identity = {
        "schema_version": "pol-study-run-identity-v5",
        **execution_device_policy(),
        "environment": numerical_environment_fingerprint(),
        "study": _scientific_spec(spec),
        "dataset_artifact_id": dataset.artifact_id,
        "dataset_split_hash": dataset.split_hash,
        "dataset_binding_kind": dataset.binding_kind,
        "dataset_binding_status": dataset.binding_status,
        "dataset_target_reference_validation_status": (
            dataset.target_reference_validation_status
        ),
        "dataset_binding_proof_hash": dataset.binding_proof_hash,
    }
    run_hash = stable_object_hash(identity)
    final_dir = spec.output_root / spec.name / f"{spec.profile}-{run_hash[:12]}"
    if final_dir.exists() and not force:
        manifest = verify_study_run(final_dir)
        if manifest.get("identity") != identity:
            raise ValueError(
                "existing study run does not match the requested content identity"
            )
        if plots_only:
            created = regenerate_plots(spec, final_dir)
            return StudyRunResult(
                final_dir,
                True,
                {"status": "plots_regenerated", "created": created},
            )
        summary = json.loads((final_dir / "run_summary.json").read_text(encoding="utf-8"))
        return StudyRunResult(final_dir, True, summary)
    if plots_only:
        raise ValueError("--plots-only requires an existing verified run")

    cases, expansion_skipped = _build_cases(spec)
    oversized = [
        (case.case_id, int(case.trial.input.n_tar))
        for case in cases
        if int(case.trial.input.n_tar) > dataset.reference_nx
    ]
    if oversized:
        details = ", ".join(
            f"{case_id}: n_tar={n_tar}" for case_id, n_tar in oversized
        )
        raise ValueError(
            "study finite-data resolution exceeds the dataset target "
            f"reference_nx={dataset.reference_nx}: {details}"
        )
    cache = FeatureStateCache(
        artifact_root=spec.artifact_root,
        enabled=spec.execution.cache_states,
        batch_size=int(spec.execution.batch_size),
    )
    engine = TrialEngine(dataset, spec, cache)
    all_evaluations: dict[tuple[str, str], CandidateEvaluation] = {}
    outcomes: dict[str, SearchOutcome] = {}
    validation_rows: list[dict[str, Any]] = []
    selection_cases: dict[str, Any] = {}
    all_skipped = list(expansion_skipped)

    for case in cases:
        outcome = run_search(
            engine,
            case.trial,
            case.search,
            metric=spec.selection.metric,
            tolerance=spec.selection.tie_tolerance,
            invalid_policy=spec.execution.invalid_trial_policy,
        )
        outcomes[case.case_id] = outcome
        all_skipped.extend(
            {"case_id": case.case_id, **item} for item in outcome.skipped
        )
        for evaluation in outcome.evaluations:
            all_evaluations[(case.case_id, evaluation.candidate_id)] = evaluation
            for readout_id, base_row in evaluation.rows.items():
                validation_rows.append(
                    {
                        "case_id": case.case_id,
                        "variant_id": case.variant_id,
                        "variant_display_name": case.variant_display_name,
                        "global_values": json.dumps(case.global_values, sort_keys=True),
                        "search_stages": ";".join(
                            outcome.stages_by_candidate.get(evaluation.candidate_id, ())
                        ),
                        "selected": outcome.selected_by_readout.get(readout_id)
                        == evaluation.candidate_id,
                        **base_row,
                    }
                )
        representative_id = outcome.selected_by_readout[
            spec.selection.representative_readout
        ]
        selection_cases[case.case_id] = {
            "variant_id": case.variant_id,
            "global_values": case.global_values,
            "selected_by_readout": outcome.selected_by_readout,
            "representative_readout": spec.selection.representative_readout,
            "representative_candidate_id": representative_id,
            "inner_selections": {
                readout_id: all_evaluations[(case.case_id, candidate_id)].inner_selections[
                    readout_id
                ]
                for readout_id, candidate_id in outcome.selected_by_readout.items()
            },
            "validation_metrics": {
                readout_id: all_evaluations[(case.case_id, candidate_id)].rows[
                    readout_id
                ][spec.selection.metric]
                for readout_id, candidate_id in outcome.selected_by_readout.items()
            },
        }

    convergence_rows: list[dict[str, Any]] = []
    convergence_statuses: dict[str, str] = {}
    for case in cases:
        outcome = outcomes[case.case_id]
        representative_id = outcome.selected_by_readout[
            spec.selection.representative_readout
        ]
        selected = all_evaluations[(case.case_id, representative_id)]
        model = selected.frozen_models[spec.selection.representative_readout]
        status, rows = _evaluate_convergence(
            spec=spec,
            dataset=dataset,
            cache=cache,
            case_id=case.case_id,
            selected=selected,
            model=model,
        )
        convergence_statuses[case.case_id] = status
        convergence_rows.extend(rows)
    if any(status == "fail" for status in convergence_statuses.values()):
        raise RuntimeError(
            "surrogate-resolution convergence failed before test evaluation: "
            f"{convergence_statuses}"
        )

    selection_record = {
        "schema_version": "pol-selection-record-v5",
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
        "cases": selection_cases,
        "convergence": convergence_statuses,
        "test_data_used": False,
    }
    _assert_selection_record_safe(selection_record)
    selection_hash = stable_object_hash(selection_record)

    # Freeze only validation-selected candidates.  Keeping unselected models out
    # of the test plan makes the test split a final evaluation rather than a
    # second exploratory sweep.
    frozen_models: dict[str, Any] = {}
    for case in cases:
        outcome = outcomes[case.case_id]
        for readout_id, candidate_id in outcome.selected_by_readout.items():
            evaluation = all_evaluations[(case.case_id, candidate_id)]
            key = _model_key(case.case_id, candidate_id, readout_id)
            frozen_models[key] = {
                "case_id": case.case_id,
                "variant_id": case.variant_id,
                "candidate_id": candidate_id,
                "readout_id": readout_id,
                "trial": evaluation.trial.model_dump(mode="python"),
                "model": evaluation.frozen_models[readout_id],
            }
    frozen_archive = {
        "schema_version": "pol-frozen-model-archive-v5",
        **execution_device_policy(),
        "selection_record_hash": selection_hash,
        "models": frozen_models,
    }
    require_cpu_tensors(
        frozen_archive,
        boundary="frozen model archive publication",
        name="archive",
    )

    transaction = RunTransaction(final_dir)
    staging = transaction.begin()
    events: list[dict[str, Any]] = []
    try:
        write_strict_json(staging / "resolved_study.json", _scientific_spec(spec))
        write_strict_json(
            staging / "dataset_reference.json",
            {
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
            },
        )
        write_csv(
            staging / "validation_trials.csv",
            validation_rows,
            fieldnames=_row_fields(validation_rows),
        )
        write_csv(
            staging / "convergence.csv",
            convergence_rows,
            fieldnames=_row_fields(convergence_rows),
        )
        write_strict_json(staging / "selection_record.json", selection_record)
        write_strict_json(staging / "skipped_trials.json", all_skipped)
        events.append({"event": "selection_complete", "selection_record_hash": selection_hash})
        events.append({"event": "convergence_complete", "status": convergence_statuses})
        atomic_torch_save(staging / "frozen_models.pt", frozen_archive)
        model_file_hash = file_sha256(staging / "frozen_models.pt")
        frozen_plan = {
            "schema_version": "pol-frozen-evaluation-plan-v5",
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
            "frozen_models_file": "frozen_models.pt",
            "frozen_models_sha256": model_file_hash,
            "cases": {
                case_id: {
                    "selected_by_readout": value["selected_by_readout"],
                    "representative_candidate_id": value[
                        "representative_candidate_id"
                    ],
                    "decoder_diagnostics_by_readout": {
                        readout_id: _decoder_diagnostic_fields(inner)
                        for readout_id, inner in value[
                            "inner_selections"
                        ].items()
                        if has_fixed_fourier_decoder_diagnostic(inner)
                    },
                }
                for case_id, value in selection_cases.items()
            },
            "test_evaluation_contract": _test_evaluation_contract(),
            "test_data_used": False,
        }
        frozen_plan_hash = stable_object_hash(frozen_plan)
        frozen_plan["plan_content_hash"] = frozen_plan_hash
        write_strict_json(staging / "frozen_evaluation_plan.json", frozen_plan)
        events.append({"event": "freeze_written", "plan_content_hash": frozen_plan_hash})

        # Required durability boundary: read the exact files back before any
        # test state solve or test metric is permitted.
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
        if stored_plan_hash != frozen_plan_hash or recomputed_plan_hash != frozen_plan_hash:
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
        if loaded_plan.get(
            "dataset_target_reference_validation_status"
        ) != dataset.target_reference_validation_status:
            raise ValueError("frozen plan dataset target-validation status mismatch")
        if loaded_plan.get(
            "dataset_binding_proof_hash"
        ) != dataset.binding_proof_hash:
            raise ValueError("frozen plan dataset binding-proof hash mismatch")
        if loaded_plan.get("selection_record_hash") != selection_hash:
            raise ValueError("frozen plan selection binding mismatch")
        if loaded_plan.get("test_evaluation_contract") != _test_evaluation_contract():
            raise ValueError("frozen plan test-evaluation contract mismatch")
        if file_sha256(staging / loaded_plan["frozen_models_file"]) != loaded_plan[
            "frozen_models_sha256"
        ]:
            raise ValueError("frozen model archive read-back hash mismatch")
        loaded_archive = torch.load(
            staging / loaded_plan["frozen_models_file"],
            map_location="cpu",
            weights_only=True,
        )
        if loaded_archive.get("schema_version") != "pol-frozen-model-archive-v5":
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
        direct_diagnostic_count, direct_zero_fill_count = (
            _verify_frozen_decoder_bindings(
                loaded_archive["models"],
                selection=loaded_selection,
                plan=loaded_plan,
            )
        )
        events.append({"event": "freeze_read_back", "plan_content_hash": frozen_plan_hash})

        test_rows: list[dict[str, Any]] = []
        random_seed_rows: list[dict[str, Any]] = []
        random_ensemble_rows: list[dict[str, Any]] = []
        first_test = True
        for entry in loaded_archive["models"].values():
            if first_test:
                events.append(
                    {"event": "first_test_state_solve", "plan_content_hash": frozen_plan_hash}
                )
            trial = TrialSpec.model_validate(entry["trial"])
            evaluated = engine.evaluate_test(
                trial,
                entry["model"],
                readout_id=entry["readout_id"],
                candidate_id=entry["candidate_id"],
            )
            if first_test:
                events.append(
                    {"event": "first_test_metric", "plan_content_hash": frozen_plan_hash}
                )
                first_test = False
            row = {
                "case_id": entry["case_id"],
                "variant_id": entry["variant_id"],
                "selected": True,
                "selection_record_hash": selection_hash,
                "frozen_plan_hash": frozen_plan_hash,
                **evaluated.primary_row,
            }
            test_rows.append(row)
            random_seed_rows.extend(
                {
                    "case_id": entry["case_id"],
                    "variant_id": entry["variant_id"],
                    "selected": True,
                    "selection_record_hash": selection_hash,
                    "frozen_plan_hash": frozen_plan_hash,
                    **seed_row,
                }
                for seed_row in evaluated.seed_rows
            )
            if evaluated.ensemble_row is not None:
                random_ensemble_rows.append(
                    {
                        "case_id": entry["case_id"],
                        "variant_id": entry["variant_id"],
                        "selected": True,
                        "selection_record_hash": selection_hash,
                        "frozen_plan_hash": frozen_plan_hash,
                        **evaluated.ensemble_row,
                    }
                )
        write_csv(
            staging / "test_metrics.csv",
            test_rows,
            fieldnames=_row_fields(test_rows),
        )
        write_csv(
            staging / "random_feature_seed_metrics.csv",
            random_seed_rows,
            fieldnames=_row_fields(random_seed_rows),
        )
        write_csv(
            staging / "random_feature_ensemble_metrics.csv",
            random_ensemble_rows,
            fieldnames=_row_fields(random_ensemble_rows),
        )

        multiplier_rows: list[dict[str, Any]] = []
        noise_rows: list[dict[str, Any]] = []
        for case in cases:
            outcome = outcomes[case.case_id]
            for readout_id, candidate_id in outcome.selected_by_readout.items():
                evaluation = all_evaluations[(case.case_id, candidate_id)]
                model = evaluation.frozen_models[readout_id]
                for diagnostic in spec.diagnostics:
                    if isinstance(diagnostic, HeatMultiplierDiagnosticSpec):
                        multiplier_rows.extend(
                            heat_multiplier_rows(
                                diagnostic,
                                dataset=dataset,
                                trial=evaluation.trial,
                                model=model,
                                case_id=case.case_id,
                                readout_id=readout_id,
                            )
                        )
                    elif isinstance(diagnostic, NoiseDiagnosticSpec):
                        noise_rows.extend(
                            noise_robustness_rows(
                                diagnostic,
                                dataset=dataset,
                                cache=cache,
                                trial=evaluation.trial,
                                model=model,
                                case_id=case.case_id,
                                readout_id=readout_id,
                            )
                        )
                    else:
                        raise TypeError(
                            f"unsupported diagnostic type: {type(diagnostic).__name__}"
                        )
        write_csv(
            staging / "heat_multiplier.csv",
            multiplier_rows,
            fieldnames=_row_fields(multiplier_rows),
        )
        write_csv(
            staging / "noise_robustness.csv",
            noise_rows,
            fieldnames=_row_fields(noise_rows),
        )

        created_figures: list[str] = []
        if spec.execution.generate_plots:
            created_figures = generate_reporters(
                spec.reporters,
                validation_rows=validation_rows,
                test_rows=test_rows,
                noise_rows=noise_rows,
                output_dir=staging / "figures",
            )
        summary = {
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
            "case_count": len(cases),
            "validation_row_count": len(validation_rows),
            "primary_test_row_count": len(test_rows),
            "random_feature_seed_row_count": len(random_seed_rows),
            "random_feature_ensemble_row_count": len(random_ensemble_rows),
            "direct_decoder_diagnostic_count": direct_diagnostic_count,
            "direct_decoder_zero_fill_count": direct_zero_fill_count,
            "direct_decoder_zero_fill_applied": direct_zero_fill_count > 0,
            "selection_record_hash": selection_hash,
            "frozen_plan_hash": frozen_plan_hash,
            "convergence": convergence_statuses,
            "cache": cache.stats(),
            "skipped_trial_count": len(all_skipped),
            "figures": created_figures,
        }
        write_strict_json(staging / "events.json", events)
        write_strict_json(staging / "run_summary.json", summary)
        _run_manifest(staging, identity=identity)
        transaction.publish(lambda root: verify_study_run(root))
    except BaseException:
        transaction.cleanup()
        raise
    return StudyRunResult(final_dir, False, summary)
