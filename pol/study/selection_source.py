from __future__ import annotations

from dataclasses import dataclass
import copy
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from pol.config.loader import (
    load_dataset_spec,
    load_study_spec,
    load_validation_spec,
)
from pol.config.models import (
    CompletedStudySelectionSourceSpec,
    StudySpec,
    TrialSpec,
)
from pol.data.dataset import dataset_reference, load_dataset
from pol.runtime.artifacts import manifest_records
from pol.runtime.hashing import stable_object_hash
from pol.runtime.io import file_sha256
from pol.validation.binding import evaluate_dataset_binding
from pol.validation.certificates import load_validation_certificate
from pol.validation.runner import validation_reference
from .cases import build_study_run_identity, scientific_study_spec
from .protocol import assert_selection_record_safe
from .verification import verify_study_run


class SelectionDependencyError(ValueError):
    """A completed-study selection dependency is invalid or inconsistent."""


class MissingSelectionDependencyError(SelectionDependencyError):
    """A required verified artifact or completed study run does not exist."""


@dataclass(frozen=True)
class ResolvedSelectionBindings:
    spec: StudySpec
    provenance_by_variant: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class VerifiedCompletedRun:
    path: Path
    run_hash: str
    scientific_identity_hash: str
    identity: dict[str, Any]
    dataset: Any


@dataclass(frozen=True)
class FrozenSourceBoundary:
    selection: dict[str, Any]
    selection_hash: str
    plan: dict[str, Any]
    plan_hash: str
    archive: dict[str, Any]
    archive_hash: str


_SUPPORTED_STUDY_MANIFESTS = {
    "pol-study-run-manifest-v8",
    "pol-study-run-manifest-v9",
    "pol-study-run-manifest-v10",
    "pol-study-run-manifest-v11",
    "pol-study-run-manifest-v12",
    "pol-study-run-manifest-v13",
    "pol-study-run-manifest-v14",
    "pol-study-run-manifest-v15",
    "pol-study-run-manifest-v16",
}


def _dependency_key(spec: StudySpec) -> str:
    return stable_object_hash(scientific_study_spec(spec))


def _load_existing_dataset(spec: StudySpec, *, repo_root: Path) -> Any:
    dataset_spec = load_dataset_spec(spec.dataset_spec, repo_root=repo_root)
    validation_spec = load_validation_spec(
        dataset_spec.validation_spec,
        repo_root=repo_root,
    )
    validation_ref = validation_reference(validation_spec)
    if not validation_ref.path.is_dir():
        raise MissingSelectionDependencyError(
            "selection dependency is missing its expected validation artifact "
            f"{validation_ref.artifact_id}"
        )
    try:
        certificate = load_validation_certificate(validation_ref.path)
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        raise SelectionDependencyError(
            "selection dependency validation artifact failed verification: "
            f"{validation_ref.path}: {exc}"
        ) from exc
    binding_proof = evaluate_dataset_binding(
        certificate,
        validation_spec,
        dataset_spec,
    )
    reference = dataset_reference(
        dataset_spec,
        validation_artifact_id=validation_ref.artifact_id,
        binding_proof=binding_proof,
    )
    if not reference.path.is_dir():
        raise MissingSelectionDependencyError(
            "selection dependency is missing its expected dataset artifact "
            f"{reference.artifact_id}"
        )
    try:
        return load_dataset(reference.path)
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        raise SelectionDependencyError(
            "selection dependency dataset artifact failed verification: "
            f"{reference.path}: {exc}"
        ) from exc


def _preflight_study_run(
    root: Path,
    *,
    expected_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify exact source bytes and frozen non-test records without parsing tests."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"not a safe study run directory: {root}")
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("study run has no regular manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") not in _SUPPORTED_STUDY_MANIFESTS:
        raise ValueError("unsupported study-run manifest")
    if manifest.get("identity") != dict(expected_identity):
        raise ValueError(
            "study-run manifest does not match the expected identity"
        )
    records = manifest.get("files")
    if not isinstance(records, list):
        raise ValueError("study-run files must be a list")
    names: list[str] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("study-run manifest record must be an object")
        name = record.get("relative_path")
        if (
            not isinstance(name, str)
            or Path(name).is_absolute()
            or Path(name).as_posix() != name
            or Path(name).name == "manifest.json"
            or any(part in {"", ".", ".."} for part in Path(name).parts)
        ):
            raise ValueError("study-run manifest has an unsafe file name")
        names.append(name)
    if len(names) != len(set(names)):
        raise ValueError("study-run manifest has duplicate file records")
    actual_names: list[str] = []
    for item in root.rglob("*"):
        if item.is_symlink():
            raise ValueError(f"study run contains a symlink: {item}")
        if item.is_file() and item.name != "manifest.json":
            actual_names.append(item.relative_to(root).as_posix())
    if sorted(actual_names) != sorted(names):
        raise ValueError("study-run file tree differs from manifest")
    if manifest_records(root, names) != records:
        raise ValueError("study-run bytes differ from manifest")
    resolved = json.loads(
        (root / "resolved_study.json").read_text(encoding="utf-8")
    )
    if resolved != expected_identity.get("study"):
        raise ValueError("resolved study disagrees with expected identity")
    summary = json.loads(
        (root / "run_summary.json").read_text(encoding="utf-8")
    )
    if (
        summary.get("status") != "pass"
        or summary.get("run_hash")
        != stable_object_hash(dict(expected_identity))
    ):
        raise ValueError("study-run summary is not a completed expected run")
    return manifest


def _verify_completed_run(
    spec: StudySpec,
    *,
    repo_root: Path,
    provenance_by_variant: Mapping[str, Mapping[str, Any]],
    full_verify: bool = True,
) -> VerifiedCompletedRun:
    dataset = _load_existing_dataset(spec, repo_root=repo_root)
    verify_selection_dataset_bindings(provenance_by_variant, dataset=dataset)
    identity = build_study_run_identity(
        spec,
        dataset=dataset,
        selection_source_provenance=provenance_by_variant,
    )
    run_hash = stable_object_hash(identity)
    run_path = (
        spec.output_root / spec.name / f"{spec.profile}-{run_hash[:12]}"
    )
    if not run_path.is_dir():
        raise MissingSelectionDependencyError(
            "completed-study selection dependency is missing; expected "
            f"run_hash={run_hash} at {run_path}"
        )
    try:
        manifest = (
            verify_study_run(run_path)
            if full_verify
            else _preflight_study_run(
                run_path,
                expected_identity=identity,
            )
        )
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        raise SelectionDependencyError(
            "completed-study selection dependency failed verification: "
            f"{run_path}: {exc}"
        ) from exc
    if manifest.get("identity") != identity:
        raise SelectionDependencyError(
            "completed-study selection dependency does not match its expected "
            "content-addressed identity"
        )
    completed = VerifiedCompletedRun(
        path=run_path,
        run_hash=run_hash,
        scientific_identity_hash=stable_object_hash(identity["study"]),
        identity=identity,
        dataset=dataset,
    )
    if not full_verify:
        # This validates selection, frozen plan, and frozen model bytes without
        # opening any test result table.
        _read_frozen_source_payload(completed)
    return completed


def _read_frozen_source_payload(
    completed: VerifiedCompletedRun,
) -> tuple[dict[str, Any], str, dict[str, Any], str, dict[str, Any], str]:
    selection = json.loads(
        (completed.path / "selection_record.json").read_text(encoding="utf-8")
    )
    assert_selection_record_safe(selection)
    if selection.get("schema_version") not in {
        "pol-selection-record-v7",
        "pol-selection-record-v8",
        "pol-selection-record-v9",
    }:
        raise SelectionDependencyError(
            "selection source uses an unsupported selection-record schema"
        )
    if selection.get("test_data_used") is not False:
        raise SelectionDependencyError(
            "selection source record is not validation/test isolated"
        )
    selection_hash = stable_object_hash(selection)

    plan = json.loads(
        (completed.path / "frozen_evaluation_plan.json").read_text(
            encoding="utf-8"
        )
    )
    if plan.get("schema_version") not in {
        "pol-frozen-evaluation-plan-v7",
        "pol-frozen-evaluation-plan-v8",
        "pol-frozen-evaluation-plan-v9",
        "pol-frozen-evaluation-plan-v10",
    }:
        raise SelectionDependencyError(
            "selection source uses an unsupported frozen-plan schema"
        )
    plan_hash = plan.get("plan_content_hash")
    without_hash = dict(plan)
    without_hash.pop("plan_content_hash", None)
    if (
        not isinstance(plan_hash, str)
        or stable_object_hash(without_hash) != plan_hash
    ):
        raise SelectionDependencyError(
            "selection source frozen-plan content hash mismatch"
        )
    if (
        plan.get("selection_record_hash") != selection_hash
        or plan.get("test_data_used") is not False
    ):
        raise SelectionDependencyError(
            "selection source frozen plan does not match the selection record"
        )

    archive_name = plan.get("frozen_models_file")
    if (
        not isinstance(archive_name, str)
        or Path(archive_name).name != archive_name
    ):
        raise SelectionDependencyError(
            "selection source frozen archive filename is unsafe"
        )
    archive_path = completed.path / archive_name
    archive_hash = file_sha256(archive_path)
    if archive_hash != plan.get("frozen_models_sha256"):
        raise SelectionDependencyError(
            "selection source frozen archive does not match the frozen plan"
        )
    archive = torch.load(
        archive_path,
        map_location="cpu",
        weights_only=True,
    )
    if archive.get("schema_version") not in {
        "pol-frozen-model-archive-v7",
        "pol-frozen-model-archive-v8",
        "pol-frozen-model-archive-v9",
        "pol-frozen-model-archive-v10",
    }:
        raise SelectionDependencyError(
            "selection source uses an unsupported frozen-model schema"
        )
    if archive.get("selection_record_hash") != selection_hash:
        raise SelectionDependencyError(
            "selection source frozen archive selection hash mismatch"
        )
    return (
        selection,
        selection_hash,
        plan,
        plan_hash,
        archive,
        archive_hash,
    )


def read_frozen_source_boundary(
    completed: VerifiedCompletedRun,
) -> FrozenSourceBoundary:
    """Read back the validated selection/freeze boundary of a source run."""
    selection, selection_hash, plan, plan_hash, archive, archive_hash = (
        _read_frozen_source_payload(completed)
    )
    return FrozenSourceBoundary(
        selection=selection,
        selection_hash=selection_hash,
        plan=plan,
        plan_hash=plan_hash,
        archive=archive,
        archive_hash=archive_hash,
    )


def _source_case(
    selection: Mapping[str, Any],
    *,
    variant_id: str,
) -> tuple[str, Mapping[str, Any]]:
    cases = selection.get("cases")
    if not isinstance(cases, Mapping):
        raise SelectionDependencyError("selection source has no case records")
    matches = [
        (str(case_id), case)
        for case_id, case in cases.items()
        if isinstance(case, Mapping) and case.get("variant_id") == variant_id
    ]
    if not matches:
        raise SelectionDependencyError(
            f"selection source variant does not exist: {variant_id}"
        )
    if len(matches) != 1:
        raise SelectionDependencyError(
            "selection source variant expands to multiple cases; a unique "
            f"representative case is required: {variant_id}"
        )
    return matches[0]


def _get_import_value(condition: Mapping[str, Any], path: str) -> Any:
    current: Any = condition
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise SelectionDependencyError(
                f"selected feature condition has no import path {path}"
            )
        current = current[part]
    return copy.deepcopy(current)


def _extract_source_provenance(
    completed: VerifiedCompletedRun,
    source: CompletedStudySelectionSourceSpec,
) -> dict[str, Any]:
    (
        selection,
        selection_hash,
        plan,
        plan_hash,
        archive,
        archive_hash,
    ) = _read_frozen_source_payload(completed)
    metric = selection.get("selection_metric")
    if not isinstance(metric, str) or not metric.startswith("validation_"):
        raise SelectionDependencyError(
            "selection source metric is not a validation metric"
        )
    case_id, case = _source_case(
        selection,
        variant_id=source.source_variant_id,
    )
    readout_id = source.source_readout_id
    selected_by_readout = case.get("selected_by_readout")
    if not isinstance(selected_by_readout, Mapping):
        raise SelectionDependencyError(
            "selection source has no selected candidates by readout"
        )
    candidate_id = selected_by_readout.get(readout_id)
    if not isinstance(candidate_id, str):
        raise SelectionDependencyError(
            f"selection source readout was not selected: {readout_id}"
        )
    if (
        case.get("representative_readout") != readout_id
        or case.get("representative_candidate_id") != candidate_id
    ):
        raise SelectionDependencyError(
            "selection source readout/candidate is not the representative "
            "feature condition"
        )
    condition = case.get("representative_feature_condition")
    if not isinstance(condition, Mapping):
        raise SelectionDependencyError(
            "selection source representative feature condition is missing"
        )
    metrics = case.get("validation_metrics")
    if not isinstance(metrics, Mapping) or readout_id not in metrics:
        raise SelectionDependencyError(
            "selection source validation metric value is missing"
        )
    validation_metric_value = metrics[readout_id]
    if (
        condition.get("selection_split") != "validation"
        or condition.get("selection_metric") != metric
        or condition.get("selection_metric_value") != validation_metric_value
        or condition.get("representative_readout") != readout_id
        or condition.get("candidate_id") != candidate_id
    ):
        raise SelectionDependencyError(
            "selection source representative condition disagrees with "
            "validation selection"
        )

    plan_cases = plan.get("cases")
    plan_case = (
        plan_cases.get(case_id) if isinstance(plan_cases, Mapping) else None
    )
    if (
        not isinstance(plan_case, Mapping)
        or plan_case.get("selected_by_readout") != selected_by_readout
        or plan_case.get("representative_feature_condition") != condition
    ):
        raise SelectionDependencyError(
            "selection source frozen plan disagrees with the selected candidate"
        )
    models = archive.get("models")
    if not isinstance(models, Mapping):
        raise SelectionDependencyError(
            "selection source frozen archive has no models"
        )
    entries = [
        entry
        for entry in models.values()
        if isinstance(entry, Mapping)
        and entry.get("case_id") == case_id
        and entry.get("variant_id") == source.source_variant_id
        and entry.get("readout_id") == readout_id
        and entry.get("candidate_id") == candidate_id
    ]
    if len(entries) != 1:
        raise SelectionDependencyError(
            "selection source frozen archive does not contain exactly one "
            "selected model"
        )
    frozen_trial = TrialSpec.model_validate(entries[0].get("trial"))
    if (
        frozen_trial.input.model_dump(mode="json")
        != condition.get("finite_input")
        or frozen_trial.feature.model_dump(mode="json")
        != condition.get("feature")
        or frozen_trial.output.model_dump(mode="json")
        != condition.get("output")
    ):
        raise SelectionDependencyError(
            "selection source frozen trial disagrees with the representative "
            "feature condition"
        )

    imports = {
        path: _get_import_value(condition, path)
        for path in source.import_paths
    }
    return {
        "kind": "resolved_completed_study_selection",
        "source_study_name": completed.identity["study"]["name"],
        "source_profile": completed.identity["study"]["profile"],
        "source_study_run_hash": completed.run_hash,
        "source_study_scientific_identity_hash": (
            completed.scientific_identity_hash
        ),
        "source_selection_record_hash": selection_hash,
        "source_frozen_plan_hash": plan_hash,
        "source_frozen_model_archive_hash": archive_hash,
        "source_dataset_artifact_id": completed.dataset.artifact_id,
        "source_dataset_split_hash": completed.dataset.split_hash,
        "source_case_id": case_id,
        "source_variant_id": source.source_variant_id,
        "source_readout_id": readout_id,
        "source_candidate_id": candidate_id,
        "import_paths": list(source.import_paths),
        "resolved_imported_feature_condition": imports,
        "selection_metric": metric,
        "validation_metric_value": validation_metric_value,
        "validation_metric": {
            "name": metric,
            "value": validation_metric_value,
        },
    }


def _paths_overlap(left: str, right: str) -> bool:
    return (
        left == right
        or left.startswith(f"{right}.")
        or right.startswith(f"{left}.")
    )


def _reject_swept_imports(
    spec: StudySpec,
    *,
    variant_id: str,
    import_paths: tuple[str, ...],
) -> None:
    variant = next(item for item in spec.variants if item.id == variant_id)
    axis_paths = [axis.path for axis in spec.global_axes]
    if variant.search.kind != "static":
        axis_paths.extend(axis.path for axis in variant.search.axes)
    conflicts = sorted(
        {
            (import_path, axis_path)
            for import_path in import_paths
            for axis_path in axis_paths
            if _paths_overlap(import_path, axis_path)
        }
    )
    if conflicts:
        raise SelectionDependencyError(
            "selection-bound feature conditions cannot also be swept: "
            + ", ".join(
                f"{import_path} vs {axis_path}"
                for import_path, axis_path in conflicts
            )
        )


def _apply_provenance(
    spec: StudySpec,
    provenance_by_variant: Mapping[str, Mapping[str, Any]],
) -> StudySpec:
    variants = []
    for variant in spec.variants:
        provenance = provenance_by_variant.get(variant.id)
        if provenance is None:
            variants.append(variant)
            continue
        _reject_swept_imports(
            spec,
            variant_id=variant.id,
            import_paths=tuple(provenance["import_paths"]),
        )
        overrides = dict(variant.overrides)
        overrides.update(
            copy.deepcopy(
                provenance["resolved_imported_feature_condition"]
            )
        )
        variants.append(variant.model_copy(update={"overrides": overrides}))
    payload = spec.model_dump(mode="python")
    payload["variants"] = [
        variant.model_dump(mode="python") for variant in variants
    ]
    return StudySpec.model_validate(payload)


def resolve_selection_bindings(
    spec: StudySpec,
    *,
    repo_root: Path,
    _stack: tuple[str, ...] = (),
    _full_verify: bool = True,
) -> ResolvedSelectionBindings:
    current_key = _dependency_key(spec)
    if current_key in _stack:
        raise SelectionDependencyError(
            "completed-study selection dependency cycle detected"
        )
    stack = (*_stack, current_key)
    provenance_by_variant: dict[str, dict[str, Any]] = {}
    for variant in spec.variants:
        source = variant.selection_source
        if source is None:
            continue
        source_path = source.source_study_spec
        if not source_path.is_file():
            raise MissingSelectionDependencyError(
                f"selection source study spec does not exist: {source_path}"
            )
        source_spec = load_study_spec(source_path, repo_root=repo_root)
        source_key = _dependency_key(source_spec)
        if source_key in stack:
            raise SelectionDependencyError(
                "completed-study selection self dependency or cycle detected: "
                f"{source_path}"
            )
        if source_spec.profile != spec.profile:
            raise SelectionDependencyError(
                "selection source profile mismatch: "
                f"source={source_spec.profile}, downstream={spec.profile}"
            )
        resolved_source = resolve_selection_bindings(
            source_spec,
            repo_root=repo_root,
            _stack=stack,
            _full_verify=_full_verify,
        )
        completed = _verify_completed_run(
            resolved_source.spec,
            repo_root=repo_root,
            provenance_by_variant=(
                resolved_source.provenance_by_variant
            ),
            full_verify=_full_verify,
        )
        provenance_by_variant[variant.id] = _extract_source_provenance(
            completed,
            source,
        )
    return ResolvedSelectionBindings(
        spec=_apply_provenance(spec, provenance_by_variant),
        provenance_by_variant=provenance_by_variant,
    )


def verify_selection_dataset_bindings(
    provenance_by_variant: Mapping[str, Mapping[str, Any]],
    *,
    dataset: Any,
) -> None:
    for variant_id, provenance in provenance_by_variant.items():
        if provenance.get("source_dataset_artifact_id") != dataset.artifact_id:
            raise SelectionDependencyError(
                "selection source/downstream dataset artifact mismatch for "
                f"variant {variant_id}"
            )
        if provenance.get("source_dataset_split_hash") != dataset.split_hash:
            raise SelectionDependencyError(
                "selection source/downstream split mismatch for "
                f"variant {variant_id}"
            )


def preflight_downstream_dataset_binding(
    spec: StudySpec,
    *,
    provenance_by_variant: Mapping[str, Mapping[str, Any]],
    repo_root: Path,
) -> None:
    if not provenance_by_variant:
        return
    dataset = _load_existing_dataset(spec, repo_root=repo_root)
    verify_selection_dataset_bindings(
        provenance_by_variant,
        dataset=dataset,
    )


def inspect_selection_dependencies(
    spec: StudySpec,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    declarations = [
        {
            "variant_id": variant.id,
            "kind": variant.selection_source.kind,
            "source_variant_id": variant.selection_source.source_variant_id,
            "source_readout_id": variant.selection_source.source_readout_id,
            "import_paths": list(variant.selection_source.import_paths),
        }
        for variant in spec.variants
        if variant.selection_source is not None
    ]
    if not declarations:
        return {
            "status": "not_required",
            "dependencies": [],
            "scientific_conditions_resolved": True,
        }
    try:
        resolved = resolve_selection_bindings(spec, repo_root=repo_root)
    except MissingSelectionDependencyError as exc:
        return {
            "status": "missing",
            "dependencies": declarations,
            "scientific_conditions_resolved": False,
            "reason": str(exc),
        }
    return {
        "status": "completed",
        "dependencies": [
            {
                **declaration,
                "status": "completed",
                "provenance": resolved.provenance_by_variant[
                    declaration["variant_id"]
                ],
            }
            for declaration in declarations
        ],
        "scientific_conditions_resolved": True,
    }


def inspect_completed_study_selection(
    spec: StudySpec,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    resolved = resolve_selection_bindings(spec, repo_root=repo_root)
    completed = _verify_completed_run(
        resolved.spec,
        repo_root=repo_root,
        provenance_by_variant=resolved.provenance_by_variant,
    )
    selection, selection_hash, plan, plan_hash, _, archive_hash = (
        _read_frozen_source_payload(completed)
    )
    cases = selection.get("cases", {})
    return {
        "status": "completed",
        "study": spec.name,
        "profile": spec.profile,
        "source_study_run_hash": completed.run_hash,
        "source_study_scientific_identity_hash": (
            completed.scientific_identity_hash
        ),
        "source_selection_record_hash": selection_hash,
        "source_frozen_plan_hash": plan_hash,
        "source_frozen_model_archive_hash": archive_hash,
        "dataset_artifact_id": completed.dataset.artifact_id,
        "dataset_split_hash": completed.dataset.split_hash,
        "selections": [
            {
                "case_id": case_id,
                "variant_id": case["variant_id"],
                "representative_readout": case[
                    "representative_readout"
                ],
                "representative_candidate_id": case[
                    "representative_candidate_id"
                ],
                "selection_metric": selection["selection_metric"],
                "validation_metric_value": case["validation_metrics"][
                    case["representative_readout"]
                ],
                "feature_condition": case[
                    "representative_feature_condition"
                ]["feature"],
            }
            for case_id, case in cases.items()
        ],
        "test_tables_used_for_condition_selection": False,
    }


def verify_downstream_selection(
    spec: StudySpec,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    resolved = resolve_selection_bindings(spec, repo_root=repo_root)
    dataset = _load_existing_dataset(resolved.spec, repo_root=repo_root)
    verify_selection_dataset_bindings(
        resolved.provenance_by_variant,
        dataset=dataset,
    )
    identity = build_study_run_identity(
        resolved.spec,
        dataset=dataset,
        selection_source_provenance=resolved.provenance_by_variant,
    )
    return {
        "status": "pass",
        "study": spec.name,
        "profile": spec.profile,
        "run_hash": stable_object_hash(identity),
        "study_scientific_identity_hash": stable_object_hash(
            identity["study"]
        ),
        "dataset_artifact_id": dataset.artifact_id,
        "dataset_split_hash": dataset.split_hash,
        "selection_source_provenance": resolved.provenance_by_variant,
        "filesystem_mutation": False,
        "test_tables_used_for_condition_selection": False,
    }


def resolve_verified_completed_run(
    spec: StudySpec,
    *,
    repo_root: Path,
) -> VerifiedCompletedRun:
    """Resolve dependencies and return the exact verified completed run.

    This is the read-only source boundary shared by cross-run reporting.  It
    loads only existing validation/dataset artifacts and completed study runs;
    it never builds or executes an upstream scientific workflow.
    """
    resolved = resolve_selection_bindings(spec, repo_root=repo_root)
    return _verify_completed_run(
        resolved.spec,
        repo_root=repo_root,
        provenance_by_variant=resolved.provenance_by_variant,
    )


def resolve_preflight_completed_run(
    spec: StudySpec,
    *,
    repo_root: Path,
) -> VerifiedCompletedRun:
    """Resolve exact source bytes and frozen non-test state without test parsing."""
    resolved = resolve_selection_bindings(
        spec,
        repo_root=repo_root,
        _full_verify=False,
    )
    return _verify_completed_run(
        resolved.spec,
        repo_root=repo_root,
        provenance_by_variant=resolved.provenance_by_variant,
        full_verify=False,
    )
