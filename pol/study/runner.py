from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
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
from pol.plotting.reporters import generate_reporters
from pol.runtime.artifacts import RunTransaction, manifest_records
from pol.runtime.environment import numerical_environment_fingerprint
from pol.runtime.hashing import stable_object_hash, tensor_sha256
from pol.runtime.io import atomic_torch_save, file_sha256, write_csv, write_strict_json
from .cache import FeatureStateCache
from .convergence import check_convergence
from .diagnostics import heat_multiplier_rows, noise_robustness_rows
from .overrides import apply_trial_overrides
from .search import SearchOutcome, run_search
from .trial import CandidateEvaluation, TrialEngine


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
            "schema_version": "pol-study-run-manifest-v1",
            "identity": dict(identity),
            "files": manifest_records(root, names),
        },
    )


def _verify_study_semantics(root: Path, manifest: Mapping[str, Any]) -> None:
    identity = manifest.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("study-run identity must be an object")
    run_hash = stable_object_hash(dict(identity))
    resolved_study = json.loads(
        (root / "resolved_study.json").read_text(encoding="utf-8")
    )
    if resolved_study != identity.get("study"):
        raise ValueError("resolved study does not match manifest identity")

    summary = json.loads((root / "run_summary.json").read_text(encoding="utf-8"))
    if summary.get("schema_version") != "pol-study-run-summary-v1":
        raise ValueError("unsupported study-run summary schema")
    if summary.get("run_hash") != run_hash:
        raise ValueError("study-run summary hash does not match manifest identity")
    if summary.get("study") != resolved_study.get("name"):
        raise ValueError("study-run summary name mismatch")
    if summary.get("profile") != resolved_study.get("profile"):
        raise ValueError("study-run summary profile mismatch")

    selection = json.loads(
        (root / "selection_record.json").read_text(encoding="utf-8")
    )
    if selection.get("schema_version") != "pol-selection-record-v1":
        raise ValueError("unsupported selection-record schema")
    _assert_selection_record_safe(selection)
    if selection.get("test_data_used") is not False:
        raise ValueError("selection record is not test-isolated")
    selection_hash = stable_object_hash(selection)
    if summary.get("selection_record_hash") != selection_hash:
        raise ValueError("selection-record hash mismatch")

    plan = json.loads(
        (root / "frozen_evaluation_plan.json").read_text(encoding="utf-8")
    )
    if plan.get("schema_version") != "pol-frozen-evaluation-plan-v1":
        raise ValueError("unsupported frozen evaluation plan schema")
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

    dataset_reference = json.loads(
        (root / "dataset_reference.json").read_text(encoding="utf-8")
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

    model_name = plan.get("frozen_models_file")
    if not isinstance(model_name, str) or Path(model_name).name != model_name:
        raise ValueError("unsafe frozen model filename")
    model_path = root / model_name
    if file_sha256(model_path) != plan.get("frozen_models_sha256"):
        raise ValueError("frozen model archive hash mismatch")
    archive = torch.load(model_path, map_location="cpu", weights_only=True)
    if archive.get("schema_version") != "pol-frozen-model-archive-v1":
        raise ValueError("unsupported frozen model archive schema")
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
    if actual != expected:
        raise ValueError("frozen model archive does not match selected candidates")

    validation_rows = _load_rows(root / "validation_trials.csv")
    test_rows = _load_rows(root / "test_metrics.csv")
    if summary.get("validation_row_count") != len(validation_rows):
        raise ValueError("run summary validation-row count mismatch")
    if summary.get("test_row_count") != len(test_rows):
        raise ValueError("run summary test-row count mismatch")
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
    for row in test_rows:
        if str(row.get("selected", "")).lower() != "true":
            raise ValueError("test row is not marked as selected")
        if row.get("selection_record_hash") != selection_hash:
            raise ValueError("test row selection binding mismatch")
        if row.get("frozen_plan_hash") != stored_plan_hash:
            raise ValueError("test row frozen-plan binding mismatch")

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
    if manifest.get("schema_version") != "pol-study-run-manifest-v1":
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
    if dataset.validation_ids.numel() == 0 or dataset.test_ids.numel() == 0:
        raise ValueError(
            "operator-learning studies require nonempty validation and test splits"
        )
    identity = {
        "schema_version": "pol-study-run-identity-v1",
        "environment": numerical_environment_fingerprint(),
        "study": _scientific_spec(spec),
        "dataset_artifact_id": dataset.artifact_id,
        "dataset_split_hash": dataset.split_hash,
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
            "study finite-data resolution exceeds the validated target "
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
        "schema_version": "pol-selection-record-v1",
        "study": spec.name,
        "profile": spec.profile,
        "dataset_artifact_id": dataset.artifact_id,
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
        "schema_version": "pol-frozen-model-archive-v1",
        "selection_record_hash": selection_hash,
        "models": frozen_models,
    }

    transaction = RunTransaction(final_dir)
    staging = transaction.begin()
    events: list[dict[str, Any]] = []
    try:
        write_strict_json(staging / "resolved_study.json", _scientific_spec(spec))
        write_strict_json(
            staging / "dataset_reference.json",
            {
                "artifact_id": dataset.artifact_id,
                "split_hash": dataset.split_hash,
                "validation_artifact_id": dataset.validation_artifact_id,
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
            "schema_version": "pol-frozen-evaluation-plan-v1",
            "study": spec.name,
            "dataset_artifact_id": dataset.artifact_id,
            "dataset_split_hash": dataset.split_hash,
            "selection_record_hash": selection_hash,
            "frozen_models_file": "frozen_models.pt",
            "frozen_models_sha256": model_file_hash,
            "cases": {
                case_id: {
                    "selected_by_readout": value["selected_by_readout"],
                    "representative_candidate_id": value[
                        "representative_candidate_id"
                    ],
                }
                for case_id, value in selection_cases.items()
            },
            "test_data_used": False,
        }
        frozen_plan_hash = stable_object_hash(frozen_plan)
        frozen_plan["plan_content_hash"] = frozen_plan_hash
        write_strict_json(staging / "frozen_evaluation_plan.json", frozen_plan)
        events.append({"event": "freeze_written", "plan_content_hash": frozen_plan_hash})

        # Required durability boundary: read the exact files back before any
        # test state solve or test metric is permitted.
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
        if loaded_plan.get("dataset_split_hash") != dataset.split_hash:
            raise ValueError("frozen plan split binding mismatch")
        if loaded_plan.get("selection_record_hash") != selection_hash:
            raise ValueError("frozen plan selection binding mismatch")
        if file_sha256(staging / loaded_plan["frozen_models_file"]) != loaded_plan[
            "frozen_models_sha256"
        ]:
            raise ValueError("frozen model archive read-back hash mismatch")
        loaded_archive = torch.load(
            staging / loaded_plan["frozen_models_file"],
            map_location="cpu",
            weights_only=True,
        )
        if loaded_archive.get("schema_version") != "pol-frozen-model-archive-v1":
            raise ValueError("unsupported frozen model archive schema")
        if loaded_archive.get("selection_record_hash") != selection_hash:
            raise ValueError("frozen model archive selection binding mismatch")
        events.append({"event": "freeze_read_back", "plan_content_hash": frozen_plan_hash})

        test_rows: list[dict[str, Any]] = []
        random_seed_rows: list[dict[str, Any]] = []
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
                **evaluated.aggregate_row,
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
            "schema_version": "pol-study-run-summary-v1",
            "status": "pass",
            "study": spec.name,
            "profile": spec.profile,
            "run_hash": run_hash,
            "dataset_artifact_id": dataset.artifact_id,
            "case_count": len(cases),
            "validation_row_count": len(validation_rows),
            "test_row_count": len(test_rows),
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
