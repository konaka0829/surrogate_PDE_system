from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

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
from pol.runtime.artifacts import RunTransaction
from pol.runtime.device import (
    execution_device_policy,
    require_cpu_tensors,
    verify_execution_device_policy,
)
from pol.runtime.environment import numerical_environment_fingerprint
from pol.runtime.hashing import stable_object_hash
from pol.runtime.io import write_strict_json
from .cache import FeatureStateCache
from .cases import (
    StudyCase,
    build_cases,
    plan_study,
    scientific_study_spec,
)
from .convergence import check_convergence
from .diagnostics import heat_multiplier_rows, noise_robustness_rows
from .evaluation import CandidateEvaluation
from .protocol import persist_and_read_back_freeze, prepare_freeze
from .results import (
    bind_test_evaluation,
    build_run_summary,
    load_reporter_inputs,
    validation_result_rows,
    write_completion_records,
    write_diagnostic_tables,
    write_pre_freeze_results,
    write_run_manifest,
    write_skipped_trials,
    write_test_tables,
)
from .search import SearchOutcome, run_search
from .trial import TrialEngine
from .verification import verify_study_run


@dataclass(frozen=True)
class StudyRunResult:
    path: Path
    reused: bool
    summary: dict[str, Any]


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
    dataset: Any,
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
            {"case_id": case_id, "rerun": rerun, **row}
            for row in outcome.rows
        ]
        if outcome.status == "pass":
            return "pass", final_rows
    return "fail", final_rows


def regenerate_plots(spec: StudySpec, run_dir: Path) -> list[str]:
    """Regenerate figures without ever invalidating the verified source run."""
    existing_manifest = verify_study_run(run_dir)
    identity = existing_manifest["identity"]
    if identity.get("study") != scientific_study_spec(spec):
        raise ValueError(
            "plot specification does not match the verified study-run identity"
        )
    transaction = RunTransaction(run_dir)
    staging = transaction.begin()
    try:
        shutil.copytree(run_dir, staging, dirs_exist_ok=True)
        reporter_inputs = load_reporter_inputs(staging)
        figures = staging / "figures"
        if figures.exists():
            shutil.rmtree(figures)
        created = generate_reporters(
            spec.reporters,
            validation_rows=reporter_inputs.validation_rows,
            test_rows=reporter_inputs.test_rows,
            noise_rows=reporter_inputs.noise_rows,
            output_dir=figures,
        )
        summary_path = staging / "run_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["figures"] = created
        write_strict_json(summary_path, summary)
        write_run_manifest(staging, identity=identity)
        transaction.publish(lambda root: verify_study_run(root))
    except BaseException:
        transaction.cleanup()
        raise
    return created


def _prepare_dataset_and_identity(
    spec: StudySpec,
    *,
    repo_root: Path,
) -> tuple[Any, dict[str, Any], str, Path]:
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
        "study": scientific_study_spec(spec),
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
    final_dir = (
        spec.output_root / spec.name / f"{spec.profile}-{run_hash[:12]}"
    )
    return dataset, identity, run_hash, final_dir


def _validate_case_resolutions(
    cases: list[StudyCase],
    *,
    reference_nx: int,
) -> None:
    oversized = [
        (case.case_id, int(case.trial.input.n_tar))
        for case in cases
        if int(case.trial.input.n_tar) > reference_nx
    ]
    if oversized:
        details = ", ".join(
            f"{case_id}: n_tar={n_tar}" for case_id, n_tar in oversized
        )
        raise ValueError(
            "study finite-data resolution exceeds the dataset target "
            f"reference_nx={reference_nx}: {details}"
        )


def _run_selection(
    *,
    spec: StudySpec,
    cases: list[StudyCase],
    engine: TrialEngine,
) -> tuple[
    dict[tuple[str, str], CandidateEvaluation],
    dict[str, SearchOutcome],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    evaluations: dict[tuple[str, str], CandidateEvaluation] = {}
    outcomes: dict[str, SearchOutcome] = {}
    validation_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
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
        skipped.extend(
            {"case_id": case.case_id, **item}
            for item in outcome.skipped
        )
        for evaluation in outcome.evaluations:
            evaluations[(case.case_id, evaluation.candidate_id)] = evaluation
        validation_rows.extend(
            validation_result_rows(case=case, outcome=outcome)
        )
    return evaluations, outcomes, validation_rows, skipped


def _run_convergence_checks(
    *,
    spec: StudySpec,
    dataset: Any,
    cache: FeatureStateCache,
    cases: list[StudyCase],
    outcomes: Mapping[str, SearchOutcome],
    evaluations: Mapping[tuple[str, str], CandidateEvaluation],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    statuses: dict[str, str] = {}
    for case in cases:
        outcome = outcomes[case.case_id]
        representative_id = outcome.selected_by_readout[
            spec.selection.representative_readout
        ]
        selected = evaluations[(case.case_id, representative_id)]
        model = selected.frozen_models[
            spec.selection.representative_readout
        ]
        status, case_rows = _evaluate_convergence(
            spec=spec,
            dataset=dataset,
            cache=cache,
            case_id=case.case_id,
            selected=selected,
            model=model,
        )
        statuses[case.case_id] = status
        rows.extend(case_rows)
    if any(status == "fail" for status in statuses.values()):
        raise RuntimeError(
            "surrogate-resolution convergence failed before test evaluation: "
            f"{statuses}"
        )
    return statuses, rows


def _run_diagnostics(
    *,
    spec: StudySpec,
    dataset: Any,
    cache: FeatureStateCache,
    cases: list[StudyCase],
    outcomes: Mapping[str, SearchOutcome],
    evaluations: Mapping[tuple[str, str], CandidateEvaluation],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    multiplier_rows: list[dict[str, Any]] = []
    noise_rows: list[dict[str, Any]] = []
    for case in cases:
        outcome = outcomes[case.case_id]
        for readout_id, candidate_id in outcome.selected_by_readout.items():
            evaluation = evaluations[(case.case_id, candidate_id)]
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
                        "unsupported diagnostic type: "
                        f"{type(diagnostic).__name__}"
                    )
    return multiplier_rows, noise_rows


def run_study(
    spec: StudySpec,
    *,
    repo_root: Path,
    force: bool = False,
    plots_only: bool = False,
) -> StudyRunResult:
    if spec.execution.torch_threads is not None:
        torch.set_num_threads(int(spec.execution.torch_threads))
    dataset, identity, run_hash, final_dir = _prepare_dataset_and_identity(
        spec,
        repo_root=repo_root,
    )
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
        summary = json.loads(
            (final_dir / "run_summary.json").read_text(encoding="utf-8")
        )
        return StudyRunResult(final_dir, True, summary)
    if plots_only:
        raise ValueError("--plots-only requires an existing verified run")

    cases, expansion_skipped = build_cases(spec)
    _validate_case_resolutions(cases, reference_nx=dataset.reference_nx)
    cache = FeatureStateCache(
        artifact_root=spec.artifact_root,
        enabled=spec.execution.cache_states,
        batch_size=int(spec.execution.batch_size),
    )
    engine = TrialEngine(dataset, spec, cache)
    (
        all_evaluations,
        outcomes,
        validation_rows,
        search_skipped,
    ) = _run_selection(spec=spec, cases=cases, engine=engine)
    all_skipped = [*expansion_skipped, *search_skipped]
    convergence_statuses, convergence_rows = _run_convergence_checks(
        spec=spec,
        dataset=dataset,
        cache=cache,
        cases=cases,
        outcomes=outcomes,
        evaluations=all_evaluations,
    )
    preparation = prepare_freeze(
        spec=spec,
        dataset=dataset,
        cases=cases,
        outcomes=outcomes,
        evaluations=all_evaluations,
        convergence_statuses=convergence_statuses,
    )

    transaction = RunTransaction(final_dir)
    staging = transaction.begin()
    try:
        write_pre_freeze_results(
            staging,
            resolved_study=scientific_study_spec(spec),
            dataset=dataset,
            validation_rows=validation_rows,
            convergence_rows=convergence_rows,
        )
        persisted = persist_and_read_back_freeze(
            staging,
            preparation=preparation,
            spec=spec,
            dataset=dataset,
            convergence_statuses=convergence_statuses,
        )
        write_skipped_trials(staging, all_skipped)

        events = list(persisted.events)
        test_rows: list[dict[str, Any]] = []
        random_seed_rows: list[dict[str, Any]] = []
        random_ensemble_rows: list[dict[str, Any]] = []
        first_test = True
        for entry in persisted.archive["models"].values():
            if first_test:
                events.append(
                    {
                        "event": "first_test_state_solve",
                        "plan_content_hash": persisted.frozen_plan_hash,
                    }
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
                    {
                        "event": "first_test_metric",
                        "plan_content_hash": persisted.frozen_plan_hash,
                    }
                )
                first_test = False
            bound = bind_test_evaluation(
                entry,
                evaluated,
                selection_hash=persisted.selection_hash,
                frozen_plan_hash=persisted.frozen_plan_hash,
            )
            test_rows.append(bound.primary_row)
            random_seed_rows.extend(bound.seed_rows)
            if bound.ensemble_row is not None:
                random_ensemble_rows.append(bound.ensemble_row)
        write_test_tables(
            staging,
            test_rows=test_rows,
            random_seed_rows=random_seed_rows,
            random_ensemble_rows=random_ensemble_rows,
        )

        multiplier_rows, noise_rows = _run_diagnostics(
            spec=spec,
            dataset=dataset,
            cache=cache,
            cases=cases,
            outcomes=outcomes,
            evaluations=all_evaluations,
        )
        write_diagnostic_tables(
            staging,
            multiplier_rows=multiplier_rows,
            noise_rows=noise_rows,
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
        summary = build_run_summary(
            spec=spec,
            run_hash=run_hash,
            dataset=dataset,
            case_count=len(cases),
            validation_rows=validation_rows,
            test_rows=test_rows,
            random_seed_rows=random_seed_rows,
            random_ensemble_rows=random_ensemble_rows,
            direct_diagnostic_count=persisted.direct_diagnostic_count,
            direct_zero_fill_count=persisted.direct_zero_fill_count,
            selection_hash=persisted.selection_hash,
            frozen_plan_hash=persisted.frozen_plan_hash,
            convergence_statuses=convergence_statuses,
            cache_stats=cache.stats(),
            skipped_trial_count=len(all_skipped),
            created_figures=created_figures,
        )
        write_completion_records(
            staging,
            events=events,
            summary=summary,
            identity=identity,
        )
        transaction.publish(lambda root: verify_study_run(root))
    except BaseException:
        transaction.cleanup()
        raise
    return StudyRunResult(final_dir, False, summary)
