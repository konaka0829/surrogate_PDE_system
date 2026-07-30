#!/usr/bin/env python3
"""Read-only strict parse/plan audit for every declared main workflow."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pol.config.loader import (
    load_dataset_spec,
    load_digital_baseline_spec,
    load_report_spec,
    load_study_spec,
    load_validation_spec,
)
from pol.digital_baselines.protocol import plan_digital_baseline
from pol.study.cases import plan_study as plan_case_expansion
from pol.study.selection_source import resolve_verified_completed_run
from pol.study.selection_source import (
    inspect_selection_dependencies,
    resolve_selection_bindings,
)


VALIDATIONS = (
    "configs/validation/foundation_main.json",
    "configs/validation/heat_main.json",
    "configs/validation/reaction_diffusion_main.json",
)
DATASETS = (
    "configs/datasets/heat_main.json",
    "configs/datasets/burgers_main.json",
)
STUDIES = (
    "studies/heat_readout_calibration.json",
    "studies/surrogate_parameter_time_coordinate_search.json",
    "studies/surrogate_parameter_time_landscape.json",
    "studies/dynamic_feature_baseline_comparison.json",
    "studies/readout_stability_noise.json",
    "studies/learning_curve.json",
    "studies/random_feature_seed_statistics.json",
    "studies/observation_output_budget.json",
    "studies/input_simulation_resolution.json",
)
REPORTS = ("reports/surrogate_operator_summary.json",)
DIGITAL_BASELINES = ("digital_baselines/fno1d.json",)


def _random_feature_counts(spec: Any) -> dict[str, int]:
    readouts = list(spec.base_trial.readouts)
    random_readouts = [
        readout for readout in readouts
        if readout.kind == "random_feature_ridge"
    ]
    return {
        "readout_count_per_case": len(readouts),
        "random_feature_readout_count_per_case": len(random_readouts),
        "random_feature_selection_seed_count_per_case": sum(
            len(readout.selection_seeds) for readout in random_readouts
        ),
        "random_feature_evaluation_seed_count_per_case": sum(
            len(readout.evaluation_seeds) for readout in random_readouts
        ),
    }


def _source_run_status(spec: Any, *, root: Path) -> dict[str, Any]:
    try:
        completed = resolve_verified_completed_run(spec, repo_root=root)
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        return {"status": "missing_or_invalid", "reason": str(exc)}
    return {
        "status": "verified_completed",
        "run_hash": completed.run_hash,
        "study_scientific_identity_hash": completed.scientific_identity_hash,
    }


def _plan_study_read_only(spec: Any, *, root: Path) -> dict[str, Any]:
    dependencies = inspect_selection_dependencies(spec, repo_root=root)
    planned_spec = spec
    if dependencies["status"] == "completed":
        planned_spec = resolve_selection_bindings(
            spec,
            repo_root=root,
        ).spec
    plan = plan_case_expansion(planned_spec)
    plan["selection_dependencies"] = dependencies
    plan["scientific_conditions_resolved"] = dependencies[
        "scientific_conditions_resolved"
    ]
    return plan


def audit(root: Path) -> dict[str, Any]:
    validation_rows: list[dict[str, Any]] = []
    for relative in VALIDATIONS:
        spec = load_validation_spec(root / relative, repo_root=root)
        if spec.profile != "main":
            raise ValueError(f"main validation has non-main profile: {relative}")
        target = spec.target_reference
        validation_rows.append(
            {
                "path": relative,
                "parse_status": "pass",
                "main_marker": True,
                "profile": spec.profile,
                "sample_count": spec.samples.total_samples,
                "reference_candidate_count": len(
                    target.reference_nx_candidates
                ),
                "numerical_condition_candidate_count": (
                    len(target.time_candidates)
                    if hasattr(target, "time_candidates")
                    else 1
                ),
                "expected_artifacts": [
                    "resolved_spec.json",
                    "checks.json",
                    "certificate.json",
                    "reference_convergence.csv",
                    "master_initial_conditions.pt",
                    "manifest.json",
                ],
            }
        )

    dataset_rows: list[dict[str, Any]] = []
    for relative in DATASETS:
        spec = load_dataset_spec(root / relative, repo_root=root)
        dataset_rows.append(
            {
                "path": relative,
                "parse_status": "pass",
                "main_marker": True,
                "reference_nx": spec.reference_nx,
                "binding_kind": spec.binding.kind,
                "upstream_validation_spec": str(
                    spec.validation_spec.relative_to(root)
                ),
                "expected_artifacts": [
                    "resolved_spec.json",
                    "metadata.json",
                    "binding_proof.json",
                    "dataset.pt",
                    "manifest.json",
                ],
            }
        )

    study_rows: list[dict[str, Any]] = []
    study_specs: dict[Path, Any] = {}
    for relative in STUDIES:
        spec = load_study_spec(root / relative, repo_root=root)
        study_specs[(root / relative).resolve()] = spec
        if spec.profile != "main":
            raise ValueError(f"main study has non-main profile: {relative}")
        plan = _plan_study_read_only(spec, root=root)
        candidate_upper_bound = sum(
            int(case["candidate_upper_bound"]) for case in plan["cases"]
        )
        counts = _random_feature_counts(spec)
        study_rows.append(
            {
                "path": relative,
                "name": spec.name,
                "parse_status": "pass",
                "plan_status": "pass",
                "main_marker": True,
                "profile": spec.profile,
                "case_count": plan["case_count"],
                "candidate_upper_bound": candidate_upper_bound,
                "planned_cartesian_cell_count": plan[
                    "planned_cartesian_cell_count"
                ],
                **counts,
                "random_feature_evaluation_realization_upper_bound": (
                    plan["case_count"]
                    * counts[
                        "random_feature_evaluation_seed_count_per_case"
                    ]
                ),
                "upstream_dependency_status": plan[
                    "selection_dependencies"
                ]["status"],
                "scientific_conditions_resolved": plan[
                    "scientific_conditions_resolved"
                ],
                "filesystem_mutation": plan["filesystem_mutation"],
                "expected_output_tables_artifacts": [
                    "resolved_study.json",
                    "dataset_reference.json",
                    "validation_trials.csv",
                    "selection_record.json",
                    "frozen_models.pt",
                    "frozen_evaluation_plan.json",
                    "test_metrics.csv",
                    "random_feature_seed_metrics.csv",
                    "random_feature_ensemble_metrics.csv",
                    "run_summary.json",
                    "manifest.json",
                ],
            }
        )

    report_rows: list[dict[str, Any]] = []
    for relative in REPORTS:
        spec = load_report_spec(root / relative, repo_root=root)
        if spec.profile != "main":
            raise ValueError(f"main report has non-main profile: {relative}")
        source_statuses = []
        for source in spec.sources:
            source_spec = study_specs.get(source.study_spec.resolve())
            if source_spec is None:
                source_spec = load_study_spec(
                    source.study_spec,
                    repo_root=root,
                )
            source_statuses.append(
                {
                    "source_id": source.id,
                    "study": source_spec.name,
                    **_source_run_status(source_spec, root=root),
                }
            )
        report_rows.append(
            {
                "path": relative,
                "name": spec.name,
                "parse_status": "pass",
                "main_marker": True,
                "profile": spec.profile,
                "source_count": len(spec.sources),
                "reporter_count": len(spec.reporters),
                "upstream_dependency_status": source_statuses,
                "expected_output_tables_artifacts": [
                    "resolved_report_spec.json",
                    "source_references.json",
                    "machine_readable_tables/*.csv",
                    "formatted_tables/*",
                    "figures/*",
                    "report_summary.json",
                    "manifest.json",
                ],
            }
        )

    digital_rows: list[dict[str, Any]] = []
    for relative in DIGITAL_BASELINES:
        spec = load_digital_baseline_spec(root / relative, repo_root=root)
        if spec.profile != "main":
            raise ValueError(
                f"main digital baseline has non-main profile: {relative}"
            )
        dataset_spec = load_dataset_spec(spec.dataset_spec, repo_root=root)
        validation_spec = load_validation_spec(
            dataset_spec.validation_spec,
            repo_root=root,
        )
        plan = plan_digital_baseline(
            spec,
            n_train=int(validation_spec.samples.n_train),
        )
        physical_spec = study_specs.get(
            spec.physical_comparison.source_study_spec.resolve()
        )
        if physical_spec is None:
            physical_spec = load_study_spec(
                spec.physical_comparison.source_study_spec,
                repo_root=root,
            )
        digital_rows.append(
            {
                "path": relative,
                "name": spec.name,
                "parse_status": "pass",
                "plan_status": "pass",
                "main_marker": True,
                "profile": spec.profile,
                **{
                    key: value
                    for key, value in plan.items()
                    if key
                    not in {
                        "schema_version",
                        "name",
                        "profile",
                        "filesystem_mutation",
                        "main_execution",
                    }
                },
                "upstream_physical_source_status": _source_run_status(
                    physical_spec,
                    root=root,
                ),
                "filesystem_mutation": False,
                "expected_output_tables_artifacts": [
                    "resolved_spec.json",
                    "dataset_reference.json",
                    "physical_source_reference.json",
                    "selection_training_history.csv",
                    "selection_record.json",
                    "evaluation_training_history.csv",
                    "frozen_checkpoints.pt",
                    "frozen_evaluation_plan.json",
                    "test_seed_metrics.csv",
                    "test_metrics.csv",
                    "prediction_ensemble_metrics.csv",
                    "fairness_comparison.csv",
                    "training_compute.json",
                    "run_summary.json",
                    "manifest.json",
                ],
            }
        )

    return {
        "schema_version": "pol-production-plan-audit-v2",
        "status": "pass",
        "mode": "read_only_plan",
        "main_execution": False,
        "filesystem_mutation": False,
        "validation_specs": validation_rows,
        "dataset_specs": dataset_rows,
        "study_specs": study_rows,
        "digital_baseline_specs": digital_rows,
        "report_specs": report_rows,
    }


def main() -> int:
    root = Path.cwd().resolve()
    try:
        result = audit(root)
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        print(f"main plan audit: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
