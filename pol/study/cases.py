from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Any, Mapping

from pol.config.models import (
    ReadoutStabilityNoiseDiagnosticSpec,
    StudySpec,
    TrialSpec,
)
from pol.runtime.device import execution_device_policy
from pol.runtime.environment import numerical_environment_fingerprint
from .overrides import apply_trial_overrides


@dataclass(frozen=True)
class StudyCase:
    case_id: str
    variant_id: str
    variant_display_name: str
    global_values: dict[str, Any]
    trial: TrialSpec
    search: Any


def scientific_study_spec(
    spec: StudySpec,
    *,
    selection_source_provenance: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    payload = spec.model_dump(mode="json")
    payload.pop("output_root", None)
    payload.pop("artifact_root", None)
    payload.pop("dataset_spec", None)
    for variant in payload["variants"]:
        declaration = variant.pop("selection_source", None)
        if declaration is None:
            continue
        variant_id = str(variant["id"])
        if selection_source_provenance is None:
            declaration.pop("source_study_spec", None)
            variant["selection_source"] = declaration
            continue
        provenance = selection_source_provenance.get(variant_id)
        if provenance is None:
            raise ValueError(
                f"resolved selection provenance is missing for variant {variant_id}"
            )
        variant["selection_source"] = dict(provenance)
    return payload


def build_study_run_identity(
    spec: StudySpec,
    *,
    dataset: Any,
    selection_source_provenance: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "pol-study-run-identity-v15",
        **execution_device_policy(),
        "environment": numerical_environment_fingerprint(),
        "study": scientific_study_spec(
            spec,
            selection_source_provenance=selection_source_provenance,
        ),
        "dataset_artifact_id": dataset.artifact_id,
        "dataset_split_hash": dataset.split_hash,
        "dataset_binding_kind": dataset.binding_kind,
        "dataset_binding_status": dataset.binding_status,
        "dataset_target_reference_validation_status": (
            dataset.target_reference_validation_status
        ),
        "dataset_binding_proof_hash": dataset.binding_proof_hash,
    }


def _axis_combinations(spec: StudySpec) -> list[dict[str, Any]]:
    if not spec.global_axes:
        return [{}]
    paths = [axis.path for axis in spec.global_axes]
    return [
        dict(zip(paths, values, strict=True))
        for values in itertools.product(*(axis.values for axis in spec.global_axes))
    ]


def build_cases(
    spec: StudySpec,
) -> tuple[list[StudyCase], list[dict[str, Any]]]:
    cases: list[StudyCase] = []
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
                StudyCase(
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


def preflight_grid_searches(
    cases: list[StudyCase],
    *,
    invalid_policy: str,
) -> None:
    """Validate declared grid cells before any dataset or feature computation."""
    if invalid_policy != "error":
        return
    for case in cases:
        if case.search.kind != "grid":
            continue
        paths = [axis.path for axis in case.search.axes]
        for values in itertools.product(
            *(axis.values for axis in case.search.axes)
        ):
            overrides = dict(zip(paths, values, strict=True))
            try:
                apply_trial_overrides(case.trial, overrides)
            except ValueError as exc:
                raise ValueError(
                    f"invalid declared grid cell for {case.case_id}: "
                    f"{overrides}: {exc}"
                ) from exc


def _candidate_upper_bound(case: StudyCase) -> int:
    search = case.search
    if search.kind == "static":
        return 1
    if search.kind == "grid":
        count = 1
        for axis in search.axes:
            count *= len(axis.values)
        return count
    return (
        sum(len(axis.values) for axis in search.axes)
        * (search.rounds + 1)
        * len(case.trial.readouts)
    )


def _case_workload(
    case: StudyCase,
    *,
    candidate_upper_bound: int,
    canonical_n_train: int | None,
) -> dict[str, Any]:
    trial = case.trial
    affine_zeta_fits = 0
    affine_zero_zeta_svds = 0
    random_map_count = 0
    random_lift_count = 0
    random_selection_ridge_fits = 0
    random_selection_zero_svds = 0
    random_selected_evaluation_fits = 0
    random_selected_evaluation_zero_svds = 0
    random_eager_evaluation_fits = 0
    maximum_lifted_dimension: int | None = None
    random_readout_count = 0
    for readout in trial.readouts:
        if readout.kind == "affine_ridge":
            affine_zeta_fits += (
                candidate_upper_bound * len(readout.zetas)
            )
            affine_zero_zeta_svds += candidate_upper_bound * sum(
                float(zeta) == 0.0 for zeta in readout.zetas
            )
        elif readout.kind == "random_feature_ridge":
            random_readout_count += 1
            structure_count = (
                len(readout.widths)
                * len(readout.weight_scales)
                * len(readout.bias_scales)
            )
            map_count = (
                candidate_upper_bound
                * structure_count
                * len(readout.selection_seeds)
            )
            selection_fit_count = map_count * len(readout.zetas)
            selected_evaluation_count = len(readout.evaluation_seeds)
            eager_evaluation_count = (
                candidate_upper_bound * selected_evaluation_count
            )
            random_map_count += map_count
            random_lift_count += 2 * map_count
            random_selection_ridge_fits += selection_fit_count
            random_selection_zero_svds += map_count * sum(
                float(zeta) == 0.0 for zeta in readout.zetas
            )
            random_selected_evaluation_fits += selected_evaluation_count
            random_eager_evaluation_fits += eager_evaluation_count
            if any(float(zeta) == 0.0 for zeta in readout.zetas):
                random_selected_evaluation_zero_svds += (
                    selected_evaluation_count
                )
            lifted = int(trial.feature.observation.J) + max(
                int(width) for width in readout.widths
            )
            maximum_lifted_dimension = (
                lifted
                if maximum_lifted_dimension is None
                else max(maximum_lifted_dimension, lifted)
            )
    n_train = (
        int(trial.training_subset.n_train)
        if trial.training_subset is not None
        else canonical_n_train
    )
    return {
        "schema_version": "pol-study-workload-case-v1",
        "case_id": case.case_id,
        "candidate_trial_upper_bound": candidate_upper_bound,
        "feature_state_solve_upper_bound": candidate_upper_bound,
        "configured_readout_count": len(trial.readouts),
        "candidate_readout_evaluation_upper_bound": (
            candidate_upper_bound * len(trial.readouts)
        ),
        "affine": {
            "zeta_fit_count": affine_zeta_fits,
            "zero_zeta_svd_count": affine_zero_zeta_svds,
        },
        "random_feature": {
            "configured_readout_count": random_readout_count,
            "unique_random_map_count": random_map_count,
            "train_validation_lift_count": random_lift_count,
            "selection_seed_ridge_fit_count": (
                random_selection_ridge_fits
            ),
            "ridge_fit_count": random_selection_ridge_fits,
            "selected_candidate_evaluation_member_fit_count": (
                random_selected_evaluation_fits
            ),
            "eager_legacy_evaluation_member_fit_count": (
                random_eager_evaluation_fits
            ),
            "lazy_total_ridge_fit_count": (
                random_selection_ridge_fits
                + random_selected_evaluation_fits
            ),
            "eager_legacy_total_ridge_fit_count": (
                random_selection_ridge_fits
                + random_eager_evaluation_fits
            ),
            "selection_zero_zeta_svd_count": (
                random_selection_zero_svds
            ),
            "selected_evaluation_zero_zeta_svd_upper_bound": (
                random_selected_evaluation_zero_svds
            ),
            "zero_zeta_svd_upper_bound": (
                random_selection_zero_svds
                + random_selected_evaluation_zero_svds
            ),
            "maximum_lifted_dimension": maximum_lifted_dimension,
            "maximum_target_dimension": int(trial.output.q),
            "maximum_training_sample_count": n_train,
        },
        "convergence": {
            "feature_state_solve_upper_bound": 0,
            "comparison_row_upper_bound": 0,
        },
    }


def _sum_nested(
    workloads: list[dict[str, Any]],
    section: str,
    field: str,
) -> int:
    return sum(int(item[section][field]) for item in workloads)


def _study_workload(
    spec: StudySpec,
    *,
    cases: list[StudyCase],
    candidate_upper_bounds: list[int],
    canonical_n_train: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_case = [
        _case_workload(
            case,
            candidate_upper_bound=candidate_upper_bound,
            canonical_n_train=canonical_n_train,
        )
        for case, candidate_upper_bound in zip(
            cases,
            candidate_upper_bounds,
            strict=True,
        )
    ]
    convergence_solves = 0
    convergence_rows = 0
    if spec.convergence is not None:
        base_count = len(spec.convergence.n_sur_candidates)
        attempts = int(spec.convergence.max_auto_reruns) + 1
        per_case_solves = sum(base_count + rerun for rerun in range(attempts))
        per_case_rows = sum(
            base_count + rerun - 1 for rerun in range(attempts)
        )
        convergence_solves = len(cases) * per_case_solves
        convergence_rows = len(cases) * per_case_rows
        for workload in per_case:
            workload["convergence"] = {
                "feature_state_solve_upper_bound": per_case_solves,
                "comparison_row_upper_bound": per_case_rows,
            }
    noise_coordinate_count = 0
    noise_feature_state_solve_count = 0
    for diagnostic in spec.diagnostics:
        if isinstance(diagnostic, ReadoutStabilityNoiseDiagnosticSpec):
            selected_model_count = sum(
                len(case.trial.readouts) for case in cases
            )
            noise_coordinate_count += (
                selected_model_count
                * len(diagnostic.levels)
                * int(diagnostic.repeats)
            )
            noise_feature_state_solve_count += selected_model_count
    maximum_training_counts = [
        item["random_feature"]["maximum_training_sample_count"]
        for item in per_case
        if item["random_feature"]["configured_readout_count"] > 0
    ]
    training_count_resolved = all(
        value is not None for value in maximum_training_counts
    )
    workload = {
        "schema_version": "pol-study-workload-plan-v1",
        "status": (
            "resolved"
            if training_count_resolved
            else "unresolved_canonical_training_sample_count"
        ),
        "count_semantics": "declared_operation_upper_bounds_v1",
        "case_count": len(cases),
        "candidate_trial_upper_bound": sum(candidate_upper_bounds),
        "feature_state_solve_upper_bound": sum(candidate_upper_bounds),
        "configured_readout_count": sum(
            len(case.trial.readouts) for case in cases
        ),
        "candidate_readout_evaluation_upper_bound": sum(
            candidate * len(case.trial.readouts)
            for case, candidate in zip(
                cases,
                candidate_upper_bounds,
                strict=True,
            )
        ),
        "affine": {
            "zeta_fit_count": _sum_nested(
                per_case, "affine", "zeta_fit_count"
            ),
            "zero_zeta_svd_count": _sum_nested(
                per_case, "affine", "zero_zeta_svd_count"
            ),
        },
        "random_feature": {
            "unique_random_map_count": _sum_nested(
                per_case, "random_feature", "unique_random_map_count"
            ),
            "train_validation_lift_count": _sum_nested(
                per_case,
                "random_feature",
                "train_validation_lift_count",
            ),
            "selection_seed_ridge_fit_count": _sum_nested(
                per_case,
                "random_feature",
                "selection_seed_ridge_fit_count",
            ),
            "ridge_fit_count": _sum_nested(
                per_case,
                "random_feature",
                "ridge_fit_count",
            ),
            "selected_candidate_evaluation_member_fit_count": _sum_nested(
                per_case,
                "random_feature",
                "selected_candidate_evaluation_member_fit_count",
            ),
            "eager_legacy_evaluation_member_fit_count": _sum_nested(
                per_case,
                "random_feature",
                "eager_legacy_evaluation_member_fit_count",
            ),
            "lazy_total_ridge_fit_count": _sum_nested(
                per_case,
                "random_feature",
                "lazy_total_ridge_fit_count",
            ),
            "eager_legacy_total_ridge_fit_count": _sum_nested(
                per_case,
                "random_feature",
                "eager_legacy_total_ridge_fit_count",
            ),
            "selection_zero_zeta_svd_count": _sum_nested(
                per_case,
                "random_feature",
                "selection_zero_zeta_svd_count",
            ),
            "selected_evaluation_zero_zeta_svd_upper_bound": _sum_nested(
                per_case,
                "random_feature",
                "selected_evaluation_zero_zeta_svd_upper_bound",
            ),
            "zero_zeta_svd_upper_bound": _sum_nested(
                per_case,
                "random_feature",
                "zero_zeta_svd_upper_bound",
            ),
            "maximum_lifted_dimension": max(
                (
                    int(value)
                    for value in (
                        item["random_feature"][
                            "maximum_lifted_dimension"
                        ]
                        for item in per_case
                    )
                    if value is not None
                ),
                default=None,
            ),
            "maximum_target_dimension": max(
                (
                    int(item["random_feature"][
                        "maximum_target_dimension"
                    ])
                    for item in per_case
                    if item["random_feature"][
                        "configured_readout_count"
                    ]
                    > 0
                ),
                default=None,
            ),
            "maximum_training_sample_count": (
                max(int(value) for value in maximum_training_counts)
                if maximum_training_counts and training_count_resolved
                else None
            ),
        },
        "convergence": {
            "feature_state_solve_upper_bound": convergence_solves,
            "comparison_row_upper_bound": convergence_rows,
        },
        "diagnostics": {
            "noise_coordinate_evaluation_upper_bound": (
                noise_coordinate_count
            ),
            "feature_state_solve_upper_bound": (
                noise_feature_state_solve_count
            ),
        },
    }
    return workload, per_case


def plan_study(
    spec: StudySpec,
    *,
    canonical_n_train: int | None = None,
) -> dict[str, Any]:
    cases, skipped = build_cases(spec)
    preflight_grid_searches(
        cases,
        invalid_policy=spec.execution.invalid_trial_policy,
    )
    planned: list[dict[str, Any]] = []
    candidate_upper_bounds = [
        _candidate_upper_bound(case) for case in cases
    ]
    workload, case_workloads = _study_workload(
        spec,
        cases=cases,
        candidate_upper_bounds=candidate_upper_bounds,
        canonical_n_train=canonical_n_train,
    )
    for case, candidate_upper_bound, case_workload in zip(
        cases,
        candidate_upper_bounds,
        case_workloads,
        strict=True,
    ):
        search = case.search
        planned.append(
            {
                "case_id": case.case_id,
                "variant_id": case.variant_id,
                "global_values": case.global_values,
                "search_kind": search.kind,
                "candidate_upper_bound": candidate_upper_bound,
                "planned_cartesian_cell_count": (
                    candidate_upper_bound
                    if search.kind == "grid"
                    else None
                ),
                "search_axes": (
                    [
                        {
                            "path": axis.path,
                            "values": list(axis.values),
                        }
                        for axis in search.axes
                    ]
                    if search.kind != "static"
                    else []
                ),
                "readout_ids": [
                    readout.id for readout in case.trial.readouts
                ],
                "workload": case_workload,
            }
        )
    return {
        "schema_version": "pol-study-plan-v4",
        "study": spec.name,
        "profile": spec.profile,
        "case_count": len(cases),
        "planned_cartesian_cell_count": sum(
            int(case["planned_cartesian_cell_count"] or 0)
            for case in planned
        ),
        "cases": planned,
        "workload": workload,
        "skipped": skipped,
        "filesystem_mutation": False,
    }
