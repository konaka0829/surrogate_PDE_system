from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Any, Mapping

from pol.config.models import StudySpec, TrialSpec
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
        "schema_version": "pol-study-run-identity-v13",
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


def plan_study(spec: StudySpec) -> dict[str, Any]:
    cases, skipped = build_cases(spec)
    preflight_grid_searches(
        cases,
        invalid_policy=spec.execution.invalid_trial_policy,
    )
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
            }
        )
    return {
        "schema_version": "pol-study-plan-v3",
        "study": spec.name,
        "profile": spec.profile,
        "case_count": len(cases),
        "planned_cartesian_cell_count": sum(
            int(case["planned_cartesian_cell_count"] or 0)
            for case in planned
        ),
        "cases": planned,
        "skipped": skipped,
        "filesystem_mutation": False,
    }
