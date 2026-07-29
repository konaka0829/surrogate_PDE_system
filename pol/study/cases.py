from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Any

from pol.config.models import StudySpec, TrialSpec
from .overrides import apply_trial_overrides


@dataclass(frozen=True)
class StudyCase:
    case_id: str
    variant_id: str
    variant_display_name: str
    global_values: dict[str, Any]
    trial: TrialSpec
    search: Any


def scientific_study_spec(spec: StudySpec) -> dict[str, Any]:
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


def plan_study(spec: StudySpec) -> dict[str, Any]:
    cases, skipped = build_cases(spec)
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
                "readout_ids": [
                    readout.id for readout in case.trial.readouts
                ],
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
