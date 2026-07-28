from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Any

from pol.config.models import (
    CoordinateSearchSpec,
    GridSearchSpec,
    SearchSpec,
    StaticSearchSpec,
    TrialSpec,
)
from .overrides import apply_trial_overrides
from .trial import CandidateEvaluation, TrialEngine


@dataclass(frozen=True)
class SearchOutcome:
    evaluations: tuple[CandidateEvaluation, ...]
    selected_by_readout: dict[str, str]
    stages_by_candidate: dict[str, tuple[str, ...]]
    skipped: tuple[dict[str, Any], ...]


class _Registry:
    def __init__(self, engine: TrialEngine, invalid_policy: str) -> None:
        self.engine = engine
        self.invalid_policy = invalid_policy
        self.evaluations: dict[str, CandidateEvaluation] = {}
        self.stages: dict[str, list[str]] = {}
        self.skipped: list[dict[str, Any]] = []

    def evaluate(self, trial: TrialSpec, stage: str) -> CandidateEvaluation:
        result = self.engine.evaluate_selection(trial)
        self.evaluations[result.candidate_id] = result
        self.stages.setdefault(result.candidate_id, [])
        if stage not in self.stages[result.candidate_id]:
            self.stages[result.candidate_id].append(stage)
        return result

    def apply(
        self,
        base: TrialSpec,
        overrides: dict[str, Any],
        stage: str,
    ) -> CandidateEvaluation | None:
        try:
            trial = apply_trial_overrides(base, overrides)
            return self.evaluate(trial, stage)
        except ValueError as exc:
            if self.invalid_policy == "error":
                raise
            self.skipped.append(
                {"stage": stage, "overrides": overrides, "reason": str(exc)}
            )
            return None


def _select(
    evaluations: list[CandidateEvaluation],
    *,
    readout_id: str,
    metric: str,
    tolerance: float,
) -> CandidateEvaluation:
    candidates = [value for value in evaluations if readout_id in value.rows]
    if not candidates:
        raise ValueError(f"no valid candidates for readout {readout_id}")
    best = min(float(value.rows[readout_id][metric]) for value in candidates)
    return next(
        value
        for value in candidates
        if float(value.rows[readout_id][metric]) <= best + tolerance
    )


def _grid_overrides(search: GridSearchSpec) -> list[dict[str, Any]]:
    paths = [axis.path for axis in search.axes]
    return [
        dict(zip(paths, values, strict=True))
        for values in itertools.product(*(axis.values for axis in search.axes))
    ]


def run_search(
    engine: TrialEngine,
    base: TrialSpec,
    search: SearchSpec,
    *,
    metric: str,
    tolerance: float,
    invalid_policy: str,
) -> SearchOutcome:
    registry = _Registry(engine, invalid_policy)
    readout_ids = [readout.id for readout in base.readouts]
    selected: dict[str, str] = {}
    if isinstance(search, StaticSearchSpec):
        result = registry.evaluate(base, "static")
        selected = {readout_id: result.candidate_id for readout_id in readout_ids}
    elif isinstance(search, GridSearchSpec):
        evaluated: list[CandidateEvaluation] = []
        for index, overrides in enumerate(_grid_overrides(search)):
            result = registry.apply(base, overrides, f"grid:{index}")
            if result is not None:
                evaluated.append(result)
        for readout_id in readout_ids:
            chosen = _select(
                evaluated,
                readout_id=readout_id,
                metric=metric,
                tolerance=tolerance,
            )
            selected[readout_id] = chosen.candidate_id
    elif isinstance(search, CoordinateSearchSpec):
        first, second = search.axes
        for readout_id in readout_ids:
            current_first = first.anchor
            current_second = second.anchor
            first_candidates: list[CandidateEvaluation] = []
            for index, value in enumerate(first.values):
                result = registry.apply(
                    base,
                    {first.path: value, second.path: current_second},
                    f"coordinate:{readout_id}:axis0:round0:{index}",
                )
                if result is not None:
                    first_candidates.append(result)
            chosen_first = _select(
                first_candidates,
                readout_id=readout_id,
                metric=metric,
                tolerance=tolerance,
            )
            # Obtain the selected first-axis value from the validated model.
            payload = chosen_first.trial.model_dump(mode="python")
            current_first = _get_dotted(payload, first.path)

            second_candidates: list[CandidateEvaluation] = []
            for index, value in enumerate(second.values):
                result = registry.apply(
                    base,
                    {first.path: current_first, second.path: value},
                    f"coordinate:{readout_id}:axis1:round0:{index}",
                )
                if result is not None:
                    second_candidates.append(result)
            chosen_second = _select(
                second_candidates,
                readout_id=readout_id,
                metric=metric,
                tolerance=tolerance,
            )
            current_second = _get_dotted(
                chosen_second.trial.model_dump(mode="python"), second.path
            )
            selected_result = chosen_second
            for round_index in range(1, search.rounds + 1):
                first_candidates = []
                for index, value in enumerate(first.values):
                    result = registry.apply(
                        base,
                        {first.path: value, second.path: current_second},
                        f"coordinate:{readout_id}:axis0:round{round_index}:{index}",
                    )
                    if result is not None:
                        first_candidates.append(result)
                selected_result = _select(
                    first_candidates,
                    readout_id=readout_id,
                    metric=metric,
                    tolerance=tolerance,
                )
                current_first = _get_dotted(
                    selected_result.trial.model_dump(mode="python"), first.path
                )
                second_candidates = []
                for index, value in enumerate(second.values):
                    result = registry.apply(
                        base,
                        {first.path: current_first, second.path: value},
                        f"coordinate:{readout_id}:axis1:round{round_index}:{index}",
                    )
                    if result is not None:
                        second_candidates.append(result)
                selected_result = _select(
                    second_candidates,
                    readout_id=readout_id,
                    metric=metric,
                    tolerance=tolerance,
                )
                current_second = _get_dotted(
                    selected_result.trial.model_dump(mode="python"), second.path
                )
            selected[readout_id] = selected_result.candidate_id
    else:
        raise TypeError(f"unsupported search type: {type(search).__name__}")

    return SearchOutcome(
        evaluations=tuple(registry.evaluations.values()),
        selected_by_readout=selected,
        stages_by_candidate={key: tuple(value) for key, value in registry.stages.items()},
        skipped=tuple(registry.skipped),
    )


def _get_dotted(root: dict[str, Any], path: str) -> Any:
    current: Any = root
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ValueError(f"path does not exist in trial: {path}")
        current = current[part]
    return current
