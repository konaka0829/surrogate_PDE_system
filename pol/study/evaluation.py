from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Any, Mapping

from scipy.stats import t as student_t
import torch

from pol.config.models import TrialSpec
from pol.learning.metrics import (
    fourier_prediction_metrics,
    fourier_representation_floor,
)


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate_id: str
    trial: TrialSpec
    rows: dict[str, dict[str, Any]]
    frozen_models: dict[str, dict[str, Any]]
    inner_selections: dict[str, dict[str, Any]]
    feature_cache_id: str


@dataclass(frozen=True)
class ValidationReadoutEvaluation:
    row: dict[str, Any]
    inner_selection: dict[str, Any]


@dataclass(frozen=True)
class TestEvaluation:
    primary_row: dict[str, Any]
    seed_rows: tuple[dict[str, Any], ...]
    ensemble_row: dict[str, Any] | None


@dataclass(frozen=True)
class FrozenPredictions:
    """Predictions from a frozen deterministic model or random realizations."""

    single_model_prediction: torch.Tensor | None
    per_seed_predictions: tuple[tuple[int, torch.Tensor], ...]

    def prediction_ensemble(self) -> torch.Tensor:
        """Explicitly form the prediction-average ensemble over random seeds."""
        if not self.per_seed_predictions:
            raise ValueError("prediction ensemble requires per-seed predictions")
        return torch.stack(
            [prediction for _, prediction in self.per_seed_predictions], dim=0
        ).mean(0)


def trial_parameters(trial: TrialSpec) -> dict[str, Any]:
    if trial.feature.kind == "static_input":
        return {
            "n_tar": int(trial.input.n_tar),
            "n_sur": int(trial.feature.n_sur),
            "J": int(trial.feature.observation.J),
            "q": int(trial.output.q),
            "feature_system": "static_input",
            "feature_nu": None,
            "feature_alpha": None,
            "feature_beta": None,
            "feature_time": 0.0,
            "feature_solver": "none",
        }
    evolution = trial.feature.evolution
    if evolution is None:  # defensive guard for manually constructed objects
        raise ValueError("pde_dynamics feature has no evolution")
    system = evolution.system
    return {
        "n_tar": int(trial.input.n_tar),
        "n_sur": int(trial.feature.n_sur),
        "J": int(trial.feature.observation.J),
        "q": int(trial.output.q),
        "feature_system": system.kind,
        "feature_nu": float(getattr(system, "nu", float("nan"))),
        "feature_alpha": (
            None if not hasattr(system, "alpha") else float(getattr(system, "alpha"))
        ),
        "feature_beta": (
            None if not hasattr(system, "beta") else float(getattr(system, "beta"))
        ),
        "feature_time": float(evolution.time),
        "feature_solver": str(getattr(system, "solver", "spectral_exact")),
    }


def prefix_metrics(
    metrics: Mapping[str, float],
    prefix: str,
) -> dict[str, float]:
    return {f"{prefix}_{key}": float(value) for key, value in metrics.items()}


def mean_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        raise ValueError("cannot average an empty metric list")
    return {
        key: sum(float(item[key]) for item in items) / len(items)
        for key in items[0]
    }


def summarize_independent_seed_metrics(
    items: list[dict[str, float]],
) -> dict[str, float]:
    """Summarize per-seed metrics with a two-sided Student-t mean interval."""
    if len(items) < 2:
        raise ValueError("at least two seed metric rows are required")
    confidence_level = 0.95
    keys = tuple(items[0])
    if any(tuple(item) != keys for item in items[1:]):
        raise ValueError("per-seed metric rows must have identical ordered keys")
    count = len(items)
    critical = float(student_t.ppf(0.5 + confidence_level / 2.0, count - 1))
    summary: dict[str, float] = {}
    for key in keys:
        values = [float(item[key]) for item in items]
        mean = math.fsum(values) / count
        variance = math.fsum((value - mean) ** 2 for value in values) / (count - 1)
        standard_deviation = math.sqrt(variance)
        margin = critical * standard_deviation / math.sqrt(count)
        summary[key] = mean
        summary[f"{key}_seed_mean"] = mean
        summary[f"{key}_seed_std"] = standard_deviation
        summary[f"{key}_seed_ci95_low"] = mean - margin
        summary[f"{key}_seed_ci95_high"] = mean + margin
    return summary


def evaluate_coefficients(
    prediction: torch.Tensor,
    target_coefficients: torch.Tensor,
    data_target_field: torch.Tensor,
    reference_target_field: torch.Tensor,
    *,
    n_tar: int,
    n_ref: int,
    domain_length: float,
) -> dict[str, float]:
    return fourier_prediction_metrics(
        prediction,
        target_coefficients,
        data_target_field,
        reference_target_field,
        n_data=n_tar,
        n_reference=n_ref,
        domain_length=domain_length,
    )


def representation_floor(
    coefficients: torch.Tensor,
    data_target_field: torch.Tensor,
    reference_target_field: torch.Tensor,
    *,
    n_tar: int,
    n_ref: int,
    domain_length: float,
) -> dict[str, float]:
    return fourier_representation_floor(
        coefficients,
        data_target_field,
        reference_target_field,
        n_data=n_tar,
        n_reference=n_ref,
        domain_length=domain_length,
    )


def build_validation_evaluation(
    *,
    candidate_id: str,
    readout_id: str,
    readout_kind: str,
    trial: TrialSpec,
    readout_values: dict[str, Any],
    inner_selection: dict[str, Any],
    floor_metrics: Mapping[str, float],
) -> ValidationReadoutEvaluation:
    # Preserve any alias between the selected metric dictionary and its
    # candidate-history entry without mutating the readout result supplied by
    # the caller. That alias is part of the established selection payload.
    values, bound_inner_selection = copy.deepcopy(
        (readout_values, inner_selection)
    )
    values.update(prefix_metrics(floor_metrics, "validation"))
    return ValidationReadoutEvaluation(
        row={
            "candidate_id": candidate_id,
            "readout_id": readout_id,
            "readout_kind": readout_kind,
            **trial_parameters(trial),
            **values,
        },
        inner_selection=bound_inner_selection,
    )


def build_test_evaluation(
    predictions: FrozenPredictions,
    *,
    model_kind: str,
    candidate_id: str,
    readout_id: str,
    trial: TrialSpec,
    feature_cache_id: str,
    target_coefficients: torch.Tensor,
    data_target_field: torch.Tensor,
    reference_target_field: torch.Tensor,
    n_tar: int,
    n_ref: int,
    domain_length: float,
    floor_metrics: Mapping[str, float],
    direct_diagnostic: Mapping[str, Any] | None = None,
) -> TestEvaluation:
    base_row = {
        "candidate_id": candidate_id,
        "readout_id": readout_id,
        **trial_parameters(trial),
        "feature_cache_id": feature_cache_id,
    }
    if direct_diagnostic is not None:
        base_row.update(direct_diagnostic)

    if model_kind == "random_feature_ridge":
        seed_rows: list[dict[str, Any]] = []
        seed_metrics: list[dict[str, float]] = []
        for seed, prediction in predictions.per_seed_predictions:
            seed_metric = evaluate_coefficients(
                prediction,
                target_coefficients,
                data_target_field,
                reference_target_field,
                n_tar=n_tar,
                n_ref=n_ref,
                domain_length=domain_length,
            )
            prefixed_seed_metric = {
                **prefix_metrics(seed_metric, "test"),
                **prefix_metrics(floor_metrics, "test"),
            }
            seed_metrics.append(prefixed_seed_metric)
            seed_rows.append(
                {
                    **base_row,
                    "seed": seed,
                    "test_result_kind": "independent_seed_realization",
                    **prefixed_seed_metric,
                }
            )
        primary_row = {
            **base_row,
            "test_result_kind": "independent_seed_metric_summary",
            "test_seed_count": len(seed_rows),
            "test_seed_std_ddof": 1,
            "test_confidence_level": 0.95,
            "test_confidence_interval_method": "student_t",
            **summarize_independent_seed_metrics(seed_metrics),
        }
        ensemble_prediction = predictions.prediction_ensemble()
        ensemble_metrics = evaluate_coefficients(
            ensemble_prediction,
            target_coefficients,
            data_target_field,
            reference_target_field,
            n_tar=n_tar,
            n_ref=n_ref,
            domain_length=domain_length,
        )
        ensemble_row = {
            **base_row,
            "test_result_kind": "prediction_ensemble",
            "ensemble_member_count": len(seed_rows),
            **prefix_metrics(ensemble_metrics, "test_ensemble"),
        }
        return TestEvaluation(
            primary_row=primary_row,
            seed_rows=tuple(seed_rows),
            ensemble_row=ensemble_row,
        )

    if predictions.single_model_prediction is None:
        raise ValueError("deterministic frozen model returned no prediction")
    metrics = evaluate_coefficients(
        predictions.single_model_prediction,
        target_coefficients,
        data_target_field,
        reference_target_field,
        n_tar=n_tar,
        n_ref=n_ref,
        domain_length=domain_length,
    )
    primary_row = {
        **base_row,
        "test_result_kind": "single_model",
        **prefix_metrics(metrics, "test"),
        **prefix_metrics(floor_metrics, "test"),
    }
    return TestEvaluation(
        primary_row=primary_row,
        seed_rows=(),
        ensemble_row=None,
    )
