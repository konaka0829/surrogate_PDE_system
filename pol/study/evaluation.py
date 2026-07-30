from __future__ import annotations

import copy
from dataclasses import dataclass, field
import json
import math
from typing import Any, Mapping

from scipy.stats import t as student_t
import torch

from pol.config.models import TrialSpec
from pol.learning.metrics import (
    fourier_prediction_metrics,
    fourier_representation_floor,
)
from pol.runtime.hashing import stable_object_hash, tensor_sha256
from .training_subsets import training_subset_result_fields


READOUT_SELECTION_FIELDS = (
    "selected_ridge_zeta",
    "selected_random_feature_activation",
    "selected_random_feature_width",
    "selected_random_feature_weight_scale",
    "selected_random_feature_bias_scale",
    "selected_random_feature_evaluation_seed_count",
    "selected_random_feature_evaluation_seeds_hash",
)


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate_id: str
    trial: TrialSpec
    rows: dict[str, dict[str, Any]]
    selection_models: dict[str, dict[str, Any]]
    inner_selections: dict[str, dict[str, Any]]
    feature_cache_id: str
    training_subset: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationReadoutEvaluation:
    row: dict[str, Any]
    inner_selection: dict[str, Any]


@dataclass(frozen=True)
class TestEvaluation:
    primary_row: dict[str, Any]
    seed_rows: tuple[dict[str, Any], ...]
    ensemble_row: dict[str, Any] | None
    predictions: FrozenPredictions


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
            "feature_family": "static_input",
            "feature_is_dynamic": False,
            "feature_nu": None,
            "feature_alpha": None,
            "feature_beta": None,
            "feature_time": 0.0,
            "feature_solver": "none",
            "feature_advection_coefficient": None,
            "feature_dt": None,
            "feature_fine_dt": None,
            "feature_dealias": None,
            "feature_nonlinear_filter": None,
            "feature_system_parameters": "{}",
        }
    evolution = trial.feature.evolution
    if evolution is None:  # defensive guard for manually constructed objects
        raise ValueError("pde_dynamics feature has no evolution")
    system = evolution.system
    system_parameters = system.model_dump(mode="json")
    return {
        "n_tar": int(trial.input.n_tar),
        "n_sur": int(trial.feature.n_sur),
        "J": int(trial.feature.observation.J),
        "q": int(trial.output.q),
        "feature_system": system.kind,
        "feature_family": system.kind,
        "feature_is_dynamic": True,
        "feature_nu": float(getattr(system, "nu", float("nan"))),
        "feature_alpha": (
            None if not hasattr(system, "alpha") else float(getattr(system, "alpha"))
        ),
        "feature_beta": (
            None if not hasattr(system, "beta") else float(getattr(system, "beta"))
        ),
        "feature_time": float(evolution.time),
        "feature_solver": str(getattr(system, "solver", "spectral_exact")),
        "feature_advection_coefficient": (
            None
            if not hasattr(system, "advection_coefficient")
            else float(getattr(system, "advection_coefficient"))
        ),
        "feature_dt": (
            None if not hasattr(system, "dt") else float(getattr(system, "dt"))
        ),
        "feature_fine_dt": (
            None
            if getattr(system, "fine_dt", None) is None
            else float(getattr(system, "fine_dt"))
        ),
        "feature_dealias": (
            None
            if not hasattr(system, "dealias")
            else bool(getattr(system, "dealias"))
        ),
        "feature_nonlinear_filter": getattr(
            system,
            "nonlinear_filter",
            None,
        ),
        "feature_system_parameters": json.dumps(
            system_parameters,
            sort_keys=True,
            separators=(",", ":"),
        ),
    }


def feature_system_condition_hash(trial: TrialSpec) -> str:
    if trial.feature.kind == "static_input":
        return stable_object_hash({})
    evolution = trial.feature.evolution
    if evolution is None:
        raise ValueError("pde_dynamics feature has no evolution")
    return stable_object_hash(evolution.system.model_dump(mode="json"))


def selected_readout_parameter_fields(
    model: Mapping[str, Any],
) -> dict[str, Any]:
    kind = model.get("kind")
    if kind == "direct_fourier_decoder":
        return {}
    if kind == "affine_ridge":
        return {
            "selected_ridge_zeta": float(model["zeta"]),
        }
    if kind == "random_feature_ridge":
        members = model.get("members")
        if isinstance(members, list):
            seeds = [int(member["seed"]) for member in members]
        elif (
            model.get("members_materialized") is False
            and isinstance(model.get("evaluation_seeds"), list)
        ):
            seeds = [int(seed) for seed in model["evaluation_seeds"]]
        else:
            raise ValueError(
                "random-feature selection model has no evaluation seeds"
            )
        return {
            "selected_ridge_zeta": float(model["zeta"]),
            "selected_random_feature_activation": str(model["activation"]),
            "selected_random_feature_width": int(model["width"]),
            "selected_random_feature_weight_scale": float(
                model["weight_scale"]
            ),
            "selected_random_feature_bias_scale": float(
                model["bias_scale"]
            ),
            "selected_random_feature_evaluation_seed_count": len(seeds),
            "selected_random_feature_evaluation_seeds_hash": (
                stable_object_hash(seeds)
            ),
        }
    raise ValueError(f"unknown frozen model kind: {kind}")


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
    """Summarize independent seeds without conflating CIs and quantiles."""
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
        ordered = sorted(values)

        def linear_quantile(probability: float) -> float:
            position = probability * (count - 1)
            lower = math.floor(position)
            upper = math.ceil(position)
            if lower == upper:
                return ordered[lower]
            fraction = position - lower
            return (
                (1.0 - fraction) * ordered[lower]
                + fraction * ordered[upper]
            )

        summary[key] = mean
        summary[f"{key}_seed_mean"] = mean
        summary[f"{key}_seed_std"] = standard_deviation
        summary[f"{key}_seed_ci95_low"] = mean - margin
        summary[f"{key}_seed_ci95_high"] = mean + margin
        summary[f"{key}_seed_q25"] = linear_quantile(0.25)
        summary[f"{key}_seed_median"] = linear_quantile(0.5)
        summary[f"{key}_seed_q75"] = linear_quantile(0.75)
    return summary


def random_feature_map_parameter_hash(
    model: Mapping[str, Any],
    member: Mapping[str, Any],
) -> str:
    """Content identity for one random map, independent of storage."""
    A = member.get("A")
    c = member.get("c")
    if not isinstance(A, torch.Tensor) or not isinstance(c, torch.Tensor):
        raise ValueError("frozen random-feature member has invalid A/c tensors")
    return stable_object_hash(
        {
            "seed": int(member["seed"]),
            "A_sha256": tensor_sha256(A),
            "c_sha256": tensor_sha256(c),
            "activation": str(model["activation"]),
            "weight_scale": float(model["weight_scale"]),
            "bias_scale": float(model["bias_scale"]),
        }
    )


def random_feature_member_parameter_hash(
    model: Mapping[str, Any],
    member: Mapping[str, Any],
) -> str:
    """Content identity for a frozen random map and its fitted readout."""
    W = member.get("W")
    b = member.get("b")
    if not isinstance(W, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise ValueError("frozen random-feature member has invalid W/b tensors")
    validation_metrics = member.get("evaluation_seed_validation_metrics")
    if validation_metrics is not None and not isinstance(
        validation_metrics,
        Mapping,
    ):
        raise ValueError("invalid evaluation-seed validation metrics")
    zeta = member.get("zeta", model.get("zeta"))
    return stable_object_hash(
        {
            "random_map_parameter_hash": random_feature_map_parameter_hash(
                model,
                member,
            ),
            "W_sha256": tensor_sha256(W),
            "b_sha256": tensor_sha256(b),
            "zeta": None if zeta is None else float(zeta),
            "solver": member.get("solver"),
            "svd_rcond": member.get("svd_rcond"),
            "singular_value_cutoff": member.get("singular_value_cutoff"),
            "numerical_rank": member.get("numerical_rank"),
            "evaluation_seed_validation_metrics": (
                None
                if validation_metrics is None
                else {
                    str(key): float(value)
                    for key, value in validation_metrics.items()
                }
            ),
        }
    )


def random_feature_member_result_fields(
    model: Mapping[str, Any],
    member: Mapping[str, Any],
) -> dict[str, Any]:
    """Auditable per-realization fields shared by result and diagnostics."""
    W = member.get("W")
    b = member.get("b")
    validation_metrics = member.get("evaluation_seed_validation_metrics")
    if not isinstance(W, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise ValueError("invalid frozen random-feature realization")
    map_hash = random_feature_map_parameter_hash(model, member)
    fields: dict[str, Any] = {
        "random_map_parameter_hash": map_hash,
        "frozen_member_parameter_hash": random_feature_member_parameter_hash(
            model,
            member,
        ),
        "evaluation_seed_validation_used_for_selection": False,
        "readout_stability_link_field": "random_map_parameter_hash",
        "readout_stability_link_value": map_hash,
        "weight_frobenius_norm": float(
            torch.linalg.matrix_norm(W, ord="fro")
        ),
        "weight_operator_norm": float(torch.linalg.matrix_norm(W, ord=2)),
        "bias_norm": float(torch.linalg.vector_norm(b)),
    }
    if member.get("numerical_rank") is not None:
        fields["readout_numerical_rank"] = int(member["numerical_rank"])
    if "singular_value_cutoff" in member:
        fields["readout_singular_value_cutoff"] = member[
            "singular_value_cutoff"
        ]
    if isinstance(validation_metrics, Mapping):
        fields.update({
            f"evaluation_seed_validation_{key}": float(value)
            for key, value in validation_metrics.items()
        })
    return fields


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
    frozen_model: Mapping[str, Any],
    readout_values: dict[str, Any],
    inner_selection: dict[str, Any],
    floor_metrics: Mapping[str, float],
    training_subset: Mapping[str, Any],
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
            **training_subset_result_fields(training_subset),
            "feature_system_condition_hash": feature_system_condition_hash(
                trial
            ),
            **selected_readout_parameter_fields(frozen_model),
            **values,
        },
        inner_selection=bound_inner_selection,
    )


def build_test_evaluation(
    predictions: FrozenPredictions,
    *,
    model: Mapping[str, Any],
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
    training_subset: Mapping[str, Any],
    direct_diagnostic: Mapping[str, Any] | None = None,
) -> TestEvaluation:
    model_kind = str(model["kind"])
    base_row = {
        "candidate_id": candidate_id,
        "readout_id": readout_id,
        "readout_kind": model_kind,
        **trial_parameters(trial),
        **training_subset_result_fields(training_subset),
        "feature_system_condition_hash": feature_system_condition_hash(trial),
        **selected_readout_parameter_fields(model),
        "feature_cache_id": feature_cache_id,
    }
    if direct_diagnostic is not None:
        base_row.update(direct_diagnostic)

    if model_kind == "random_feature_ridge":
        seed_rows: list[dict[str, Any]] = []
        seed_metrics: list[dict[str, float]] = []
        members = model.get("members")
        if not isinstance(members, list):
            raise ValueError("frozen random-feature model members must be a list")
        members_by_seed = {int(member["seed"]): member for member in members}
        if len(members_by_seed) != len(members):
            raise ValueError("frozen random-feature member seeds must be unique")
        member_hashes: list[str] = []
        for seed, prediction in predictions.per_seed_predictions:
            member = members_by_seed.get(int(seed))
            if not isinstance(member, Mapping):
                raise ValueError(
                    "prediction seed is absent from the frozen random-feature "
                    "members"
                )
            member_fields = random_feature_member_result_fields(model, member)
            member_hashes.append(
                str(member_fields["frozen_member_parameter_hash"])
            )
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
                    **member_fields,
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
            "test_seed_descriptive_quantiles": "[0.25,0.5,0.75]",
            "test_seed_quantile_method": "linear",
            "test_seed_quantiles_are_uncertainty_interval": False,
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
            "ensemble_member_seeds_hash": stable_object_hash(
                [int(seed) for seed, _ in predictions.per_seed_predictions]
            ),
            "ensemble_member_parameters_hash": stable_object_hash(
                member_hashes
            ),
            **prefix_metrics(ensemble_metrics, "test_ensemble"),
            **prefix_metrics(floor_metrics, "test_ensemble"),
        }
        return TestEvaluation(
            primary_row=primary_row,
            seed_rows=tuple(seed_rows),
            ensemble_row=ensemble_row,
            predictions=predictions,
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
        predictions=predictions,
    )
