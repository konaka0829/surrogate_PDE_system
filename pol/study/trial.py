from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Any, Mapping

from scipy.stats import t as student_t
import torch

from pol.config.models import (
    AffineRidgeReadoutSpec,
    DirectReadoutSpec,
    RandomFeatureRidgeReadoutSpec,
    ReadoutSpec,
    StudySpec,
    TrialSpec,
)
from pol.data.dataset import ReferenceDataset
from pol.data.finite import FiniteDataView, derive_finite_view
from pol.learning.direct import decode_point_observation_to_real_fourier
from pol.learning.metrics import (
    fourier_prediction_metrics,
    fourier_representation_floor,
)
from pol.learning.observations import observe_equispaced_periodic
from pol.learning.random_features import RandomFeatureMap
from pol.learning.ridge import fit_centered_affine_ridge
from pol.runtime.device import (
    require_cpu_tensor,
    require_cpu_tensors,
    verify_execution_device_policy,
)
from pol.runtime.hashing import stable_object_hash
from .cache import FeatureStateCache


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate_id: str
    trial: TrialSpec
    rows: dict[str, dict[str, Any]]
    frozen_models: dict[str, dict[str, Any]]
    inner_selections: dict[str, dict[str, Any]]
    feature_cache_id: str


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


def _prefix(metrics: Mapping[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{key}": float(value) for key, value in metrics.items()}


def _mean_metrics(items: list[dict[str, float]]) -> dict[str, float]:
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


def _evaluate_coefficients(
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


def _representation_floor(
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


def _serialize_affine(readout, *, zeta: float) -> dict[str, Any]:
    require_cpu_tensors(
        {"W": readout.W, "b": readout.b},
        boundary="frozen affine readout publication",
        name="readout",
    )
    return {
        "kind": "affine_ridge",
        "zeta": float(zeta),
        "W": readout.W.detach().cpu(),
        "b": readout.b.detach().cpu(),
        "solver": readout.solver,
        "svd_rcond": readout.svd_rcond,
        "singular_value_cutoff": readout.singular_value_cutoff,
        "numerical_rank": readout.numerical_rank,
    }


def _predict_affine(model: Mapping[str, Any], features: torch.Tensor) -> torch.Tensor:
    W: torch.Tensor = model["W"]
    b: torch.Tensor = model["b"]
    return features.to(dtype=W.dtype, device=W.device) @ W.T + b


def predict_frozen(
    model: Mapping[str, Any],
    features: torch.Tensor,
    *,
    q: int,
    domain_length: float,
) -> FrozenPredictions:
    require_cpu_tensor(
        features,
        boundary="frozen model prediction input",
        name="features",
    )
    require_cpu_tensors(
        model,
        boundary="frozen model prediction archive",
        name="model",
    )
    kind = model["kind"]
    if kind == "direct_fourier_decoder":
        prediction = decode_point_observation_to_real_fourier(
            features, q, domain_length=domain_length
        )
        require_cpu_tensor(
            prediction,
            boundary="frozen direct-model prediction",
            name="prediction",
        )
        return FrozenPredictions(prediction, ())
    if kind == "affine_ridge":
        prediction = _predict_affine(model, features)
        require_cpu_tensor(
            prediction,
            boundary="frozen affine-model prediction",
            name="prediction",
        )
        return FrozenPredictions(prediction, ())
    if kind == "random_feature_ridge":
        predictions: list[tuple[int, torch.Tensor]] = []
        for member in model["members"]:
            random_map = RandomFeatureMap(
                A=member["A"],
                c=member["c"],
                activation=model["activation"],
                seed=int(member["seed"]),
                weight_scale=float(model["weight_scale"]),
                bias_scale=float(model["bias_scale"]),
            )
            lifted = random_map(features.to(member["A"].device, member["A"].dtype))
            prediction = _predict_affine(member, lifted)
            require_cpu_tensor(
                prediction,
                boundary="frozen random-feature-model prediction",
                name=f"prediction_seed_{member['seed']}",
            )
            predictions.append((int(member["seed"]), prediction))
        return FrozenPredictions(None, tuple(predictions))
    raise ValueError(f"unknown frozen model kind: {kind}")


class TrialEngine:
    """Evaluate all readouts for one validated trial.

    Selection views are constructed from train and validation sample IDs only.
    Test IDs are accepted only by :meth:`evaluate_test`, which the study runner
    calls after the frozen plan has been written and read back.
    """

    def __init__(
        self,
        dataset: ReferenceDataset,
        study: StudySpec,
        cache: FeatureStateCache,
    ) -> None:
        self.dataset = dataset
        self.study = study
        self.cache = cache
        self.selection_ids = torch.cat(
            [dataset.train_ids, dataset.validation_ids]
        ).to(torch.long)
        self.n_train = int(dataset.train_ids.numel())
        self._finite_cache: dict[tuple[str, int, int], FiniteDataView] = {}
        self._candidate_cache: dict[str, CandidateEvaluation] = {}
        verify_execution_device_policy(
            dataset.__dict__,
            boundary="trial-engine dataset",
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
            boundary="trial-engine dataset",
            name="dataset",
        )

    def _finite(self, ids: torch.Tensor, trial: TrialSpec) -> FiniteDataView:
        key = (
            stable_object_hash([int(value) for value in ids.tolist()]),
            int(trial.input.n_tar),
            int(trial.output.q),
        )
        if key not in self._finite_cache:
            inputs, targets = self.dataset.tensors_for(ids)
            self._finite_cache[key] = derive_finite_view(
                ids,
                inputs,
                targets,
                n_tar=int(trial.input.n_tar),
                q=int(trial.output.q),
                domain_length=self.dataset.domain_length,
            )
            require_cpu_tensors(
                self._finite_cache[key].__dict__,
                boundary="finite data view",
                name="finite",
            )
        return self._finite_cache[key]

    def evaluate_selection(self, trial: TrialSpec) -> CandidateEvaluation:
        candidate_id = stable_object_hash(trial.model_dump(mode="json"))
        if candidate_id in self._candidate_cache:
            return self._candidate_cache[candidate_id]
        finite = self._finite(self.selection_ids, trial)
        state = self.cache.get_or_solve(self.dataset, self.selection_ids, trial)
        features = observe_equispaced_periodic(
            state.values,
            int(trial.feature.observation.J),
            domain_length=self.dataset.domain_length,
            l2_scale=trial.feature.observation.l2_scale,
        )
        require_cpu_tensors(
            {
                "state": state.values,
                "features": features,
                "target_coefficients": finite.target_coefficients,
                "targets": finite.targets,
                "targets_reference": finite.targets_reference,
            },
            boundary="readout selection inputs",
            name="selection",
        )
        x_train = features[: self.n_train]
        x_validation = features[self.n_train :]
        y_train = finite.target_coefficients[: self.n_train]
        y_validation = finite.target_coefficients[self.n_train :]
        target_validation = finite.targets[self.n_train :]
        target_reference_validation = finite.targets_reference[self.n_train :]
        floor = _representation_floor(
            y_validation,
            target_validation,
            target_reference_validation,
            n_tar=int(trial.input.n_tar),
            n_ref=finite.n_ref,
            domain_length=self.dataset.domain_length,
        )
        rows: dict[str, dict[str, Any]] = {}
        frozen: dict[str, dict[str, Any]] = {}
        selections: dict[str, dict[str, Any]] = {}
        for readout in trial.readouts:
            row, model, inner = self._fit_readout(
                readout,
                x_train=x_train,
                y_train=y_train,
                x_validation=x_validation,
                y_validation=y_validation,
                target_validation=target_validation,
                target_reference_validation=target_reference_validation,
                trial=trial,
            )
            row.update(_prefix(floor, "validation"))
            rows[readout.id] = {
                "candidate_id": candidate_id,
                "readout_id": readout.id,
                "readout_kind": readout.kind,
                **trial_parameters(trial),
                **row,
            }
            frozen[readout.id] = model
            selections[readout.id] = inner
        result = CandidateEvaluation(
            candidate_id=candidate_id,
            trial=trial,
            rows=rows,
            frozen_models=frozen,
            inner_selections=selections,
            feature_cache_id=state.cache_id,
        )
        self._candidate_cache[candidate_id] = result
        return result

    def _fit_readout(
        self,
        readout: ReadoutSpec,
        *,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_validation: torch.Tensor,
        y_validation: torch.Tensor,
        target_validation: torch.Tensor,
        target_reference_validation: torch.Tensor,
        trial: TrialSpec,
    ) -> tuple[dict[str, float], dict[str, Any], dict[str, Any]]:
        require_cpu_tensors(
            {
                "x_train": x_train,
                "y_train": y_train,
                "x_validation": x_validation,
                "y_validation": y_validation,
                "target_validation": target_validation,
                "target_reference_validation": target_reference_validation,
            },
            boundary="readout fitting inputs",
            name="fit",
        )
        q = int(trial.output.q)
        n_tar = int(trial.input.n_tar)
        L = self.dataset.domain_length
        metric_name = self.study.selection.metric
        if isinstance(readout, DirectReadoutSpec):
            prediction = decode_point_observation_to_real_fourier(
                x_validation, q, domain_length=L
            )
            metrics = _evaluate_coefficients(
                prediction,
                y_validation,
                target_validation,
                target_reference_validation,
                n_tar=n_tar,
                n_ref=self.dataset.reference_nx,
                domain_length=L,
            )
            return (
                _prefix(metrics, "validation"),
                {
                    "kind": "direct_fourier_decoder",
                    "q": q,
                    "domain_length": L,
                },
                {"kind": "fixed", "parameter_count": 0},
            )
        if isinstance(readout, AffineRidgeReadoutSpec):
            candidates: list[tuple[float, dict[str, float], Any]] = []
            for zeta in readout.zetas:
                fitted = fit_centered_affine_ridge(
                    x_train,
                    y_train,
                    float(zeta),
                    svd_rcond=readout.svd_rcond,
                )
                prediction = fitted(x_validation)
                metrics = _evaluate_coefficients(
                    prediction,
                    y_validation,
                    target_validation,
                    target_reference_validation,
                    n_tar=n_tar,
                    n_ref=self.dataset.reference_nx,
                    domain_length=L,
                )
                candidates.append((float(zeta), _prefix(metrics, "validation"), fitted))
            best = min(float(metrics[metric_name]) for _, metrics, _ in candidates)
            eligible = [
                item
                for item in candidates
                if float(item[1][metric_name]) <= best + readout.tie_tolerance
            ]
            chosen = (
                max(eligible, key=lambda item: item[0])
                if readout.tie_break == "largest_zeta"
                else eligible[0]
            )
            zeta, metrics, fitted = chosen
            return (
                metrics,
                _serialize_affine(fitted, zeta=zeta),
                {
                    "zeta": zeta,
                    "candidate_metrics": [
                        {"zeta": value, **metric} for value, metric, _ in candidates
                    ],
                },
            )
        if isinstance(readout, RandomFeatureRidgeReadoutSpec):
            structural: list[dict[str, Any]] = []
            for width, weight_scale, bias_scale, zeta in itertools.product(
                readout.widths,
                readout.weight_scales,
                readout.bias_scales,
                readout.zetas,
            ):
                seed_metrics: list[dict[str, float]] = []
                for seed in readout.selection_seeds:
                    random_map = RandomFeatureMap.create(
                        x_train.shape[1],
                        int(width),
                        activation=readout.activation,
                        seed=int(seed),
                        weight_scale=float(weight_scale),
                        bias_scale=float(bias_scale),
                        dtype=x_train.dtype,
                        device=x_train.device,
                    )
                    train_lift = random_map(x_train)
                    validation_lift = random_map(x_validation)
                    fitted = fit_centered_affine_ridge(
                        train_lift,
                        y_train,
                        float(zeta),
                        svd_rcond=readout.svd_rcond,
                    )
                    prediction = fitted(validation_lift)
                    seed_metrics.append(
                        _prefix(
                            _evaluate_coefficients(
                                prediction,
                                y_validation,
                                target_validation,
                                target_reference_validation,
                                n_tar=n_tar,
                                n_ref=self.dataset.reference_nx,
                                domain_length=L,
                            ),
                            "validation",
                        )
                    )
                structural.append(
                    {
                        "width": int(width),
                        "weight_scale": float(weight_scale),
                        "bias_scale": float(bias_scale),
                        "zeta": float(zeta),
                        "metrics": _mean_metrics(seed_metrics),
                        "selection_seed_metrics": seed_metrics,
                    }
                )
            best = min(float(item["metrics"][metric_name]) for item in structural)
            chosen = next(
                item
                for item in structural
                if float(item["metrics"][metric_name]) <= best + readout.tie_tolerance
            )
            members: list[dict[str, Any]] = []
            for seed in readout.evaluation_seeds:
                random_map = RandomFeatureMap.create(
                    x_train.shape[1],
                    chosen["width"],
                    activation=readout.activation,
                    seed=int(seed),
                    weight_scale=chosen["weight_scale"],
                    bias_scale=chosen["bias_scale"],
                    dtype=x_train.dtype,
                    device=x_train.device,
                )
                fitted = fit_centered_affine_ridge(
                    random_map(x_train),
                    y_train,
                    chosen["zeta"],
                    svd_rcond=readout.svd_rcond,
                )
                members.append(
                    {
                        **_serialize_affine(fitted, zeta=chosen["zeta"]),
                        "seed": int(seed),
                        "A": random_map.A.detach().cpu(),
                        "c": random_map.c.detach().cpu(),
                    }
                )
            model = {
                "kind": "random_feature_ridge",
                "activation": readout.activation,
                "width": chosen["width"],
                "weight_scale": chosen["weight_scale"],
                "bias_scale": chosen["bias_scale"],
                "zeta": chosen["zeta"],
                "members": members,
            }
            require_cpu_tensors(
                model,
                boundary="frozen random-feature readout publication",
                name="model",
            )
            return (
                chosen["metrics"],
                model,
                {
                    "width": chosen["width"],
                    "weight_scale": chosen["weight_scale"],
                    "bias_scale": chosen["bias_scale"],
                    "zeta": chosen["zeta"],
                    "candidate_metrics": structural,
                },
            )
        raise TypeError(f"unsupported readout spec: {type(readout).__name__}")

    def evaluate_test(
        self,
        trial: TrialSpec,
        model: Mapping[str, Any],
        *,
        readout_id: str,
        candidate_id: str,
    ) -> TestEvaluation:
        ids = self.dataset.test_ids.to(torch.long)
        finite = self._finite(ids, trial)
        state = self.cache.get_or_solve(self.dataset, ids, trial)
        features = observe_equispaced_periodic(
            state.values,
            int(trial.feature.observation.J),
            domain_length=self.dataset.domain_length,
            l2_scale=trial.feature.observation.l2_scale,
        )
        require_cpu_tensors(
            {
                "state": state.values,
                "features": features,
                "target_coefficients": finite.target_coefficients,
                "targets": finite.targets,
                "targets_reference": finite.targets_reference,
                "model": model,
            },
            boundary="test evaluation inputs",
            name="test",
        )
        predictions = predict_frozen(
            model,
            features,
            q=int(trial.output.q),
            domain_length=self.dataset.domain_length,
        )
        floor = _representation_floor(
            finite.target_coefficients,
            finite.targets,
            finite.targets_reference,
            n_tar=int(trial.input.n_tar),
            n_ref=finite.n_ref,
            domain_length=self.dataset.domain_length,
        )
        base_row = {
            "candidate_id": candidate_id,
            "readout_id": readout_id,
            **trial_parameters(trial),
            "feature_cache_id": state.cache_id,
        }
        if model["kind"] == "random_feature_ridge":
            seed_rows: list[dict[str, Any]] = []
            seed_metrics: list[dict[str, float]] = []
            for seed, prediction in predictions.per_seed_predictions:
                seed_metric = _evaluate_coefficients(
                    prediction,
                    finite.target_coefficients,
                    finite.targets,
                    finite.targets_reference,
                    n_tar=int(trial.input.n_tar),
                    n_ref=finite.n_ref,
                    domain_length=self.dataset.domain_length,
                )
                prefixed_seed_metric = {
                    **_prefix(seed_metric, "test"),
                    **_prefix(floor, "test"),
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
            ensemble_metrics = _evaluate_coefficients(
                ensemble_prediction,
                finite.target_coefficients,
                finite.targets,
                finite.targets_reference,
                n_tar=int(trial.input.n_tar),
                n_ref=finite.n_ref,
                domain_length=self.dataset.domain_length,
            )
            ensemble_row = {
                **base_row,
                "test_result_kind": "prediction_ensemble",
                "ensemble_member_count": len(seed_rows),
                **_prefix(ensemble_metrics, "test_ensemble"),
            }
            return TestEvaluation(
                primary_row=primary_row,
                seed_rows=tuple(seed_rows),
                ensemble_row=ensemble_row,
            )

        if predictions.single_model_prediction is None:
            raise ValueError("deterministic frozen model returned no prediction")
        metrics = _evaluate_coefficients(
            predictions.single_model_prediction,
            finite.target_coefficients,
            finite.targets,
            finite.targets_reference,
            n_tar=int(trial.input.n_tar),
            n_ref=finite.n_ref,
            domain_length=self.dataset.domain_length,
        )
        primary_row = {
            **base_row,
            "test_result_kind": "single_model",
            **_prefix(metrics, "test"),
            **_prefix(floor, "test"),
        }
        return TestEvaluation(
            primary_row=primary_row,
            seed_rows=(),
            ensemble_row=None,
        )
