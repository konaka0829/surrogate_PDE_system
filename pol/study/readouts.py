from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Any, Mapping

import torch

from pol.config.models import (
    AffineRidgeReadoutSpec,
    DirectReadoutSpec,
    RandomFeatureRidgeReadoutSpec,
    ReadoutSpec,
)
from pol.learning.direct import (
    decode_point_observation_to_real_fourier,
    fixed_fourier_decoder_bandwidth,
    verify_fixed_fourier_decoder_diagnostic,
)
from pol.learning.random_features import RandomFeatureMap
from pol.learning.ridge import AffineReadout, fit_centered_affine_ridge
from pol.runtime.device import require_cpu_tensor, require_cpu_tensors
from .evaluation import (
    FrozenPredictions,
    evaluate_coefficients,
    mean_metrics,
    prefix_metrics,
)


@dataclass(frozen=True)
class ReadoutFitInputs:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_validation: torch.Tensor
    y_validation: torch.Tensor
    data_target_validation: torch.Tensor
    reference_target_validation: torch.Tensor
    observation_count: int
    q: int
    n_tar: int
    n_ref: int
    domain_length: float
    selection_metric: str


@dataclass(frozen=True)
class FittedReadout:
    validation_values: dict[str, Any]
    selection_model: dict[str, Any]
    inner_selection: dict[str, Any]


def serialize_affine(readout: AffineReadout, *, zeta: float) -> dict[str, Any]:
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


def predict_affine(
    model: Mapping[str, Any],
    features: torch.Tensor,
) -> torch.Tensor:
    W: torch.Tensor = model["W"]
    b: torch.Tensor = model["b"]
    return features.to(dtype=W.dtype, device=W.device) @ W.T + b


def _fit_direct(
    readout: DirectReadoutSpec,
    inputs: ReadoutFitInputs,
) -> FittedReadout:
    del readout
    diagnostic = fixed_fourier_decoder_bandwidth(
        inputs.observation_count,
        inputs.q,
    ).as_artifact_fields()
    prediction = decode_point_observation_to_real_fourier(
        inputs.x_validation,
        inputs.q,
        domain_length=inputs.domain_length,
    )
    metrics = evaluate_coefficients(
        prediction,
        inputs.y_validation,
        inputs.data_target_validation,
        inputs.reference_target_validation,
        n_tar=inputs.n_tar,
        n_ref=inputs.n_ref,
        domain_length=inputs.domain_length,
    )
    return FittedReadout(
        validation_values={
            **prefix_metrics(metrics, "validation"),
            **diagnostic,
        },
        selection_model={
            "kind": "direct_fourier_decoder",
            "q": inputs.q,
            "domain_length": inputs.domain_length,
            **diagnostic,
        },
        inner_selection={
            "kind": "fixed",
            "parameter_count": 0,
            **diagnostic,
        },
    )


def _fit_affine(
    readout: AffineRidgeReadoutSpec,
    inputs: ReadoutFitInputs,
) -> FittedReadout:
    candidates: list[tuple[float, dict[str, float], AffineReadout]] = []
    for zeta in readout.zetas:
        fitted = fit_centered_affine_ridge(
            inputs.x_train,
            inputs.y_train,
            float(zeta),
            svd_rcond=readout.svd_rcond,
        )
        prediction = fitted(inputs.x_validation)
        metrics = evaluate_coefficients(
            prediction,
            inputs.y_validation,
            inputs.data_target_validation,
            inputs.reference_target_validation,
            n_tar=inputs.n_tar,
            n_ref=inputs.n_ref,
            domain_length=inputs.domain_length,
        )
        candidates.append(
            (float(zeta), prefix_metrics(metrics, "validation"), fitted)
        )
    metric_values = [
        float(metrics[inputs.selection_metric])
        for _, metrics, _ in candidates
    ]
    if not all(math.isfinite(value) for value in metric_values):
        raise ValueError(
            "affine readout produced a non-finite validation selection metric"
        )
    best = min(metric_values)
    eligible = [
        item
        for item in candidates
        if float(item[1][inputs.selection_metric])
        <= best + readout.tie_tolerance
    ]
    chosen = (
        max(eligible, key=lambda item: item[0])
        if readout.tie_break == "largest_zeta"
        else eligible[0]
    )
    zeta, metrics, fitted = chosen
    return FittedReadout(
        validation_values=metrics,
        selection_model=serialize_affine(fitted, zeta=zeta),
        inner_selection={
            "zeta": zeta,
            "candidate_metrics": [
                {"zeta": value, **metric}
                for value, metric, _ in candidates
            ],
        },
    )


def materialize_random_feature_readout(
    readout: RandomFeatureRidgeReadoutSpec,
    inputs: ReadoutFitInputs,
    recipe: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit evaluation-seed members after study-level candidate selection."""
    if (
        recipe.get("kind") != "random_feature_ridge"
        or recipe.get("members_materialized") is not False
    ):
        raise ValueError("random-feature materialization requires a recipe")
    if (
        recipe.get("activation") != readout.activation
        or list(recipe.get("evaluation_seeds", []))
        != list(readout.evaluation_seeds)
    ):
        raise ValueError(
            "random-feature recipe disagrees with the selected readout"
        )
    members: list[dict[str, Any]] = []
    for seed in readout.evaluation_seeds:
        random_map = RandomFeatureMap.create(
            inputs.x_train.shape[1],
            int(recipe["width"]),
            activation=readout.activation,
            seed=int(seed),
            weight_scale=float(recipe["weight_scale"]),
            bias_scale=float(recipe["bias_scale"]),
            dtype=inputs.x_train.dtype,
            device=inputs.x_train.device,
        )
        fitted = fit_centered_affine_ridge(
            random_map(inputs.x_train),
            inputs.y_train,
            float(recipe["zeta"]),
            svd_rcond=readout.svd_rcond,
        )
        validation_prediction = fitted(random_map(inputs.x_validation))
        validation_metrics = evaluate_coefficients(
            validation_prediction,
            inputs.y_validation,
            inputs.data_target_validation,
            inputs.reference_target_validation,
            n_tar=inputs.n_tar,
            n_ref=inputs.n_ref,
            domain_length=inputs.domain_length,
        )
        members.append(
            {
                **serialize_affine(fitted, zeta=float(recipe["zeta"])),
                "seed": int(seed),
                "A": random_map.A.detach().cpu(),
                "c": random_map.c.detach().cpu(),
                "evaluation_seed_validation_metrics": validation_metrics,
            }
        )
    model = {
        key: value
        for key, value in recipe.items()
        if key not in {"evaluation_seeds", "members_materialized"}
    }
    model.update(
        {
            "members_materialized": True,
            "materialization_split": "train_validation_only",
            "evaluation_seed_metrics_used_for_selection": False,
            "members": members,
        }
    )
    require_cpu_tensors(
        model,
        boundary="frozen random-feature readout publication",
        name="model",
    )
    return model


def _fit_random_feature(
    readout: RandomFeatureRidgeReadoutSpec,
    inputs: ReadoutFitInputs,
) -> FittedReadout:
    structural: list[dict[str, Any]] = []
    lift_cache: dict[
        tuple[int, float, float, int],
        tuple[torch.Tensor, torch.Tensor],
    ] = {}
    for width, weight_scale, bias_scale in itertools.product(
        readout.widths,
        readout.weight_scales,
        readout.bias_scales,
    ):
        for seed in readout.selection_seeds:
            cache_key = (
                int(width),
                float(weight_scale),
                float(bias_scale),
                int(seed),
            )
            random_map = RandomFeatureMap.create(
                inputs.x_train.shape[1],
                int(width),
                activation=readout.activation,
                seed=int(seed),
                weight_scale=float(weight_scale),
                bias_scale=float(bias_scale),
                dtype=inputs.x_train.dtype,
                device=inputs.x_train.device,
            )
            lift_cache[cache_key] = (
                random_map(inputs.x_train),
                random_map(inputs.x_validation),
            )
        for zeta in readout.zetas:
            seed_metrics: list[dict[str, float]] = []
            for seed in readout.selection_seeds:
                train_lift, validation_lift = lift_cache[
                    (
                        int(width),
                        float(weight_scale),
                        float(bias_scale),
                        int(seed),
                    )
                ]
                fitted = fit_centered_affine_ridge(
                    train_lift,
                    inputs.y_train,
                    float(zeta),
                    svd_rcond=readout.svd_rcond,
                )
                prediction = fitted(validation_lift)
                seed_metrics.append(
                    prefix_metrics(
                        evaluate_coefficients(
                            prediction,
                            inputs.y_validation,
                            inputs.data_target_validation,
                            inputs.reference_target_validation,
                            n_tar=inputs.n_tar,
                            n_ref=inputs.n_ref,
                            domain_length=inputs.domain_length,
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
                    "metrics": mean_metrics(seed_metrics),
                    "selection_seed_metrics": seed_metrics,
                }
            )
    metric_values = [
        float(item["metrics"][inputs.selection_metric])
        for item in structural
    ]
    if not all(math.isfinite(value) for value in metric_values):
        raise ValueError(
            "random-feature readout produced a non-finite validation "
            "selection metric"
        )
    best = min(metric_values)
    chosen = next(
        item
        for item in structural
        if float(item["metrics"][inputs.selection_metric])
        <= best + readout.tie_tolerance
    )
    model = {
        "kind": "random_feature_ridge",
        "activation": readout.activation,
        "width": chosen["width"],
        "weight_scale": chosen["weight_scale"],
        "bias_scale": chosen["bias_scale"],
        "zeta": chosen["zeta"],
        "evaluation_seeds": list(readout.evaluation_seeds),
        "members_materialized": False,
    }
    return FittedReadout(
        validation_values=chosen["metrics"],
        selection_model=model,
        inner_selection={
            "width": chosen["width"],
            "weight_scale": chosen["weight_scale"],
            "bias_scale": chosen["bias_scale"],
            "zeta": chosen["zeta"],
            "candidate_metrics": structural,
        },
    )


def fit_readout(
    readout: ReadoutSpec,
    inputs: ReadoutFitInputs,
) -> FittedReadout:
    require_cpu_tensors(
        {
            "x_train": inputs.x_train,
            "y_train": inputs.y_train,
            "x_validation": inputs.x_validation,
            "y_validation": inputs.y_validation,
            "target_validation": inputs.data_target_validation,
            "target_reference_validation": inputs.reference_target_validation,
        },
        boundary="readout fitting inputs",
        name="fit",
    )
    if isinstance(readout, DirectReadoutSpec):
        return _fit_direct(readout, inputs)
    if isinstance(readout, AffineRidgeReadoutSpec):
        return _fit_affine(readout, inputs)
    if isinstance(readout, RandomFeatureRidgeReadoutSpec):
        return _fit_random_feature(readout, inputs)
    raise TypeError(f"unsupported readout spec: {type(readout).__name__}")


def frozen_readout_diagnostic(
    model: Mapping[str, Any],
    *,
    observation_count: int,
    q: int,
    boundary: str,
) -> dict[str, Any] | None:
    if model.get("kind") != "direct_fourier_decoder":
        return None
    return verify_fixed_fourier_decoder_diagnostic(
        model,
        observation_count=observation_count,
        requested_q=q,
        boundary=boundary,
    ).as_artifact_fields()


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
        frozen_readout_diagnostic(
            model,
            observation_count=int(features.shape[-1]),
            q=q,
            boundary="frozen direct model",
        )
        prediction = decode_point_observation_to_real_fourier(
            features,
            q,
            domain_length=domain_length,
        )
        require_cpu_tensor(
            prediction,
            boundary="frozen direct-model prediction",
            name="prediction",
        )
        return FrozenPredictions(prediction, ())
    if kind == "affine_ridge":
        prediction = predict_affine(model, features)
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
            lifted = random_map(
                features.to(member["A"].device, member["A"].dtype)
            )
            prediction = predict_affine(member, lifted)
            require_cpu_tensor(
                prediction,
                boundary="frozen random-feature-model prediction",
                name=f"prediction_seed_{member['seed']}",
            )
            predictions.append((int(member["seed"]), prediction))
        return FrozenPredictions(None, tuple(predictions))
    raise ValueError(f"unknown frozen model kind: {kind}")
