from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from pol.config.models import ConvergenceSpec, TrialSpec
from pol.data.dataset import ReferenceDataset
from pol.learning.metrics import compare_fields_on_common_grid
from pol.learning.observations import observe_equispaced_periodic
from pol.math.fourier import real_fourier_synthesis
from .cache import FeatureStateCache
from .overrides import apply_trial_overrides
from .trial import predict_frozen


@dataclass(frozen=True)
class ConvergenceOutcome:
    status: str
    rows: tuple[dict[str, Any], ...]


def _feature_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    denominator = torch.linalg.vector_norm(b, dim=-1).clamp_min(
        torch.finfo(b.dtype).eps
    )
    relative = torch.linalg.vector_norm(a - b, dim=-1) / denominator
    return {"mean": float(relative.mean()), "max": float(relative.max())}


def check_convergence(
    *,
    dataset: ReferenceDataset,
    cache: FeatureStateCache,
    trial: TrialSpec,
    model: Mapping[str, Any],
    spec: ConvergenceSpec,
) -> ConvergenceOutcome:
    configured_ids = torch.tensor(spec.sample_ids, dtype=torch.long)
    selection_ids = set(torch.cat([dataset.train_ids, dataset.validation_ids]).tolist())
    forbidden = [int(value) for value in configured_ids.tolist() if int(value) not in selection_ids]
    if forbidden:
        raise ValueError(
            "convergence sample_ids must belong to train/validation only; "
            f"forbidden IDs: {forbidden}"
        )
    states: list[torch.Tensor] = []
    features: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    prediction_semantics: str | None = None
    valid_n_sur: list[int] = []
    for n_sur in spec.n_sur_candidates:
        refined = apply_trial_overrides(trial, {"feature.n_sur": int(n_sur)})
        state = cache.get_or_solve(dataset, configured_ids, refined)
        phi = observe_equispaced_periodic(
            state.values,
            int(refined.feature.observation.J),
            domain_length=dataset.domain_length,
            l2_scale=refined.feature.observation.l2_scale,
        )
        frozen_predictions = predict_frozen(
            model,
            phi,
            q=int(refined.output.q),
            domain_length=dataset.domain_length,
        )
        if frozen_predictions.single_model_prediction is not None:
            coefficients = frozen_predictions.single_model_prediction
            active_semantics = "single_model"
        else:
            coefficients = frozen_predictions.prediction_ensemble()
            active_semantics = "prediction_ensemble"
        if prediction_semantics is None:
            prediction_semantics = active_semantics
        elif prediction_semantics != active_semantics:
            raise ValueError("prediction semantics changed across convergence resolutions")
        field = real_fourier_synthesis(
            coefficients,
            int(refined.input.n_tar),
            domain_length=dataset.domain_length,
        )
        valid_n_sur.append(int(n_sur))
        states.append(state.values)
        features.append(phi)
        predictions.append(field)
    rows: list[dict[str, Any]] = []
    tolerance = spec.tolerances
    for index in range(len(valid_n_sur) - 1):
        terminal = compare_fields_on_common_grid(
            states[index],
            states[index + 1],
            domain_length=dataset.domain_length,
        )
        feature = _feature_metrics(features[index], features[index + 1])
        prediction = compare_fields_on_common_grid(
            predictions[index],
            predictions[index + 1],
            domain_length=dataset.domain_length,
        )
        passed = (
            terminal["relative_l2_mean"] <= tolerance.terminal_mean
            and terminal["relative_l2_max"] <= tolerance.terminal_max
            and feature["mean"] <= tolerance.feature_mean
            and feature["max"] <= tolerance.feature_max
            and prediction["relative_l2_mean"] <= tolerance.prediction_mean
            and prediction["relative_l2_max"] <= tolerance.prediction_max
        )
        rows.append(
            {
                "coarse_n_sur": valid_n_sur[index],
                "fine_n_sur": valid_n_sur[index + 1],
                "terminal_relative_l2_mean": terminal["relative_l2_mean"],
                "terminal_relative_l2_max": terminal["relative_l2_max"],
                "feature_relative_l2_mean": feature["mean"],
                "feature_relative_l2_max": feature["max"],
                "prediction_relative_l2_mean": prediction["relative_l2_mean"],
                "prediction_relative_l2_max": prediction["relative_l2_max"],
                "prediction_semantics": prediction_semantics,
                "status": "pass" if passed else "fail",
            }
        )
    status = "pass" if rows and all(row["status"] == "pass" for row in rows) else "fail"
    return ConvergenceOutcome(status=status, rows=tuple(rows))
