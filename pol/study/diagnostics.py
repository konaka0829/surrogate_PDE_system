from __future__ import annotations

from typing import Any, Mapping

import torch

from pol.config.models import HeatMultiplierDiagnosticSpec, NoiseDiagnosticSpec, TrialSpec
from pol.data.dataset import ReferenceDataset
from pol.data.finite import derive_finite_view
from pol.learning.metrics import fourier_prediction_metrics
from pol.learning.observations import observe_equispaced_periodic
from pol.learning.ridge import l2_synthesis_matrix
from pol.systems.heat import heat_multiplier_vector
from .cache import FeatureStateCache
from .trial import predict_frozen


def heat_multiplier_rows(
    diagnostic: HeatMultiplierDiagnosticSpec,
    *,
    dataset: ReferenceDataset,
    trial: TrialSpec,
    model: Mapping[str, Any],
    case_id: str,
    readout_id: str,
) -> list[dict[str, Any]]:
    if model.get("kind") != "affine_ridge":
        return []
    if dataset.target_metadata.get("kind") != "heat":
        return []
    evolution = trial.feature.evolution
    if trial.feature.kind != "pde_dynamics" or evolution is None:
        return []
    if evolution.system.kind != "heat":
        return []
    q = int(trial.output.q)
    J = int(trial.feature.observation.J)
    synthesis = l2_synthesis_matrix(
        q,
        J,
        domain_length=dataset.domain_length,
        dtype=model["W"].dtype,
        device=model["W"].device,
    )
    effective = model["W"] @ synthesis
    ideal = heat_multiplier_vector(
        q,
        target_nu=float(dataset.target_metadata["nu"]),
        target_time=float(dataset.target_metadata["time"]),
        surrogate_nu=float(evolution.system.nu),
        surrogate_time=float(evolution.time),
        domain_length=dataset.domain_length,
        dtype=effective.dtype,
        device=effective.device,
    )
    rows: list[dict[str, Any]] = []
    for index in range(q):
        off = effective[index].clone()
        off[index] = 0
        rows.append(
            {
                "case_id": case_id,
                "readout_id": readout_id,
                "coefficient_index": index,
                "effective_diagonal": float(effective[index, index]),
                "ideal_diagonal": float(ideal[index]),
                "absolute_diagonal_error": float(
                    torch.abs(effective[index, index] - ideal[index])
                ),
                "off_diagonal_l2": float(torch.linalg.vector_norm(off)),
                "identifiable": bool(
                    torch.abs(ideal[index]) >= diagnostic.identifiable_variance_floor
                ),
            }
        )
    return rows


@torch.no_grad()
def noise_robustness_rows(
    diagnostic: NoiseDiagnosticSpec,
    *,
    dataset: ReferenceDataset,
    cache: FeatureStateCache,
    trial: TrialSpec,
    model: Mapping[str, Any],
    case_id: str,
    readout_id: str,
) -> list[dict[str, Any]]:
    ids = dataset.test_ids.to(torch.long)
    inputs, targets = dataset.tensors_for(ids)
    finite = derive_finite_view(
        ids,
        inputs,
        targets,
        n_tar=int(trial.input.n_tar),
        q=int(trial.output.q),
        domain_length=dataset.domain_length,
    )
    state = cache.get_or_solve(dataset, ids, trial)
    features = observe_equispaced_periodic(
        state.values,
        int(trial.feature.observation.J),
        domain_length=dataset.domain_length,
        l2_scale=trial.feature.observation.l2_scale,
    )
    rms = torch.sqrt(features.square().mean()).clamp_min(torch.finfo(features.dtype).eps)
    rows: list[dict[str, Any]] = []
    for level_index, level in enumerate(diagnostic.levels):
        metrics: list[dict[str, float]] = []
        for repeat in range(diagnostic.repeats):
            generator = torch.Generator(device="cpu").manual_seed(
                int(diagnostic.seed) + 1009 * level_index + repeat
            )
            noise = torch.randn(
                features.shape,
                generator=generator,
                dtype=features.dtype,
                device="cpu",
            ).to(features.device)
            noisy = features + float(level) * rms * noise
            predictions = predict_frozen(
                model,
                noisy,
                q=int(trial.output.q),
                domain_length=dataset.domain_length,
            )
            if predictions.single_model_prediction is not None:
                prediction = predictions.single_model_prediction
                prediction_semantics = "single_model"
            else:
                prediction = predictions.prediction_ensemble()
                prediction_semantics = "prediction_ensemble"
            metrics.append(
                fourier_prediction_metrics(
                    prediction,
                    finite.target_coefficients,
                    finite.targets,
                    finite.targets_reference,
                    n_data=finite.n_tar,
                    n_reference=finite.n_ref,
                    domain_length=dataset.domain_length,
                )
            )
        row: dict[str, Any] = {
            "case_id": case_id,
            "readout_id": readout_id,
            "noise_level": float(level),
            "repeats": int(diagnostic.repeats),
            "prediction_semantics": prediction_semantics,
        }
        for key in metrics[0]:
            row[key] = sum(item[key] for item in metrics) / len(metrics)
        rows.append(row)
    return rows
