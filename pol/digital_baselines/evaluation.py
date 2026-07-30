"""Training, checkpoint selection, and shared-metric evaluation."""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Mapping

import torch

from pol.data.finite import FiniteDataView
from pol.learning.metrics import (
    fourier_prediction_metrics,
    fourier_representation_floor,
)
from pol.math.fourier import real_fourier_analysis
from pol.runtime.device import require_cpu_tensors
from pol.runtime.hashing import stable_object_hash, tensor_sha256
from pol.study.evaluation import summarize_independent_seed_metrics

from .datasets import (
    denormalize_targets,
    normalize_inputs,
    normalize_targets,
)
from .fno1d import FNO1d, parameter_count
from .protocol import DigitalTrainingSpec, FNO1dCandidateSpec


@dataclass(frozen=True)
class TrainingOutcome:
    candidate_id: str
    seed: int
    seed_role: str
    best_epoch: int
    best_validation_metrics: dict[str, float]
    state_dict: dict[str, torch.Tensor]
    state_dict_hash: str
    parameter_count: int
    history: tuple[dict[str, Any], ...]
    wall_time_seconds: float
    process_time_seconds: float


def state_dict_content_hash(
    state_dict: Mapping[str, torch.Tensor],
) -> str:
    records = []
    for name, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError("FNO state_dict entries must be tensors")
        records.append(
            {
                "name": str(name),
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "sha256": tensor_sha256(value),
            }
        )
    return stable_object_hash(records)


def _clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    result = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    require_cpu_tensors(
        result,
        boundary="digital baseline checkpoint snapshot",
        name="state_dict",
    )
    return result


def build_fno(
    candidate: FNO1dCandidateSpec,
    *,
    n_tar: int,
    dtype: torch.dtype,
    seed: int,
    coordinate_channel: str,
    domain_length: float,
) -> FNO1d:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = FNO1d(
            candidate,
            n_tar=n_tar,
            coordinate_channel=coordinate_channel,
            domain_length=domain_length,
        ).to(
            device=torch.device("cpu"),
            dtype=dtype,
        )
    return model


def load_fno_checkpoint(
    candidate: FNO1dCandidateSpec,
    *,
    n_tar: int,
    dtype: torch.dtype,
    state_dict: Mapping[str, torch.Tensor],
    expected_hash: str,
    coordinate_channel: str,
    domain_length: float,
) -> FNO1d:
    actual_hash = state_dict_content_hash(state_dict)
    if actual_hash != expected_hash:
        raise ValueError("frozen FNO checkpoint content hash mismatch")
    model = FNO1d(
        candidate,
        n_tar=n_tar,
        coordinate_channel=coordinate_channel,
        domain_length=domain_length,
    ).to(
        device=torch.device("cpu"),
        dtype=dtype,
    )
    model.load_state_dict(dict(state_dict), strict=True)
    model.eval()
    return model


@torch.no_grad()
def predict_coefficients(
    model: FNO1d,
    finite_inputs: torch.Tensor,
    normalization: Mapping[str, object],
    *,
    q: int,
    domain_length: float,
    batch_size: int,
) -> torch.Tensor:
    predictions: list[torch.Tensor] = []
    for start in range(0, finite_inputs.shape[0], int(batch_size)):
        batch = normalize_inputs(
            finite_inputs[start : start + int(batch_size)],
            dict(normalization),
        )
        normalized_field = model(batch)
        normalized_coefficients = real_fourier_analysis(
            normalized_field,
            int(q),
            domain_length=domain_length,
        )
        predictions.append(
            denormalize_targets(
                normalized_coefficients,
                dict(normalization),
            )
        )
    result = torch.cat(predictions, dim=0)
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("FNO prediction contains NaN or Inf")
    return result


def prediction_metrics(
    prediction: torch.Tensor,
    view: FiniteDataView,
    *,
    domain_length: float,
) -> dict[str, float]:
    metrics = fourier_prediction_metrics(
        prediction,
        view.target_coefficients,
        view.targets,
        view.targets_reference,
        n_data=view.n_tar,
        n_reference=view.n_ref,
        domain_length=domain_length,
    )
    if not all(math.isfinite(float(value)) for value in metrics.values()):
        raise RuntimeError("digital baseline metric contains NaN or Inf")
    return metrics


def representation_floor(
    view: FiniteDataView,
    *,
    domain_length: float,
) -> dict[str, float]:
    return fourier_representation_floor(
        view.target_coefficients,
        view.targets,
        view.targets_reference,
        n_data=view.n_tar,
        n_reference=view.n_ref,
        domain_length=domain_length,
    )


def _prefixed(values: Mapping[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{key}": float(value) for key, value in values.items()}


def train_one_seed(
    candidate: FNO1dCandidateSpec,
    training: DigitalTrainingSpec,
    *,
    seed: int,
    seed_role: str,
    train_view: FiniteDataView,
    validation_view: FiniteDataView,
    normalization: Mapping[str, object],
    domain_length: float,
    coordinate_channel: str,
) -> TrainingOutcome:
    """Train for a fixed budget and select a checkpoint using validation only."""
    model = build_fno(
        candidate,
        n_tar=train_view.n_tar,
        dtype=train_view.inputs.dtype,
        seed=int(seed),
        coordinate_channel=coordinate_channel,
        domain_length=domain_length,
    )
    count = parameter_count(model)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training.optimizer.learning_rate),
        weight_decay=float(training.optimizer.weight_decay),
    )
    normalized_train_inputs = normalize_inputs(
        train_view.inputs,
        dict(normalization),
    )
    normalized_train_targets = normalize_targets(
        train_view.target_coefficients,
        dict(normalization),
    )
    started_wall = time.perf_counter()
    started_process = time.process_time()
    best_value = math.inf
    best_epoch = 0
    best_metrics: dict[str, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []

    for epoch in range(1, int(training.epochs) + 1):
        model.train()
        generator = torch.Generator(device="cpu").manual_seed(
            int(seed) + epoch * 1_000_003
        )
        order = torch.randperm(
            normalized_train_inputs.shape[0],
            generator=generator,
        )
        losses: list[float] = []
        for start in range(0, order.numel(), int(training.batch_size)):
            indices = order[start : start + int(training.batch_size)]
            batch_inputs = normalized_train_inputs.index_select(0, indices)
            batch_targets = normalized_train_targets.index_select(0, indices)
            optimizer.zero_grad(set_to_none=True)
            normalized_field = model(batch_inputs)
            predicted = real_fourier_analysis(
                normalized_field,
                train_view.q,
                domain_length=domain_length,
            )
            loss = torch.mean((predicted - batch_targets).square())
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(
                    "digital baseline training loss contains NaN or Inf"
                )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))

        model.eval()
        validation_prediction = predict_coefficients(
            model,
            validation_view.inputs,
            normalization,
            q=validation_view.q,
            domain_length=domain_length,
            batch_size=int(training.batch_size),
        )
        metrics = _prefixed(
            prediction_metrics(
                validation_prediction,
                validation_view,
                domain_length=domain_length,
            ),
            "validation",
        )
        value = float(metrics[training.checkpoint_metric])
        history.append(
            {
                "candidate_id": candidate.id,
                "seed": int(seed),
                "seed_role": seed_role,
                "epoch": epoch,
                "train_standardized_coefficient_mse": (
                    math.fsum(losses) / len(losses)
                ),
                **metrics,
            }
        )
        if value < best_value - float(training.checkpoint_tie_tolerance):
            best_value = value
            best_epoch = epoch
            best_metrics = dict(metrics)
            best_state = _clone_state_dict(model)

    if best_state is None or best_metrics is None:
        raise RuntimeError("digital baseline training selected no checkpoint")
    wall = time.perf_counter() - started_wall
    process = time.process_time() - started_process
    return TrainingOutcome(
        candidate_id=candidate.id,
        seed=int(seed),
        seed_role=seed_role,
        best_epoch=best_epoch,
        best_validation_metrics=best_metrics,
        state_dict=best_state,
        state_dict_hash=state_dict_content_hash(best_state),
        parameter_count=count,
        history=tuple(history),
        wall_time_seconds=float(wall),
        process_time_seconds=float(process),
    )


def summarize_seed_metrics(
    rows: list[dict[str, float]],
) -> dict[str, float]:
    return summarize_independent_seed_metrics(rows)


__all__ = [
    "TrainingOutcome",
    "load_fno_checkpoint",
    "parameter_count",
    "predict_coefficients",
    "prediction_metrics",
    "representation_floor",
    "state_dict_content_hash",
    "summarize_seed_metrics",
    "train_one_seed",
]
