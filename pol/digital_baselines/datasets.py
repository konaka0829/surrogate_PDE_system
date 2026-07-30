"""Finite-interface dataset views and train-only normalization."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from pol.data.dataset import ReferenceDataset
from pol.data.finite import FiniteDataView, derive_finite_view
from pol.runtime.device import require_cpu_tensors


@dataclass(frozen=True)
class SelectionViews:
    train: FiniteDataView
    validation: FiniteDataView


def _finite_view(
    dataset: ReferenceDataset,
    ids: torch.Tensor,
    *,
    n_tar: int,
    q: int,
) -> FiniteDataView:
    inputs, targets = dataset.tensors_for(ids)
    view = derive_finite_view(
        ids,
        inputs,
        targets,
        n_tar=n_tar,
        q=q,
        domain_length=dataset.domain_length,
    )
    require_cpu_tensors(
        view.__dict__,
        boundary="digital baseline finite-interface view",
        name="finite_view",
    )
    return view


def build_selection_views(
    dataset: ReferenceDataset,
    *,
    n_tar: int,
    q: int,
) -> SelectionViews:
    """Build only train/validation finite views before the freeze boundary."""
    return SelectionViews(
        train=_finite_view(
            dataset,
            dataset.train_ids,
            n_tar=n_tar,
            q=q,
        ),
        validation=_finite_view(
            dataset,
            dataset.validation_ids,
            n_tar=n_tar,
            q=q,
        ),
    )


def build_test_view(
    dataset: ReferenceDataset,
    *,
    n_tar: int,
    q: int,
) -> FiniteDataView:
    """The sole digital-baseline test-tensor request boundary."""
    return _finite_view(
        dataset,
        dataset.test_ids,
        n_tar=n_tar,
        q=q,
    )


def train_normalization(
    train: FiniteDataView,
    *,
    epsilon: float,
) -> dict[str, torch.Tensor | float | int | str]:
    """Fit deterministic standard-score statistics on the train split only."""
    input_mean = train.inputs.mean()
    input_std_raw = train.inputs.std(unbiased=False)
    target_mean = train.target_coefficients.mean(dim=0)
    target_std_raw = train.target_coefficients.std(dim=0, unbiased=False)
    input_scale = input_std_raw.clamp_min(float(epsilon))
    target_scale = target_std_raw.clamp_min(float(epsilon))
    result: dict[str, torch.Tensor | float | int | str] = {
        "schema_version": "pol-digital-normalization-v1",
        "kind": "train_standard_score",
        "epsilon": float(epsilon),
        "input_mean": input_mean.detach().clone(),
        "input_scale": input_scale.detach().clone(),
        "target_mean": target_mean.detach().clone(),
        "target_scale": target_scale.detach().clone(),
        "input_clamped_count": int(input_std_raw < float(epsilon)),
        "target_clamped_count": int(
            (target_std_raw < float(epsilon)).sum().item()
        ),
    }
    require_cpu_tensors(
        result,
        boundary="digital baseline train-only normalization",
        name="normalization",
    )
    return result


def normalize_inputs(
    inputs: torch.Tensor,
    normalization: dict[str, object],
) -> torch.Tensor:
    return (
        inputs - normalization["input_mean"]  # type: ignore[operator]
    ) / normalization["input_scale"]  # type: ignore[operator]


def normalize_targets(
    coefficients: torch.Tensor,
    normalization: dict[str, object],
) -> torch.Tensor:
    return (
        coefficients - normalization["target_mean"]  # type: ignore[operator]
    ) / normalization["target_scale"]  # type: ignore[operator]


def denormalize_targets(
    coefficients: torch.Tensor,
    normalization: dict[str, object],
) -> torch.Tensor:
    return (
        coefficients * normalization["target_scale"]  # type: ignore[operator]
        + normalization["target_mean"]  # type: ignore[operator]
    )


__all__ = [
    "SelectionViews",
    "build_selection_views",
    "build_test_view",
    "denormalize_targets",
    "normalize_inputs",
    "normalize_targets",
    "train_normalization",
]
