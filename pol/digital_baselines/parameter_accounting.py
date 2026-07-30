"""Auditable real-scalar parameter accounting for fairness comparisons."""
from __future__ import annotations

from typing import Any, Mapping

import torch


PARAMETER_COUNT_DEFINITION_VERSION = "pol-real-scalar-parameter-count-v1"
PARAMETER_COUNT_SCOPE = "per_independent_model_realization"
PARAMETER_SCALAR_POLICY = "real_numel_complex_numel_times_two"


def _tensor(
    model: Mapping[str, Any],
    name: str,
) -> torch.Tensor:
    value = model.get(name)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"frozen model has no tensor {name}")
    if value.is_complex():
        raise ValueError(
            f"physical frozen tensor {name} must contain real scalars"
        )
    return value


def _base_counts(
    *,
    trainable: int,
    fixed_random: int,
    before: int | None,
    after: int | None,
    realization_count: int,
) -> dict[str, Any]:
    if realization_count <= 0:
        raise ValueError("frozen realization count must be positive")
    total = trainable + fixed_random
    return {
        "parameter_count_scope": PARAMETER_COUNT_SCOPE,
        "parameter_scalar_policy": PARAMETER_SCALAR_POLICY,
        "parameter_count_definition_version": (
            PARAMETER_COUNT_DEFINITION_VERSION
        ),
        "trainable_parameter_count": trainable,
        "fixed_random_parameter_count": fixed_random,
        "total_stored_parameter_count": total,
        "feature_dimension_before_readout": before,
        "feature_dimension_after_lift": after,
        "frozen_independent_realization_count": realization_count,
        "all_frozen_realizations_total_stored_parameter_count": (
            realization_count * total
        ),
        "primary_count_seed_multiplier_applied": False,
    }


def physical_parameter_counts(
    model: Mapping[str, Any],
    *,
    observation_count: int,
    q: int,
) -> dict[str, Any]:
    """Cross-check frozen physical tensors against semantic count formulas."""
    J = int(observation_count)
    output_dimension = int(q)
    if J <= 0 or output_dimension <= 0:
        raise ValueError("physical parameter dimensions must be positive")
    kind = model.get("kind")
    if kind == "direct_fourier_decoder":
        if (
            int(model.get("q", -1)) != output_dimension
            or int(model.get("decoder_observation_count", -1)) != J
        ):
            raise ValueError("direct decoder dimension mismatch")
        return _base_counts(
            trainable=0,
            fixed_random=0,
            before=J,
            after=J,
            realization_count=1,
        )
    if kind == "affine_ridge":
        W = _tensor(model, "W")
        b = _tensor(model, "b")
        if tuple(W.shape) != (output_dimension, J) or tuple(b.shape) != (
            output_dimension,
        ):
            raise ValueError("frozen affine W/b shape mismatch")
        formula = output_dimension * (J + 1)
        tensor_count = int(W.numel() + b.numel())
        if tensor_count != formula:
            raise ValueError("frozen affine tensor count/formula mismatch")
        return _base_counts(
            trainable=tensor_count,
            fixed_random=0,
            before=J,
            after=J,
            realization_count=1,
        )
    if kind == "random_feature_ridge":
        width = int(model.get("width", -1))
        members = model.get("members")
        if width <= 0 or not isinstance(members, list) or not members:
            raise ValueError("frozen random-feature model has invalid members")
        expected_fixed = width * (J + 1)
        expected_trainable = output_dimension * (J + width + 1)
        for member in members:
            if not isinstance(member, Mapping):
                raise ValueError("frozen random-feature member is invalid")
            A = _tensor(member, "A")
            c = _tensor(member, "c")
            W = _tensor(member, "W")
            b = _tensor(member, "b")
            if tuple(A.shape) != (width, J) or tuple(c.shape) != (width,):
                raise ValueError("frozen random-feature A/c shape mismatch")
            if tuple(W.shape) != (
                output_dimension,
                J + width,
            ) or tuple(b.shape) != (output_dimension,):
                raise ValueError("frozen random-feature W/b shape mismatch")
            fixed = int(A.numel() + c.numel())
            trainable = int(W.numel() + b.numel())
            if (
                fixed != expected_fixed
                or trainable != expected_trainable
            ):
                raise ValueError(
                    "frozen random-feature tensor count/formula mismatch"
                )
        return _base_counts(
            trainable=expected_trainable,
            fixed_random=expected_fixed,
            before=J,
            after=J + width,
            realization_count=len(members),
        )
    raise ValueError(f"unknown frozen physical readout kind: {kind}")


def state_dict_real_scalar_count(
    state_dict: Mapping[str, Any],
) -> int:
    """Count stored real scalars, counting each complex entry as two."""
    if not state_dict:
        raise ValueError("frozen state_dict must not be empty")
    total = 0
    for name, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"frozen state_dict entry is not a tensor: {name}")
        total += int(value.numel()) * (2 if value.is_complex() else 1)
    return total


def fno_parameter_counts(
    archive: Mapping[str, Any],
) -> dict[str, Any]:
    """Cross-check each independent FNO checkpoint against training counts."""
    models = archive.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("digital checkpoint archive has no model realizations")
    counts: list[int] = []
    for model in models:
        if not isinstance(model, Mapping):
            raise ValueError("digital frozen model record is invalid")
        state = model.get("state_dict")
        if not isinstance(state, Mapping):
            raise ValueError("digital frozen model has no state_dict")
        count = state_dict_real_scalar_count(state)
        if count != int(model.get("parameter_count", -1)):
            raise ValueError(
                "FNO frozen state_dict count disagrees with training outcome"
            )
        counts.append(count)
    if len(set(counts)) != 1:
        raise ValueError("FNO realizations have different parameter counts")
    return _base_counts(
        trainable=counts[0],
        fixed_random=0,
        before=None,
        after=None,
        realization_count=len(models),
    )


__all__ = [
    "PARAMETER_COUNT_DEFINITION_VERSION",
    "PARAMETER_COUNT_SCOPE",
    "PARAMETER_SCALAR_POLICY",
    "fno_parameter_counts",
    "physical_parameter_counts",
    "state_dict_real_scalar_count",
]
