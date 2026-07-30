from __future__ import annotations

import pytest
import torch

from pol.digital_baselines.parameter_accounting import (
    PARAMETER_COUNT_DEFINITION_VERSION,
    PARAMETER_COUNT_SCOPE,
    fno_parameter_counts,
    physical_parameter_counts,
    state_dict_real_scalar_count,
)


def _random_model(J: int, M: int, q: int, seeds: int = 2):
    return {
        "kind": "random_feature_ridge",
        "width": M,
        "members": [
            {
                "A": torch.zeros(M, J),
                "c": torch.zeros(M),
                "W": torch.zeros(q, J + M),
                "b": torch.zeros(q),
            }
            for _ in range(seeds)
        ],
    }


@pytest.mark.parametrize(
    ("J", "M", "q"),
    [(2, 1, 1), (5, 3, 7), (16, 4, 17), (11, 8, 5)],
)
def test_random_feature_parameter_formula_matches_frozen_tensors(
    J: int,
    M: int,
    q: int,
) -> None:
    counts = physical_parameter_counts(
        _random_model(J, M, q),
        observation_count=J,
        q=q,
    )
    fixed = M * (J + 1)
    trainable = q * (J + M + 1)
    assert counts["fixed_random_parameter_count"] == fixed
    assert counts["trainable_parameter_count"] == trainable
    assert counts["total_stored_parameter_count"] == fixed + trainable
    assert counts["feature_dimension_before_readout"] == J
    assert counts["feature_dimension_after_lift"] == J + M
    assert counts["parameter_count_scope"] == PARAMETER_COUNT_SCOPE
    assert (
        counts["parameter_count_definition_version"]
        == PARAMETER_COUNT_DEFINITION_VERSION
    )
    assert counts["primary_count_seed_multiplier_applied"] is False
    assert counts[
        "all_frozen_realizations_total_stored_parameter_count"
    ] == 2 * (fixed + trainable)


def test_smoke_random_feature_count_is_425_per_realization() -> None:
    counts = physical_parameter_counts(
        _random_model(16, 4, 17),
        observation_count=16,
        q=17,
    )
    assert counts["fixed_random_parameter_count"] == 68
    assert counts["trainable_parameter_count"] == 357
    assert counts["total_stored_parameter_count"] == 425


def test_direct_and_affine_counts_cross_check_frozen_tensors() -> None:
    direct = physical_parameter_counts(
        {
            "kind": "direct_fourier_decoder",
            "q": 9,
            "decoder_observation_count": 6,
        },
        observation_count=6,
        q=9,
    )
    assert direct["trainable_parameter_count"] == 0
    assert direct["fixed_random_parameter_count"] == 0
    assert direct["total_stored_parameter_count"] == 0

    affine = physical_parameter_counts(
        {
            "kind": "affine_ridge",
            "W": torch.zeros(9, 6),
            "b": torch.zeros(9),
        },
        observation_count=6,
        q=9,
    )
    assert affine["trainable_parameter_count"] == 9 * 7
    assert affine["fixed_random_parameter_count"] == 0
    assert affine["total_stored_parameter_count"] == 9 * 7


@pytest.mark.parametrize(
    "mutation",
    ["A_shape", "W_shape", "readout_kind"],
)
def test_physical_parameter_accounting_rejects_tensor_or_kind_tampering(
    mutation: str,
) -> None:
    model = _random_model(5, 3, 7)
    if mutation == "A_shape":
        model["members"][0]["A"] = torch.zeros(3, 4)
    elif mutation == "W_shape":
        model["members"][0]["W"] = torch.zeros(7, 7)
    else:
        model["kind"] = "tampered_readout_kind"
    with pytest.raises(ValueError):
        physical_parameter_counts(model, observation_count=5, q=7)


def test_fno_count_uses_real_scalars_and_per_realization_scope() -> None:
    state = {
        "real": torch.zeros(3),
        "complex": torch.zeros(2, dtype=torch.complex64),
    }
    assert state_dict_real_scalar_count(state) == 7
    archive = {
        "models": [
            {"state_dict": state, "parameter_count": 7},
            {
                "state_dict": {
                    key: value.clone() for key, value in state.items()
                },
                "parameter_count": 7,
            },
        ]
    }
    counts = fno_parameter_counts(archive)
    assert counts["trainable_parameter_count"] == 7
    assert counts["fixed_random_parameter_count"] == 0
    assert counts["total_stored_parameter_count"] == 7
    assert counts["frozen_independent_realization_count"] == 2
    assert counts[
        "all_frozen_realizations_total_stored_parameter_count"
    ] == 14
    assert counts["primary_count_seed_multiplier_applied"] is False


def test_fno_count_rejects_training_outcome_or_shape_count_mismatch() -> None:
    archive = {
        "models": [
            {
                "state_dict": {"weight": torch.zeros(2, 3)},
                "parameter_count": 5,
            }
        ]
    }
    with pytest.raises(ValueError, match="training outcome"):
        fno_parameter_counts(archive)
