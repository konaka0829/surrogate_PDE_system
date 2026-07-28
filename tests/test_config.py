from __future__ import annotations

import json
from pathlib import Path

import pytest

from pol.config.loader import load_study_spec
from pol.config.models import (
    InterfaceDimensionsSpec,
    RandomFeatureRidgeReadoutSpec,
    TrialSpec,
)
from tests.helpers import write_tiny_stack


def test_dimensions_allow_n_tar_smaller_than_observation_dimension() -> None:
    value = InterfaceDimensionsSpec(n_tar=16, n_sur=64, J=32, q=9)
    assert value.n_tar < value.J


def test_dimensions_reject_only_representability_violations() -> None:
    with pytest.raises(ValueError, match="J must be <= n_sur"):
        InterfaceDimensionsSpec(n_tar=32, n_sur=16, J=17, q=9)
    with pytest.raises(ValueError, match="q must be <= n_tar"):
        InterfaceDimensionsSpec(n_tar=8, n_sur=32, J=16, q=9)


def test_unknown_configuration_key_reports_json_path(tmp_path: Path) -> None:
    _, _, study_path = write_tiny_stack(tmp_path)
    raw = json.loads(study_path.read_text(encoding="utf-8"))
    raw["base_trial"]["feature"]["unknown"] = 1
    study_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match=r"\$\.base_trial\.feature\.unknown"):
        load_study_spec(study_path, repo_root=tmp_path)


def test_cli_override_must_name_existing_field(tmp_path: Path) -> None:
    from pol.config.loader import load_study_with_overrides

    _, _, study_path = write_tiny_stack(tmp_path)
    with pytest.raises(ValueError, match="override path does not exist"):
        load_study_with_overrides(
            study_path,
            repo_root=tmp_path,
            overrides=["base_trial.feature.missing=4"],
        )


def test_trial_validation_keeps_dimensions_independent() -> None:
    trial = TrialSpec.model_validate(
        {
            "input": {"n_tar": 16},
            "feature": {
                "evolution": {"system": {"kind": "heat", "nu": 0.1}, "time": 0.2},
                "n_sur": 64,
                "observation": {"J": 32},
            },
            "output": {"q": 9},
            "readouts": [{"id": "direct", "kind": "direct_fourier_decoder"}],
        }
    )
    assert trial.input.n_tar == 16 and trial.feature.observation.J == 32


def test_static_feature_generator_has_no_hidden_evolution() -> None:
    trial = TrialSpec.model_validate(
        {
            "input": {"n_tar": 16},
            "feature": {
                "kind": "static_input",
                "n_sur": 32,
                "observation": {"J": 16},
            },
            "output": {"q": 9},
            "readouts": [{"id": "affine", "kind": "affine_ridge", "zetas": [0.0]}],
        }
    )
    assert trial.feature.kind == "static_input"
    assert trial.feature.evolution is None

    with pytest.raises(ValueError, match="must not define evolution"):
        TrialSpec.model_validate(
            {
                "input": {"n_tar": 16},
                "feature": {
                    "kind": "static_input",
                    "evolution": {
                        "system": {"kind": "heat", "nu": 0.1},
                        "time": 0.1,
                    },
                    "n_sur": 32,
                    "observation": {"J": 16},
                },
                "output": {"q": 9},
                "readouts": [
                    {"id": "affine", "kind": "affine_ridge", "zetas": [0.0]}
                ],
            }
        )


def _random_feature_readout(**overrides):
    values = {
        "id": "random",
        "kind": "random_feature_ridge",
        "widths": [4],
        "weight_scales": [0.5],
        "bias_scales": [0.1],
        "selection_seeds": [11],
        "evaluation_seeds": [21, 22],
        "zetas": [1e-8],
    }
    values.update(overrides)
    return values


def test_random_feature_readout_rejects_one_evaluation_seed() -> None:
    with pytest.raises(ValueError, match="at least two"):
        RandomFeatureRidgeReadoutSpec.model_validate(
            _random_feature_readout(evaluation_seeds=[21])
        )


def test_random_feature_readout_rejects_duplicate_seeds() -> None:
    with pytest.raises(ValueError, match="evaluation_seeds must be unique"):
        RandomFeatureRidgeReadoutSpec.model_validate(
            _random_feature_readout(evaluation_seeds=[21, 21])
        )


def test_random_feature_readout_rejects_selection_evaluation_overlap() -> None:
    with pytest.raises(ValueError, match="must be disjoint"):
        RandomFeatureRidgeReadoutSpec.model_validate(
            _random_feature_readout(
                selection_seeds=[11, 12],
                evaluation_seeds=[12, 21],
            )
        )
