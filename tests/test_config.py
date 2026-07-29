from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from pol.config.loader import load_dataset_spec, load_study_spec, load_validation_spec
from pol.config.models import (
    InterfaceDimensionsSpec,
    RandomFeatureRidgeReadoutSpec,
    TrialSpec,
)
from pol.data.splits import build_data_split, calibration_split_provenance
from tests.helpers import write_json, write_tiny_heat_stack, write_tiny_stack


def test_validation_modules_import_without_a_package_cycle() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import pol.validation.conditions; "
                "import pol.validation.binding; "
                "import pol.validation.runner; "
                "from pol.validation import ensure_validation; "
                "from pol.data import ensure_dataset"
            ),
        ],
        cwd=repo_root,
        check=True,
    )


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


def test_dataset_binding_rejects_unknown_key(tmp_path: Path) -> None:
    _, dataset_path, _ = write_tiny_stack(tmp_path)
    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    raw["binding"]["unknown"] = True
    dataset_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match=r"\$\.binding.*unknown"):
        load_dataset_spec(dataset_path, repo_root=tmp_path)


def test_foundation_only_binding_requires_nonempty_reason(tmp_path: Path) -> None:
    _, dataset_path, _ = write_tiny_stack(tmp_path)
    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    raw["binding"].pop("reason")
    dataset_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match=r"\$\.binding.*reason"):
        load_dataset_spec(dataset_path, repo_root=tmp_path)

    raw["binding"]["reason"] = "   "
    dataset_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="must not be blank"):
        load_dataset_spec(dataset_path, repo_root=tmp_path)


def test_heat_analytic_validation_spec_parses_without_time_candidates(
    tmp_path: Path,
) -> None:
    validation_path, _, _ = write_tiny_heat_stack(tmp_path)
    spec = load_validation_spec(validation_path, repo_root=tmp_path)
    assert spec.schema_version == "pol-validation-v5"
    assert spec.target_reference.kind == "heat_analytic"
    assert spec.target_reference.reference_evolution.system.kind == "heat"
    assert not hasattr(spec.target_reference, "time_candidates")


def test_heat_analytic_validation_rejects_burgers_fields_and_system(
    tmp_path: Path,
) -> None:
    validation_path, _, _ = write_tiny_heat_stack(tmp_path)
    raw = json.loads(validation_path.read_text(encoding="utf-8"))
    raw["target_reference"]["time_candidates"] = [
        {
            "solver": "split_step",
            "dt": 0.01,
            "fine_dt": 0.005,
            "dealias": True,
        }
    ]
    validation_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match=r"target_reference.*time_candidates"):
        load_validation_spec(validation_path, repo_root=tmp_path)

    raw["target_reference"].pop("time_candidates")
    raw["target_reference"]["reference_evolution"]["system"] = {
        "kind": "burgers",
        "nu": 0.1,
        "solver": "split_step",
        "dt": 0.01,
        "fine_dt": 0.005,
        "dealias": True,
    }
    validation_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="requires a heat reference evolution"):
        load_validation_spec(validation_path, repo_root=tmp_path)


def test_target_reference_union_rejects_unknown_kind_and_key(
    tmp_path: Path,
) -> None:
    validation_path, _, _ = write_tiny_heat_stack(tmp_path)
    raw = json.loads(validation_path.read_text(encoding="utf-8"))
    raw["target_reference"]["kind"] = "unknown_reference"
    validation_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match=r"target_reference.*kind"):
        load_validation_spec(validation_path, repo_root=tmp_path)

    raw["target_reference"]["kind"] = "heat_analytic"
    raw["target_reference"]["unknown"] = True
    validation_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match=r"target_reference.*unknown"):
        load_validation_spec(validation_path, repo_root=tmp_path)


def test_burgers_validation_uses_discriminated_target_reference(
    tmp_path: Path,
) -> None:
    validation_path, _, _ = write_tiny_stack(tmp_path)
    spec = load_validation_spec(validation_path, repo_root=tmp_path)
    assert spec.target_reference.kind == "burgers_convergence"
    assert len(spec.target_reference.time_candidates) == 2


def _validation_with_time_candidates(
    tmp_path: Path,
    candidates: list[dict[str, object]],
):
    validation_path, _, _ = write_tiny_stack(tmp_path)
    raw = json.loads(validation_path.read_text(encoding="utf-8"))
    target = raw["target_reference"]
    target["time_candidates"] = candidates
    target["reference_evolution"]["system"].update(candidates[-1])
    write_json(validation_path, raw)
    return validation_path


def _enabled_cross_solver_validation() -> dict[str, object]:
    return {
        "schema_version": "pol-burgers-cross-solver-spec-v1",
        "enabled": True,
        "context": {
            "system_kind": "burgers",
            "nu": 0.05,
            "advection_coefficient": 1.0,
            "final_time": 0.02,
            "domain_length": 1.0,
            "dtype": "float64",
            "dealias": True,
        },
        "solvers": {
            "split_step": {
                "candidates": [
                    {
                        "solver": "semi_implicit",
                        "dt": 0.01,
                        "fine_dt": 0.005,
                        "dealias": True,
                    },
                    {
                        "solver": "split_step",
                        "dt": 0.01,
                        "fine_dt": 0.0025,
                        "dealias": True,
                    },
                ]
            },
            "etdrk4": {
                "candidates": [
                    {
                        "solver": "fourier_pseudospectral_etdrk4",
                        "dt": 0.01,
                        "fine_dt": None,
                        "dealias": True,
                    },
                    {
                        "solver": "etdrk4",
                        "dt": 0.005,
                        "fine_dt": None,
                        "dealias": True,
                    },
                ]
            },
        },
        "tolerances": {
            "mean_relative_l2": 1.0,
            "max_relative_l2": 1.0,
            "low_mode_relative_l2": 1.0,
        },
    }


def _validation_with_cross_solver(
    tmp_path: Path,
    diagnostic: dict[str, object],
) -> Path:
    validation_path, _, _ = write_tiny_stack(tmp_path)
    raw = json.loads(validation_path.read_text(encoding="utf-8"))
    raw["target_reference"]["cross_solver_validation"] = diagnostic
    write_json(validation_path, raw)
    return validation_path


def test_cross_solver_spec_accepts_independent_canonical_families(
    tmp_path: Path,
) -> None:
    path = _validation_with_cross_solver(
        tmp_path,
        _enabled_cross_solver_validation(),
    )
    diagnostic = load_validation_spec(
        path,
        repo_root=tmp_path,
    ).target_reference.cross_solver_validation
    assert diagnostic.enabled
    assert len(diagnostic.solvers.split_step.candidates) == 2
    assert len(diagnostic.solvers.etdrk4.candidates) == 2


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("family_mixing", "solver family"),
        ("duplicate_etdrk4", "strictly decreasing"),
        ("reversed_etdrk4", "strictly decreasing"),
        ("etdrk4_fine_dt", "fine_dt=null"),
        ("dealias_mismatch", "dealias policy"),
        ("primary_dealias_mismatch", "primary reference"),
        ("pde_mismatch", "PDE parameters and final time"),
        ("time_mismatch", "PDE parameters and final time"),
        ("domain_mismatch", "domain length and dtype"),
        ("dtype_mismatch", "domain length and dtype"),
    ],
)
def test_cross_solver_spec_rejects_malformed_or_mismatched_diagnostic(
    tmp_path: Path,
    change: str,
    message: str,
) -> None:
    diagnostic = _enabled_cross_solver_validation()
    solvers = diagnostic["solvers"]
    context = diagnostic["context"]
    if change == "family_mixing":
        solvers["split_step"]["candidates"][1] = {
            "solver": "etdrk4",
            "dt": 0.005,
            "fine_dt": None,
            "dealias": True,
        }
    elif change == "duplicate_etdrk4":
        solvers["etdrk4"]["candidates"][1]["dt"] = 0.01
    elif change == "reversed_etdrk4":
        solvers["etdrk4"]["candidates"].reverse()
    elif change == "etdrk4_fine_dt":
        solvers["etdrk4"]["candidates"][1]["fine_dt"] = 0.001
    elif change == "dealias_mismatch":
        solvers["etdrk4"]["candidates"][1]["dealias"] = False
    elif change == "primary_dealias_mismatch":
        context["dealias"] = False
        for family in ("split_step", "etdrk4"):
            for candidate in solvers[family]["candidates"]:
                candidate["dealias"] = False
    elif change == "pde_mismatch":
        context["nu"] = 0.04
    elif change == "time_mismatch":
        context["final_time"] = 0.01
    elif change == "domain_mismatch":
        context["domain_length"] = 2.0
    else:
        context["dtype"] = "float32"
    path = _validation_with_cross_solver(tmp_path, diagnostic)
    with pytest.raises(ValueError, match=message):
        load_validation_spec(path, repo_root=tmp_path)


def test_cross_solver_spec_rejects_unknown_key_and_solver(
    tmp_path: Path,
) -> None:
    diagnostic = _enabled_cross_solver_validation()
    diagnostic["unknown"] = True
    path = _validation_with_cross_solver(tmp_path, diagnostic)
    with pytest.raises(ValueError, match="unknown"):
        load_validation_spec(path, repo_root=tmp_path)

    diagnostic = _enabled_cross_solver_validation()
    diagnostic["solvers"]["split_step"]["candidates"][0]["solver"] = (
        "unknown"
    )
    path = _validation_with_cross_solver(tmp_path, diagnostic)
    with pytest.raises(ValueError, match="solver"):
        load_validation_spec(path, repo_root=tmp_path)


def test_split_step_refinement_accepts_effective_substep_only(
    tmp_path: Path,
) -> None:
    path = _validation_with_time_candidates(
        tmp_path,
        [
            {
                "solver": "semi_implicit",
                "dt": 0.01,
                "fine_dt": 0.005,
                "dealias": True,
            },
            {
                "solver": "split_step",
                "dt": 0.01,
                "fine_dt": 0.0025,
                "dealias": True,
            },
        ],
    )
    spec = load_validation_spec(path, repo_root=tmp_path)
    assert spec.target_reference.time_candidates[0].dt == (
        spec.target_reference.time_candidates[1].dt
    )


def test_split_step_refinement_accepts_outer_and_effective_steps(
    tmp_path: Path,
) -> None:
    path = _validation_with_time_candidates(
        tmp_path,
        [
            {
                "solver": "split_step",
                "dt": 0.02,
                "fine_dt": 0.01,
                "dealias": True,
            },
            {
                "solver": "split_step",
                "dt": 0.01,
                "fine_dt": 0.0025,
                "dealias": True,
            },
        ],
    )
    load_validation_spec(path, repo_root=tmp_path)


@pytest.mark.parametrize(
    ("candidates", "message"),
    [
        (
            [
                {
                    "solver": "split_step",
                    "dt": 0.01,
                    "fine_dt": 0.0025,
                    "dealias": True,
                },
                {
                    "solver": "split_step",
                    "dt": 0.01,
                    "fine_dt": 0.005,
                    "dealias": True,
                },
            ],
            "coarse-to-fine order",
        ),
        (
            [
                {
                    "solver": "split_step",
                    "dt": 0.01,
                    "fine_dt": 0.006,
                    "dealias": True,
                },
                {
                    "solver": "split_step",
                    "dt": 0.01,
                    "fine_dt": 0.0051,
                    "dealias": True,
                },
            ],
            "actual effective substep",
        ),
        (
            [
                {
                    "solver": "split_step",
                    "dt": 0.01,
                    "fine_dt": 0.005,
                    "dealias": True,
                },
                {
                    "solver": "etdrk4",
                    "dt": 0.005,
                    "fine_dt": None,
                    "dealias": True,
                },
            ],
            "solver family",
        ),
        (
            [
                {
                    "solver": "split_step",
                    "dt": 0.01,
                    "fine_dt": 0.005,
                    "dealias": True,
                },
                {
                    "solver": "split_step",
                    "dt": 0.01,
                    "fine_dt": 0.0025,
                    "dealias": False,
                },
            ],
            "dealias policy",
        ),
        (
            [
                {
                    "solver": "split_step",
                    "dt": 0.01,
                    "fine_dt": 0.005,
                    "dealias": True,
                },
                {
                    "solver": "split_step",
                    "dt": 0.006,
                    "fine_dt": 0.002,
                    "dealias": True,
                },
            ],
            "must align",
        ),
    ],
)
def test_burgers_refinement_rejects_malformed_sequence_before_solve(
    tmp_path: Path,
    candidates: list[dict[str, object]],
    message: str,
) -> None:
    path = _validation_with_time_candidates(tmp_path, candidates)
    with pytest.raises(ValueError, match=message):
        load_validation_spec(path, repo_root=tmp_path)


def test_etdrk4_candidate_rejects_nonnull_fine_dt(
    tmp_path: Path,
) -> None:
    path = _validation_with_time_candidates(
        tmp_path,
        [
            {
                "solver": "etdrk4",
                "dt": 0.01,
                "fine_dt": None,
                "dealias": True,
            },
            {
                "solver": "etdrk4",
                "dt": 0.005,
                "fine_dt": 0.001,
                "dealias": True,
            },
        ],
    )
    with pytest.raises(ValueError, match="fine_dt=null"):
        load_validation_spec(path, repo_root=tmp_path)


def test_etdrk4_refinement_accepts_strictly_decreasing_dt(
    tmp_path: Path,
) -> None:
    path = _validation_with_time_candidates(
        tmp_path,
        [
            {
                "solver": "fourier_pseudospectral_etdrk4",
                "dt": 0.01,
                "fine_dt": None,
                "dealias": True,
            },
            {
                "solver": "etdrk4",
                "dt": 0.005,
                "fine_dt": None,
                "dealias": True,
            },
        ],
    )
    load_validation_spec(path, repo_root=tmp_path)


def _reaction_validation_with_candidates(
    tmp_path: Path,
    candidates: list[dict[str, object]],
) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    raw = json.loads(
        (
            repo_root
            / "configs/validation/reaction_diffusion_smoke.json"
        ).read_text(encoding="utf-8")
    )
    raw["name"] = "tiny_reaction_diffusion_config_test"
    raw["artifact_root"] = str(tmp_path / "artifacts")
    raw["target_reference"]["time_candidates"] = candidates
    raw["target_reference"]["reference_evolution"]["system"].update(
        candidates[-1]
    )
    return write_json(tmp_path / "reaction_validation.json", raw)


def test_reaction_diffusion_refinement_accepts_strictly_decreasing_dt(
    tmp_path: Path,
) -> None:
    path = _reaction_validation_with_candidates(
        tmp_path,
        [
            {
                "solver": "semi_implicit_spectral_euler",
                "dt": 0.005,
                "nonlinear_filter": "two_thirds",
            },
            {
                "solver": "semi_implicit_spectral_euler",
                "dt": 0.0025,
                "nonlinear_filter": "two_thirds",
            },
        ],
    )
    spec = load_validation_spec(path, repo_root=tmp_path)
    assert spec.schema_version == "pol-validation-v6"
    assert spec.target_reference.kind == "reaction_diffusion_convergence"


@pytest.mark.parametrize(
    ("candidates", "message"),
    [
        (
            [
                {
                    "solver": "semi_implicit_spectral_euler",
                    "dt": 0.0025,
                    "nonlinear_filter": "two_thirds",
                },
                {
                    "solver": "semi_implicit_spectral_euler",
                    "dt": 0.005,
                    "nonlinear_filter": "two_thirds",
                },
            ],
            "strictly decreasing",
        ),
        (
            [
                {
                    "solver": "semi_implicit_spectral_euler",
                    "dt": 0.0025,
                    "nonlinear_filter": "two_thirds",
                },
                {
                    "solver": "semi_implicit_spectral_euler",
                    "dt": 0.0025,
                    "nonlinear_filter": "two_thirds",
                },
            ],
            "strictly decreasing",
        ),
        (
            [
                {
                    "solver": "semi_implicit_spectral_euler",
                    "dt": 0.005,
                    "nonlinear_filter": "none",
                },
                {
                    "solver": "semi_implicit_spectral_euler",
                    "dt": 0.0025,
                    "nonlinear_filter": "two_thirds",
                },
            ],
            "filter switching is not refinement",
        ),
        (
            [
                {
                    "solver": "semi_implicit_spectral_euler",
                    "dt": 0.005,
                    "nonlinear_filter": "two_thirds",
                },
                {
                    "solver": "semi_implicit_spectral_euler",
                    "dt": 0.003,
                    "nonlinear_filter": "two_thirds",
                },
            ],
            "align",
        ),
    ],
)
def test_reaction_diffusion_refinement_rejects_invalid_sequence(
    tmp_path: Path,
    candidates: list[dict[str, object]],
    message: str,
) -> None:
    path = _reaction_validation_with_candidates(tmp_path, candidates)
    with pytest.raises(ValueError, match=message):
        load_validation_spec(path, repo_root=tmp_path)


def test_reaction_diffusion_reference_condition_must_equal_finest_candidate(
    tmp_path: Path,
) -> None:
    path = _reaction_validation_with_candidates(
        tmp_path,
        [
            {
                "solver": "semi_implicit_spectral_euler",
                "dt": 0.005,
                "nonlinear_filter": "two_thirds",
            },
            {
                "solver": "semi_implicit_spectral_euler",
                "dt": 0.0025,
                "nonlinear_filter": "two_thirds",
            },
        ],
    )
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["target_reference"]["reference_evolution"]["system"]["dt"] = 0.005
    write_json(path, raw)
    with pytest.raises(ValueError, match="must equal the finest"):
        load_validation_spec(path, repo_root=tmp_path)


@pytest.mark.parametrize("field", ["alpha", "beta", "dt"])
def test_reaction_diffusion_spec_rejects_nonfinite_parameters(
    tmp_path: Path,
    field: str,
) -> None:
    path = _reaction_validation_with_candidates(
        tmp_path,
        [
            {
                "solver": "semi_implicit_spectral_euler",
                "dt": 0.005,
                "nonlinear_filter": "two_thirds",
            },
            {
                "solver": "semi_implicit_spectral_euler",
                "dt": 0.0025,
                "nonlinear_filter": "two_thirds",
            },
        ],
    )
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["target_reference"]["reference_evolution"]["system"][field] = (
        float("nan")
    )
    write_json(path, raw)
    with pytest.raises(ValueError, match="finite|greater than"):
        load_validation_spec(path, repo_root=tmp_path)


def test_legacy_dataset_schema_has_actionable_migration_error(
    tmp_path: Path,
) -> None:
    _, dataset_path, _ = write_tiny_stack(tmp_path)
    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    raw["schema_version"] = "pol-dataset-v1"
    dataset_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported legacy dataset schema"):
        load_dataset_spec(dataset_path, repo_root=tmp_path)


def test_legacy_validation_schema_has_actionable_migration_error(
    tmp_path: Path,
) -> None:
    validation_path, _, _ = write_tiny_stack(tmp_path)
    raw = json.loads(validation_path.read_text(encoding="utf-8"))
    raw["schema_version"] = "pol-validation-v4"
    validation_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported legacy validation schema"):
        load_validation_spec(validation_path, repo_root=tmp_path)


def test_checked_in_dataset_bindings_are_target_specific() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    burgers_smoke = load_validation_spec(
        repo_root / "configs/validation/foundation_smoke.json",
        repo_root=repo_root,
    )
    burgers_main = load_validation_spec(
        repo_root / "configs/validation/foundation_main.json",
        repo_root=repo_root,
    )
    load_validation_spec(
        repo_root / "configs/validation/heat_smoke.json",
        repo_root=repo_root,
    )
    load_validation_spec(
        repo_root / "configs/validation/heat_main.json",
        repo_root=repo_root,
    )
    reaction_smoke = load_validation_spec(
        repo_root
        / "configs/validation/reaction_diffusion_smoke.json",
        repo_root=repo_root,
    )
    reaction_main = load_validation_spec(
        repo_root
        / "configs/validation/reaction_diffusion_main.json",
        repo_root=repo_root,
    )
    assert reaction_smoke.profile == "smoke"
    assert reaction_main.profile == "main"
    assert burgers_smoke.target_reference.cross_solver_validation.enabled
    assert not burgers_main.target_reference.cross_solver_validation.enabled
    assert {
        family: [
            candidate.solver
            for candidate in getattr(
                burgers_smoke.target_reference.cross_solver_validation.solvers,
                family,
            ).candidates
        ]
        for family in ("split_step", "etdrk4")
    } == {
        "split_step": ["semi_implicit", "split_step"],
        "etdrk4": ["fourier_pseudospectral_etdrk4", "etdrk4"],
    }
    expected = {
        "burgers_smoke.json": ("validated_reference", None),
        "burgers_main.json": ("validated_reference", None),
        "heat_smoke.json": ("validated_reference", None),
        "heat_main.json": ("validated_reference", None),
    }
    for name, (kind, reason_field) in expected.items():
        spec = load_dataset_spec(
            repo_root / "configs/datasets" / name,
            repo_root=repo_root,
        )
        assert spec.binding.kind == kind
        if reason_field is not None:
            assert spec.binding.reason.strip()


@pytest.mark.parametrize(
    ("filename", "expected_ids", "expected_membership"),
    [
        (
            "foundation_smoke.json",
            (0, 1),
            {"0": "train", "1": "train"},
        ),
        (
            "foundation_main.json",
            (0, 1, 4, 6),
            {"0": "train", "1": "train", "4": "train", "6": "train"},
        ),
        (
            "heat_smoke.json",
            (0, 1),
            {"0": "train", "1": "train"},
        ),
        (
            "heat_main.json",
            (0, 1, 4, 6),
            {"0": "train", "1": "train", "4": "train", "6": "train"},
        ),
        (
            "reaction_diffusion_smoke.json",
            (0, 1),
            {"0": "train", "1": "train"},
        ),
        (
            "reaction_diffusion_main.json",
            (0, 1, 4, 6),
            {"0": "train", "1": "train", "4": "train", "6": "train"},
        ),
    ],
)
def test_checked_in_calibration_ids_are_deterministically_non_test(
    filename: str,
    expected_ids: tuple[int, ...],
    expected_membership: dict[str, str],
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    spec = load_validation_spec(
        repo_root / "configs/validation" / filename,
        repo_root=repo_root,
    )
    samples = spec.samples
    split = build_data_split(
        total_samples=int(samples.total_samples),
        n_train=int(samples.n_train),
        n_validation=int(samples.n_validation),
        n_test=int(samples.n_test),
        seed=int(samples.seed),
    )
    provenance = calibration_split_provenance(
        split,
        spec.target_reference.calibration_sample_ids,
        sample_ids=torch.arange(split.total_samples, dtype=torch.long),
    )
    assert spec.target_reference.calibration_sample_ids == expected_ids
    assert provenance["calibration_split_membership"] == expected_membership
    assert provenance["calibration_test_overlap_count"] == 0
