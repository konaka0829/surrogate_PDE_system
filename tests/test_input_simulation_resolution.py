from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from pol.config.loader import load_dataset_spec, load_study_spec
from pol.data.dataset import ensure_dataset
from pol.data.finite import build_feature_initial_state
from pol.math.periodic import periodic_grid, spectral_resample_periodic
from pol.runtime.artifacts import manifest_records
from pol.runtime.hashing import stable_object_hash
from pol.runtime.io import write_csv, write_strict_json
from pol.study.cache import FeatureStateCache
from pol.study.cases import build_cases, scientific_study_spec
from pol.study.overrides import apply_trial_overrides
from pol.study.runner import regenerate_plots, run_study, verify_study_run
from tests.helpers import write_json, write_tiny_stack


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _checked_in_spec(name: str):
    root = Path(__file__).resolve().parents[1]
    return load_study_spec(root / "studies" / name, repo_root=root)


def _refresh_manifest_record(run_path: Path, relative_path: str) -> None:
    manifest_path = run_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for index, record in enumerate(manifest["files"]):
        if record["relative_path"] == relative_path:
            manifest["files"][index] = manifest_records(
                run_path, [relative_path]
            )[0]
            break
    else:
        raise AssertionError(f"manifest has no record for {relative_path}")
    write_strict_json(manifest_path, manifest)


@pytest.mark.parametrize(
    ("filename", "axis_values", "case_count"),
    [
        ("input_simulation_resolution_smoke.json", ((9, 16), (9, 16)), 8),
        (
            "input_simulation_resolution.json",
            ((64, 128, 256, 512), (64, 128, 256, 512)),
            32,
        ),
    ],
)
def test_checked_in_resolution_is_rectangular_and_fixes_J_q(
    filename: str,
    axis_values: tuple[tuple[int, ...], tuple[int, ...]],
    case_count: int,
) -> None:
    spec = _checked_in_spec(filename)
    assert spec.name == "input_simulation_resolution"
    assert tuple(axis.path for axis in spec.global_axes) == (
        "input.n_tar",
        "feature.n_sur",
    )
    assert tuple(tuple(axis.values) for axis in spec.global_axes) == axis_values
    assert {variant.id for variant in spec.variants} == {
        "burgers",
        "reaction_diffusion",
    }
    assert {
        readout.kind for readout in spec.base_trial.readouts
    } == {
        "direct_fourier_decoder",
        "affine_ridge",
        "random_feature_ridge",
    }
    for variant in spec.variants:
        source = variant.selection_source
        assert source is not None
        assert source.source_variant_id == variant.id
        assert tuple(source.import_paths) == (
            "feature.evolution.system",
            "feature.evolution.time",
        )

    cases, skipped = build_cases(spec)
    assert skipped == []
    assert len(cases) == case_count
    assert {
        int(case.trial.feature.observation.J) for case in cases
    } == {int(spec.base_trial.feature.observation.J)}
    assert {int(case.trial.output.q) for case in cases} == {
        int(spec.base_trial.output.q)
    }
    assert {
        (int(case.trial.input.n_tar), int(case.trial.feature.n_sur))
        for case in cases
    } == set(
        (n_tar, n_sur)
        for n_tar in axis_values[0]
        for n_sur in axis_values[1]
    )


def test_smoke_resolution_exercises_parity_and_all_size_relations() -> None:
    spec = _checked_in_spec("input_simulation_resolution_smoke.json")
    cases, _ = build_cases(spec)
    assert {
        int(case.trial.input.n_tar) % 2 for case in cases
    } == {0, 1}
    assert {
        int(case.trial.feature.n_sur) % 2 for case in cases
    } == {0, 1}
    assert {
        (
            int(case.trial.input.n_tar)
            > int(case.trial.feature.n_sur)
        )
        - (
            int(case.trial.input.n_tar)
            < int(case.trial.feature.n_sur)
        )
        for case in cases
    } == {-1, 0, 1}


def test_discarded_reference_modes_are_isolated_for_each_n_sur() -> None:
    x = periodic_grid(64, 1.0)
    low = 0.3 + torch.cos(4.0 * torch.pi * x)
    high = 0.25 * torch.cos(20.0 * torch.pi * x)
    finite = spectral_resample_periodic(
        torch.stack([low, low + high]),
        9,
        domain_length=1.0,
    )
    assert torch.allclose(finite[0], finite[1], atol=1e-11, rtol=1e-11)
    for n_sur in (9, 16):
        initial = build_feature_initial_state(
            finite,
            n_sur=n_sur,
            domain_length=1.0,
        )
        assert initial.shape == (2, n_sur)
        assert torch.allclose(
            initial[0],
            initial[1],
            atol=1e-11,
            rtol=1e-11,
        )


def test_feature_cache_identity_separates_finite_and_simulation_inputs(
    tmp_path: Path,
) -> None:
    _, dataset_path, study_path = write_tiny_stack(
        tmp_path,
        observation_J=8,
    )
    dataset = ensure_dataset(
        load_dataset_spec(dataset_path, repo_root=tmp_path),
        repo_root=tmp_path,
    )
    trial = load_study_spec(
        study_path,
        repo_root=tmp_path,
    ).base_trial
    cache = FeatureStateCache(
        artifact_root=tmp_path / "cache",
        enabled=False,
    )
    ids = dataset.train_ids[:2]
    identities = [
        cache.identity(dataset, ids, configured)
        for configured in (
            trial,
            apply_trial_overrides(trial, {"input.n_tar": 9}),
            apply_trial_overrides(trial, {"feature.n_sur": 16}),
            apply_trial_overrides(
                trial,
                {"feature.evolution.time": 0.2},
            ),
        )
    ]
    identities.append(cache.identity(dataset, ids.flip(0), trial))
    assert all(
        {
            "dataset_artifact_id",
            "sample_ids",
            "n_tar",
            "n_sur",
            "feature_generator",
        }
        <= identity.keys()
        for identity in identities
    )
    assert len({stable_object_hash(identity) for identity in identities}) == 5


def test_resolution_schema_rejects_unknown_scientific_key(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "studies" / "input_simulation_resolution_smoke.json"
    raw = json.loads(source.read_text(encoding="utf-8"))
    raw["base_trial"]["input"]["unknown_resolution_key"] = True
    path = write_json(tmp_path / "invalid_resolution.json", raw)
    with pytest.raises(ValueError, match="unknown_resolution_key"):
        load_study_spec(path, repo_root=root)


def test_dimension_studies_have_distinct_scientific_identities() -> None:
    resolution = _checked_in_spec("input_simulation_resolution_smoke.json")
    observation = _checked_in_spec("observation_output_budget_smoke.json")
    resolution_identity = scientific_study_spec(resolution)
    observation_identity = scientific_study_spec(observation)
    assert resolution_identity["name"] != observation_identity["name"]
    assert resolution_identity["global_axes"] != observation_identity[
        "global_axes"
    ]
    assert stable_object_hash(resolution_identity) != stable_object_hash(
        observation_identity
    )


def test_tiny_resolution_complete_verify_metrics_and_report_regeneration(
    tmp_path: Path,
) -> None:
    _, _, study_path = write_tiny_stack(
        tmp_path,
        include_diagnostics=False,
        generate_plots=True,
        observation_J=8,
    )
    raw = json.loads(study_path.read_text(encoding="utf-8"))
    raw["name"] = "tiny_input_simulation_resolution"
    raw["base_trial"]["feature"]["n_sur"] = 16
    raw["global_axes"] = [
        {"path": "input.n_tar", "values": [9, 16]},
        {"path": "feature.n_sur", "values": [8, 16]},
    ]
    raw["reporters"] = [
        {
            "kind": "metric_map",
            "filename": f"validation_resolution_{readout_id}",
            "x": "n_tar",
            "y": "n_sur",
            "x_values": [9, 16],
            "y_values": [8, 16],
            "metric": "validation_field_relative_l2_mean",
            "split": "validation",
            "readout_id": readout_id,
            "variant_id": "heat",
            "mark_selected": False,
            "formats": ["png"],
            "dpi": 60,
        }
        for readout_id in ("direct", "affine", "random")
    ]
    write_json(study_path, raw)
    spec = load_study_spec(study_path, repo_root=tmp_path)
    result = run_study(spec, repo_root=tmp_path)
    verify_study_run(result.path)

    validation_rows = _read_csv(result.path / "validation_trials.csv")
    test_rows = _read_csv(result.path / "test_metrics.csv")
    assert len(validation_rows) == len(test_rows) == 12
    assert {(row["J"], row["q"]) for row in validation_rows} == {
        ("8", "9")
    }
    assert {
        (row["n_tar"], row["n_sur"]) for row in validation_rows
    } == {
        ("9", "8"),
        ("9", "16"),
        ("16", "8"),
        ("16", "16"),
    }
    assert all(
        row["validation_field_relative_l2_mean"]
        and row["validation_data_field_relative_l2_mean"]
        and row["validation_representation_floor_relative_l2_mean"]
        and row["validation_data_representation_floor_relative_l2_mean"]
        for row in validation_rows
    )
    assert all(
        row["test_field_relative_l2_mean"]
        and row["test_data_field_relative_l2_mean"]
        and row["test_representation_floor_relative_l2_mean"]
        and row["test_data_representation_floor_relative_l2_mean"]
        for row in test_rows
    )
    assert len({row["feature_system_condition_hash"] for row in validation_rows}) == 1
    assert len({row["feature_cache_id"] for row in test_rows}) == 4
    assert regenerate_plots(spec, result.path) == [
        "validation_resolution_direct.png",
        "validation_resolution_affine.png",
        "validation_resolution_random.png",
    ]

    affine_index = next(
        index
        for index, row in enumerate(validation_rows)
        if row["readout_id"] == "affine"
    )
    validation_rows[affine_index]["J"] = "7"
    write_csv(
        result.path / "validation_trials.csv",
        validation_rows,
        fieldnames=list(validation_rows[0]),
    )
    _refresh_manifest_record(result.path, "validation_trials.csv")
    with pytest.raises(ValueError, match="fixed J"):
        verify_study_run(result.path)
