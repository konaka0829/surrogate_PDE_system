from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from pol.config.loader import load_study_spec
from pol.config.models import MetricMapReporterSpec
from pol.learning.direct import fixed_fourier_decoder_bandwidth
from pol.plotting.reporters import build_metric_map_data
from pol.study.cases import build_cases
from pol.study.runner import regenerate_plots, run_study, verify_study_run
from tests.helpers import write_json, write_tiny_stack


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _checked_in_spec(name: str):
    root = Path(__file__).resolve().parents[1]
    return load_study_spec(root / "studies" / name, repo_root=root)


@pytest.mark.parametrize(
    ("filename", "axis_sizes", "case_count"),
    [
        ("observation_output_budget_smoke.json", (2, 2), 8),
        ("observation_output_budget.json", (5, 4), 40),
    ],
)
def test_checked_in_budget_is_rectangular_and_changes_only_J_and_q(
    filename: str,
    axis_sizes: tuple[int, int],
    case_count: int,
) -> None:
    spec = _checked_in_spec(filename)
    assert spec.name == "observation_output_budget"
    assert tuple(axis.path for axis in spec.global_axes) == (
        "feature.observation.J",
        "output.q",
    )
    assert tuple(len(axis.values) for axis in spec.global_axes) == axis_sizes
    assert {variant.id for variant in spec.variants} == {
        "burgers",
        "reaction_diffusion",
    }
    for variant in spec.variants:
        assert variant.selection_source is not None
        assert variant.selection_source.source_variant_id == variant.id
        assert tuple(variant.selection_source.import_paths) == (
            "feature.evolution.system",
            "feature.evolution.time",
        )

    cases, skipped = build_cases(spec)
    assert skipped == []
    assert len(cases) == case_count
    assert {
        readout.kind for readout in spec.base_trial.readouts
    } == {
        "direct_fourier_decoder",
        "affine_ridge",
        "random_feature_ridge",
    }
    assert {int(case.trial.input.n_tar) for case in cases} == {
        int(spec.base_trial.input.n_tar)
    }
    assert {int(case.trial.feature.n_sur) for case in cases} == {
        int(spec.base_trial.feature.n_sur)
    }

    per_variant: dict[str, list[object]] = {}
    for case in cases:
        per_variant.setdefault(case.variant_id, []).append(case)
    expected_cell_count = axis_sizes[0] * axis_sizes[1]
    for variant_cases in per_variant.values():
        assert len(variant_cases) == expected_cell_count
        first = variant_cases[0].trial
        for case in variant_cases[1:]:
            trial = case.trial
            assert trial.input == first.input
            assert trial.feature.n_sur == first.feature.n_sur
            assert trial.feature.kind == first.feature.kind
            assert trial.feature.evolution == first.feature.evolution
            assert trial.feature.observation.kind == (
                first.feature.observation.kind
            )
            assert trial.feature.observation.l2_scale == (
                first.feature.observation.l2_scale
            )
            assert trial.output.kind == first.output.kind

    learned_q_above_J = [
        case
        for case in cases
        if int(case.trial.output.q)
        > int(case.trial.feature.observation.J)
        and any(
            readout.kind in {"affine_ridge", "random_feature_ridge"}
            for readout in case.trial.readouts
        )
    ]
    assert learned_q_above_J


def test_direct_budget_diagnostic_formula_does_not_constrain_learned_cells() -> None:
    diagnostic = fixed_fourier_decoder_bandwidth(4, 9)
    assert diagnostic.observable_q == 3
    assert diagnostic.retained_q == 3
    assert diagnostic.zero_filled_mode_count == 3
    assert diagnostic.zero_filled_coefficient_count == 6
    assert diagnostic.zero_fill_applied is True


def test_budget_schema_rejects_unknown_scientific_key(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "studies" / "observation_output_budget_smoke.json"
    raw = json.loads(source.read_text(encoding="utf-8"))
    raw["base_trial"]["feature"]["observation"]["unknown_budget_key"] = True
    path = write_json(tmp_path / "invalid_budget.json", raw)
    with pytest.raises(ValueError, match="unknown_budget_key"):
        load_study_spec(path, repo_root=root)


def test_metric_map_distinguishes_missing_invalid_and_nonfinite_cells() -> None:
    spec = MetricMapReporterSpec.model_validate(
        {
            "kind": "metric_map",
            "filename": "budget",
            "x": "J",
            "y": "q",
            "x_values": [4, 8],
            "y_values": [5, 9],
            "metric": "validation_field_relative_l2_mean",
            "readout_id": "affine",
            "variant_id": "burgers",
        }
    )
    row = {
        "variant_id": "burgers",
        "readout_id": "affine",
        "J": 4,
        "q": 5,
        "validation_field_relative_l2_mean": 0.25,
        "search_kind": "static",
        "selected": False,
    }
    data = build_metric_map_data(
        spec,
        [row],
        skipped_rows=[
            {
                "scope": "variant",
                "variant_id": "burgers",
                "global_values": {
                    "feature.observation.J": 4,
                    "output.q": 9,
                },
                "reason": "synthetic invalid budget cell",
            }
        ],
    )
    assert data is not None
    assert data.invalid_cells == (
        (0, 1, "synthetic invalid budget cell"),
    )
    assert data.missing_cells == ((1, 0), (1, 1))
    assert np.isnan(data.matrix[1, 0])

    nonfinite = dict(row)
    nonfinite["validation_field_relative_l2_mean"] = float("nan")
    with pytest.raises(ValueError, match="non-finite metric"):
        build_metric_map_data(spec, [nonfinite])


def test_tiny_budget_complete_verify_and_report_regeneration(
    tmp_path: Path,
) -> None:
    _, _, study_path = write_tiny_stack(
        tmp_path,
        include_diagnostics=False,
        generate_plots=True,
    )
    raw = json.loads(study_path.read_text(encoding="utf-8"))
    raw["name"] = "tiny_observation_output_budget"
    raw["base_trial"]["feature"]["observation"]["J"] = 4
    raw["base_trial"]["output"]["q"] = 5
    raw["global_axes"] = [
        {"path": "feature.observation.J", "values": [4, 8]},
        {"path": "output.q", "values": [5, 9]},
    ]
    raw["reporters"] = [
        {
            "kind": "metric_map",
            "filename": f"validation_budget_{readout_id}",
            "x": "J",
            "y": "q",
            "x_values": [4, 8],
            "y_values": [5, 9],
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
    assert result.summary["planned_global_axis_combination_count"] == 4
    assert result.summary["planned_global_axis_case_count"] == 4
    assert result.summary["evaluated_global_axis_case_count"] == 4
    assert result.summary["skipped_global_axis_case_count"] == 0

    validation_rows = _read_csv(result.path / "validation_trials.csv")
    test_rows = _read_csv(result.path / "test_metrics.csv")
    assert len(validation_rows) == 12
    assert len(test_rows) == 12
    assert {(row["n_tar"], row["n_sur"]) for row in validation_rows} == {
        ("16", "32")
    }
    assert len(
        {row["feature_system_condition_hash"] for row in validation_rows}
    ) == 1
    learned_q_above_J = [
        row
        for row in validation_rows
        if int(row["q"]) > int(row["J"])
        and row["readout_kind"] in {
            "affine_ridge",
            "random_feature_ridge",
        }
    ]
    assert {row["readout_kind"] for row in learned_q_above_J} == {
        "affine_ridge",
        "random_feature_ridge",
    }
    direct = next(
        row
        for row in validation_rows
        if row["readout_id"] == "direct"
        and row["J"] == "4"
        and row["q"] == "9"
    )
    assert direct["decoder_observable_q"] == "3"
    assert direct["decoder_retained_q"] == "3"
    assert direct["decoder_zero_filled_coefficient_count"] == "6"
    assert all(
        row["selected_ridge_zeta"]
        for row in validation_rows
        if row["readout_kind"] != "direct_fourier_decoder"
    )
    assert all(
        row["validation_representation_floor_relative_l2_mean"]
        and row["validation_data_representation_floor_relative_l2_mean"]
        for row in validation_rows
    )
    assert all(
        row["test_representation_floor_relative_l2_mean"]
        and row["test_data_representation_floor_relative_l2_mean"]
        for row in test_rows
    )

    created = regenerate_plots(spec, result.path)
    assert created == [
        "validation_budget_direct.png",
        "validation_budget_affine.png",
        "validation_budget_random.png",
    ]
    verify_study_run(result.path)
