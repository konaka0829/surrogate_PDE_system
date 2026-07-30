from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError
from scipy.stats import t as student_t

from pol.config.loader import load_study_spec
from pol.config.models import StudySpec
from pol.runtime.artifacts import manifest_records
from pol.runtime.io import write_csv, write_strict_json
from pol.study.evaluation import summarize_independent_seed_metrics
from pol.study.runner import regenerate_plots, run_study, verify_study_run
from tests.helpers import write_json, write_tiny_stack


ROOT = Path(__file__).resolve().parents[1]


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _seed_statistics_study(root: Path) -> Path:
    _, _, study_path = write_tiny_stack(
        root,
        include_diagnostics=False,
    )
    raw = json.loads(study_path.read_text(encoding="utf-8"))
    raw["schema_version"] = "pol-study-v5"
    raw["name"] = "random_feature_seed_statistics"
    raw["base_trial"]["readouts"] = [
        readout
        for readout in raw["base_trial"]["readouts"]
        if readout["id"] in {"affine", "random"}
    ]
    random_readout = next(
        readout
        for readout in raw["base_trial"]["readouts"]
        if readout["id"] == "random"
    )
    random_readout["selection_seeds"] = [11, 12]
    random_readout["evaluation_seeds"] = [21, 22, 23]
    raw["selection"]["representative_readout"] = "random"
    raw["diagnostics"] = [
        {
            "kind": "readout_stability_noise",
            "levels": [0.0],
            "repeats": 2,
            "seed": 7201,
            "scaling": {"kind": "relative_global_feature_rms"},
            "common_random_numbers": True,
            "include_prediction_ensemble": True,
            "covariance_rcond": 1e-12,
        }
    ]
    raw["reporters"] = [
        {
            "kind": "random_feature_seed_distribution",
            "filename": f"seed_{plot}",
            "plot": plot,
            "metric": "test_field_relative_l2_mean",
            "group_by": ["variant_id", "readout_id"],
            "yscale": "log",
            "formats": ["png"],
            "dpi": 80,
        }
        for plot in ("scatter", "box", "empirical_cdf")
    ]
    raw["execution"]["generate_plots"] = True
    return write_json(study_path, raw)


def _refresh_manifest(run_path: Path, relative_path: str) -> None:
    manifest_path = run_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    replacement = manifest_records(run_path, [relative_path])[0]
    for index, record in enumerate(manifest["files"]):
        if record["relative_path"] == relative_path:
            manifest["files"][index] = replacement
            break
    else:
        raise AssertionError(f"manifest has no record for {relative_path}")
    write_strict_json(manifest_path, manifest)


def test_seed_summary_separates_student_t_interval_and_quantiles() -> None:
    summary = summarize_independent_seed_metrics(
        [{"metric": value} for value in (1.0, 2.0, 4.0, 8.0)]
    )
    mean = 3.75
    sample_std = math.sqrt(
        sum((value - mean) ** 2 for value in (1.0, 2.0, 4.0, 8.0)) / 3
    )
    margin = float(student_t.ppf(0.975, 3)) * sample_std / 2.0
    assert summary["metric"] == pytest.approx(mean)
    assert summary["metric_seed_std"] == pytest.approx(sample_std)
    assert summary["metric_seed_ci95_low"] == pytest.approx(mean - margin)
    assert summary["metric_seed_ci95_high"] == pytest.approx(mean + margin)
    assert summary["metric_seed_q25"] == pytest.approx(1.75)
    assert summary["metric_seed_median"] == pytest.approx(3.0)
    assert summary["metric_seed_q75"] == pytest.approx(5.0)


def test_seed_statistics_completion_links_members_and_rejects_map_tamper(
    tmp_path: Path,
) -> None:
    spec = load_study_spec(
        _seed_statistics_study(tmp_path),
        repo_root=tmp_path,
    )
    result = run_study(spec, repo_root=tmp_path)
    verify_study_run(result.path)

    seed_rows = _rows(result.path / "random_feature_seed_metrics.csv")
    primary = next(
        row
        for row in _rows(result.path / "test_metrics.csv")
        if row["readout_id"] == "random"
    )
    ensemble = _rows(
        result.path / "random_feature_ensemble_metrics.csv"
    )
    stability = _rows(result.path / "readout_stability_models.csv")
    assert {int(row["seed"]) for row in seed_rows} == {21, 22, 23}
    assert len({row["random_map_parameter_hash"] for row in seed_rows}) == 3
    assert all(row["frozen_member_parameter_hash"] for row in seed_rows)
    assert all(
        row["evaluation_seed_validation_used_for_selection"] == "False"
        for row in seed_rows
    )
    assert all(
        row["evaluation_seed_validation_field_relative_l2_mean"]
        for row in seed_rows
    )
    assert {row["random_map_parameter_hash"] for row in seed_rows} <= {
        row["random_map_parameter_hash"]
        for row in stability
        if row["random_map_parameter_hash"]
    }
    assert primary["test_seed_count"] == "3"
    assert primary["test_seed_std_ddof"] == "1"
    assert primary["test_confidence_interval_method"] == "student_t"
    assert primary["test_seed_descriptive_quantiles"] == "[0.25,0.5,0.75]"
    assert primary["test_seed_quantile_method"] == "linear"
    assert primary["test_seed_quantiles_are_uncertainty_interval"] == "False"
    assert primary["test_field_relative_l2_mean_seed_median"]
    assert primary["test_field_relative_l2_mean_seed_q25"]
    assert primary["test_field_relative_l2_mean_seed_q75"]
    assert len(ensemble) == 1
    assert ensemble[0]["ensemble_member_count"] == "3"
    assert ensemble[0]["ensemble_member_seeds_hash"]
    assert ensemble[0]["ensemble_member_parameters_hash"]
    assert sorted(
        path.name for path in (result.path / "figures").glob("*.png")
    ) == [
        "seed_box.png",
        "seed_empirical_cdf.png",
        "seed_scatter.png",
    ]
    assert regenerate_plots(spec, result.path) == [
        "seed_scatter.png",
        "seed_box.png",
        "seed_empirical_cdf.png",
    ]

    seed_rows[0]["random_map_parameter_hash"] = "0" * 64
    seed_path = result.path / "random_feature_seed_metrics.csv"
    write_csv(seed_path, seed_rows, fieldnames=list(seed_rows[0]))
    _refresh_manifest(result.path, seed_path.name)
    with pytest.raises(
        ValueError,
        match="per-seed realization random_map_parameter_hash mismatch",
    ):
        verify_study_run(result.path)


def test_seed_reporter_schema_and_checked_profiles_are_strict() -> None:
    raw = json.loads(
        (ROOT / "studies" / "random_feature_seed_statistics_smoke.json")
        .read_text(encoding="utf-8")
    )
    raw["reporters"][0]["unknown_scientific_key"] = True
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        StudySpec.model_validate(raw)

    for name, expected_seed_count in (
        ("random_feature_seed_statistics_smoke.json", 3),
        ("random_feature_seed_statistics.json", 32),
    ):
        spec = load_study_spec(ROOT / "studies" / name, repo_root=ROOT)
        random_readout = next(
            readout
            for readout in spec.base_trial.readouts
            if readout.kind == "random_feature_ridge"
        )
        assert set(random_readout.selection_seeds).isdisjoint(
            random_readout.evaluation_seeds
        )
        assert len(random_readout.evaluation_seeds) == expected_seed_count
        assert [
            reporter.plot
            for reporter in spec.reporters
            if reporter.kind == "random_feature_seed_distribution"
        ] == [
            "scatter",
            "box",
            "empirical_cdf",
        ]
        assert spec.prediction_capture is not None
        assert spec.prediction_capture.sample_selection_policy == (
            "predeclared_test_ids"
        )
        assert {
            reporter.kind for reporter in spec.reporters
        } >= {
            "representative_prediction_fields",
            "fourier_error_spectra",
        }
        assert spec.diagnostics[0].levels == (0.0,)
        assert spec.profile in {"smoke", "main"}
