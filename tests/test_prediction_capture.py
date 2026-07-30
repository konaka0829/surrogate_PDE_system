from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from pydantic import ValidationError
import torch

from pol.config.loader import load_study_spec
from pol.config.models import StudySpec
from pol.runtime.artifacts import manifest_records
from pol.runtime.io import atomic_torch_save, write_strict_json
from pol.study.prediction_capture import (
    PREDICTION_CAPTURE_FILENAME,
    load_prediction_capture,
)
from pol.study.runner import regenerate_plots, run_study, verify_study_run
from tests.helpers import write_json, write_tiny_stack


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _capture_study(root: Path) -> Path:
    _, _, study_path = write_tiny_stack(
        root,
        include_diagnostics=False,
    )
    raw = json.loads(study_path.read_text(encoding="utf-8"))
    raw["schema_version"] = "pol-study-v6"
    raw["name"] = "tiny_prediction_capture"
    raw["prediction_capture"] = {
        "kind": "predeclared_test_predictions",
        "sample_ids": [8, 5],
        "sample_selection_policy": "predeclared_test_ids",
        "readout_ids": ["direct", "affine", "random"],
        "random_feature_members": {
            "kind": "explicit_seeds",
            "seeds": [21],
        },
        "include_ensemble": True,
    }
    raw["reporters"] = [
        {
            "kind": "representative_prediction_fields",
            "filename": "representative_fields",
            "formats": ["png"],
            "dpi": 80,
        },
        {
            "kind": "fourier_error_spectra",
            "filename": "error_spectra",
            "metric": "per_mode_squared_error_sample_mean",
            "x_axis": "mode_index",
            "yscale": "log",
            "formats": ["png"],
            "dpi": 80,
        },
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


def test_capture_schema_rejects_posthoc_or_unknown_coordinates(
    tmp_path: Path,
) -> None:
    study_path = _capture_study(tmp_path)
    raw = json.loads(study_path.read_text(encoding="utf-8"))
    raw["prediction_capture"]["random_feature_members"]["seeds"] = [999]
    with pytest.raises(
        ValidationError,
        match="must be frozen evaluation seeds",
    ):
        StudySpec.model_validate(raw)

    raw = json.loads(study_path.read_text(encoding="utf-8"))
    raw["prediction_capture"]["unknown_scientific_key"] = True
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        StudySpec.model_validate(raw)


def test_capture_membership_fails_before_selection_or_test_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study_path = _capture_study(tmp_path)
    raw = json.loads(study_path.read_text(encoding="utf-8"))
    raw["prediction_capture"]["sample_ids"] = [0]
    write_json(study_path, raw)
    calls = {"selection": 0, "test": 0}

    def forbidden_selection(*args, **kwargs):
        calls["selection"] += 1
        raise AssertionError("selection must not start")

    def forbidden_test(*args, **kwargs):
        calls["test"] += 1
        raise AssertionError("test must not start")

    monkeypatch.setattr(
        "pol.study.trial.TrialEngine.evaluate_selection",
        forbidden_selection,
    )
    monkeypatch.setattr(
        "pol.study.trial.TrialEngine.evaluate_test",
        forbidden_test,
    )
    with pytest.raises(ValueError, match="predeclared test IDs"):
        run_study(
            load_study_spec(study_path, repo_root=tmp_path),
            repo_root=tmp_path,
        )
    assert calls == {"selection": 0, "test": 0}


def test_capture_metrics_semantics_and_publish_then_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pol.study.runner as runner_module

    spec = load_study_spec(
        _capture_study(tmp_path),
        repo_root=tmp_path,
    )
    original_generate = runner_module.generate_reporters
    observed_completed_numerical_run = False

    def checked_generate(*args, **kwargs):
        nonlocal observed_completed_numerical_run
        assert isinstance(kwargs["validation_rows"][0]["selected"], str)
        completed = [
            path
            for path in (tmp_path / "outputs" / spec.name).iterdir()
            if path.is_dir() and not path.name.startswith(".")
        ]
        assert len(completed) == 1
        verify_study_run(completed[0])
        numerical_summary = json.loads(
            (completed[0] / "run_summary.json").read_text(encoding="utf-8")
        )
        assert numerical_summary["report_status"] == (
            "numerical_complete_report_not_generated"
        )
        assert numerical_summary["figures"] == []
        observed_completed_numerical_run = True
        return original_generate(*args, **kwargs)

    monkeypatch.setattr(runner_module, "generate_reporters", checked_generate)
    result = run_study(spec, repo_root=tmp_path)
    assert observed_completed_numerical_run is True
    verify_study_run(result.path)
    capture = load_prediction_capture(
        result.path / PREDICTION_CAPTURE_FILENAME
    )
    assert capture["sample_ids"].tolist() == [8, 5]
    assert capture["spectrum_definition"]["stored_prediction_policy"] == (
        "predeclared_samples_plus_all_test_per_coefficient_aggregates"
    )
    assert {
        (entry["readout_id"], entry["prediction_semantics"], entry["seed"])
        for entry in capture["entries"]
    } == {
        ("direct", "single_model", None),
        ("affine", "single_model", None),
        ("random", "independent_seed_realization", 21),
        ("random", "prediction_ensemble", None),
    }
    primary = {
        row["readout_id"]: row
        for row in _rows(result.path / "test_metrics.csv")
    }
    seed = _rows(result.path / "random_feature_seed_metrics.csv")[0]
    ensemble = _rows(
        result.path / "random_feature_ensemble_metrics.csv"
    )[0]
    for entry in capture["entries"]:
        if entry["prediction_semantics"] == "single_model":
            expected = primary[entry["readout_id"]]["test_coefficient_mse"]
        elif entry["prediction_semantics"] == (
            "independent_seed_realization"
        ):
            expected = seed["test_coefficient_mse"]
        else:
            expected = ensemble["test_ensemble_coefficient_mse"]
        assert entry["test_coefficient_mse"] == pytest.approx(float(expected))
    assert result.summary["report_status"] == "complete"
    assert result.summary["report_source"] == (
        "verified_completed_run_read_only"
    )
    assert result.summary["prediction_capture_entry_count"] == 4
    assert sorted(result.summary["figures"]) == [
        "error_spectra.png",
        "representative_fields.png",
    ]


def test_report_regeneration_is_inference_free_and_tamper_evident(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pol.study.runner as runner_module

    spec = load_study_spec(
        _capture_study(tmp_path),
        repo_root=tmp_path,
    )
    result = run_study(spec, repo_root=tmp_path)
    capture_path = result.path / PREDICTION_CAPTURE_FILENAME
    protected = {
        name: (result.path / name).read_bytes()
        for name in (
            PREDICTION_CAPTURE_FILENAME,
            "test_metrics.csv",
            "random_feature_seed_metrics.csv",
            "random_feature_ensemble_metrics.csv",
            "frozen_models.pt",
        )
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("report regeneration attempted model inference")

    monkeypatch.setattr(
        "pol.study.cache.FeatureStateCache.get_or_solve",
        forbidden,
    )
    monkeypatch.setattr("pol.study.trial.predict_frozen", forbidden)
    monkeypatch.setattr("pol.study.readouts.fit_readout", forbidden)
    regenerate_plots(spec, result.path)
    regenerate_plots(spec, result.path)
    assert {
        name: (result.path / name).read_bytes() for name in protected
    } == protected

    monkeypatch.setattr(
        runner_module,
        "generate_reporters",
        forbidden,
    )
    with pytest.raises(
        AssertionError,
        match="attempted model inference",
    ):
        regenerate_plots(spec, result.path)
    verify_study_run(result.path)
    assert {
        name: (result.path / name).read_bytes() for name in protected
    } == protected

    capture = torch.load(
        capture_path,
        map_location="cpu",
        weights_only=True,
    )
    capture["entries"][0]["prediction_q_coefficients"][0, 0] += 1.0
    atomic_torch_save(capture_path, capture)
    _refresh_manifest(result.path, PREDICTION_CAPTURE_FILENAME)
    with pytest.raises(ValueError, match="capture content hash mismatch"):
        verify_study_run(result.path)


def test_failed_automatic_report_preserves_verified_numerical_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pol.study.runner as runner_module

    spec = load_study_spec(
        _capture_study(tmp_path),
        repo_root=tmp_path,
    )

    def fail_report(*args, **kwargs):
        raise RuntimeError("intentional report failure")

    monkeypatch.setattr(runner_module, "generate_reporters", fail_report)
    with pytest.raises(RuntimeError, match="intentional report failure"):
        run_study(spec, repo_root=tmp_path)
    completed = [
        path
        for path in (tmp_path / "outputs" / spec.name).iterdir()
        if path.is_dir() and not path.name.startswith(".")
    ]
    assert len(completed) == 1
    verify_study_run(completed[0])
    summary = json.loads(
        (completed[0] / "run_summary.json").read_text(encoding="utf-8")
    )
    assert summary["prediction_capture_status"] == "complete"
    assert summary["report_status"] == (
        "numerical_complete_report_not_generated"
    )
    assert summary["figures"] == []
