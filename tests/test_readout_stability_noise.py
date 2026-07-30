from __future__ import annotations

import copy
import csv
from dataclasses import replace
import json
import math
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from pol.config.loader import load_study_spec
from pol.config.models import StudySpec
from pol.runtime.artifacts import manifest_records
from pol.runtime.io import write_strict_json
from pol.study.diagnostics import (
    _model_norms,
    covariance_diagnostics,
    summarize_repeated_metrics,
)
from pol.study.runner import run_study, verify_study_run
from tests.helpers import write_json, write_tiny_stack


ROOT = Path(__file__).resolve().parents[1]


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _stability_study(root: Path) -> Path:
    _, _, study_path = write_tiny_stack(
        root,
        include_diagnostics=False,
    )
    raw = json.loads(study_path.read_text(encoding="utf-8"))
    raw["diagnostics"] = [
        {
            "kind": "readout_stability_noise",
            "levels": [0.0, 0.05],
            "repeats": 3,
            "seed": 7001,
            "scaling": {"kind": "relative_global_feature_rms"},
            "common_random_numbers": True,
            "include_prediction_ensemble": True,
            "covariance_rcond": 1e-12,
        }
    ]
    return write_json(study_path, raw)


def _refresh_manifest(run_path: Path, relative_path: str) -> None:
    manifest_path = run_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest_records(run_path, [relative_path])[0]
    for index, existing in enumerate(manifest["files"]):
        if existing["relative_path"] == relative_path:
            manifest["files"][index] = record
            break
    else:
        raise AssertionError(f"missing manifest record: {relative_path}")
    write_strict_json(manifest_path, manifest)


def test_readout_norms_match_simple_analytic_matrix() -> None:
    model = {
        "kind": "affine_ridge",
        "W": torch.diag(torch.tensor([3.0, 4.0], dtype=torch.float64)),
        "b": torch.tensor([0.0, 5.0], dtype=torch.float64),
        "zeta": 1e-4,
    }
    norms = _model_norms(model)
    assert norms["weight_frobenius_norm"] == pytest.approx(5.0)
    assert norms["weight_operator_norm"] == pytest.approx(4.0)
    assert norms["bias_norm"] == pytest.approx(5.0)
    assert norms["selected_ridge_zeta"] == pytest.approx(1e-4)

    direct = _model_norms({"kind": "direct_fourier_decoder"})
    assert direct["norm_status"] == "not_applicable_fixed_decoder"
    assert direct["weight_frobenius_norm"] is None


def test_singular_covariance_reports_raw_and_retained_conditions() -> None:
    values = torch.tensor(
        [[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]],
        dtype=torch.float64,
    )
    result = covariance_diagnostics(values, rcond=1e-12)
    assert result["covariance_rank"] == 1
    assert result["covariance_dimension"] == 2
    assert math.isinf(result["covariance_raw_condition"])
    assert result["covariance_retained_rank_condition"] == pytest.approx(1.0)
    assert result["covariance_rank_cutoff"] > 0.0


def test_repeat_summary_uses_sample_std_and_student_t_interval() -> None:
    result = summarize_repeated_metrics(
        [{"error": 1.0}, {"error": 2.0}, {"error": 3.0}],
        dimension="repeat",
    )
    assert result["error"] == pytest.approx(2.0)
    assert result["error_repeat_std"] == pytest.approx(1.0)
    assert result["error_repeat_ci95_low"] < 1.0
    assert result["error_repeat_ci95_high"] > 3.0


def test_stability_study_separates_seed_and_ensemble_and_uses_crn(
    tmp_path: Path,
) -> None:
    result = run_study(
        load_study_spec(_stability_study(tmp_path), repo_root=tmp_path),
        repo_root=tmp_path,
    )
    verify_study_run(result.path)
    repeats = _rows(result.path / "readout_stability_noise_repeats.csv")
    summaries = _rows(result.path / "readout_stability_noise_summary.csv")
    ensemble = _rows(
        result.path / "readout_stability_noise_ensemble_repeats.csv"
    )
    assert {
        row["result_kind"] for row in summaries
    } >= {
        "single_model_repeat_summary",
        "independent_seed_repeat_summary",
        "independent_seed_primary_summary",
    }
    assert {row["result_kind"] for row in ensemble} == {
        "noise_prediction_ensemble"
    }
    random_rows = [row for row in repeats if row["readout_id"] == "random"]
    by_coordinate: dict[tuple[str, str], set[str]] = {}
    for row in random_rows:
        by_coordinate.setdefault(
            (row["noise_level"], row["repeat"]),
            set(),
        ).add(row["noise_seed"])
    assert by_coordinate
    assert all(len(seeds) == 1 for seeds in by_coordinate.values())
    primary = [
        row
        for row in summaries
        if row["result_kind"] == "independent_seed_primary_summary"
    ]
    assert primary
    assert all(row["field_relative_l2_mean_seed_std"] != "" for row in primary)
    assert result.summary["readout_stability_model_row_count"] == 4


def test_diagnostics_receive_models_from_read_back_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pol.study.runner as runner_module

    original_persist = runner_module.persist_and_read_back_freeze
    original_diagnostic = runner_module.readout_stability_noise_diagnostic

    def marked_persist(*args, **kwargs):
        persisted = original_persist(*args, **kwargs)
        archive = copy.deepcopy(persisted.archive)
        for entry in archive["models"].values():
            entry["model"]["read_back_test_marker"] = True
        return replace(persisted, archive=archive)

    observed = 0

    def checked_diagnostic(*args, **kwargs):
        nonlocal observed
        assert kwargs["model"]["read_back_test_marker"] is True
        observed += 1
        return original_diagnostic(*args, **kwargs)

    monkeypatch.setattr(
        runner_module,
        "persist_and_read_back_freeze",
        marked_persist,
    )
    monkeypatch.setattr(
        runner_module,
        "readout_stability_noise_diagnostic",
        checked_diagnostic,
    )
    run_study(
        load_study_spec(_stability_study(tmp_path), repo_root=tmp_path),
        repo_root=tmp_path,
    )
    assert observed == 3


def test_stability_schema_is_strict_and_tamper_is_rejected(
    tmp_path: Path,
) -> None:
    study_path = _stability_study(tmp_path)
    raw = json.loads(study_path.read_text(encoding="utf-8"))
    raw["diagnostics"][0]["unknown_scientific_key"] = True
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        StudySpec.model_validate(raw)

    result = run_study(
        load_study_spec(study_path, repo_root=tmp_path),
        repo_root=tmp_path,
    )
    path = result.path / "readout_stability_noise_summary.csv"
    text = path.read_text(encoding="utf-8")
    path.write_text(
        text.replace(
            "pol-readout-stability-summary-v1",
            "tampered-summary-schema",
            1,
        ),
        encoding="utf-8",
    )
    _refresh_manifest(result.path, path.name)
    with pytest.raises(ValueError, match="summary schema mismatch"):
        verify_study_run(result.path)


@pytest.mark.parametrize(
    "name",
    ["readout_stability_noise_smoke.json", "readout_stability_noise.json"],
)
def test_checked_stability_study_contract(name: str) -> None:
    spec = load_study_spec(ROOT / "studies" / name, repo_root=ROOT)
    assert [diagnostic.kind for diagnostic in spec.diagnostics] == [
        "readout_stability_noise"
    ]
    diagnostic = spec.diagnostics[0]
    assert diagnostic.levels[0] == 0.0
    assert diagnostic.repeats >= 2
    assert diagnostic.common_random_numbers is True
    assert {readout.kind for readout in spec.base_trial.readouts} == {
        "direct_fourier_decoder",
        "affine_ridge",
        "random_feature_ridge",
    }
