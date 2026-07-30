from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pytest

from pol.config.loader import (
    load_digital_baseline_spec,
    load_report_spec,
    load_study_spec,
)
from pol.config.models import (
    HeatMultiplierComparisonReporterSpec,
    MetricCurveReporterSpec,
)
from pol.plotting.reporters import generate_reporters
from pol.study.results import write_run_manifest
from pol.study.runner import run_study
from pol.study.verification import verify_study_run
from tests.helpers import write_tiny_stack


ROOT = Path(__file__).resolve().parents[1]
UNSAFE_COMPONENTS = (
    "../x",
    "a/b",
    r"a\b",
    ".hidden",
    ".",
    "..",
    "",
    " ",
    " x",
    "x ",
    "x\n",
)


def _load_study(path: Path) -> object:
    return load_study_spec(path, repo_root=ROOT)


def _load_report(path: Path) -> object:
    return load_report_spec(path, repo_root=ROOT)


def _load_digital(path: Path) -> object:
    return load_digital_baseline_spec(path, repo_root=ROOT)


PATH_COMPONENT_CASES: tuple[
    tuple[str, Path, Callable[[Path], object]],
    ...,
] = (
    (
        "study",
        ROOT / "studies/surrogate_parameter_time_landscape_smoke.json",
        _load_study,
    ),
    (
        "report",
        ROOT / "reports/surrogate_operator_summary_smoke.json",
        _load_report,
    ),
    (
        "digital",
        ROOT / "digital_baselines/fno1d_smoke.json",
        _load_digital,
    ),
)


@pytest.mark.parametrize("value", UNSAFE_COMPONENTS)
@pytest.mark.parametrize("field", ("name", "profile"))
@pytest.mark.parametrize(
    ("family", "source", "loader"),
    PATH_COMPONENT_CASES,
    ids=[family for family, *_ in PATH_COMPONENT_CASES],
)
def test_path_derived_name_and_profile_fields_reject_unsafe_components(
    tmp_path: Path,
    family: str,
    source: Path,
    loader: Callable[[Path], object],
    field: str,
    value: str,
) -> None:
    del family
    raw = json.loads(source.read_text(encoding="utf-8"))
    raw[field] = value
    path = tmp_path / source.name
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        loader(path)


def test_existing_semantic_path_components_remain_valid() -> None:
    study = load_study_spec(
        ROOT / "studies/surrogate_parameter_time_landscape_smoke.json",
        repo_root=ROOT,
    )
    report = load_report_spec(
        ROOT / "reports/surrogate_operator_summary_smoke.json",
        repo_root=ROOT,
    )
    digital = load_digital_baseline_spec(
        ROOT / "digital_baselines/fno1d_smoke.json",
        repo_root=ROOT,
    )
    assert (study.name, study.profile) == (
        "surrogate_parameter_time_landscape",
        "smoke",
    )
    assert (report.name, report.profile) == (
        "surrogate_operator_summary",
        "smoke",
    )
    assert (digital.name, digital.profile) == ("fno1d_burgers", "smoke")


def test_path_escape_is_rejected_without_creating_outside_output_root(
    tmp_path: Path,
) -> None:
    _, _, study_path = write_tiny_stack(tmp_path)
    raw = json.loads(study_path.read_text(encoding="utf-8"))
    raw["name"] = "../escaped"
    study_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="safe basename"):
        load_study_spec(study_path, repo_root=tmp_path)
    assert not (tmp_path / "escaped").exists()


def _metric_reporter(
    *,
    filename: str = "metric",
    formats: tuple[str, ...] = ("png",),
    metric: str = "validation_field_relative_l2_mean",
) -> MetricCurveReporterSpec:
    return MetricCurveReporterSpec.model_validate(
        {
            "kind": "metric_curve",
            "filename": filename,
            "x": "q",
            "metric": metric,
            "formats": list(formats),
        }
    )


def test_study_reporter_filenames_are_extension_free() -> None:
    assert _metric_reporter(filename="same").filename == "same"
    with pytest.raises(ValueError, match="safe basename|ASCII"):
        _metric_reporter(filename="same.png")


@pytest.mark.parametrize("formats", ((), ("png", "png")))
def test_study_reporter_formats_are_nonempty_and_unique(
    formats: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="formats"):
        _metric_reporter(formats=formats)


def test_study_rejects_cross_type_reporter_output_collisions(
    tmp_path: Path,
) -> None:
    _, _, study_path = write_tiny_stack(tmp_path)
    raw = json.loads(study_path.read_text(encoding="utf-8"))
    raw["reporters"] = [
        {
            "kind": "metric_curve",
            "filename": "same",
            "x": "q",
            "metric": "validation_field_relative_l2_mean",
            "formats": ["png"],
        },
        {
            "kind": "readout_stability",
            "filename": "same",
            "plot": "noise_curve",
            "formats": ["png"],
        },
    ]
    study_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match=r"unique \(filename, format\)"):
        load_study_spec(study_path, repo_root=tmp_path)


def test_study_allows_same_filename_for_disjoint_formats(
    tmp_path: Path,
) -> None:
    _, _, study_path = write_tiny_stack(tmp_path)
    raw = json.loads(study_path.read_text(encoding="utf-8"))
    raw["reporters"] = [
        {
            "kind": "metric_curve",
            "filename": "same",
            "x": "q",
            "metric": "validation_field_relative_l2_mean",
            "formats": ["png"],
        },
        {
            "kind": "readout_stability",
            "filename": "same",
            "plot": "noise_curve",
            "formats": ["pdf"],
        },
    ]
    study_path.write_text(json.dumps(raw), encoding="utf-8")
    spec = load_study_spec(study_path, repo_root=tmp_path)
    assert len(spec.reporters) == 2


@pytest.mark.parametrize("case", ("empty_table", "typo_metric", "wrong_group"))
def test_configured_reporter_fails_closed_when_it_produces_no_files(
    tmp_path: Path,
    case: str,
) -> None:
    validation_rows: list[dict[str, object]] = []
    multiplier_rows: list[dict[str, object]] = []
    if case == "empty_table":
        reporter = _metric_reporter()
    elif case == "typo_metric":
        reporter = _metric_reporter(metric="validation_metric_typo")
        validation_rows = [{"q": 9, "validation_field_relative_l2_mean": 0.1}]
    else:
        reporter = HeatMultiplierComparisonReporterSpec(
            filename="wrong_group",
            readout_id="missing",
            formats=("png",),
        )
        multiplier_rows = [
            {
                "readout_id": "affine",
                "q": 9,
                "coefficient_index": 0,
                "ideal_readout_multiplier": 1.0,
                "effective_learned_diagonal": 1.0,
            }
        ]
    with pytest.raises(ValueError, match="expected exactly"):
        generate_reporters(
            [reporter],
            validation_rows=validation_rows,
            test_rows=[],
            multiplier_rows=multiplier_rows,
            noise_rows=[],
            output_dir=tmp_path / "figures",
        )


def test_multi_format_report_is_bound_to_summary_and_manifest(
    tmp_path: Path,
) -> None:
    _, _, study_path = write_tiny_stack(tmp_path, generate_plots=True)
    raw = json.loads(study_path.read_text(encoding="utf-8"))
    raw["reporters"][0]["formats"] = ["png", "pdf"]
    study_path.write_text(json.dumps(raw), encoding="utf-8")
    spec = load_study_spec(study_path, repo_root=tmp_path)

    result = run_study(spec, repo_root=tmp_path)
    manifest = verify_study_run(result.path)

    expected = ["validation_error_vs_q.pdf", "validation_error_vs_q.png"]
    assert result.summary["schema_version"] == "pol-study-run-summary-v16"
    assert result.summary["expected_figures"] == expected
    assert sorted(result.summary["figures"]) == expected
    assert result.summary["configured_reporter_count"] == 1
    assert result.summary["report_expected_file_count"] == 2
    assert result.summary["report_generated_file_count"] == 2
    assert manifest["schema_version"] == "pol-study-run-manifest-v16"
    assert {
        record["relative_path"]
        for record in manifest["files"]
        if record["relative_path"].startswith("figures/")
    } == {
        "figures/validation_error_vs_q.png",
        "figures/validation_error_vs_q.pdf",
    }


def test_zero_file_report_rolls_back_without_publishing_completion(
    tmp_path: Path,
) -> None:
    _, _, study_path = write_tiny_stack(tmp_path, generate_plots=True)
    raw = json.loads(study_path.read_text(encoding="utf-8"))
    raw["reporters"][0]["metric"] = "validation_metric_typo"
    study_path.write_text(json.dumps(raw), encoding="utf-8")
    spec = load_study_spec(study_path, repo_root=tmp_path)

    with pytest.raises(ValueError, match="expected exactly"):
        run_study(spec, repo_root=tmp_path)

    run_dirs = [
        path
        for path in (tmp_path / "outputs" / spec.name).iterdir()
        if path.is_dir() and not path.name.startswith(".")
    ]
    assert len(run_dirs) == 1
    verify_study_run(run_dirs[0])
    summary = json.loads(
        (run_dirs[0] / "run_summary.json").read_text(encoding="utf-8")
    )
    assert summary["report_status"] == (
        "numerical_complete_report_not_generated"
    )
    assert summary["figures"] == []
    assert not list(run_dirs[0].parent.glob(f".{run_dirs[0].name}.staging-*"))
    assert not list(run_dirs[0].parent.glob(f".{run_dirs[0].name}.backup-*"))


def test_verifier_rejects_tampered_expected_figure_summary(
    tmp_path: Path,
) -> None:
    _, _, study_path = write_tiny_stack(tmp_path, generate_plots=True)
    spec = load_study_spec(study_path, repo_root=tmp_path)
    result = run_study(spec, repo_root=tmp_path)
    manifest = verify_study_run(result.path)
    summary_path = result.path / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["expected_figures"] = []
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    write_run_manifest(
        result.path,
        identity=manifest["identity"],
        schema_version="pol-study-run-manifest-v16",
    )
    with pytest.raises(ValueError, match="expected figure list mismatch"):
        verify_study_run(result.path)
