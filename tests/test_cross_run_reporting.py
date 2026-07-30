from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from pol.config.loader import load_report_spec, load_study_spec
from pol.reporting.runner import run_report, verify_report
from pol.study.runner import run_study
from tests.helpers import write_json, write_tiny_stack


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _build_sources_and_report_spec(
    root: Path,
) -> tuple[Path, tuple[Path, Path]]:
    _, _, study_path = write_tiny_stack(
        root,
        observation_J=4,
        include_diagnostics=False,
    )
    base = json.loads(study_path.read_text(encoding="utf-8"))
    base["name"] = "tiny_report_grid"
    base["global_axes"] = [
        {"path": "feature.observation.J", "values": [4, 8]},
        {"path": "output.q", "values": [5, 9]},
    ]
    grid_path = write_json(root / "grid_study.json", base)
    grid_result = run_study(
        load_study_spec(grid_path, repo_root=root),
        repo_root=root,
    )

    baseline = json.loads(study_path.read_text(encoding="utf-8"))
    baseline["name"] = "tiny_report_baseline"
    baseline["global_axes"] = []
    baseline_path = write_json(root / "baseline_study.json", baseline)
    baseline_result = run_study(
        load_study_spec(baseline_path, repo_root=root),
        repo_root=root,
    )

    report_path = write_json(
        root / "report.json",
        {
            "schema_version": "pol-report-v1",
            "name": "tiny_cross_run_summary",
            "profile": "test",
            "output_root": str(root / "reports"),
            "sources": [
                {"id": "grid", "study_spec": str(grid_path)},
                {
                    "id": "baseline",
                    "study_spec": str(baseline_path),
                },
            ],
            "reporters": [
                {
                    "kind": "phase_diagram_report",
                    "source_id": "grid",
                    "filename": "validation_J_q",
                    "split": "validation",
                    "metric": "validation_field_relative_l2_mean",
                    "variant_id": "heat",
                    "readout_id": "affine",
                    "x": "J",
                    "y": "q",
                    "x_values": [4, 8],
                    "y_values": [5, 9],
                    "x_label": "observation count J",
                    "y_label": "output dimension q",
                    "metric_label": "validation relative L2 field error",
                    "xscale": "linear",
                    "yscale": "linear",
                    "mark_selected": False,
                    "formats": ["png"],
                    "dpi": 80,
                },
                {
                    "kind": "baseline_summary_table",
                    "source_id": "baseline",
                    "filename": "baseline",
                    "rows": [
                        {
                            "id": "direct",
                            "label": "Direct",
                            "variant_id": "heat",
                            "readout_id": "direct",
                        },
                        {
                            "id": "affine",
                            "label": "Affine",
                            "variant_id": "heat",
                            "readout_id": "affine",
                        },
                        {
                            "id": "random",
                            "label": "Random feature",
                            "variant_id": "heat",
                            "readout_id": "random",
                        },
                    ],
                    "field_metric": "test_field_relative_l2_mean",
                    "data_metric": "test_data_field_relative_l2_mean",
                    "field_representation_floor_metric": (
                        "test_representation_floor_relative_l2_mean"
                    ),
                    "data_representation_floor_metric": (
                        "test_data_representation_floor_relative_l2_mean"
                    ),
                    "formatted_outputs": ["markdown", "latex"],
                    "significant_digits": 4,
                },
            ],
        },
    )
    return report_path, (grid_result.path, baseline_result.path)


def test_report_schema_is_strict_and_missing_source_preflight_is_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_a = tmp_path / "missing_a.json"
    missing_b = tmp_path / "missing_b.json"
    report_path = write_json(
        tmp_path / "report.json",
        {
            "schema_version": "pol-report-v1",
            "name": "missing_sources",
            "profile": "test",
            "sources": [
                {"id": "a", "study_spec": str(missing_a)},
                {"id": "b", "study_spec": str(missing_b)},
            ],
            "reporters": [
                {
                    "kind": "baseline_summary_table",
                    "source_id": "a",
                    "filename": "table",
                    "rows": [
                        {
                            "id": "row",
                            "label": "row",
                            "variant_id": "heat",
                            "readout_id": "affine",
                        }
                    ],
                }
            ],
        },
    )
    raw = json.loads(report_path.read_text(encoding="utf-8"))
    raw["unknown_scientific_key"] = True
    write_json(report_path, raw)
    with pytest.raises(
        ValueError,
        match="Extra inputs are not permitted",
    ):
        load_report_spec(report_path, repo_root=tmp_path)

    raw.pop("unknown_scientific_key")
    write_json(report_path, raw)

    def forbidden(*args, **kwargs):
        raise AssertionError("missing-source preflight started numerical work")

    monkeypatch.setattr("pol.systems.registry.evolve", forbidden)
    monkeypatch.setattr("pol.study.readouts.fit_readout", forbidden)
    monkeypatch.setattr(
        "pol.study.trial.TrialEngine.evaluate_test",
        forbidden,
    )
    with pytest.raises(ValueError, match="missing"):
        run_report(
            load_report_spec(report_path, repo_root=tmp_path),
            repo_root=tmp_path,
        )


def test_cross_run_report_is_read_only_deterministic_and_separates_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path, source_paths = _build_sources_and_report_spec(tmp_path)
    source_bytes = {
        path: _tree_bytes(path) for path in source_paths
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("read-only reporting started numerical work")

    monkeypatch.setattr("pol.systems.registry.evolve", forbidden)
    monkeypatch.setattr("pol.study.readouts.fit_readout", forbidden)
    monkeypatch.setattr(
        "pol.study.trial.TrialEngine.evaluate_selection",
        forbidden,
    )
    monkeypatch.setattr(
        "pol.study.trial.TrialEngine.evaluate_test",
        forbidden,
    )
    spec = load_report_spec(report_path, repo_root=tmp_path)
    first = run_report(spec, repo_root=tmp_path)
    verify_report(first.path)
    second = run_report(spec, repo_root=tmp_path)
    assert second.reused is True
    assert second.report_id == first.report_id
    assert second.path == first.path
    for path in source_paths:
        assert _tree_bytes(path) == source_bytes[path]

    phase = _rows(
        first.path
        / "machine_readable_tables"
        / "validation_J_q.csv"
    )
    assert len(phase) == 4
    assert {row["cell_status"] for row in phase} == {"valid"}
    assert {
        (row["x_name"], row["y_name"], row["split"])
        for row in phase
    } == {("J", "q", "validation")}

    baseline = _rows(
        first.path / "machine_readable_tables" / "baseline.csv"
    )
    assert len(baseline) == 3
    assert {
        row["field_metric_name"] for row in baseline
    } == {"test_field_relative_l2_mean"}
    assert {
        row["data_metric_name"] for row in baseline
    } == {"test_data_field_relative_l2_mean"}
    assert {
        row["prediction_ensemble_in_primary"] for row in baseline
    } == {"False"}
    random_row = next(
        row for row in baseline if row["readout_kind"] == "random_feature_ridge"
    )
    assert random_row["primary_result_kind"] == (
        "independent_seed_metric_summary"
    )
    assert int(random_row["seed_count"]) == 2
    assert float(random_row["field_seed_std"]) >= 0.0
    assert float(random_row["field_seed_ci95_low"]) <= float(
        random_row["field_metric_value"]
    ) <= float(random_row["field_seed_ci95_high"])
    assert first.summary["source_count"] == 2
    assert first.summary["feature_solve"] is False
    assert first.summary["readout_fit"] is False
    assert first.summary["test_inference"] is False


def test_tampered_source_is_rejected_before_rendering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path, source_paths = _build_sources_and_report_spec(tmp_path)
    with (source_paths[0] / "test_metrics.csv").open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write("tampered\n")
    calls = {"render": 0}

    def forbidden_render(*args, **kwargs):
        calls["render"] += 1
        raise AssertionError("tampered source reached rendering")

    monkeypatch.setattr(
        "pol.reporting.runner._render_phase_diagram",
        forbidden_render,
    )
    with pytest.raises(ValueError, match="bytes differ from manifest"):
        run_report(
            load_report_spec(report_path, repo_root=tmp_path),
            repo_root=tmp_path,
        )
    assert calls["render"] == 0
    assert not (tmp_path / "reports").exists()


def test_failed_forced_report_preserves_verified_previous_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path, _ = _build_sources_and_report_spec(tmp_path)
    spec = load_report_spec(report_path, repo_root=tmp_path)
    completed = run_report(spec, repo_root=tmp_path)
    protected = _tree_bytes(completed.path)

    def fail_render(*args, **kwargs):
        raise RuntimeError("intentional reporter failure")

    monkeypatch.setattr(
        "pol.reporting.runner._render_phase_diagram",
        fail_render,
    )
    with pytest.raises(RuntimeError, match="intentional reporter failure"):
        run_report(spec, repo_root=tmp_path, force=True)
    assert _tree_bytes(completed.path) == protected
    verify_report(completed.path)
