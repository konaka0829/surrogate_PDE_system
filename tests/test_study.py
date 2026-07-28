from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from pol.config.loader import load_dataset_spec, load_study_spec
from pol.data.dataset import ensure_dataset
from pol.runtime.artifacts import manifest_records
from pol.runtime.io import write_csv, write_strict_json
from pol.study.cache import FeatureStateCache
from pol.study.runner import plan_study, regenerate_plots, run_study, verify_study_run
from tests.helpers import write_json, write_tiny_stack


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_plan_is_pure_and_scalar_is_a_one_cell_study(tmp_path: Path) -> None:
    _, _, study_path = write_tiny_stack(tmp_path)
    spec = load_study_spec(study_path, repo_root=tmp_path)
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    plan = plan_study(spec)
    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert plan["case_count"] == 1
    assert plan["filesystem_mutation"] is False
    assert before == after


def test_global_axis_uses_same_study_executor(tmp_path: Path) -> None:
    _, _, study_path = write_tiny_stack(tmp_path, global_q_values=(5, 9))
    spec = load_study_spec(study_path, repo_root=tmp_path)
    plan = plan_study(spec)
    assert plan["case_count"] == 2
    assert {case["global_values"]["output.q"] for case in plan["cases"]} == {5, 9}


def test_study_freezes_selection_before_any_test_evaluation(tmp_path: Path) -> None:
    _, _, study_path = write_tiny_stack(tmp_path)
    spec = load_study_spec(study_path, repo_root=tmp_path)
    result = run_study(spec, repo_root=tmp_path)
    verify_study_run(result.path)

    events = json.loads((result.path / "events.json").read_text(encoding="utf-8"))
    names = [item["event"] for item in events]
    assert names.index("freeze_written") < names.index("freeze_read_back")
    assert names.index("freeze_read_back") < names.index("first_test_state_solve")
    assert names.index("first_test_state_solve") <= names.index("first_test_metric")

    selection = json.loads((result.path / "selection_record.json").read_text(encoding="utf-8"))
    encoded = json.dumps(selection).lower()
    assert selection["test_data_used"] is False
    assert "test_ids" not in encoded and "test_metric" not in encoded

    validation_rows = _read_csv(result.path / "validation_trials.csv")
    test_rows = _read_csv(result.path / "test_metrics.csv")
    assert len(validation_rows) == 3
    assert len(test_rows) == 3
    assert sum(row["selected"] == "True" for row in validation_rows) == 3




def test_feature_state_batching_is_numerically_invariant(tmp_path: Path) -> None:
    _, dataset_path, study_path = write_tiny_stack(tmp_path)
    dataset_spec = load_dataset_spec(dataset_path, repo_root=tmp_path)
    dataset = ensure_dataset(dataset_spec, repo_root=tmp_path)
    study = load_study_spec(study_path, repo_root=tmp_path)
    sample_ids = dataset.sample_ids

    small = FeatureStateCache(
        artifact_root=tmp_path / "small-cache", enabled=False, batch_size=2
    ).get_or_solve(dataset, sample_ids, study.base_trial)
    large = FeatureStateCache(
        artifact_root=tmp_path / "large-cache", enabled=False, batch_size=64
    ).get_or_solve(dataset, sample_ids, study.base_trial)

    assert small.metadata == large.metadata
    assert small.values.shape == large.values.shape
    assert small.values.equal(large.values)


def test_only_validation_selected_candidates_reach_the_test_split(tmp_path: Path) -> None:
    _, _, study_path = write_tiny_stack(tmp_path)
    raw = json.loads(study_path.read_text(encoding="utf-8"))
    raw["variants"][0]["search"] = {
        "kind": "grid",
        "axes": [
            {
                "path": "feature.evolution.system.nu",
                "values": [0.025, 0.05],
            }
        ],
    }
    write_json(study_path, raw)

    spec = load_study_spec(study_path, repo_root=tmp_path)
    result = run_study(spec, repo_root=tmp_path)
    validation_rows = _read_csv(result.path / "validation_trials.csv")
    test_rows = _read_csv(result.path / "test_metrics.csv")

    assert len(validation_rows) == 6
    assert len(test_rows) == 3
    assert all(row["selected"] == "True" for row in test_rows)


def test_static_input_uses_the_same_study_executor(tmp_path: Path) -> None:
    _, _, study_path = write_tiny_stack(tmp_path)
    raw = json.loads(study_path.read_text(encoding="utf-8"))
    raw["base_trial"]["feature"] = {
        "kind": "static_input",
        "n_sur": 32,
        "observation": {
            "kind": "equispaced_points",
            "J": 32,
            "l2_scale": True,
        },
    }
    write_json(study_path, raw)

    spec = load_study_spec(study_path, repo_root=tmp_path)
    result = run_study(spec, repo_root=tmp_path)
    validation_rows = _read_csv(result.path / "validation_trials.csv")
    test_rows = _read_csv(result.path / "test_metrics.csv")
    assert validation_rows and test_rows
    assert {row["feature_system"] for row in validation_rows} == {"static_input"}
    assert {row["feature_time"] for row in validation_rows} == {"0.0"}


def test_study_rejects_finite_resolution_above_reference_grid(tmp_path: Path) -> None:
    _, _, study_path = write_tiny_stack(tmp_path)
    raw = json.loads(study_path.read_text(encoding="utf-8"))
    raw["base_trial"]["input"]["n_tar"] = 64
    raw["base_trial"]["output"]["q"] = 9
    write_json(study_path, raw)
    spec = load_study_spec(study_path, repo_root=tmp_path)
    with pytest.raises(ValueError, match="exceeds the validated target"):
        run_study(spec, repo_root=tmp_path)




def test_study_verification_checks_frozen_plan_semantics(tmp_path: Path) -> None:
    _, _, study_path = write_tiny_stack(tmp_path)
    spec = load_study_spec(study_path, repo_root=tmp_path)
    result = run_study(spec, repo_root=tmp_path)

    plan_path = result.path / "frozen_evaluation_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["plan_content_hash"] = "0" * 64
    write_strict_json(plan_path, plan)

    manifest_path = result.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for index, record in enumerate(manifest["files"]):
        if record["relative_path"] == "frozen_evaluation_plan.json":
            manifest["files"][index] = manifest_records(
                result.path, ["frozen_evaluation_plan.json"]
            )[0]
            break
    write_strict_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="plan content hash"):
        verify_study_run(result.path)


def test_study_verification_checks_test_table_bindings(tmp_path: Path) -> None:
    _, _, study_path = write_tiny_stack(tmp_path)
    spec = load_study_spec(study_path, repo_root=tmp_path)
    result = run_study(spec, repo_root=tmp_path)

    metrics_path = result.path / "test_metrics.csv"
    rows = _read_csv(metrics_path)
    rows[0]["selected"] = "False"
    write_csv(metrics_path, rows, fieldnames=rows[0].keys())

    manifest_path = result.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for index, record in enumerate(manifest["files"]):
        if record["relative_path"] == "test_metrics.csv":
            manifest["files"][index] = manifest_records(
                result.path, ["test_metrics.csv"]
            )[0]
            break
    write_strict_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="not marked as selected"):
        verify_study_run(result.path)


def test_plot_regeneration_is_transactional_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, study_path = write_tiny_stack(tmp_path, generate_plots=True)
    spec = load_study_spec(study_path, repo_root=tmp_path)
    result = run_study(spec, repo_root=tmp_path)
    before_manifest = (result.path / "manifest.json").read_bytes()

    def fail_reporters(*args, **kwargs):
        raise RuntimeError("synthetic plotting failure")

    monkeypatch.setattr("pol.study.runner.generate_reporters", fail_reporters)
    with pytest.raises(RuntimeError, match="synthetic plotting failure"):
        regenerate_plots(spec, result.path)

    verify_study_run(result.path)
    assert (result.path / "manifest.json").read_bytes() == before_manifest
    assert not list(result.path.parent.glob(f".{result.path.name}.staging-*"))
    assert not list(result.path.parent.glob(f".{result.path.name}.backup-*"))


def test_plots_only_requires_an_existing_run(tmp_path: Path) -> None:
    _, _, study_path = write_tiny_stack(tmp_path, generate_plots=True)
    spec = load_study_spec(study_path, repo_root=tmp_path)
    with pytest.raises(ValueError, match="existing verified run"):
        run_study(spec, repo_root=tmp_path, plots_only=True)


def test_study_reuses_verified_run_and_regenerates_plots(tmp_path: Path) -> None:
    _, _, study_path = write_tiny_stack(tmp_path, generate_plots=True)
    spec = load_study_spec(study_path, repo_root=tmp_path)
    first = run_study(spec, repo_root=tmp_path)
    second = run_study(spec, repo_root=tmp_path)
    assert second.reused and second.path == first.path
    created = regenerate_plots(spec, first.path)
    assert created == ["validation_error_vs_q.png"]
    verify_study_run(first.path)
