from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from pol.cli import main
from pol.config.loader import load_study_spec
from pol.runtime.artifacts import manifest_records
from pol.runtime.hashing import stable_object_hash
from pol.runtime.io import write_csv, write_strict_json
from pol.study.cases import build_cases
from pol.study.runner import plan_study, run_study, verify_study_run
from pol.study.selection_source import (
    MissingSelectionDependencyError,
    SelectionDependencyError,
    resolve_selection_bindings,
    verify_downstream_selection,
    verify_selection_dataset_bindings,
)
from tests.helpers import write_json, write_tiny_stack


def _refresh_manifest_record(run_path: Path, relative_path: str) -> None:
    manifest_path = run_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for index, record in enumerate(manifest["files"]):
        if record["relative_path"] == relative_path:
            manifest["files"][index] = manifest_records(
                run_path,
                [relative_path],
            )[0]
            break
    else:
        raise AssertionError(f"manifest has no record for {relative_path}")
    write_strict_json(manifest_path, manifest)


def _bound_stack(
    root: Path,
    *,
    source_nu: float = 0.07,
    source_time: float = 0.2,
) -> tuple[Path, Path, object]:
    _, _, source_path = write_tiny_stack(
        root,
        include_diagnostics=False,
    )
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["name"] = "tiny_feature_landscape"
    source["base_trial"]["feature"]["evolution"] = {
        "system": {"kind": "heat", "nu": source_nu},
        "time": source_time,
    }
    write_json(source_path, source)
    source_result = run_study(
        load_study_spec(source_path, repo_root=root),
        repo_root=root,
    )

    downstream = copy.deepcopy(source)
    downstream["name"] = "tiny_selection_bound"
    downstream["base_trial"]["feature"]["evolution"] = {
        "system": {"kind": "heat", "nu": 0.9},
        "time": 0.3,
    }
    downstream["variants"][0]["selection_source"] = {
        "kind": "completed_study_selection",
        "source_study_spec": str(source_path),
        "source_variant_id": "heat",
        "source_readout_id": "affine",
        "import_paths": [
            "feature.evolution.system",
            "feature.evolution.time",
        ],
    }
    downstream_path = write_json(root / "downstream.json", downstream)
    return source_path, downstream_path, source_result


def test_completed_source_imports_selected_condition_and_binds_freeze(
    tmp_path: Path,
    capsys,
) -> None:
    source_path, downstream_path, source_result = _bound_stack(tmp_path)
    downstream = load_study_spec(downstream_path, repo_root=tmp_path)
    resolved = resolve_selection_bindings(
        downstream,
        repo_root=tmp_path,
    )
    cases, _ = build_cases(resolved.spec)
    trial = cases[0].trial
    assert trial.feature.evolution is not None
    assert trial.feature.evolution.system.kind == "heat"
    assert float(trial.feature.evolution.system.nu) == 0.07
    assert float(trial.feature.evolution.time) == 0.2
    assert float(downstream.base_trial.feature.evolution.system.nu) == 0.9
    assert float(downstream.base_trial.feature.evolution.time) == 0.3

    result = run_study(downstream, repo_root=tmp_path)
    verify_study_run(result.path)
    resolved_study = json.loads(
        (result.path / "resolved_study.json").read_text(encoding="utf-8")
    )
    provenance = resolved_study["variants"][0]["selection_source"]
    assert provenance["source_study_run_hash"] == (
        json.loads(
            (source_result.path / "run_summary.json").read_text(
                encoding="utf-8"
            )
        )["run_hash"]
    )
    assert provenance["selection_metric"].startswith("validation_")
    assert provenance["resolved_imported_feature_condition"] == {
        "feature.evolution.system": {"kind": "heat", "nu": 0.07},
        "feature.evolution.time": 0.2,
    }
    selection = json.loads(
        (result.path / "selection_record.json").read_text(encoding="utf-8")
    )
    plan = json.loads(
        (result.path / "frozen_evaluation_plan.json").read_text(
            encoding="utf-8"
        )
    )
    assert selection["selection_source_provenance"]["heat"] == provenance
    assert plan["selection_source_provenance"]["heat"] == provenance
    expected_source_fields = {
        "selected_condition_source_provenance_hash": stable_object_hash(
            provenance
        ),
        "selected_condition_source_study_run_hash": provenance[
            "source_study_run_hash"
        ],
        "selected_condition_source_selection_record_hash": provenance[
            "source_selection_record_hash"
        ],
        "selected_condition_source_frozen_plan_hash": provenance[
            "source_frozen_plan_hash"
        ],
        "selected_condition_source_frozen_model_archive_hash": provenance[
            "source_frozen_model_archive_hash"
        ],
        "selected_condition_source_feature_system_hash": stable_object_hash(
            {"kind": "heat", "nu": 0.07}
        ),
    }
    for name in (
        "validation_trials.csv",
        "test_metrics.csv",
        "random_feature_seed_metrics.csv",
        "random_feature_ensemble_metrics.csv",
    ):
        with (result.path / name).open(
            newline="",
            encoding="utf-8",
        ) as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            assert row["result_row_schema_version"] == (
                    "pol-study-result-row-v3"
            )
            assert row["feature_system_condition_hash"] == (
                expected_source_fields[
                    "selected_condition_source_feature_system_hash"
                ]
            )
            assert {
                field: row[field] for field in expected_source_fields
            } == expected_source_fields
            assert row["readout_kind"] in {
                "direct_fourier_decoder",
                "affine_ridge",
                "random_feature_ridge",
            }
            assert row["n_tar"] == "16"
            assert row["n_sur"] == "32"

    assert main(["selection", "inspect", str(source_path)]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["status"] == "completed"
    assert inspected["test_tables_used_for_condition_selection"] is False
    assert main(["selection", "verify", str(downstream_path)]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["status"] == "pass"
    assert verified["filesystem_mutation"] is False

    table_path = result.path / "validation_trials.csv"
    with table_path.open(newline="", encoding="utf-8") as handle:
        validation_rows = list(csv.DictReader(handle))
    validation_rows[0]["feature_system_condition_hash"] = "0" * 64
    write_csv(
        table_path,
        validation_rows,
        fieldnames=validation_rows[0].keys(),
    )
    _refresh_manifest_record(result.path, "validation_trials.csv")
    with pytest.raises(ValueError, match="feature-system hash mismatch"):
        verify_study_run(result.path)


def test_selection_source_schema_rejects_unauthorized_and_unknown_keys(
    tmp_path: Path,
) -> None:
    _, downstream_path, _ = _bound_stack(tmp_path)
    raw = json.loads(downstream_path.read_text(encoding="utf-8"))
    raw["variants"][0]["selection_source"]["import_paths"] = ["input.n_tar"]
    unauthorized = write_json(tmp_path / "unauthorized.json", raw)
    with pytest.raises(ValueError, match="import_paths"):
        load_study_spec(unauthorized, repo_root=tmp_path)

    raw["variants"][0]["selection_source"]["import_paths"] = [
        "feature.evolution.time"
    ]
    raw["variants"][0]["selection_source"]["unknown"] = True
    unknown = write_json(tmp_path / "unknown.json", raw)
    with pytest.raises(ValueError, match="unknown"):
        load_study_spec(unknown, repo_root=tmp_path)

    legacy = json.loads(downstream_path.read_text(encoding="utf-8"))
    legacy["schema_version"] = "pol-study-v2"
    legacy_path = write_json(tmp_path / "legacy.json", legacy)
    with pytest.raises(ValueError, match="migrate to pol-study-v3"):
        load_study_spec(legacy_path, repo_root=tmp_path)


def test_missing_source_is_visible_in_plan_and_fails_before_test_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, study_path = write_tiny_stack(tmp_path)
    raw = json.loads(study_path.read_text(encoding="utf-8"))
    raw["variants"][0]["selection_source"] = {
        "kind": "completed_study_selection",
        "source_study_spec": str(tmp_path / "missing.json"),
        "source_variant_id": "heat",
        "source_readout_id": "affine",
        "import_paths": ["feature.evolution.time"],
    }
    downstream_path = write_json(tmp_path / "downstream.json", raw)
    spec = load_study_spec(downstream_path, repo_root=tmp_path)
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    plan = plan_study(spec, repo_root=tmp_path)
    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert plan["selection_dependencies"]["status"] == "missing"
    assert plan["scientific_conditions_resolved"] is False
    assert before == after

    def forbidden_test_access(*args, **kwargs):
        raise AssertionError("downstream test access must not start")

    monkeypatch.setattr(
        "pol.study.trial.TrialEngine.evaluate_test",
        forbidden_test_access,
    )
    with pytest.raises(MissingSelectionDependencyError, match="does not exist"):
        run_study(spec, repo_root=tmp_path)


def test_profile_mismatch_and_dependency_cycle_are_rejected(
    tmp_path: Path,
) -> None:
    source_path, downstream_path, _ = _bound_stack(tmp_path)
    downstream = json.loads(downstream_path.read_text(encoding="utf-8"))
    downstream["profile"] = "main"
    wrong_profile = write_json(tmp_path / "wrong_profile.json", downstream)
    with pytest.raises(SelectionDependencyError, match="profile mismatch"):
        resolve_selection_bindings(
            load_study_spec(wrong_profile, repo_root=tmp_path),
            repo_root=tmp_path,
        )

    base = json.loads(source_path.read_text(encoding="utf-8"))
    first_path = tmp_path / "cycle_a.json"
    second_path = tmp_path / "cycle_b.json"
    for name, path, dependency in (
        ("cycle_a", first_path, second_path),
        ("cycle_b", second_path, first_path),
    ):
        raw = copy.deepcopy(base)
        raw["name"] = name
        raw["variants"][0]["selection_source"] = {
            "kind": "completed_study_selection",
            "source_study_spec": str(dependency),
            "source_variant_id": "heat",
            "source_readout_id": "affine",
            "import_paths": ["feature.evolution.time"],
        }
        write_json(path, raw)
    with pytest.raises(SelectionDependencyError, match="cycle"):
        resolve_selection_bindings(
            load_study_spec(first_path, repo_root=tmp_path),
            repo_root=tmp_path,
        )


def test_source_tamper_fails_before_downstream_feature_or_test_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, downstream_path, source_result = _bound_stack(tmp_path)
    selection_path = source_result.path / "selection_record.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["cases"]["heat"]["validation_metrics"]["affine"] = -1.0
    write_strict_json(selection_path, selection)

    def forbidden_work(*args, **kwargs):
        raise AssertionError("downstream numerical/test work must not start")

    monkeypatch.setattr(
        "pol.study.trial.TrialEngine.__init__",
        forbidden_work,
    )
    with pytest.raises(SelectionDependencyError, match="failed verification"):
        run_study(
            load_study_spec(downstream_path, repo_root=tmp_path),
            repo_root=tmp_path,
        )


def test_source_selection_record_with_test_field_is_rejected(
    tmp_path: Path,
) -> None:
    _, downstream_path, source_result = _bound_stack(tmp_path)
    selection_path = source_result.path / "selection_record.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["test_metric_leak"] = 0.0
    write_strict_json(selection_path, selection)
    _refresh_manifest_record(source_result.path, "selection_record.json")
    with pytest.raises(
        SelectionDependencyError,
        match="selection record contains test binding",
    ):
        resolve_selection_bindings(
            load_study_spec(downstream_path, repo_root=tmp_path),
            repo_root=tmp_path,
        )


def test_dataset_and_split_mismatch_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, downstream_path, _ = _bound_stack(tmp_path)
    resolved = resolve_selection_bindings(
        load_study_spec(downstream_path, repo_root=tmp_path),
        repo_root=tmp_path,
    )
    provenance = resolved.provenance_by_variant
    source = provenance["heat"]
    with pytest.raises(SelectionDependencyError, match="dataset artifact"):
        verify_selection_dataset_bindings(
            provenance,
            dataset=SimpleNamespace(
                artifact_id="different",
                split_hash=source["source_dataset_split_hash"],
            ),
        )
    with pytest.raises(SelectionDependencyError, match="split mismatch"):
        verify_selection_dataset_bindings(
            provenance,
            dataset=SimpleNamespace(
                artifact_id=source["source_dataset_artifact_id"],
                split_hash="different",
            ),
        )

    different_dataset = json.loads(
        (tmp_path / "dataset.json").read_text(encoding="utf-8")
    )
    different_dataset["reference_nx"] = 16
    different_dataset_path = write_json(
        tmp_path / "different_dataset.json",
        different_dataset,
    )
    downstream = json.loads(
        downstream_path.read_text(encoding="utf-8")
    )
    downstream["dataset_spec"] = str(different_dataset_path)
    mismatched_path = write_json(
        tmp_path / "mismatched_dataset_study.json",
        downstream,
    )

    def forbidden_dataset_build(*args, **kwargs):
        raise AssertionError("a mismatched downstream dataset must not build")

    monkeypatch.setattr(
        "pol.study.runner.ensure_dataset",
        forbidden_dataset_build,
    )
    with pytest.raises(
        MissingSelectionDependencyError,
        match="expected dataset artifact",
    ):
        run_study(
            load_study_spec(mismatched_path, repo_root=tmp_path),
            repo_root=tmp_path,
        )


def test_downstream_identity_is_storage_independent_and_source_sensitive(
    tmp_path: Path,
) -> None:
    source_path, downstream_path, source_result = _bound_stack(tmp_path)
    baseline = verify_downstream_selection(
        load_study_spec(downstream_path, repo_root=tmp_path),
        repo_root=tmp_path,
    )

    relocated_root = tmp_path / "relocated_outputs"
    relocated_run = (
        relocated_root
        / "tiny_feature_landscape"
        / source_result.path.name
    )
    relocated_run.parent.mkdir(parents=True)
    shutil.copytree(source_result.path, relocated_run)
    relocated_source = json.loads(source_path.read_text(encoding="utf-8"))
    relocated_source["output_root"] = str(relocated_root)
    relocated_source_path = write_json(
        tmp_path / "relocated_source.json",
        relocated_source,
    )
    relocated_downstream = json.loads(
        downstream_path.read_text(encoding="utf-8")
    )
    relocated_downstream["variants"][0]["selection_source"][
        "source_study_spec"
    ] = str(relocated_source_path)
    relocated_downstream_path = write_json(
        tmp_path / "relocated_downstream.json",
        relocated_downstream,
    )
    relocated = verify_downstream_selection(
        load_study_spec(relocated_downstream_path, repo_root=tmp_path),
        repo_root=tmp_path,
    )
    assert relocated["run_hash"] == baseline["run_hash"]
    assert (
        relocated["study_scientific_identity_hash"]
        == baseline["study_scientific_identity_hash"]
    )

    changed_source = json.loads(source_path.read_text(encoding="utf-8"))
    changed_source["base_trial"]["feature"]["evolution"]["system"]["nu"] = 0.08
    changed_source_path = write_json(
        tmp_path / "changed_source.json",
        changed_source,
    )
    run_study(
        load_study_spec(changed_source_path, repo_root=tmp_path),
        repo_root=tmp_path,
    )
    changed_downstream = json.loads(
        downstream_path.read_text(encoding="utf-8")
    )
    changed_downstream["variants"][0]["selection_source"][
        "source_study_spec"
    ] = str(changed_source_path)
    changed_downstream_path = write_json(
        tmp_path / "changed_downstream.json",
        changed_downstream,
    )
    changed = verify_downstream_selection(
        load_study_spec(changed_downstream_path, repo_root=tmp_path),
        repo_root=tmp_path,
    )
    assert changed["run_hash"] != baseline["run_hash"]
    assert (
        changed["study_scientific_identity_hash"]
        != baseline["study_scientific_identity_hash"]
    )


def test_source_test_metric_is_not_used_to_resolve_feature_condition(
    tmp_path: Path,
) -> None:
    _, downstream_path, source_result = _bound_stack(tmp_path)
    spec = load_study_spec(downstream_path, repo_root=tmp_path)
    baseline = verify_downstream_selection(spec, repo_root=tmp_path)

    table_path = source_result.path / "test_metrics.csv"
    with table_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    affine = next(row for row in rows if row["readout_id"] == "affine")
    affine["test_field_relative_l2_mean"] = "987654.0"
    write_csv(table_path, rows, fieldnames=rows[0].keys())
    _refresh_manifest_record(source_result.path, "test_metrics.csv")
    comparison_path = source_result.path / "selected_comparison.csv"
    with comparison_path.open(newline="", encoding="utf-8") as handle:
        comparison_rows = list(csv.DictReader(handle))
    comparison_affine = next(
        row for row in comparison_rows if row["readout_id"] == "affine"
    )
    comparison_affine["test_field_relative_l2_mean"] = "987654.0"
    write_csv(
        comparison_path,
        comparison_rows,
        fieldnames=comparison_rows[0].keys(),
    )
    _refresh_manifest_record(source_result.path, "selected_comparison.csv")
    verify_study_run(source_result.path)

    after = verify_downstream_selection(spec, repo_root=tmp_path)
    assert after["run_hash"] == baseline["run_hash"]
    assert (
        after["selection_source_provenance"]
        == baseline["selection_source_provenance"]
    )
    assert after["test_tables_used_for_condition_selection"] is False
