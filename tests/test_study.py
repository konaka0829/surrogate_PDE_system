from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from pol.config.loader import load_dataset_spec, load_study_spec
from pol.data.dataset import ensure_dataset
from pol.learning.direct import (
    DIRECT_DECODER_DIAGNOSTIC_FIELDS,
    DIRECT_DECODER_POLICY,
)
from pol.runtime.artifacts import manifest_records
from pol.runtime.hashing import stable_object_hash
from pol.runtime.io import file_sha256, write_csv, write_strict_json
from pol.study.cache import FeatureStateCache
from pol.study.runner import plan_study, regenerate_plots, run_study, verify_study_run
from tests.helpers import write_json, write_tiny_stack


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def test_readout_and_evaluation_modules_import_independently() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "\n".join(
                (
                    "import sys",
                    "import pol.study.evaluation",
                    "assert 'pol.study.runner' not in sys.modules",
                    "assert 'pol.study.trial' not in sys.modules",
                    "import pol.study.readouts",
                    "assert 'pol.study.runner' not in sys.modules",
                    "assert 'pol.study.trial' not in sys.modules",
                    "from pol.study import run_study",
                    "assert callable(run_study)",
                )
            ),
        ],
        check=True,
    )


def test_study_support_modules_import_without_runner_cycle() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "\n".join(
                (
                    "import sys",
                    "import pol.study.cases",
                    "import pol.study.protocol",
                    "import pol.study.results",
                    "import pol.study.verification",
                    "assert 'pol.study.runner' not in sys.modules",
                    "from pol.study.runner import (",
                    "    plan_study,",
                    "    regenerate_plots,",
                    "    run_study,",
                    "    verify_study_run,",
                    ")",
                    "assert all(callable(value) for value in (",
                    "    plan_study, regenerate_plots, run_study, verify_study_run",
                    "))",
                )
            ),
        ],
        check=True,
    )


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
    assert plan["schema_version"] == "pol-study-plan-v4"
    assert plan["workload"]["case_count"] == 2
    assert plan["workload"]["candidate_trial_upper_bound"] == 2


def test_checked_in_observation_output_budget_keeps_q_greater_than_J_cells() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    spec = load_study_spec(
        repo_root / "studies" / "observation_output_budget_smoke.json",
        repo_root=repo_root,
    )
    plan = plan_study(spec)
    cells = {
        (
            int(case["global_values"]["feature.observation.J"]),
            int(case["global_values"]["output.q"]),
        )
        for case in plan["cases"]
    }
    assert plan["case_count"] == 8
    assert (4, 9) in cells
    assert any(q > J for J, q in cells)


def test_direct_decoder_diagnostic_is_bound_across_study_artifacts(
    tmp_path: Path,
) -> None:
    _, _, study_path = write_tiny_stack(
        tmp_path,
        observation_J=4,
        include_diagnostics=False,
    )
    spec = load_study_spec(study_path, repo_root=tmp_path)
    result = run_study(spec, repo_root=tmp_path)
    verify_study_run(result.path)

    expected = {
        "decoder_policy": DIRECT_DECODER_POLICY,
        "decoder_observation_count": "4",
        "decoder_requested_q": "9",
        "decoder_observable_q": "3",
        "decoder_retained_q": "3",
        "decoder_requested_max_mode": "4",
        "decoder_observable_max_mode": "1",
        "decoder_zero_filled_mode_count": "3",
        "decoder_zero_filled_coefficient_count": "6",
        "decoder_zero_fill_applied": "True",
    }
    validation_rows = _read_csv(result.path / "validation_trials.csv")
    direct_validation = next(
        row for row in validation_rows if row["readout_id"] == "direct"
    )
    assert {
        field: direct_validation[field]
        for field in DIRECT_DECODER_DIAGNOSTIC_FIELDS
    } == expected
    for row in validation_rows:
        if row["readout_id"] != "direct":
            assert all(row[field] == "" for field in DIRECT_DECODER_DIAGNOSTIC_FIELDS)

    selection = json.loads(
        (result.path / "selection_record.json").read_text(encoding="utf-8")
    )
    direct_inner = selection["cases"]["heat"]["inner_selections"]["direct"]
    assert direct_inner["decoder_observable_q"] == 3
    assert direct_inner["decoder_zero_filled_coefficient_count"] == 6
    for readout_id in ("affine", "random"):
        assert not (
            set(selection["cases"]["heat"]["inner_selections"][readout_id])
            & set(DIRECT_DECODER_DIAGNOSTIC_FIELDS)
        )

    archive = torch.load(
        result.path / "frozen_models.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert archive["schema_version"] == "pol-frozen-model-archive-v10"
    entries = {
        entry["readout_id"]: entry for entry in archive["models"].values()
    }
    assert entries["direct"]["model"]["decoder_observable_q"] == 3
    assert (
        entries["direct"]["model"]["decoder_zero_filled_coefficient_count"]
        == 6
    )
    for readout_id in ("affine", "random"):
        assert not (
            set(entries[readout_id]["model"])
            & set(DIRECT_DECODER_DIAGNOSTIC_FIELDS)
        )
    assert entries["affine"]["model"]["W"].shape[0] == 9
    assert all(
        member["W"].shape[0] == 9
        for member in entries["random"]["model"]["members"]
    )

    plan = json.loads(
        (result.path / "frozen_evaluation_plan.json").read_text(encoding="utf-8")
    )
    plan_diagnostics = plan["cases"]["heat"]["decoder_diagnostics_by_readout"]
    assert set(plan_diagnostics) == {"direct"}
    assert plan_diagnostics["direct"]["decoder_zero_fill_applied"] is True

    test_rows = _read_csv(result.path / "test_metrics.csv")
    direct_test = next(row for row in test_rows if row["readout_id"] == "direct")
    assert {
        field: direct_test[field]
        for field in DIRECT_DECODER_DIAGNOSTIC_FIELDS
    } == expected
    for row in test_rows:
        if row["readout_id"] != "direct":
            assert all(row[field] == "" for field in DIRECT_DECODER_DIAGNOSTIC_FIELDS)
    assert result.summary["direct_decoder_diagnostic_count"] == 1
    assert result.summary["direct_decoder_zero_fill_count"] == 1
    assert result.summary["direct_decoder_zero_fill_applied"] is True


@pytest.mark.parametrize(
    ("readout_id", "field", "value", "error"),
    [
        (
            "direct",
            "decoder_observable_q",
            "5",
            "direct validation row decoder_observable_q",
        ),
        (
            "affine",
            "decoder_policy",
            DIRECT_DECODER_POLICY,
            "false direct-decoder diagnostic",
        ),
    ],
)
def test_study_verifier_rejects_decoder_diagnostic_tampering(
    tmp_path: Path,
    readout_id: str,
    field: str,
    value: str,
    error: str,
) -> None:
    _, _, study_path = write_tiny_stack(
        tmp_path,
        observation_J=4,
        include_diagnostics=False,
    )
    result = run_study(
        load_study_spec(study_path, repo_root=tmp_path),
        repo_root=tmp_path,
    )
    relative_path = "validation_trials.csv"
    rows = _read_csv(result.path / relative_path)
    row = next(item for item in rows if item["readout_id"] == readout_id)
    row[field] = value
    write_csv(
        result.path / relative_path,
        rows,
        fieldnames=rows[0].keys(),
    )
    _refresh_manifest_record(result.path, relative_path)
    with pytest.raises(ValueError, match=error):
        verify_study_run(result.path)


def test_frozen_decoder_mismatch_stops_before_test_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, study_path = write_tiny_stack(
        tmp_path,
        observation_J=4,
        include_diagnostics=False,
    )
    spec = load_study_spec(study_path, repo_root=tmp_path)

    import pol.study.protocol as protocol_module

    original_save = protocol_module.atomic_torch_save

    def tampering_save(path: Path, value: dict[str, object]) -> None:
        tampered = copy.deepcopy(value)
        if Path(path).name == "frozen_models.pt":
            direct = next(
                entry
                for entry in tampered["models"].values()
                if entry["readout_id"] == "direct"
            )
            direct["model"]["decoder_retained_q"] = 5
        original_save(path, tampered)

    def forbidden_test_evaluation(*args, **kwargs):
        raise AssertionError("test evaluation must not start")

    monkeypatch.setattr(protocol_module, "atomic_torch_save", tampering_save)
    monkeypatch.setattr(
        "pol.study.trial.TrialEngine.evaluate_test",
        forbidden_test_evaluation,
    )
    with pytest.raises(
        ValueError,
        match="frozen direct model decoder_retained_q",
    ):
        run_study(spec, repo_root=tmp_path)


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
    dataset_reference = json.loads(
        (result.path / "dataset_reference.json").read_text(encoding="utf-8")
    )
    assert dataset_reference["dataset_binding_kind"] == "foundation_only"
    assert dataset_reference["dataset_binding_status"] == "pass"
    assert (
        dataset_reference["dataset_target_reference_validation_status"]
        == "not_claimed"
    )
    assert dataset_reference["dataset_binding_proof_hash"]
    assert (
        result.summary["dataset_target_reference_validation_status"]
        == "not_claimed"
    )


def test_freeze_protocol_cross_hashes_and_event_payloads(tmp_path: Path) -> None:
    _, _, study_path = write_tiny_stack(tmp_path)
    spec = load_study_spec(study_path, repo_root=tmp_path)
    result = run_study(spec, repo_root=tmp_path)

    selection = json.loads(
        (result.path / "selection_record.json").read_text(encoding="utf-8")
    )
    selection_hash = stable_object_hash(selection)
    plan = json.loads(
        (result.path / "frozen_evaluation_plan.json").read_text(
            encoding="utf-8"
        )
    )
    plan_hash = plan.pop("plan_content_hash")
    assert stable_object_hash(plan) == plan_hash
    archive = torch.load(
        result.path / plan["frozen_models_file"],
        map_location="cpu",
        weights_only=True,
    )
    assert plan["selection_record_hash"] == selection_hash
    assert archive["selection_record_hash"] == selection_hash
    assert file_sha256(result.path / plan["frozen_models_file"]) == (
        plan["frozen_models_sha256"]
    )
    assert result.summary["selection_record_hash"] == selection_hash
    assert result.summary["frozen_plan_hash"] == plan_hash

    events = json.loads(
        (result.path / "events.json").read_text(encoding="utf-8")
    )
    assert [event["event"] for event in events] == [
        "selection_complete",
        "evaluation_members_materialized",
        "convergence_complete",
        "freeze_written",
        "freeze_read_back",
        "first_test_state_solve",
        "first_test_metric",
    ]
    assert events[0]["selection_record_hash"] == selection_hash
    assert events[1]["selection_record_hash"] == selection_hash
    assert events[2]["status"] == {"heat": "not_requested"}
    assert all(
        event["plan_content_hash"] == plan_hash for event in events[3:]
    )

    for name in (
        "test_metrics.csv",
        "random_feature_seed_metrics.csv",
        "random_feature_ensemble_metrics.csv",
    ):
        for row in _read_csv(result.path / name):
            assert row["selection_record_hash"] == selection_hash
            assert row["frozen_plan_hash"] == plan_hash


def test_random_feature_test_tables_bind_seed_summary_and_ensemble(
    tmp_path: Path,
) -> None:
    _, _, study_path = write_tiny_stack(tmp_path)
    spec = load_study_spec(study_path, repo_root=tmp_path)
    result = run_study(spec, repo_root=tmp_path)

    primary_rows = _read_csv(result.path / "test_metrics.csv")
    seed_rows = _read_csv(result.path / "random_feature_seed_metrics.csv")
    ensemble_rows = _read_csv(
        result.path / "random_feature_ensemble_metrics.csv"
    )
    primary = next(row for row in primary_rows if row["readout_id"] == "random")
    assert primary["test_result_kind"] == "independent_seed_metric_summary"
    assert primary["test_seed_count"] == "2"
    assert primary["test_seed_std_ddof"] == "1"
    assert primary["test_confidence_level"] == "0.95"
    assert primary["test_confidence_interval_method"] == "student_t"
    assert float(primary["test_field_relative_l2_mean"]) == pytest.approx(
        float(primary["test_field_relative_l2_mean_seed_mean"])
    )
    deterministic_rows = [
        row for row in primary_rows if row["readout_id"] != "random"
    ]
    assert all(row["test_result_kind"] == "single_model" for row in deterministic_rows)
    assert all(row["test_seed_count"] == "" for row in deterministic_rows)
    assert all(
        row["test_field_relative_l2_mean_seed_std"] == ""
        for row in deterministic_rows
    )

    archive = torch.load(
        result.path / "frozen_models.pt",
        map_location="cpu",
        weights_only=True,
    )
    random_entry = next(
        entry
        for entry in archive["models"].values()
        if entry["readout_id"] == "random"
    )
    frozen_seeds = {int(member["seed"]) for member in random_entry["model"]["members"]}
    assert {int(row["seed"]) for row in seed_rows} == frozen_seeds
    assert all(
        row["test_result_kind"] == "independent_seed_realization"
        for row in seed_rows
    )
    assert all(row["selection_record_hash"] for row in seed_rows)
    assert all(row["frozen_plan_hash"] for row in seed_rows)

    assert len(ensemble_rows) == 1
    assert ensemble_rows[0]["test_result_kind"] == "prediction_ensemble"
    assert ensemble_rows[0]["ensemble_member_count"] == "2"
    assert "test_ensemble_field_relative_l2_mean" in ensemble_rows[0]
    assert result.summary["primary_test_row_count"] == 3
    assert result.summary["random_feature_seed_row_count"] == 2
    assert result.summary["random_feature_ensemble_row_count"] == 1




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
    with pytest.raises(ValueError, match="exceeds the dataset target"):
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


def test_study_verifier_rejects_unmanifested_file_tamper(
    tmp_path: Path,
) -> None:
    _, _, study_path = write_tiny_stack(tmp_path)
    result = run_study(
        load_study_spec(study_path, repo_root=tmp_path),
        repo_root=tmp_path,
    )
    selection_path = result.path / "selection_record.json"
    selection_path.write_bytes(selection_path.read_bytes() + b" ")

    with pytest.raises(ValueError, match="bytes differ from manifest"):
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


@pytest.mark.parametrize(
    ("tamper_kind", "error"),
    [
        ("seed", "primary metric test_coefficient_mse"),
        ("summary", "primary metric test_coefficient_mse"),
        ("ensemble", "ensemble member count"),
    ],
)
def test_study_verifier_rejects_random_feature_statistical_tampering(
    tmp_path: Path,
    tamper_kind: str,
    error: str,
) -> None:
    _, _, study_path = write_tiny_stack(tmp_path)
    spec = load_study_spec(study_path, repo_root=tmp_path)
    result = run_study(spec, repo_root=tmp_path)

    if tamper_kind == "seed":
        relative_path = "random_feature_seed_metrics.csv"
        rows = _read_csv(result.path / relative_path)
        rows[0]["test_coefficient_mse"] = str(
            float(rows[0]["test_coefficient_mse"]) + 1.0
        )
    elif tamper_kind == "summary":
        relative_path = "test_metrics.csv"
        rows = _read_csv(result.path / relative_path)
        random_row = next(row for row in rows if row["readout_id"] == "random")
        random_row["test_coefficient_mse"] = str(
            float(random_row["test_coefficient_mse"]) + 1.0
        )
    else:
        relative_path = "random_feature_ensemble_metrics.csv"
        rows = _read_csv(result.path / relative_path)
        rows[0]["ensemble_member_count"] = "3"
    write_csv(
        result.path / relative_path,
        rows,
        fieldnames=rows[0].keys(),
    )
    _refresh_manifest_record(result.path, relative_path)

    with pytest.raises(ValueError, match=error):
        verify_study_run(result.path)


def test_legacy_study_run_manifest_is_rejected_explicitly(tmp_path: Path) -> None:
    _, _, study_path = write_tiny_stack(tmp_path)
    spec = load_study_spec(study_path, repo_root=tmp_path)
    result = run_study(spec, repo_root=tmp_path)
    manifest_path = result.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "pol-study-run-manifest-v2"
    write_strict_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="unsupported study-run manifest"):
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


def test_plots_only_does_not_start_dataset_or_feature_computation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, study_path = write_tiny_stack(tmp_path, generate_plots=True)
    spec = load_study_spec(study_path, repo_root=tmp_path)
    first = run_study(spec, repo_root=tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("plots-only must not start numerical computation")

    monkeypatch.setattr("pol.study.runner.run_search", forbidden)
    monkeypatch.setattr(
        "pol.study.cache.FeatureStateCache.get_or_solve",
        forbidden,
    )
    monkeypatch.setattr("pol.data.dataset.evolve", forbidden)
    result = run_study(spec, repo_root=tmp_path, plots_only=True)

    assert result.reused
    assert result.path == first.path
    assert result.summary == {
        "status": "plots_regenerated",
        "created": ["validation_error_vs_q.png"],
    }
    verify_study_run(result.path)
