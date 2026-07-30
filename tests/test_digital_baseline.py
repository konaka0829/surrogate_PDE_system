from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from pol.config.loader import (
    load_dataset_spec,
    load_digital_baseline_spec,
    load_study_spec,
)
from pol.data.dataset import ensure_dataset
from pol.digital_baselines.datasets import build_test_view
from pol.digital_baselines.evaluation import (
    load_fno_checkpoint,
    predict_coefficients,
    prediction_metrics,
)
from pol.digital_baselines.runner import (
    run_digital_baseline,
    verify_digital_baseline_run,
)
from pol.runtime.artifacts import manifest_records
from pol.runtime.io import (
    atomic_torch_save,
    file_sha256,
    write_csv,
    write_strict_json,
)
from pol.runtime.hashing import stable_object_hash
from pol.study.runner import run_study
from tests.helpers import write_json, write_tiny_stack


ROOT = Path(__file__).resolve().parents[1]


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _replace_manifest_record(root: Path, name: str) -> None:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for index, record in enumerate(manifest["files"]):
        if record["relative_path"] == name:
            manifest["files"][index] = manifest_records(root, [name])[0]
            write_strict_json(manifest_path, manifest)
            return
    raise AssertionError(f"manifest has no record for {name}")


def _digital_spec(root: Path, dataset_path: Path, study_path: Path) -> Path:
    return write_json(
        root / "digital.json",
        {
            "schema_version": "pol-digital-baseline-v3",
            "name": "tiny_fno1d",
            "profile": "test",
            "output_root": str(root / "digital_outputs"),
            "dataset_spec": str(dataset_path),
            "input": {"n_tar": 16, "resampling": "spectral"},
            "output": {"kind": "real_fourier", "q": 9},
            "model": {
                "kind": "fno1d",
                "activation": "gelu",
                "coordinate_channel": "none",
                "candidates": [
                    {"id": "m2_w2_d1", "modes": 2, "width": 2, "depth": 1},
                    {"id": "m3_w3_d1", "modes": 3, "width": 3, "depth": 1},
                ],
            },
            "normalization": {
                "kind": "train_standard_score",
                "epsilon": 1e-12,
            },
            "training": {
                "optimizer": {
                    "kind": "adam",
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                },
                "epochs": 2,
                "batch_size": 4,
                "selection_seeds": [31, 32],
                "evaluation_seeds": [41, 42],
                "checkpoint_metric": "validation_field_relative_l2_mean",
                "checkpoint_tie_tolerance": 1e-12,
                "candidate_tie_tolerance": 1e-12,
                "candidate_tie_break": "first_in_config_order",
            },
            "physical_comparison": {
                "source_study_spec": str(study_path),
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
                        "label": "Random",
                        "variant_id": "heat",
                        "readout_id": "random",
                    },
                ],
            },
            "reporting": {
                "primary_result": "independent_training_seed_metric_summary",
                "confidence_level": 0.95,
                "confidence_interval_method": "student_t",
                "prediction_ensemble": "separate_table",
                "wall_clock_energy_comparison": (
                    "only_same_measurement_protocol"
                ),
            },
            "execution": {"device": "cpu", "torch_threads": 1},
        },
    )


@pytest.mark.parametrize("name", ["fno1d_smoke.json", "fno1d.json"])
def test_checked_in_fno_contract_is_strict_and_cpu_only(name: str) -> None:
    path = ROOT / "digital_baselines" / name
    spec = load_digital_baseline_spec(path, repo_root=ROOT)
    assert spec.model.kind == "fno1d"
    assert spec.execution.device == "cpu"
    assert spec.output.q <= spec.input.n_tar
    assert len(spec.training.evaluation_seeds) >= 2
    assert set(spec.training.selection_seeds).isdisjoint(
        spec.training.evaluation_seeds
    )

    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["unknown_scientific_key"] = True
    with pytest.raises(
        ValueError,
        match="Extra inputs are not permitted",
    ):
        from pol.digital_baselines.protocol import DigitalBaselineSpec

        DigitalBaselineSpec.model_validate(raw)

    raw.pop("unknown_scientific_key")
    raw["execution"]["device"] = "cuda"
    with pytest.raises(ValueError, match="CPU-only"):
        from pol.digital_baselines.protocol import DigitalBaselineSpec

        DigitalBaselineSpec.model_validate(raw)


def test_missing_physical_source_fails_before_dataset_training_or_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, dataset_path, study_path = write_tiny_stack(
        tmp_path,
        include_diagnostics=False,
    )
    digital_path = _digital_spec(tmp_path, dataset_path, study_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("preflight failure started numerical/test work")

    monkeypatch.setattr(
        "pol.digital_baselines.runner.ensure_dataset",
        forbidden,
    )
    monkeypatch.setattr(
        "pol.digital_baselines.runner.train_one_seed",
        forbidden,
    )
    monkeypatch.setattr(
        "pol.digital_baselines.runner.build_test_view",
        forbidden,
    )
    with pytest.raises(ValueError, match="missing"):
        run_digital_baseline(
            load_digital_baseline_spec(digital_path, repo_root=tmp_path),
            repo_root=tmp_path,
        )


def test_fno_lifecycle_freezes_before_test_and_reports_seed_statistics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, dataset_path, study_path = write_tiny_stack(
        tmp_path,
        include_diagnostics=False,
    )
    run_study(
        load_study_spec(study_path, repo_root=tmp_path),
        repo_root=tmp_path,
    )
    digital_path = _digital_spec(tmp_path, dataset_path, study_path)
    spec = load_digital_baseline_spec(digital_path, repo_root=tmp_path)

    import pol.digital_baselines.runner as runner_module

    original_test_view = runner_module.build_test_view
    original_physical_reader = runner_module._read_physical_test_rows
    observed_boundary: dict[str, bool] = {}

    def guarded_test_view(dataset, *, n_tar, q):
        run_dirs = list(
            (tmp_path / "digital_outputs" / "tiny_fno1d").glob(
                ".test-*.staging-*"
            )
        )
        assert len(run_dirs) == 1
        staging = run_dirs[0]
        assert (staging / "selection_record.json").is_file()
        assert (staging / "frozen_checkpoints.pt").is_file()
        assert (staging / "frozen_evaluation_plan.json").is_file()
        runner_module._read_frozen_boundary(
            staging,
            spec,
            domain_length=dataset.domain_length,
        )
        observed_boundary["verified"] = True
        return original_test_view(dataset, n_tar=n_tar, q=q)

    def guarded_physical_reader(path):
        if "physical_reader_verified" not in observed_boundary:
            run_dirs = list(
                (tmp_path / "digital_outputs" / "tiny_fno1d").glob(
                    ".test-*.staging-*"
                )
            )
            assert len(run_dirs) == 1
            dataset_reference = json.loads(
                (run_dirs[0] / "dataset_reference.json").read_text(
                    encoding="utf-8"
                )
            )
            runner_module._read_frozen_boundary(
                run_dirs[0],
                spec,
                domain_length=float(dataset_reference["domain_length"]),
            )
            observed_boundary["physical_reader_verified"] = True
        return original_physical_reader(path)

    monkeypatch.setattr(
        "pol.digital_baselines.runner.build_test_view",
        guarded_test_view,
    )
    monkeypatch.setattr(
        "pol.digital_baselines.runner._read_physical_test_rows",
        guarded_physical_reader,
    )
    first = run_digital_baseline(spec, repo_root=tmp_path)
    assert observed_boundary == {
        "verified": True,
        "physical_reader_verified": True,
    }
    verify_digital_baseline_run(first.path)

    primary = _rows(first.path / "test_metrics.csv")
    seeds = _rows(first.path / "test_seed_metrics.csv")
    ensemble = _rows(first.path / "prediction_ensemble_metrics.csv")
    fairness = _rows(first.path / "fairness_comparison.csv")
    assert len(primary) == 1
    assert len(seeds) == 2
    assert primary[0]["test_seed_count"] == "2"
    assert primary[0]["test_seed_std_ddof"] == "1"
    assert primary[0]["test_confidence_interval_method"] == "student_t"
    assert primary[0]["prediction_ensemble_in_primary"] == "False"
    assert ensemble[0]["test_result_kind"] == "prediction_ensemble"
    assert len(fairness) == 4
    assert {row["dataset_split_hash"] for row in fairness} == {
        fairness[0]["dataset_split_hash"]
    }
    assert {
        row["inference_path"] for row in fairness
        if row["row_id"] == "fno1d"
    } != {
        row["inference_path"] for row in fairness
        if row["row_id"] != "fno1d"
    }
    assert all(row["energy_comparison_allowed"] == "False" for row in fairness)
    fairness_by_id = {row["row_id"]: row for row in fairness}
    assert fairness_by_id["direct"]["total_stored_parameter_count"] == "0"
    affine_J = int(
        fairness_by_id["affine"]["feature_dimension_before_readout"]
    )
    affine_q = int(fairness_by_id["affine"]["output_dimension"])
    affine_count = affine_q * (affine_J + 1)
    assert fairness_by_id["affine"]["trainable_parameter_count"] == str(
        affine_count
    )
    assert fairness_by_id["affine"]["total_stored_parameter_count"] == str(
        affine_count
    )
    random_J = int(
        fairness_by_id["random"]["feature_dimension_before_readout"]
    )
    random_after = int(
        fairness_by_id["random"]["feature_dimension_after_lift"]
    )
    random_M = random_after - random_J
    random_q = int(fairness_by_id["random"]["output_dimension"])
    random_fixed = random_M * (random_J + 1)
    random_trainable = random_q * (random_J + random_M + 1)
    random_total = random_fixed + random_trainable
    assert fairness_by_id["random"]["fixed_random_parameter_count"] == str(
        random_fixed
    )
    assert fairness_by_id["random"]["trainable_parameter_count"] == str(
        random_trainable
    )
    assert fairness_by_id["random"]["total_stored_parameter_count"] == str(
        random_total
    )
    assert fairness_by_id["random"][
        "all_frozen_realizations_total_stored_parameter_count"
    ] == str(2 * random_total)
    assert fairness_by_id["random"][
        "primary_count_seed_multiplier_applied"
    ] == "False"
    assert fairness_by_id["fno1d"]["fixed_random_parameter_count"] == "0"
    assert fairness_by_id["fno1d"]["trainable_parameter_count"] == (
        fairness_by_id["fno1d"]["total_stored_parameter_count"]
    )
    assert {
        row["parameter_count_scope"] for row in fairness
    } == {"per_independent_model_realization"}
    summary = json.loads(
        (first.path / "run_summary.json").read_text(encoding="utf-8")
    )
    assert summary["parameter_count_scope"] == (
        "per_independent_model_realization"
    )
    assert summary["primary_model_parameter_counts"][
        "total_stored_parameter_count"
    ] == int(fairness_by_id["fno1d"]["total_stored_parameter_count"])
    assert summary["model_family_parameter_count_policy"][
        "physical_dynamics_included"
    ] is False

    events = json.loads(
        (first.path / "events.json").read_text(encoding="utf-8")
    )
    names = [event["event"] for event in events]
    assert names == [
        "physical_source_preflight_verified",
        "selection_complete",
        "freeze_written",
        "freeze_read_back",
        "first_test_tensor_request",
        "physical_source_test_rows_parsed",
        "first_test_metric",
        "fairness_comparison_computed",
        "physical_source_full_verified",
        "fairness_comparison_written",
        "numerical_run_complete",
    ]
    source_reference = json.loads(
        (first.path / "physical_source_reference.json").read_text(
            encoding="utf-8"
        )
    )
    selection = json.loads(
        (first.path / "selection_record.json").read_text(encoding="utf-8")
    )
    plan = json.loads(
        (first.path / "frozen_evaluation_plan.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = json.loads(
        (first.path / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["identity"]["digital_baseline"]["model"][
        "coordinate_channel"
    ] == "none"
    assert selection["coordinate_channel"] == "none"
    assert selection["lifting_input_channels"] == 1
    assert all(
        item["coordinate_channel"] == "none"
        and item["lifting_input_channels"] == 1
        for item in selection["candidate_summaries"]
    )
    assert plan["coordinate_channel"] == "none"
    assert plan["lifting_input_channels"] == 1
    source_preflight = source_reference["preflight"]
    source_postfreeze = source_reference["postfreeze"]
    assert source_preflight["test_metrics_parsed"] is False
    assert source_postfreeze["source_manifest_unchanged"] is True
    for event in events:
        assert event["source_manifest_sha256"] == source_preflight[
            "preflight_manifest_sha256"
        ]
        assert event["source_selection_record_hash"] == source_preflight[
            "selection_record_hash"
        ]
        assert event["source_frozen_plan_hash"] == source_preflight[
            "frozen_plan_hash"
        ]
    for event in events[1:]:
        assert event["selection_record_hash"] == stable_object_hash(selection)
    for event in events[2:]:
        assert event["plan_content_hash"] == plan["plan_content_hash"]
    for event in events[5:]:
        assert event["source_test_metrics_sha256"] == source_postfreeze[
            "test_metrics_sha256"
        ]

    dataset = ensure_dataset(
        load_dataset_spec(dataset_path, repo_root=tmp_path),
        repo_root=tmp_path,
    )
    test_view = build_test_view(dataset, n_tar=16, q=9)
    archive = torch.load(
        first.path / "frozen_checkpoints.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert archive["coordinate_channel"] == "none"
    assert archive["lifting_input_channels"] == 1
    assert all(
        tuple(model["state_dict"]["lifting.weight"].shape)[1] == 1
        and model["coordinate_channel"] == "none"
        for model in archive["models"]
    )
    candidate = next(
        candidate
        for candidate in spec.model.candidates
        if candidate.id == archive["selected_candidate_id"]
    )
    frozen = archive["models"][0]
    model = load_fno_checkpoint(
        candidate,
        n_tar=16,
        dtype=test_view.inputs.dtype,
        state_dict=frozen["state_dict"],
        expected_hash=frozen["state_dict_hash"],
        coordinate_channel=spec.model.coordinate_channel,
        domain_length=dataset.domain_length,
    )
    prediction = predict_coefficients(
        model,
        test_view.inputs,
        archive["normalization"],
        q=9,
        domain_length=dataset.domain_length,
        batch_size=4,
    )
    shared_metrics = prediction_metrics(
        prediction,
        test_view,
        domain_length=dataset.domain_length,
    )
    assert float(seeds[0]["test_field_relative_l2_mean"]) == pytest.approx(
        shared_metrics["field_relative_l2_mean"],
        rel=0,
        abs=0,
    )

    original_primary_bytes = (first.path / "test_metrics.csv").read_bytes()
    archive_path = first.path / "frozen_checkpoints.pt"
    plan_path = first.path / "frozen_evaluation_plan.json"
    manifest_path = first.path / "manifest.json"
    original_archive_bytes = archive_path.read_bytes()
    original_plan_bytes = plan_path.read_bytes()
    original_manifest_bytes = manifest_path.read_bytes()
    policy_tampered = torch.load(
        archive_path,
        map_location="cpu",
        weights_only=True,
    )
    policy_tampered["coordinate_channel"] = "periodic_sin_cos"
    atomic_torch_save(archive_path, policy_tampered)
    policy_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    policy_plan["frozen_checkpoints_sha256"] = file_sha256(archive_path)
    unsigned_policy_plan = dict(policy_plan)
    unsigned_policy_plan.pop("plan_content_hash")
    policy_plan["plan_content_hash"] = stable_object_hash(
        unsigned_policy_plan
    )
    write_strict_json(plan_path, policy_plan)
    _replace_manifest_record(first.path, "frozen_checkpoints.pt")
    _replace_manifest_record(first.path, "frozen_evaluation_plan.json")

    original_csv_reader = runner_module._read_csv

    def no_digital_test_table_before_policy_rejection(path):
        if path.parent == first.path and path.suffix == ".csv":
            raise AssertionError(
                "coordinate-policy mismatch reached digital test tables"
            )
        return original_csv_reader(path)

    with monkeypatch.context() as policy_patch:
        policy_patch.setattr(
            "pol.digital_baselines.runner._read_csv",
            no_digital_test_table_before_policy_rejection,
        )
        with pytest.raises(
            ValueError,
            match="checkpoint coordinate architecture mismatch",
        ):
            verify_digital_baseline_run(first.path)
    archive_path.write_bytes(original_archive_bytes)
    plan_path.write_bytes(original_plan_bytes)
    manifest_path.write_bytes(original_manifest_bytes)
    verify_digital_baseline_run(first.path)

    fairness[2]["total_stored_parameter_count"] = str(
        int(fairness[2]["total_stored_parameter_count"]) + 1
    )
    write_csv(
        first.path / "fairness_comparison.csv",
        fairness,
        fieldnames=list(fairness[0]),
    )
    _replace_manifest_record(first.path, "fairness_comparison.csv")
    with pytest.raises(ValueError, match="fairness table disagrees"):
        verify_digital_baseline_run(first.path)

    second = run_digital_baseline(spec, repo_root=tmp_path, force=True)
    assert second.reused is False
    assert (second.path / "test_metrics.csv").read_bytes() == original_primary_bytes

    second_events = json.loads(
        (second.path / "events.json").read_text(encoding="utf-8")
    )
    second_events[4]["source_manifest_sha256"] = "0" * 64
    write_strict_json(second.path / "events.json", second_events)
    _replace_manifest_record(second.path, "events.json")
    with pytest.raises(ValueError, match="physical-source preflight binding"):
        verify_digital_baseline_run(second.path)

    third = run_digital_baseline(spec, repo_root=tmp_path, force=True)
    assert third.reused is False
    assert (third.path / "test_metrics.csv").read_bytes() == original_primary_bytes

    archive_path = third.path / "frozen_checkpoints.pt"
    tampered = torch.load(
        archive_path,
        map_location="cpu",
        weights_only=True,
    )
    state = tampered["models"][0]["state_dict"]
    first_name = next(iter(state))
    state[first_name] = state[first_name].clone()
    state[first_name].view(-1)[0] += 1.0
    atomic_torch_save(archive_path, tampered)
    plan_path = third.path / "frozen_evaluation_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["frozen_checkpoints_sha256"] = file_sha256(archive_path)
    unsigned = dict(plan)
    unsigned.pop("plan_content_hash")
    plan["plan_content_hash"] = stable_object_hash(unsigned)
    write_strict_json(plan_path, plan)
    manifest_path = third.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in ("frozen_checkpoints.pt", "frozen_evaluation_plan.json"):
        for index, record in enumerate(manifest["files"]):
            if record["relative_path"] == name:
                manifest["files"][index] = manifest_records(
                    third.path,
                    [name],
                )[0]
                break
    write_strict_json(manifest_path, manifest)
    with pytest.raises(
        ValueError,
        match="checkpoint content hash mismatch",
    ):
        verify_digital_baseline_run(third.path)


def test_physical_coordinates_are_preflighted_without_test_table_parsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, dataset_path, study_path = write_tiny_stack(
        tmp_path,
        include_diagnostics=False,
    )
    run_study(
        load_study_spec(study_path, repo_root=tmp_path),
        repo_root=tmp_path,
    )
    spec = load_digital_baseline_spec(
        _digital_spec(tmp_path, dataset_path, study_path),
        repo_root=tmp_path,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("preflight parsed physical test values")

    monkeypatch.setattr(
        "pol.digital_baselines.runner._read_physical_test_rows",
        forbidden,
    )
    import pol.digital_baselines.runner as runner_module

    physical = runner_module._physical_source_preflight(
        spec,
        repo_root=tmp_path,
    )
    assert [
        (row["variant_id"], row["readout_id"])
        for row in physical.coordinates
    ] == [
        ("heat", "direct"),
        ("heat", "affine"),
        ("heat", "random"),
    ]
    assert all(row["case_id"] for row in physical.coordinates)
    assert all(row["candidate_id"] for row in physical.coordinates)


def test_physical_test_value_tamper_cannot_affect_frozen_digital_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, dataset_path, study_path = write_tiny_stack(
        tmp_path,
        include_diagnostics=False,
    )
    run_study(
        load_study_spec(study_path, repo_root=tmp_path),
        repo_root=tmp_path,
    )
    spec = load_digital_baseline_spec(
        _digital_spec(tmp_path, dataset_path, study_path),
        repo_root=tmp_path,
    )
    import pol.digital_baselines.runner as runner_module

    original_reader = runner_module._read_physical_test_rows
    original_full_verify = runner_module.resolve_verified_completed_run
    frozen_hashes: list[tuple[str, str]] = []

    def staging_hashes() -> tuple[str, str]:
        run_dirs = list(
            (tmp_path / "digital_outputs" / "tiny_fno1d").glob(
                ".test-*.staging-*"
            )
        )
        assert len(run_dirs) == 1
        staging = run_dirs[0]
        selection = json.loads(
            (staging / "selection_record.json").read_text(encoding="utf-8")
        )
        return (
            stable_object_hash(selection),
            file_sha256(staging / "frozen_checkpoints.pt"),
        )

    def extreme_value_reader(path):
        frozen_hashes.append(staging_hashes())
        rows = _rows(path)
        rows[0]["test_field_relative_l2_mean"] = "1e200"
        write_csv(path, rows, fieldnames=list(rows[0]))
        return original_reader(path)

    def guarded_full_verify(*args, **kwargs):
        frozen_hashes.append(staging_hashes())
        return original_full_verify(*args, **kwargs)

    monkeypatch.setattr(
        "pol.digital_baselines.runner._read_physical_test_rows",
        extreme_value_reader,
    )
    monkeypatch.setattr(
        "pol.digital_baselines.runner.resolve_verified_completed_run",
        guarded_full_verify,
    )
    with pytest.raises(ValueError, match="bytes differ from manifest"):
        run_digital_baseline(spec, repo_root=tmp_path)
    assert len(frozen_hashes) == 2
    assert frozen_hashes[0] == frozen_hashes[1]
    output = tmp_path / "digital_outputs" / "tiny_fno1d"
    assert not any(path.is_dir() for path in output.glob("test-*"))
    assert not any(output.glob(".test-*.staging-*"))


def test_physical_manifest_race_rolls_back_before_test_parse_or_fairness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, dataset_path, study_path = write_tiny_stack(
        tmp_path,
        include_diagnostics=False,
    )
    physical_run = run_study(
        load_study_spec(study_path, repo_root=tmp_path),
        repo_root=tmp_path,
    )
    spec = load_digital_baseline_spec(
        _digital_spec(tmp_path, dataset_path, study_path),
        repo_root=tmp_path,
    )
    import pol.digital_baselines.runner as runner_module

    original_test_view = runner_module.build_test_view
    parsed = {"physical": False}

    def race_test_view(*args, **kwargs):
        view = original_test_view(*args, **kwargs)
        manifest_path = physical_run.path / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["race_marker"] = "changed_after_digital_freeze"
        write_strict_json(manifest_path, manifest)
        return view

    def forbidden_reader(*args, **kwargs):
        parsed["physical"] = True
        raise AssertionError("manifest race reached physical test parser")

    monkeypatch.setattr(
        "pol.digital_baselines.runner.build_test_view",
        race_test_view,
    )
    monkeypatch.setattr(
        "pol.digital_baselines.runner._read_physical_test_rows",
        forbidden_reader,
    )
    with pytest.raises(ValueError, match="manifest changed after digital preflight"):
        run_digital_baseline(spec, repo_root=tmp_path)
    assert parsed["physical"] is False
    output = tmp_path / "digital_outputs" / "tiny_fno1d"
    assert not any(path.is_dir() for path in output.glob("test-*"))
    assert not any(output.glob(".test-*.staging-*"))
    assert not any(output.rglob("fairness_comparison.csv"))


def test_physical_study_package_does_not_import_digital_adapter() -> None:
    study_root = ROOT / "pol" / "study"
    assert all(
        "digital_baselines" not in path.read_text(encoding="utf-8")
        for path in study_root.glob("*.py")
    )
