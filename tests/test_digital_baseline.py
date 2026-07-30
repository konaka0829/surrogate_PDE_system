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
    write_strict_json,
)
from pol.runtime.hashing import stable_object_hash
from pol.study.runner import run_study
from tests.helpers import write_json, write_tiny_stack


ROOT = Path(__file__).resolve().parents[1]


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _digital_spec(root: Path, dataset_path: Path, study_path: Path) -> Path:
    return write_json(
        root / "digital.json",
        {
            "schema_version": "pol-digital-baseline-v1",
            "name": "tiny_fno1d",
            "profile": "test",
            "output_root": str(root / "digital_outputs"),
            "dataset_spec": str(dataset_path),
            "input": {"n_tar": 16, "resampling": "spectral"},
            "output": {"kind": "real_fourier", "q": 9},
            "model": {
                "kind": "fno1d",
                "activation": "gelu",
                "coordinate_channel": "unit_periodic",
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
        runner_module._read_frozen_boundary(staging, spec)
        observed_boundary["verified"] = True
        return original_test_view(dataset, n_tar=n_tar, q=q)

    monkeypatch.setattr(
        "pol.digital_baselines.runner.build_test_view",
        guarded_test_view,
    )
    first = run_digital_baseline(spec, repo_root=tmp_path)
    assert observed_boundary == {"verified": True}
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

    events = json.loads(
        (first.path / "events.json").read_text(encoding="utf-8")
    )
    names = [event["event"] for event in events]
    assert names.index("selection_complete") < names.index("freeze_written")
    assert names.index("freeze_written") < names.index("freeze_read_back")
    assert names.index("freeze_read_back") < names.index(
        "first_test_tensor_request"
    )
    assert names.index("first_test_tensor_request") < names.index(
        "first_test_metric"
    )

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
    second = run_digital_baseline(spec, repo_root=tmp_path, force=True)
    assert second.reused is False
    assert (second.path / "test_metrics.csv").read_bytes() == original_primary_bytes

    archive_path = second.path / "frozen_checkpoints.pt"
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
    plan_path = second.path / "frozen_evaluation_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["frozen_checkpoints_sha256"] = file_sha256(archive_path)
    unsigned = dict(plan)
    unsigned.pop("plan_content_hash")
    plan["plan_content_hash"] = stable_object_hash(unsigned)
    write_strict_json(plan_path, plan)
    manifest_path = second.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in ("frozen_checkpoints.pt", "frozen_evaluation_plan.json"):
        for index, record in enumerate(manifest["files"]):
            if record["relative_path"] == name:
                manifest["files"][index] = manifest_records(
                    second.path,
                    [name],
                )[0]
                break
    write_strict_json(manifest_path, manifest)
    with pytest.raises(
        ValueError,
        match="checkpoint content hash mismatch",
    ):
        verify_digital_baseline_run(second.path)


def test_physical_study_package_does_not_import_digital_adapter() -> None:
    study_root = ROOT / "pol" / "study"
    assert all(
        "digital_baselines" not in path.read_text(encoding="utf-8")
        for path in study_root.glob("*.py")
    )
