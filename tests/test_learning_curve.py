from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from pol.config.loader import load_dataset_spec, load_study_spec
from pol.config.models import StudySpec, TrialSpec
from pol.data.dataset import ensure_dataset
from pol.runtime.artifacts import manifest_records
from pol.runtime.io import write_strict_json
from pol.study.runner import regenerate_plots, run_study, verify_study_run
from pol.study.training_subsets import resolve_training_subset
from tests.helpers import write_json, write_tiny_stack


ROOT = Path(__file__).resolve().parents[1]


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _learning_study(root: Path, *, sizes: list[int] | None = None) -> Path:
    _, _, study_path = write_tiny_stack(
        root,
        include_diagnostics=False,
    )
    raw = json.loads(study_path.read_text(encoding="utf-8"))
    raw["schema_version"] = "pol-study-v5"
    raw["name"] = "tiny_learning_curve"
    raw["base_trial"]["training_subset"] = {
        "kind": "nested_train_prefix",
        "n_train": 2,
        "policy_version": 1,
    }
    raw["learning_curve"] = {
        "kind": "learning_curve",
        "training_axis_path": "training_subset.n_train",
        "subset_policy": "canonical_train_order_prefix_v1",
    }
    raw["global_axes"] = [
        {
            "path": "training_subset.n_train",
            "values": sizes or [2, 4, 8],
        }
    ]
    raw["reporters"] = [
        {
            "kind": "learning_curve",
            "filename": "test_learning_curve",
            "metric": "test_field_relative_l2_mean",
            "split": "test",
            "group_by": ["variant_id", "readout_id"],
            "xscale": "log",
            "yscale": "log",
            "formats": ["png"],
            "dpi": 80,
        }
    ]
    raw["execution"]["generate_plots"] = True
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
        raise AssertionError(relative_path)
    write_strict_json(manifest_path, manifest)


def test_nested_prefix_inclusion_hash_stability_and_disjointness(
    tmp_path: Path,
) -> None:
    _, dataset_path, study_path = write_tiny_stack(
        tmp_path,
        include_diagnostics=False,
    )
    dataset = ensure_dataset(
        load_dataset_spec(dataset_path, repo_root=tmp_path),
        repo_root=tmp_path,
    )
    base = load_study_spec(study_path, repo_root=tmp_path).base_trial
    records = []
    ids_by_size = []
    for size in (2, 4, 8):
        payload = base.model_dump(mode="python")
        payload["training_subset"] = {
            "kind": "nested_train_prefix",
            "n_train": size,
            "policy_version": 1,
        }
        trial = TrialSpec.model_validate(payload)
        ids, record = resolve_training_subset(dataset, trial)
        repeated_ids, repeated = resolve_training_subset(dataset, trial)
        assert torch.equal(ids, repeated_ids)
        assert record == repeated
        assert not bool(torch.isin(ids, dataset.validation_ids).any())
        assert not bool(torch.isin(ids, dataset.test_ids).any())
        ids_by_size.append(ids.tolist())
        records.append(record)
    assert ids_by_size[1][:2] == ids_by_size[0]
    assert ids_by_size[2][:4] == ids_by_size[1]
    assert len({record["training_subset_hash"] for record in records}) == 3


def test_training_subset_boundary_and_schema_fail_before_study_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _learning_study(tmp_path, sizes=[2, 99])
    raw = json.loads(path.read_text(encoding="utf-8"))
    unknown = json.loads(json.dumps(raw))
    unknown["base_trial"]["training_subset"]["ids"] = [0, 1]
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        StudySpec.model_validate(unknown)

    def forbidden(*args, **kwargs):
        raise AssertionError("feature fitting or test evaluation must not start")

    monkeypatch.setattr(
        "pol.study.trial.TrialEngine.evaluate_selection",
        forbidden,
    )
    monkeypatch.setattr(
        "pol.study.trial.TrialEngine.evaluate_test",
        forbidden,
    )
    with pytest.raises(ValueError, match="canonical train count"):
        run_study(
            load_study_spec(path, repo_root=tmp_path),
            repo_root=tmp_path,
        )


def test_learning_curve_freezes_all_sizes_reuses_cache_and_regenerates_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pol.study.runner as runner_module

    path = _learning_study(tmp_path)
    spec = load_study_spec(path, repo_root=tmp_path)
    original_persist = runner_module.persist_and_read_back_freeze
    original_test = runner_module.TrialEngine.evaluate_test
    frozen = False

    def checked_persist(*args, **kwargs):
        nonlocal frozen
        persisted = original_persist(*args, **kwargs)
        assert len(persisted.archive["models"]) == 9
        frozen = True
        return persisted

    def checked_test(*args, **kwargs):
        assert frozen
        return original_test(*args, **kwargs)

    monkeypatch.setattr(
        runner_module,
        "persist_and_read_back_freeze",
        checked_persist,
    )
    monkeypatch.setattr(
        runner_module.TrialEngine,
        "evaluate_test",
        checked_test,
    )
    result = run_study(spec, repo_root=tmp_path)
    verify_study_run(result.path)
    validation = _rows(result.path / "validation_trials.csv")
    test = _rows(result.path / "test_metrics.csv")
    direct = sorted(
        (row for row in test if row["readout_id"] == "direct"),
        key=lambda row: int(row["n_train"]),
    )
    assert [int(row["n_train"]) for row in direct] == [2, 4, 8]
    assert len({row["test_field_relative_l2_mean"] for row in direct}) == 1
    assert len({row["feature_cache_id"] for row in direct}) == 1
    direct_validation = [
        row for row in validation if row["readout_id"] == "direct"
    ]
    assert len(
        {row["selection_feature_cache_id"] for row in direct_validation}
    ) == 1
    random_primary = [
        row for row in test if row["readout_id"] == "random"
    ]
    assert all(row["test_seed_count"] == "2" for row in random_primary)
    assert all(
        row["test_field_relative_l2_mean_seed_std"] != ""
        for row in random_primary
    )
    assert (result.path / "figures" / "test_learning_curve.png").is_file()
    regenerated = regenerate_plots(spec, result.path)
    assert regenerated == ["test_learning_curve.png"]

    selection = json.loads(
        (result.path / "selection_record.json").read_text(encoding="utf-8")
    )
    subset_ids = [
        case["training_subsets_by_readout"]["affine"]["subset_ids"]
        for case in selection["cases"].values()
    ]
    assert sorted(map(len, subset_ids)) == [2, 4, 8]
    assert "test_ids" not in json.dumps(selection).lower()

    table = result.path / "test_metrics.csv"
    text = table.read_text(encoding="utf-8")
    first_hash = direct[0]["training_subset_hash"]
    table.write_text(
        text.replace(first_hash, "0" * 64, 1),
        encoding="utf-8",
    )
    _refresh_manifest(result.path, table.name)
    with pytest.raises(ValueError, match="training_subset_hash"):
        verify_study_run(result.path)


@pytest.mark.parametrize(
    "name",
    ["learning_curve_smoke.json", "learning_curve.json"],
)
def test_checked_learning_curve_contract(name: str) -> None:
    spec = load_study_spec(ROOT / "studies" / name, repo_root=ROOT)
    assert spec.learning_curve is not None
    assert spec.global_axes[0].path == "training_subset.n_train"
    sizes = [int(value) for value in spec.global_axes[0].values]
    assert sizes == sorted(sizes)
    assert len(spec.variants) == 1
    assert spec.variants[0].id == "burgers"
    assert spec.variants[0].selection_source is not None
