from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from pol.config.loader import (
    load_dataset_spec,
    load_study_spec,
    load_validation_spec,
)
from pol.config.models import SampleSpec
from pol.data.dataset import ensure_dataset
from pol.data.initial_conditions import generate_grf_archive, resolve_device
from pol.runtime.artifacts import manifest_records
from pol.runtime.device import (
    execution_device_policy,
    require_cpu_tensor,
)
from pol.runtime.environment import numerical_environment_fingerprint
from pol.runtime.hashing import stable_object_hash, tensor_sha256
from pol.runtime.io import write_strict_json
from pol.study.runner import run_study, verify_study_run
from pol.validation.binding import verify_binding_proof
from pol.validation.runner import ensure_validation
from tests.helpers import write_json, write_tiny_stack


def _assert_all_tensors_cpu(value: Any) -> None:
    if isinstance(value, torch.Tensor):
        assert value.device.type == "cpu"
    elif isinstance(value, dict):
        for item in value.values():
            _assert_all_tensors_cpu(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_all_tensors_cpu(item)


def _refresh_manifest_record(root: Path, relative_path: str) -> None:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for index, record in enumerate(manifest["files"]):
        if record["relative_path"] == relative_path:
            manifest["files"][index] = manifest_records(
                root, [relative_path]
            )[0]
            break
    else:
        raise AssertionError(f"manifest has no record for {relative_path}")
    write_strict_json(manifest_path, manifest)


def test_sample_spec_accepts_only_cpu() -> None:
    spec = SampleSpec(
        total_samples=3,
        n_train=1,
        n_validation=1,
        n_test=1,
        device="cpu",
    )
    assert spec.device == "cpu"
    assert SampleSpec(
        total_samples=3,
        n_train=1,
        n_validation=1,
        n_test=1,
    ).device == "cpu"


@pytest.mark.parametrize("available", [False, True])
@pytest.mark.parametrize("device", ["cuda", "auto"])
def test_config_load_rejects_cuda_and_auto_regardless_of_availability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    available: bool,
    device: str,
) -> None:
    validation_path, _, _ = write_tiny_stack(tmp_path)
    raw = json.loads(validation_path.read_text(encoding="utf-8"))
    raw["samples"]["device"] = device
    write_json(validation_path, raw)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)

    with pytest.raises(ValueError, match="CPU-only"):
        load_validation_spec(validation_path, repo_root=tmp_path)


def test_config_load_rejects_unknown_device(tmp_path: Path) -> None:
    validation_path, _, _ = write_tiny_stack(tmp_path)
    raw = json.loads(validation_path.read_text(encoding="utf-8"))
    raw["samples"]["device"] = "tpu"
    write_json(validation_path, raw)
    with pytest.raises(ValueError, match="CPU-only"):
        load_validation_spec(validation_path, repo_root=tmp_path)


def test_device_resolution_never_consults_cuda_availability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden() -> bool:
        raise AssertionError("CPU-only device resolution must not query CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    assert resolve_device("cpu") == torch.device("cpu")
    for device in ("cuda", "auto", "unknown"):
        with pytest.raises(ValueError, match="CPU-only"):
            resolve_device(device)


def test_environment_fingerprint_separates_policy_from_cuda_build_metadata() -> None:
    fingerprint = numerical_environment_fingerprint()
    assert fingerprint["schema_version"] == "pol-numerical-environment-v2"
    assert {
        key: fingerprint[key] for key in execution_device_policy()
    } == execution_device_policy()
    assert "torch_cuda_version" in fingerprint


def test_cpu_policy_is_published_and_verified_across_artifacts(
    tmp_path: Path,
) -> None:
    validation_path, dataset_path, study_path = write_tiny_stack(tmp_path)
    validation = ensure_validation(
        load_validation_spec(validation_path, repo_root=tmp_path)
    )
    master = torch.load(
        validation.reference.path / "master_initial_conditions.pt",
        map_location="cpu",
        weights_only=True,
    )
    _assert_all_tensors_cpu(master)
    assert {
        key: master[key] for key in execution_device_policy()
    } == execution_device_policy()
    assert {
        key: validation.certificate[key] for key in execution_device_policy()
    } == execution_device_policy()
    assert {
        key: validation.certificate["foundation_contract"][key]
        for key in execution_device_policy()
    } == execution_device_policy()

    dataset = ensure_dataset(
        load_dataset_spec(dataset_path, repo_root=tmp_path),
        repo_root=tmp_path,
    )
    dataset_archive = torch.load(
        dataset.path / "dataset.pt",
        map_location="cpu",
        weights_only=True,
    )
    _assert_all_tensors_cpu(dataset_archive)
    _assert_all_tensors_cpu(dataset.__dict__)
    assert dataset.execution_device_policy == "cpu_only"
    assert dataset.compute_device == "cpu"
    assert {
        key: dataset.binding_proof[key] for key in execution_device_policy()
    } == execution_device_policy()

    tampered_proof = copy.deepcopy(dataset.binding_proof)
    tampered_proof["compute_device"] = "cuda"
    unsigned = {
        key: value
        for key, value in tampered_proof.items()
        if key != "proof_hash"
    }
    tampered_proof["proof_hash"] = stable_object_hash(unsigned)
    with pytest.raises(ValueError, match="execution-device policy mismatch"):
        verify_binding_proof(tampered_proof)

    result = run_study(
        load_study_spec(study_path, repo_root=tmp_path),
        repo_root=tmp_path,
    )
    manifest = verify_study_run(result.path)
    summary = json.loads(
        (result.path / "run_summary.json").read_text(encoding="utf-8")
    )
    assert {
        key: manifest["identity"][key] for key in execution_device_policy()
    } == execution_device_policy()
    assert {
        key: summary[key] for key in execution_device_policy()
    } == execution_device_policy()

    feature_archives = sorted(
        (tmp_path / "artifacts" / "feature_states").glob("*/state.pt")
    )
    assert feature_archives
    for archive_path in feature_archives:
        state = torch.load(archive_path, map_location="cpu", weights_only=True)
        metadata = json.loads(
            (archive_path.parent / "metadata.json").read_text(encoding="utf-8")
        )
        _assert_all_tensors_cpu(state)
        assert {
            key: state[key] for key in execution_device_policy()
        } == execution_device_policy()
        assert {
            key: metadata[key] for key in execution_device_policy()
        } == execution_device_policy()

    frozen = torch.load(
        result.path / "frozen_models.pt",
        map_location="cpu",
        weights_only=True,
    )
    _assert_all_tensors_cpu(frozen)
    assert {
        key: frozen[key] for key in execution_device_policy()
    } == execution_device_policy()

    summary["compute_device"] = "cuda"
    write_strict_json(result.path / "run_summary.json", summary)
    _refresh_manifest_record(result.path, "run_summary.json")
    with pytest.raises(ValueError, match="execution-device policy mismatch"):
        verify_study_run(result.path)


def test_cpu_workflow_boundary_rejects_non_cpu_tensor_without_copying() -> None:
    unexpected = torch.empty(1, device="meta")
    with pytest.raises(
        RuntimeError,
        match="dataset target-generation batch requires CPU tensor",
    ):
        require_cpu_tensor(
            unexpected,
            boundary="dataset target-generation batch",
            name="inputs",
        )


def test_p0_04_cpu_deterministic_archive_regression() -> None:
    archive = generate_grf_archive(
        total_samples=2,
        nx=8,
        seed=2024,
        gamma=2.25,
        tau=4.0,
        sigma=7.0,
        mean=0.5,
        domain_length=1.0,
        dtype="float64",
        device="cpu",
    )
    assert tensor_sha256(archive.values) == (
        "92357619df6ce687a3064f33d592a8bcd53ceb63a31e1b26f67a693f62ad2edf"
    )
    assert tensor_sha256(archive.fourier) == (
        "cd734ae54fe54785822b9e8f208fce383f88e1203d852cf4287bc6269ec2d9e5"
    )
