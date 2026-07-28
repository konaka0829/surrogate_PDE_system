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
from pol.config.models import DomainSpec
from pol.data.dataset import dataset_reference, ensure_dataset, load_dataset
from pol.runtime.artifacts import manifest_records
from pol.runtime.hashing import stable_object_hash
from pol.runtime.io import atomic_torch_save, write_strict_json
from pol.study.runner import run_study, verify_study_run
from pol.validation.binding import (
    DatasetBindingError,
    evaluate_dataset_binding,
    verify_binding_proof,
)
from pol.validation.runner import ensure_validation, load_validation_certificate
from tests.helpers import write_json, write_tiny_stack


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _refresh_artifact_record(root: Path, relative_path: str) -> None:
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path)
    for index, record in enumerate(manifest["files"]):
        if record["relative_path"] == relative_path:
            manifest["files"][index] = manifest_records(
                root, [relative_path]
            )[0]
            break
    else:
        raise AssertionError(f"manifest has no record for {relative_path}")
    write_strict_json(manifest_path, manifest)


def _validated_burgers_dataset_path(root: Path) -> tuple[Path, Path]:
    validation_path, dataset_path, _ = write_tiny_stack(root)
    validation_raw = _read_json(validation_path)
    validation_raw["reference_nx_candidates"] = [32, 64]
    write_json(validation_path, validation_raw)
    raw = _read_json(dataset_path)
    raw.update(
        {
            "name": "tiny_burgers_dataset",
            "binding": {"kind": "validated_reference"},
            "reference_nx": 64,
            "target": {
                "system": {
                    "kind": "burgers",
                    "nu": 0.05,
                    "advection_coefficient": 1.0,
                    "solver": "split_step",
                    "dt": 0.01,
                    "fine_dt": 0.0025,
                    "dealias": True,
                },
                "time": 0.02,
            },
        }
    )
    write_json(dataset_path, raw)
    return validation_path, dataset_path


def _binding_inputs(
    root: Path,
) -> tuple[Any, Any, Any]:
    validation_path, dataset_path = _validated_burgers_dataset_path(root)
    validation_spec = load_validation_spec(validation_path, repo_root=root)
    certificate = ensure_validation(validation_spec).certificate
    dataset_spec = load_dataset_spec(dataset_path, repo_root=root)
    return certificate, validation_spec, dataset_spec


def _dataset_with_change(root: Path, dotted: str, value: Any):
    certificate, validation_spec, dataset_spec = _binding_inputs(root)
    raw = dataset_spec.model_dump(mode="json")
    raw["validation_spec"] = str(dataset_spec.validation_spec)
    current: Any = raw
    parts = dotted.split(".")
    for part in parts[:-1]:
        current = current[part]
    current[parts[-1]] = value
    path = write_json(root / "changed_dataset.json", raw)
    changed = load_dataset_spec(path, repo_root=root)
    return certificate, validation_spec, changed


def test_certificate_records_self_consistent_selected_suffix_contract(
    tmp_path: Path,
) -> None:
    certificate, _, _ = _binding_inputs(tmp_path)
    target = certificate["target_reference_contract"]
    reference = target["reference_resolution"]
    time = target["time_discretization"]
    relation = target["allowed_refinement_relation"]

    assert reference["selected_candidate_index"] == 0
    assert reference["selected_value"] == reference["candidates"][0] == 32
    assert relation["reference_nx_allowed_indices"] == [0, 1]
    assert relation["reference_nx_allowed_values"] == [32, 64]
    assert time["selected_candidate_index"] == 0
    assert time["selected_candidate"] == time["candidates"][0]
    assert relation["time_candidate_allowed_indices"] == [0, 1]
    assert relation["time_candidate_allowed_values"] == time["candidates"]
    master = certificate["foundation_contract"]["master_initial_conditions"]
    assert set(master["tensor_hashes"]) == {"sample_ids", "values", "fourier"}
    assert len(master["archive_identity_hash"]) == 64
    assert certificate["foundation_contract"]["domain_length"] == (
        certificate["foundation_contract"]["grf_sampler_domain_length"]
        == master["domain_length"]
        == master["metadata"]["domain_length"]
    )


def test_validated_reference_accepts_only_an_actual_finer_suffix_member(
    tmp_path: Path,
) -> None:
    certificate, validation_spec, dataset_spec = _binding_inputs(tmp_path)
    proof = evaluate_dataset_binding(certificate, validation_spec, dataset_spec)
    assert proof["binding_kind"] == "validated_reference"
    assert proof["target_reference_validation_status"] == "validated"
    assert proof["matched_reference_candidate_index"] == 1
    assert proof["matched_time_candidate_index"] == 1
    assert proof["dataset_condition"]["reference_nx"] == 64
    assert proof["proof_hash"] == stable_object_hash(
        {key: value for key, value in proof.items() if key != "proof_hash"}
    )


@pytest.mark.parametrize(
    "reference_nx",
    [48, 16],
)
def test_validated_reference_rejects_non_suffix_reference_resolution(
    tmp_path: Path,
    reference_nx: int,
) -> None:
    certificate, validation_spec, changed = _dataset_with_change(
        tmp_path, "reference_nx", reference_nx
    )
    with pytest.raises(
        DatasetBindingError,
        match=r"field_path=\$\.dataset\.reference_nx.*binding_kind=validated_reference",
    ):
        evaluate_dataset_binding(certificate, validation_spec, changed)


@pytest.mark.parametrize(
    ("path", "value", "error_path"),
    [
        ("target.system.nu", 0.04, "nu"),
        ("target.time", 0.03, "time"),
        ("target.system.solver", "semi_implicit", "solver"),
        ("target.system.dealias", False, "dealias"),
        ("target.system.dt", 0.02, "dt"),
        ("target.system.fine_dt", 0.00125, "fine_dt"),
    ],
)
def test_validated_reference_rejects_target_condition_mismatch(
    tmp_path: Path,
    path: str,
    value: Any,
    error_path: str,
) -> None:
    certificate, validation_spec, changed = _dataset_with_change(
        tmp_path, path, value
    )
    with pytest.raises(
        DatasetBindingError,
        match=rf"field_path=\$\.dataset\.target\..*{error_path}",
    ):
        evaluate_dataset_binding(certificate, validation_spec, changed)


@pytest.mark.parametrize("field", ["dtype", "domain_length"])
def test_validated_reference_rejects_dtype_or_domain_mismatch(
    tmp_path: Path,
    field: str,
) -> None:
    certificate, validation_spec, dataset_spec = _binding_inputs(tmp_path)
    if field == "dtype":
        changed_validation = validation_spec.model_copy(
            update={
                "samples": validation_spec.samples.model_copy(
                    update={"dtype": "float32"}
                )
            }
        )
    else:
        changed_validation = validation_spec.model_copy(
            update={"domain": DomainSpec(length=2.0)}
        )
    with pytest.raises(DatasetBindingError, match=field):
        evaluate_dataset_binding(
            certificate, changed_validation, dataset_spec
        )


def test_validated_reference_rejects_heat_target(tmp_path: Path) -> None:
    validation_path, dataset_path, _ = write_tiny_stack(tmp_path)
    raw = _read_json(dataset_path)
    raw["binding"] = {"kind": "validated_reference"}
    write_json(dataset_path, raw)
    validation_spec = load_validation_spec(validation_path, repo_root=tmp_path)
    certificate = ensure_validation(validation_spec).certificate
    dataset_spec = load_dataset_spec(dataset_path, repo_root=tmp_path)
    with pytest.raises(
        DatasetBindingError,
        match=r"field_path=\$\.dataset\.target\.system\.kind",
    ):
        evaluate_dataset_binding(certificate, validation_spec, dataset_spec)


def test_foundation_only_heat_status_is_persisted_and_loaded(
    tmp_path: Path,
) -> None:
    _, dataset_path, _ = write_tiny_stack(tmp_path)
    dataset_spec = load_dataset_spec(dataset_path, repo_root=tmp_path)
    dataset = ensure_dataset(dataset_spec, repo_root=tmp_path)
    loaded = load_dataset(dataset.path)

    assert loaded.binding_kind == "foundation_only"
    assert loaded.binding_status == "pass"
    assert loaded.target_reference_validation_status == "not_claimed"
    assert loaded.binding_proof["reason"]
    assert loaded.binding_proof_hash == loaded.binding_proof["proof_hash"]
    metadata = _read_json(dataset.path / "metadata.json")
    resolved = _read_json(dataset.path / "resolved_spec.json")
    archive = torch.load(
        dataset.path / "dataset.pt", map_location="cpu", weights_only=True
    )
    for copy_payload in (metadata, resolved, archive):
        assert copy_payload["binding_kind"] == "foundation_only"
        assert copy_payload["target_reference_validation_status"] == "not_claimed"
        assert copy_payload["binding_proof_hash"] == loaded.binding_proof_hash
        assert copy_payload["binding_proof"] == loaded.binding_proof


def test_binding_failure_precedes_target_evolution_and_publishes_no_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, dataset_path = _validated_burgers_dataset_path(tmp_path)
    raw = _read_json(dataset_path)
    raw["reference_nx"] = 48
    write_json(dataset_path, raw)
    dataset_spec = load_dataset_spec(dataset_path, repo_root=tmp_path)
    called = False

    def forbidden_evolve(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("target evolve must not be called")

    monkeypatch.setattr("pol.data.dataset.evolve", forbidden_evolve)
    with pytest.raises(DatasetBindingError, match="reference_nx"):
        ensure_dataset(dataset_spec, repo_root=tmp_path)
    assert not called
    assert not (tmp_path / "artifacts" / "datasets").exists()


@pytest.mark.parametrize(
    ("copy_name", "field", "value"),
    [
        ("metadata.json", "binding_proof_hash", "0" * 64),
        ("resolved_spec.json", "target_reference_validation_status", "validated"),
        ("dataset.pt", "binding_kind", "validated_reference"),
    ],
)
def test_load_dataset_rejects_binding_copy_tamper(
    tmp_path: Path,
    copy_name: str,
    field: str,
    value: str,
) -> None:
    _, dataset_path, _ = write_tiny_stack(tmp_path)
    dataset = ensure_dataset(
        load_dataset_spec(dataset_path, repo_root=tmp_path),
        repo_root=tmp_path,
    )
    target = dataset.path / copy_name
    if target.suffix == ".pt":
        payload = torch.load(target, map_location="cpu", weights_only=True)
        payload[field] = value
        atomic_torch_save(target, payload)
    else:
        payload = _read_json(target)
        payload[field] = value
        write_strict_json(target, payload)
    _refresh_artifact_record(dataset.path, copy_name)
    with pytest.raises(ValueError, match="binding mismatch"):
        load_dataset(dataset.path)


def test_load_dataset_rejects_legacy_archive_revision(tmp_path: Path) -> None:
    _, dataset_path, _ = write_tiny_stack(tmp_path)
    dataset = ensure_dataset(
        load_dataset_spec(dataset_path, repo_root=tmp_path),
        repo_root=tmp_path,
    )
    archive_path = dataset.path / "dataset.pt"
    archive = torch.load(archive_path, map_location="cpu", weights_only=True)
    archive["schema_version"] = "pol-reference-dataset-v2"
    atomic_torch_save(archive_path, archive)
    _refresh_artifact_record(dataset.path, "dataset.pt")
    with pytest.raises(ValueError, match="unsupported dataset archive schema"):
        load_dataset(dataset.path)


def test_study_verifier_rejects_dataset_validation_status_tamper(
    tmp_path: Path,
) -> None:
    _, _, study_path = write_tiny_stack(tmp_path)
    result = run_study(
        load_study_spec(study_path, repo_root=tmp_path),
        repo_root=tmp_path,
    )
    reference_path = result.path / "dataset_reference.json"
    reference = _read_json(reference_path)
    reference["dataset_target_reference_validation_status"] = "validated"
    write_strict_json(reference_path, reference)
    _refresh_artifact_record(result.path, "dataset_reference.json")
    with pytest.raises(ValueError, match="dataset validation binding mismatch"):
        verify_study_run(result.path)
def test_binding_proof_changes_dataset_artifact_identity(tmp_path: Path) -> None:
    certificate, validation_spec, dataset_spec = _binding_inputs(tmp_path)
    proof = evaluate_dataset_binding(certificate, validation_spec, dataset_spec)
    changed = copy.deepcopy(proof)
    changed["per_field_checks"][0]["comparison"] = "tampered"
    unsigned = {key: value for key, value in changed.items() if key != "proof_hash"}
    changed["proof_hash"] = stable_object_hash(unsigned)
    first = dataset_reference(
        dataset_spec,
        validation_artifact_id=certificate["artifact_id"],
        binding_proof=proof,
    )
    second = dataset_reference(
        dataset_spec,
        validation_artifact_id=certificate["artifact_id"],
        binding_proof=changed,
    )
    assert first.artifact_id != second.artifact_id


@pytest.mark.parametrize("binding_kind", ["validated_reference", "foundation_only"])
def test_binding_proof_rejects_grf_sampler_domain_mismatch(
    tmp_path: Path,
    binding_kind: str,
) -> None:
    if binding_kind == "validated_reference":
        certificate, validation_spec, dataset_spec = _binding_inputs(tmp_path)
    else:
        validation_path, dataset_path, _ = write_tiny_stack(tmp_path)
        validation_spec = load_validation_spec(
            validation_path, repo_root=tmp_path
        )
        certificate = ensure_validation(validation_spec).certificate
        dataset_spec = load_dataset_spec(dataset_path, repo_root=tmp_path)
    proof = evaluate_dataset_binding(certificate, validation_spec, dataset_spec)
    changed = copy.deepcopy(proof)
    changed["grf_sampler_domain_length"] = 2.0
    unsigned = {
        key: value for key, value in changed.items() if key != "proof_hash"
    }
    changed["proof_hash"] = stable_object_hash(unsigned)
    with pytest.raises(ValueError, match="GRF sampler domain mismatch"):
        verify_binding_proof(changed)


def test_certificate_loader_rejects_allowed_suffix_tamper(tmp_path: Path) -> None:
    validation_path, _, _ = write_tiny_stack(tmp_path)
    outcome = ensure_validation(
        load_validation_spec(validation_path, repo_root=tmp_path)
    )
    certificate_path = outcome.reference.path / "certificate.json"
    certificate = _read_json(certificate_path)
    certificate["target_reference_contract"]["allowed_refinement_relation"][
        "reference_nx_allowed_values"
    ] = [32, 48, 64]
    write_strict_json(certificate_path, certificate)
    _refresh_artifact_record(outcome.reference.path, "certificate.json")
    with pytest.raises(ValueError, match="certificate contract"):
        load_validation_certificate(outcome.reference.path)


def test_certificate_loader_rejects_master_sampler_domain_tamper(
    tmp_path: Path,
) -> None:
    validation_path, _, _ = write_tiny_stack(tmp_path)
    outcome = ensure_validation(
        load_validation_spec(validation_path, repo_root=tmp_path)
    )
    master_path = outcome.reference.path / "master_initial_conditions.pt"
    master = torch.load(master_path, map_location="cpu", weights_only=True)
    master["metadata"]["domain_length"] = 2.0
    atomic_torch_save(master_path, master)
    _refresh_artifact_record(
        outcome.reference.path, "master_initial_conditions.pt"
    )
    with pytest.raises(ValueError, match="sampler domain mismatch"):
        load_validation_certificate(outcome.reference.path)


def test_certificate_loader_rejects_p0_02_certificate_revision(
    tmp_path: Path,
) -> None:
    validation_path, _, _ = write_tiny_stack(tmp_path)
    outcome = ensure_validation(
        load_validation_spec(validation_path, repo_root=tmp_path)
    )
    certificate_path = outcome.reference.path / "certificate.json"
    certificate = _read_json(certificate_path)
    certificate["schema_version"] = "pol-validation-certificate-v2"
    write_strict_json(certificate_path, certificate)
    _refresh_artifact_record(outcome.reference.path, "certificate.json")
    with pytest.raises(ValueError, match="P0-03 requires"):
        load_validation_certificate(outcome.reference.path)
