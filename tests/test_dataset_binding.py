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
from tests.helpers import write_json, write_tiny_heat_stack, write_tiny_stack


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
    validation_raw["target_reference"]["reference_nx_candidates"] = [32, 64]
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


def _cross_binding_inputs(
    root: Path,
) -> tuple[Any, Any, Any]:
    validation_path, dataset_path = _validated_burgers_dataset_path(root)
    validation = _read_json(validation_path)
    validation["target_reference"]["cross_solver_validation"] = {
        "schema_version": "pol-burgers-cross-solver-spec-v1",
        "enabled": True,
        "context": {
            "system_kind": "burgers",
            "nu": 0.05,
            "advection_coefficient": 1.0,
            "final_time": 0.02,
            "domain_length": 1.0,
            "dtype": "float64",
            "dealias": True,
        },
        "solvers": {
            "split_step": {
                "candidates": [
                    {
                        "solver": "split_step",
                        "dt": 0.01,
                        "fine_dt": 0.005,
                        "dealias": True,
                    },
                    {
                        "solver": "split_step",
                        "dt": 0.01,
                        "fine_dt": 0.0025,
                        "dealias": True,
                    },
                ]
            },
            "etdrk4": {
                "candidates": [
                    {
                        "solver": "etdrk4",
                        "dt": 0.01,
                        "fine_dt": None,
                        "dealias": True,
                    },
                    {
                        "solver": "etdrk4",
                        "dt": 0.005,
                        "fine_dt": None,
                        "dealias": True,
                    },
                ]
            },
        },
        "tolerances": {
            "mean_relative_l2": 1.0,
            "max_relative_l2": 1.0,
            "low_mode_relative_l2": 1.0,
        },
    }
    write_json(validation_path, validation)
    validation_spec = load_validation_spec(validation_path, repo_root=root)
    certificate = ensure_validation(validation_spec).certificate
    dataset_spec = load_dataset_spec(dataset_path, repo_root=root)
    return certificate, validation_spec, dataset_spec


def _heat_binding_inputs(
    root: Path,
) -> tuple[Any, Any, Any]:
    validation_path, dataset_path, _ = write_tiny_heat_stack(root)
    validation_spec = load_validation_spec(validation_path, repo_root=root)
    certificate = ensure_validation(validation_spec).certificate
    dataset_spec = load_dataset_spec(dataset_path, repo_root=root)
    return certificate, validation_spec, dataset_spec


def _reaction_diffusion_binding_inputs(
    root: Path,
) -> tuple[Any, Any, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    validation_raw = _read_json(
        repo_root
        / "configs/validation/reaction_diffusion_smoke.json"
    )
    validation_raw["name"] = "tiny_reaction_diffusion_binding"
    validation_raw["artifact_root"] = str(root / "artifacts")
    validation_path = write_json(
        root / "reaction_validation.json",
        validation_raw,
    )
    dataset_path = write_json(
        root / "reaction_dataset.json",
        {
            "schema_version": "pol-dataset-v3",
            "name": "tiny_reaction_diffusion_dataset",
            "artifact_root": str(root / "artifacts"),
            "validation_spec": str(validation_path),
            "binding": {"kind": "validated_reference"},
            "reference_nx": 64,
            "target": {
                "system": {
                    "kind": "reaction_diffusion",
                    "nu": 0.05,
                    "alpha": 1.0,
                    "beta": 1.0,
                    "solver": "semi_implicit_spectral_euler",
                    "dt": 0.00125,
                    "nonlinear_filter": "two_thirds",
                },
                "time": 0.04,
            },
            "batch_size": 24,
        },
    )
    validation_spec = load_validation_spec(
        validation_path,
        repo_root=root,
    )
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


def _heat_dataset_with_change(root: Path, dotted: str, value: Any):
    certificate, validation_spec, dataset_spec = _heat_binding_inputs(root)
    raw = dataset_spec.model_dump(mode="json")
    raw["validation_spec"] = str(dataset_spec.validation_spec)
    current: Any = raw
    parts = dotted.split(".")
    for part in parts[:-1]:
        current = current[part]
    current[parts[-1]] = value
    path = write_json(root / "changed_heat_dataset.json", raw)
    changed = load_dataset_spec(path, repo_root=root)
    return certificate, validation_spec, changed


def test_certificate_records_self_consistent_selected_suffix_contract(
    tmp_path: Path,
) -> None:
    certificate, _, _ = _binding_inputs(tmp_path)
    target = certificate["target_reference_contract"]
    reference = target["reference_resolution"]
    method = target["numerical_method_validation"]
    relation = target["allowed_refinement_relation"]

    assert reference["selected_candidate_index"] == 0
    assert reference["selected_value"] == reference["candidates"][0] == 32
    assert relation["reference_nx_allowed_indices"] == [0, 1]
    assert relation["reference_nx_allowed_values"] == [32, 64]
    assert method["kind"] == "candidate_refinement"
    assert method["temporal_status"] == "converged"
    assert method["selected_candidate_index"] == 0
    assert method["selected_condition"] == method["candidates"][0]
    assert relation["numerical_condition_allowed_indices"] == [0, 1]
    assert (
        relation["numerical_condition_allowed_values"]
        == method["candidates"]
    )
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
    assert proof["matched_numerical_condition_index"] == 1
    assert proof["dataset_condition"]["reference_nx"] == 64
    calibration = certificate["foundation_contract"]["calibration_provenance"]
    assert proof["schema_version"] == "pol-dataset-binding-proof-v7"
    assert proof["calibration_provenance"] == calibration
    assert proof["dataset_condition"]["split"] == {
        key: calibration[key]
        for key in (
            "split_policy",
            "split_policy_version",
            "total_samples",
            "n_train",
            "n_validation",
            "n_test",
            "seed",
            "split_hash",
        )
    }
    assert proof["proof_hash"] == stable_object_hash(
        {key: value for key, value in proof.items() if key != "proof_hash"}
    )


def test_heat_validated_reference_builds_and_loads_target_specific_proof(
    tmp_path: Path,
) -> None:
    _, _, dataset_spec = _heat_binding_inputs(tmp_path)
    dataset = ensure_dataset(dataset_spec, repo_root=tmp_path)
    loaded = load_dataset(dataset.path)
    proof = loaded.binding_proof
    assert loaded.binding_kind == "validated_reference"
    assert loaded.target_reference_validation_status == "validated"
    assert proof["validated_condition"]["system_kind"] == "heat"
    assert proof["matched_reference_candidate_index"] == 1
    assert proof["matched_numerical_condition_index"] == 0
    assert proof["dataset_condition"]["target"]["system"] == {
        "kind": "heat",
        "nu": 0.1,
    }
    assert proof["allowed_refinement_relation"][
        "numerical_condition_allowed_values"
    ] == [{"solver": "spectral_exact"}]


def test_reaction_diffusion_temporary_dataset_builds_validated_binding(
    tmp_path: Path,
) -> None:
    _, _, dataset_spec = _reaction_diffusion_binding_inputs(tmp_path)
    dataset = ensure_dataset(dataset_spec, repo_root=tmp_path)
    loaded = load_dataset(dataset.path)
    proof = loaded.binding_proof
    assert loaded.binding_kind == "validated_reference"
    assert loaded.target_reference_validation_status == "validated"
    assert proof["schema_version"] == "pol-dataset-binding-proof-v7"
    assert proof["validated_condition"]["system_kind"] == (
        "reaction_diffusion"
    )
    assert proof["validated_condition"]["invariant_parameters"] == {
        "nu": 0.05,
        "alpha": 1.0,
        "beta": 1.0,
    }
    assert proof["matched_reference_candidate_index"] == 1
    assert proof["matched_numerical_condition_index"] == 2
    assert proof["dataset_condition"]["target"]["system"][
        "nonlinear_filter"
    ] == "two_thirds"
    assert proof["allowed_refinement_relation"][
        "numerical_condition_allowed_values"
    ] == [
        {
            "solver": "semi_implicit_spectral_euler",
            "dt": 0.0025,
            "nonlinear_filter": "two_thirds",
        },
        {
            "solver": "semi_implicit_spectral_euler",
            "dt": 0.00125,
            "nonlinear_filter": "two_thirds",
        },
    ]


def test_reaction_diffusion_binding_rejects_every_exact_condition_mismatch(
    tmp_path: Path,
) -> None:
    certificate, validation_spec, dataset_spec = (
        _reaction_diffusion_binding_inputs(tmp_path)
    )
    changes = [
        ("target.system.nu", 0.06),
        ("target.system.alpha", 1.1),
        ("target.system.beta", 0.9),
        ("target.time", 0.08),
        ("target.system.dt", 0.005),
        ("target.system.nonlinear_filter", "none"),
        ("reference_nx", 48),
    ]
    base = dataset_spec.model_dump(mode="json")
    base["validation_spec"] = str(dataset_spec.validation_spec)
    for index, (dotted, value) in enumerate(changes):
        raw = copy.deepcopy(base)
        current: Any = raw
        parts = dotted.split(".")
        for part in parts[:-1]:
            current = current[part]
        current[parts[-1]] = value
        changed = load_dataset_spec(
            write_json(
                tmp_path / f"changed_reaction_{index}.json",
                raw,
            ),
            repo_root=tmp_path,
        )
        with pytest.raises(DatasetBindingError, match="field_path"):
            evaluate_dataset_binding(
                certificate,
                validation_spec,
                changed,
            )

    changed_dtype = validation_spec.model_copy(
        update={
            "samples": validation_spec.samples.model_copy(
                update={"dtype": "float32"}
            )
        }
    )
    changed_domain = validation_spec.model_copy(
        update={"domain": DomainSpec(length=2.0)}
    )
    with pytest.raises(DatasetBindingError, match="dtype"):
        evaluate_dataset_binding(
            certificate,
            changed_dtype,
            dataset_spec,
        )
    with pytest.raises(DatasetBindingError, match="domain"):
        evaluate_dataset_binding(
            certificate,
            changed_domain,
            dataset_spec,
        )


def test_reaction_diffusion_binding_rejects_mixed_filter_dt_pseudo_condition(
    tmp_path: Path,
) -> None:
    certificate, validation_spec, dataset_spec = (
        _reaction_diffusion_binding_inputs(tmp_path)
    )
    raw = dataset_spec.model_dump(mode="json")
    raw["validation_spec"] = str(dataset_spec.validation_spec)
    raw["target"]["system"]["dt"] = 0.0025
    raw["target"]["system"]["nonlinear_filter"] = "none"
    changed = load_dataset_spec(
        write_json(tmp_path / "reaction_pseudo_condition.json", raw),
        repo_root=tmp_path,
    )
    with pytest.raises(
        DatasetBindingError,
        match=r"target\.system\.nonlinear_filter",
    ):
        evaluate_dataset_binding(
            certificate,
            validation_spec,
            changed,
        )


@pytest.mark.parametrize(
    ("path", "value", "error_path"),
    [
        ("target.system.nu", 0.2, "nu"),
        ("target.time", 0.2, "time"),
        ("reference_nx", 24, "reference_nx"),
    ],
)
def test_heat_binding_rejects_target_or_resolution_mismatch(
    tmp_path: Path,
    path: str,
    value: Any,
    error_path: str,
) -> None:
    certificate, validation_spec, changed = _heat_dataset_with_change(
        tmp_path,
        path,
        value,
    )
    with pytest.raises(
        DatasetBindingError,
        match=rf"field_path=.*{error_path}",
    ):
        evaluate_dataset_binding(certificate, validation_spec, changed)


@pytest.mark.parametrize("field", ["dtype", "domain_length"])
def test_heat_binding_rejects_dtype_or_domain_mismatch(
    tmp_path: Path,
    field: str,
) -> None:
    certificate, validation_spec, dataset_spec = _heat_binding_inputs(tmp_path)
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
            certificate,
            changed_validation,
            dataset_spec,
        )


def test_heat_binding_rejects_noncanonical_solver_contract(
    tmp_path: Path,
) -> None:
    certificate, validation_spec, dataset_spec = _heat_binding_inputs(tmp_path)
    changed = copy.deepcopy(certificate)
    contract = changed["target_reference_contract"]
    method = contract["numerical_method_validation"]
    relation = contract["allowed_refinement_relation"]
    method["selected_condition"] = {"solver": "split_step"}
    method["candidates"] = [{"solver": "split_step"}]
    relation["numerical_condition_allowed_values"] = [
        {"solver": "split_step"}
    ]
    with pytest.raises(
        DatasetBindingError,
        match="target_reference_contract",
    ):
        evaluate_dataset_binding(changed, validation_spec, dataset_spec)


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


def test_validated_reference_rejects_solver_family_mismatch(
    tmp_path: Path,
) -> None:
    certificate, validation_spec, dataset_spec = _binding_inputs(tmp_path)
    raw = dataset_spec.model_dump(mode="json")
    raw["validation_spec"] = str(dataset_spec.validation_spec)
    raw["target"]["system"]["solver"] = "etdrk4"
    raw["target"]["system"]["fine_dt"] = None
    changed = load_dataset_spec(
        write_json(tmp_path / "changed_solver_dataset.json", raw),
        repo_root=tmp_path,
    )
    with pytest.raises(
        DatasetBindingError,
        match=r"field_path=\$\.dataset\.target\..*solver",
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


def test_heat_and_burgers_certificates_cannot_cross_bind(
    tmp_path: Path,
) -> None:
    heat_root = tmp_path / "heat"
    burgers_root = tmp_path / "burgers"
    heat_certificate, heat_validation, heat_dataset = _heat_binding_inputs(
        heat_root
    )
    (
        burgers_certificate,
        burgers_validation,
        burgers_dataset,
    ) = _binding_inputs(burgers_root)
    with pytest.raises(DatasetBindingError, match="system.kind"):
        evaluate_dataset_binding(
            heat_certificate,
            heat_validation,
            burgers_dataset,
        )
    with pytest.raises(DatasetBindingError, match="system.kind"):
        evaluate_dataset_binding(
            burgers_certificate,
            burgers_validation,
            heat_dataset,
        )


def test_reaction_diffusion_certificate_cannot_cross_bind_heat_or_burgers(
    tmp_path: Path,
) -> None:
    reaction_certificate, reaction_validation, reaction_dataset = (
        _reaction_diffusion_binding_inputs(tmp_path / "reaction")
    )
    heat_certificate, heat_validation, heat_dataset = _heat_binding_inputs(
        tmp_path / "heat"
    )
    burgers_certificate, burgers_validation, burgers_dataset = (
        _binding_inputs(tmp_path / "burgers")
    )
    with pytest.raises(DatasetBindingError, match="system.kind"):
        evaluate_dataset_binding(
            heat_certificate,
            heat_validation,
            reaction_dataset,
        )
    with pytest.raises(DatasetBindingError, match="system.kind"):
        evaluate_dataset_binding(
            burgers_certificate,
            burgers_validation,
            reaction_dataset,
        )
    with pytest.raises(DatasetBindingError, match="system.kind"):
        evaluate_dataset_binding(
            reaction_certificate,
            reaction_validation,
            heat_dataset,
        )
    with pytest.raises(DatasetBindingError, match="system.kind"):
        evaluate_dataset_binding(
            reaction_certificate,
            reaction_validation,
            burgers_dataset,
        )


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


def test_heat_binding_failure_precedes_target_evolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, dataset_path, _ = write_tiny_heat_stack(tmp_path)
    raw = _read_json(dataset_path)
    raw["target"]["system"]["nu"] = 0.2
    write_json(dataset_path, raw)
    dataset_spec = load_dataset_spec(dataset_path, repo_root=tmp_path)
    calls = 0

    def forbidden_evolve(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("target evolve must not be called")

    monkeypatch.setattr("pol.data.dataset.evolve", forbidden_evolve)
    with pytest.raises(DatasetBindingError, match="nu"):
        ensure_dataset(dataset_spec, repo_root=tmp_path)
    assert calls == 0
    assert not (tmp_path / "artifacts" / "datasets").exists()


def test_reaction_diffusion_binding_failure_precedes_target_evolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, dataset_spec = _reaction_diffusion_binding_inputs(tmp_path)
    raw = dataset_spec.model_dump(mode="json")
    raw["validation_spec"] = str(dataset_spec.validation_spec)
    raw["target"]["system"]["alpha"] = 1.1
    changed = load_dataset_spec(
        write_json(tmp_path / "wrong_reaction_dataset.json", raw),
        repo_root=tmp_path,
    )
    calls = 0

    def forbidden_evolve(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("target evolve must not be called")

    monkeypatch.setattr("pol.data.dataset.evolve", forbidden_evolve)
    with pytest.raises(DatasetBindingError, match="alpha"):
        ensure_dataset(changed, repo_root=tmp_path)
    assert calls == 0
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


def test_dataset_binding_rejects_certificate_split_hash_mismatch(
    tmp_path: Path,
) -> None:
    certificate, validation_spec, dataset_spec = _binding_inputs(tmp_path)
    changed = copy.deepcopy(certificate)
    foundation = changed["foundation_contract"]
    foundation["calibration_provenance"]["split_hash"] = "0" * 64
    changed["foundation_contract_hash"] = stable_object_hash(foundation)
    with pytest.raises(
        DatasetBindingError,
        match=r"foundation_contract\.calibration_provenance",
    ):
        evaluate_dataset_binding(changed, validation_spec, dataset_spec)


def test_binding_proof_rejects_rehashed_split_semantics_tamper(
    tmp_path: Path,
) -> None:
    certificate, validation_spec, dataset_spec = _binding_inputs(tmp_path)
    proof = evaluate_dataset_binding(certificate, validation_spec, dataset_spec)
    changed = copy.deepcopy(proof)
    changed["dataset_condition"]["split"]["split_hash"] = "0" * 64
    unsigned = {
        key: value for key, value in changed.items() if key != "proof_hash"
    }
    changed["proof_hash"] = stable_object_hash(unsigned)
    with pytest.raises(ValueError, match="split condition mismatch"):
        verify_binding_proof(changed)


def test_heat_binding_proof_rejects_rehashed_solver_tamper(
    tmp_path: Path,
) -> None:
    certificate, validation_spec, dataset_spec = _heat_binding_inputs(tmp_path)
    proof = evaluate_dataset_binding(certificate, validation_spec, dataset_spec)
    changed = copy.deepcopy(proof)
    changed["validated_condition"]["numerical_method_validation"][
        "selected_condition"
    ] = {"solver": "split_step"}
    unsigned = {
        key: value for key, value in changed.items() if key != "proof_hash"
    }
    changed["proof_hash"] = stable_object_hash(unsigned)
    with pytest.raises(ValueError, match="condition binding mismatch"):
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


@pytest.mark.parametrize(
    "tamper",
    ["selected_index", "effective_substep", "substep_count", "candidate_order"],
)
def test_target_contract_rejects_rehashed_refinement_semantics_tamper(
    tmp_path: Path,
    tamper: str,
) -> None:
    certificate, validation_spec, dataset_spec = _binding_inputs(tmp_path)
    changed = copy.deepcopy(certificate)
    contract = changed["target_reference_contract"]
    method = contract["numerical_method_validation"]
    if tamper == "selected_index":
        method["selected_candidate_index"] = 1
        method["selected_condition"] = copy.deepcopy(
            method["candidates"][1]
        )
        relation = contract["allowed_refinement_relation"]
        relation["numerical_condition_allowed_indices"] = [1]
        relation["numerical_condition_allowed_values"] = [
            copy.deepcopy(method["candidates"][1])
        ]
    elif tamper == "effective_substep":
        method["candidates"][1]["effective_substep"] = 0.001
    elif tamper == "substep_count":
        method["candidates"][1]["substeps_per_outer"] = 99
    else:
        method["candidates"].reverse()
    changed["target_reference_contract_hash"] = stable_object_hash(contract)
    with pytest.raises(
        DatasetBindingError,
        match="target_reference_contract",
    ):
        evaluate_dataset_binding(changed, validation_spec, dataset_spec)


def test_dataset_binding_rejects_mixed_field_pseudo_candidate(
    tmp_path: Path,
) -> None:
    validation_path, dataset_path = _validated_burgers_dataset_path(tmp_path)
    validation = _read_json(validation_path)
    validation["target_reference"]["time_candidates"] = [
        {
            "solver": "split_step",
            "dt": 0.02,
            "fine_dt": 0.01,
            "dealias": True,
        },
        {
            "solver": "split_step",
            "dt": 0.01,
            "fine_dt": 0.0025,
            "dealias": True,
        },
    ]
    validation["target_reference"]["reference_evolution"]["system"].update(
        validation["target_reference"]["time_candidates"][-1]
    )
    write_json(validation_path, validation)
    dataset = _read_json(dataset_path)
    dataset["target"]["system"]["dt"] = 0.02
    dataset["target"]["system"]["fine_dt"] = 0.0025
    write_json(dataset_path, dataset)
    validation_spec = load_validation_spec(
        validation_path,
        repo_root=tmp_path,
    )
    certificate = ensure_validation(validation_spec).certificate
    dataset_spec = load_dataset_spec(dataset_path, repo_root=tmp_path)
    with pytest.raises(
        DatasetBindingError,
        match=r"target\.numerical_condition",
    ):
        evaluate_dataset_binding(
            certificate,
            validation_spec,
            dataset_spec,
        )


def test_certificate_loader_rejects_convergence_csv_row_tamper(
    tmp_path: Path,
) -> None:
    validation_path, _, _ = write_tiny_stack(tmp_path)
    outcome = ensure_validation(
        load_validation_spec(validation_path, repo_root=tmp_path)
    )
    csv_path = outcome.reference.path / "reference_convergence.csv"
    text = csv_path.read_text(encoding="utf-8")
    csv_path.write_text(
        text.replace(
            "spatial,reference_resolution",
            "tampered,reference_resolution",
            1,
        ),
        encoding="utf-8",
    )
    _refresh_artifact_record(
        outcome.reference.path,
        "reference_convergence.csv",
    )
    with pytest.raises(ValueError, match="CSV rows disagree"):
        load_validation_certificate(outcome.reference.path)


@pytest.mark.parametrize(
    "tamper",
    [
        "condition",
        "metric",
        "status",
        "missing_self_evidence",
        "row_hash",
        "runtime_substep_count",
    ],
)
def test_certificate_loader_rejects_cross_solver_evidence_tamper(
    tmp_path: Path,
    tamper: str,
) -> None:
    validation_path, _ = _validated_burgers_dataset_path(tmp_path)
    validation = _read_json(validation_path)
    validation["target_reference"]["cross_solver_validation"] = (
        _read_json(
            Path("configs/validation/foundation_smoke.json")
        )["target_reference"]["cross_solver_validation"]
    )
    validation["target_reference"]["cross_solver_validation"]["context"].update(
        {
            "nu": 0.05,
            "domain_length": 1.0,
            "dtype": "float64",
        }
    )
    validation["target_reference"]["cross_solver_validation"][
        "tolerances"
    ] = {
        "mean_relative_l2": 1.0,
        "max_relative_l2": 1.0,
        "low_mode_relative_l2": 1.0,
    }
    write_json(validation_path, validation)
    outcome = ensure_validation(
        load_validation_spec(validation_path, repo_root=tmp_path)
    )
    checks_path = outcome.reference.path / "checks.json"
    checks = _read_json(checks_path)
    block = checks["cross_solver_validation"]
    if tamper == "condition":
        block["self_convergence"]["etdrk4"]["ordered_candidates"][1][
            "effective_substep"
        ] = 0.004
    elif tamper == "metric":
        block["discrepancy_metrics"]["mean_relative_l2"] = -1.0
    elif tamper == "status":
        block["status"] = "fail"
    elif tamper == "missing_self_evidence":
        block["self_convergence"].pop("etdrk4")
    elif tamper == "row_hash":
        block["self_convergence"]["split_step"]["rows"][0][
            "row_hash"
        ] = "0" * 64
    else:
        block["self_convergence"]["split_step"][
            "runtime_solver_metadata"
        ][1]["substeps_per_outer"] = 99
    write_strict_json(checks_path, checks)
    _refresh_artifact_record(outcome.reference.path, "checks.json")
    with pytest.raises(ValueError, match="cross-solver|convergence row"):
        load_validation_certificate(outcome.reference.path)


def test_cross_solver_condition_cannot_be_injected_into_dataset_suffix(
    tmp_path: Path,
) -> None:
    certificate, validation_spec, dataset_spec = _cross_binding_inputs(
        tmp_path
    )
    changed = copy.deepcopy(certificate)
    contract = changed["target_reference_contract"]
    relation = contract["allowed_refinement_relation"]
    relation["numerical_condition_allowed_indices"].append(2)
    relation["numerical_condition_allowed_values"].append(
        copy.deepcopy(
            changed["cross_solver_validation"]["finest_conditions"][
                "etdrk4"
            ]
        )
    )
    changed["target_reference_contract_hash"] = stable_object_hash(contract)
    with pytest.raises(
        DatasetBindingError,
        match="target_reference_contract",
    ):
        evaluate_dataset_binding(changed, validation_spec, dataset_spec)


def test_heat_certificate_loader_rejects_analytic_check_tamper(
    tmp_path: Path,
) -> None:
    validation_path, _, _ = write_tiny_heat_stack(tmp_path)
    outcome = ensure_validation(
        load_validation_spec(validation_path, repo_root=tmp_path)
    )
    checks_path = outcome.reference.path / "checks.json"
    checks = _read_json(checks_path)
    checks["heat_analytic"]["cases"][0]["expected_multiplier"] = 0.5
    write_strict_json(checks_path, checks)
    _refresh_artifact_record(outcome.reference.path, "checks.json")
    with pytest.raises(ValueError, match="heat analytic check"):
        load_validation_certificate(outcome.reference.path)


def test_reaction_diffusion_certificate_loader_rejects_characterization_tamper(
    tmp_path: Path,
) -> None:
    _, validation_spec, _ = _reaction_diffusion_binding_inputs(tmp_path)
    outcome = ensure_validation(validation_spec)
    checks_path = outcome.reference.path / "checks.json"
    checks = _read_json(checks_path)
    checks["reaction_diffusion_characterization"][
        "beta_zero_linear_modes"
    ][0]["expected_multiplier"] = 0.5
    write_strict_json(checks_path, checks)
    _refresh_artifact_record(outcome.reference.path, "checks.json")
    with pytest.raises(
        ValueError,
        match="reaction-diffusion characterization",
    ):
        load_validation_certificate(outcome.reference.path)


def test_reaction_diffusion_target_contract_rejects_filter_tamper(
    tmp_path: Path,
) -> None:
    certificate, validation_spec, dataset_spec = (
        _reaction_diffusion_binding_inputs(tmp_path)
    )
    changed = copy.deepcopy(certificate)
    contract = changed["target_reference_contract"]
    contract["numerical_method_validation"]["candidates"][1][
        "nonlinear_filter"
    ] = "none"
    changed["target_reference_contract_hash"] = stable_object_hash(contract)
    with pytest.raises(
        DatasetBindingError,
        match="target_reference_contract",
    ):
        evaluate_dataset_binding(
            changed,
            validation_spec,
            dataset_spec,
        )


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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (
            "calibration_split_membership",
            {"0": "test", "1": "train"},
        ),
        ("calibration_test_overlap_count", 1),
        ("split_hash", "0" * 64),
    ],
)
def test_certificate_loader_rejects_calibration_provenance_tamper(
    tmp_path: Path,
    field: str,
    value: Any,
) -> None:
    validation_path, _, _ = write_tiny_stack(tmp_path)
    outcome = ensure_validation(
        load_validation_spec(validation_path, repo_root=tmp_path)
    )
    certificate_path = outcome.reference.path / "certificate.json"
    certificate = _read_json(certificate_path)
    foundation = certificate["foundation_contract"]
    foundation["calibration_provenance"][field] = value
    certificate["foundation_contract_hash"] = stable_object_hash(foundation)
    write_strict_json(certificate_path, certificate)
    _refresh_artifact_record(outcome.reference.path, "certificate.json")
    with pytest.raises(ValueError, match="certificate contract"):
        load_validation_certificate(outcome.reference.path)


def test_certificate_loader_rejects_pre_review_gate_b_certificate_revision(
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
    with pytest.raises(ValueError, match="Phase 2-05B requires"):
        load_validation_certificate(outcome.reference.path)
