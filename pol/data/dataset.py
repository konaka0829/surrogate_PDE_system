from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import torch

from pol.artifacts import ArtifactRef, ArtifactStore, verify_artifact
from pol.config.loader import load_validation_spec
from pol.config.models import DatasetSpec
from pol.math.periodic import spectral_resample_periodic
from pol.runtime.environment import numerical_environment_fingerprint
from pol.runtime.hashing import stable_object_hash, tensor_sha256
from pol.runtime.io import atomic_torch_save, write_strict_json
from pol.systems.registry import evolve
from pol.validation.binding import evaluate_dataset_binding, verify_binding_proof
from pol.validation.runner import ensure_validation


@dataclass(frozen=True)
class ReferenceDataset:
    artifact_id: str
    path: Path
    sample_ids: torch.Tensor
    inputs_reference: torch.Tensor
    targets_reference: torch.Tensor
    train_ids: torch.Tensor
    validation_ids: torch.Tensor
    test_ids: torch.Tensor
    reference_nx: int
    domain_length: float
    dtype_name: str
    target_metadata: dict[str, Any]
    split_hash: str
    validation_artifact_id: str
    binding_kind: str
    binding_status: str
    target_reference_validation_status: str
    binding_proof: dict[str, Any]
    binding_proof_hash: str

    @property
    def total_samples(self) -> int:
        return int(self.sample_ids.numel())

    def positions(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.ndim != 1 or ids.dtype != torch.long:
            raise ValueError("sample ids must be a one-dimensional long tensor")
        lookup = {int(value): index for index, value in enumerate(self.sample_ids.tolist())}
        try:
            return torch.tensor([lookup[int(value)] for value in ids.tolist()], dtype=torch.long)
        except KeyError as exc:
            raise ValueError(f"unknown sample id: {exc.args[0]}") from exc

    def tensors_for(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        positions = self.positions(ids)
        return (
            self.inputs_reference.index_select(0, positions),
            self.targets_reference.index_select(0, positions),
        )


def _same_canonical_json(left: Any, right: Any) -> bool:
    return stable_object_hash(left) == stable_object_hash(right)


def _scientific_identity(
    spec: DatasetSpec,
    validation_artifact_id: str,
    binding_proof: dict[str, Any],
) -> dict[str, Any]:
    payload = spec.model_dump(mode="json")
    payload.pop("artifact_root", None)
    payload.pop("validation_spec", None)
    return {
        "schema_version": "pol-reference-dataset-identity-v2",
        "environment": numerical_environment_fingerprint(),
        "validation_artifact_id": validation_artifact_id,
        "binding_kind": binding_proof["binding_kind"],
        "binding_status": binding_proof["status"],
        "target_reference_validation_status": binding_proof[
            "target_reference_validation_status"
        ],
        "binding_proof_hash": binding_proof["proof_hash"],
        "binding_proof": binding_proof,
        "spec": payload,
    }


def _split_ids(total: int, n_train: int, n_validation: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    permutation = torch.randperm(total, generator=generator, dtype=torch.long)
    train = permutation[:n_train].clone()
    validation = permutation[n_train : n_train + n_validation].clone()
    test = permutation[n_train + n_validation :].clone()
    return train, validation, test


def _load_master(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("initial-condition archive payload must be an object")
    if payload.get("schema_version") != "pol-initial-condition-archive-v2":
        raise ValueError(
            "unsupported legacy initial-condition archive; P0-02 requires v2"
        )
    return payload


def dataset_reference(
    spec: DatasetSpec,
    *,
    validation_artifact_id: str,
    binding_proof: dict[str, Any],
) -> ArtifactRef:
    identity = _scientific_identity(spec, validation_artifact_id, binding_proof)
    return ArtifactStore(spec.artifact_root).reference("datasets", identity)


def ensure_dataset(
    spec: DatasetSpec,
    *,
    repo_root: Path,
    force: bool = False,
) -> ReferenceDataset:
    validation_spec = load_validation_spec(spec.validation_spec, repo_root=repo_root)
    validation = ensure_validation(validation_spec, force=False)
    binding_proof = evaluate_dataset_binding(
        validation.certificate,
        validation_spec,
        spec,
    )
    ref = dataset_reference(
        spec,
        validation_artifact_id=validation.reference.artifact_id,
        binding_proof=binding_proof,
    )
    store = ArtifactStore(spec.artifact_root)
    if not store.exists(ref) or force:
        _build_dataset(
            spec,
            validation_spec=validation_spec,
            validation_ref=validation.reference,
            binding_proof=binding_proof,
            ref=ref,
            force=force,
        )
    return load_dataset(ref.path)


def _build_dataset(
    spec: DatasetSpec,
    *,
    validation_spec,
    validation_ref: ArtifactRef,
    binding_proof: dict[str, Any],
    ref: ArtifactRef,
    force: bool,
) -> None:
    master = _load_master(validation_ref.path / "master_initial_conditions.pt")
    master_values: torch.Tensor = master["values"]
    master_ids: torch.Tensor = master["sample_ids"]
    if int(spec.reference_nx) > int(master["nx"]):
        raise ValueError(
            f"dataset reference_nx={spec.reference_nx} exceeds validated master nx={master['nx']}"
        )
    inputs = spectral_resample_periodic(
        master_values, int(spec.reference_nx), domain_length=validation_spec.domain.length
    )
    targets: list[torch.Tensor] = []
    metadata: dict[str, Any] | None = None
    evolution = spec.target.model_dump(mode="json")
    for start in range(0, inputs.shape[0], int(spec.batch_size)):
        batch = inputs[start : start + int(spec.batch_size)]
        values, batch_metadata = evolve(
            batch,
            evolution,
            domain_length=validation_spec.domain.length,
        )
        targets.append(values.detach().cpu())
        if metadata is None:
            metadata = batch_metadata
        elif {
            key: value for key, value in metadata.items() if key not in {"device"}
        } != {
            key: value for key, value in batch_metadata.items() if key not in {"device"}
        }:
            raise RuntimeError("target solver metadata changed between batches")
    target_values = torch.cat(targets, dim=0)
    samples = validation_spec.samples
    train_ids, validation_ids, test_ids = _split_ids(
        samples.total_samples,
        samples.n_train,
        samples.n_validation,
        samples.seed,
    )
    split_payload = {
        "sample_ids": master_ids.tolist(),
        "train_ids": train_ids.tolist(),
        "validation_ids": validation_ids.tolist(),
        "test_ids": test_ids.tolist(),
    }
    split_hash = stable_object_hash(split_payload)
    identity = _scientific_identity(
        spec, validation_ref.artifact_id, binding_proof
    )
    binding_fields = {
        "binding_kind": binding_proof["binding_kind"],
        "binding_status": binding_proof["status"],
        "target_reference_validation_status": binding_proof[
            "target_reference_validation_status"
        ],
        "binding_proof_hash": binding_proof["proof_hash"],
        "binding_proof": binding_proof,
    }
    metadata_payload = {
        "schema_version": "pol-reference-dataset-metadata-v2",
        "artifact_id": ref.artifact_id,
        "name": spec.name,
        "validation_artifact_id": validation_ref.artifact_id,
        **binding_fields,
        "reference_nx": int(spec.reference_nx),
        "domain_length": float(validation_spec.domain.length),
        "dtype": samples.dtype,
        "total_samples": int(samples.total_samples),
        "split_hash": split_hash,
        "target": spec.target.model_dump(mode="json"),
        "target_solver_metadata": metadata,
        "tensor_hashes": {
            "sample_ids": tensor_sha256(master_ids),
            "inputs_reference": tensor_sha256(inputs),
            "targets_reference": tensor_sha256(target_values),
            "train_ids": tensor_sha256(train_ids),
            "validation_ids": tensor_sha256(validation_ids),
            "test_ids": tensor_sha256(test_ids),
        },
    }
    archive = {
        "schema_version": "pol-reference-dataset-v2",
        "artifact_id": ref.artifact_id,
        **binding_fields,
        "sample_ids": master_ids.clone(),
        "inputs_reference": inputs.detach().cpu(),
        "targets_reference": target_values.detach().cpu(),
        "train_ids": train_ids,
        "validation_ids": validation_ids,
        "test_ids": test_ids,
        "reference_nx": int(spec.reference_nx),
        "domain_length": float(validation_spec.domain.length),
        "dtype": samples.dtype,
        "target_metadata": metadata,
        "split_hash": split_hash,
        "validation_artifact_id": validation_ref.artifact_id,
    }

    def writer(root: Path) -> Iterable[str]:
        write_strict_json(
            root / "resolved_spec.json",
            {
                "schema_version": "pol-dataset-resolved-spec-v2",
                "validation_artifact_id": validation_ref.artifact_id,
                **binding_fields,
                "spec": identity["spec"],
            },
        )
        write_strict_json(root / "metadata.json", metadata_payload)
        atomic_torch_save(root / "dataset.pt", archive)
        return "resolved_spec.json", "metadata.json", "dataset.pt"

    ArtifactStore(spec.artifact_root).publish(
        ref,
        identity=identity,
        writer=writer,
        force=force,
    )


def load_dataset(path: Path | str) -> ReferenceDataset:
    root = Path(path).resolve()
    manifest = verify_artifact(root)
    if manifest.get("kind") != "datasets":
        raise ValueError("artifact is not a dataset")
    identity = manifest.get("identity")
    if not isinstance(identity, dict) or identity.get("schema_version") != (
        "pol-reference-dataset-identity-v2"
    ):
        raise ValueError(
            "unsupported legacy dataset identity; P0-02 binding proof is required"
        )
    payload = torch.load(root / "dataset.pt", map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("dataset archive payload must be an object")
    if payload.get("schema_version") != "pol-reference-dataset-v2":
        raise ValueError(
            "unsupported dataset archive schema; P0-02 requires v2"
        )
    if payload.get("artifact_id") != manifest.get("artifact_id"):
        raise ValueError("dataset archive identity does not match manifest")
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("dataset metadata payload must be an object")
    if metadata.get("schema_version") != "pol-reference-dataset-metadata-v2":
        raise ValueError(
            "unsupported dataset metadata schema; P0-02 requires v2"
        )
    if metadata.get("artifact_id") != payload.get("artifact_id"):
        raise ValueError("dataset metadata identity mismatch")
    resolved = json.loads(
        (root / "resolved_spec.json").read_text(encoding="utf-8")
    )
    if not isinstance(resolved, dict):
        raise ValueError("dataset resolved-spec payload must be an object")
    if resolved.get("schema_version") != "pol-dataset-resolved-spec-v2":
        raise ValueError(
            "unsupported dataset resolved-spec schema; P0-02 requires v2"
        )
    if not _same_canonical_json(resolved.get("spec"), identity.get("spec")):
        raise ValueError("dataset resolved spec does not match artifact identity")
    binding_proof = identity.get("binding_proof")
    if not isinstance(binding_proof, dict):
        raise ValueError("dataset identity has no binding proof")
    verify_binding_proof(binding_proof)
    binding_fields = (
        "binding_kind",
        "binding_status",
        "target_reference_validation_status",
        "binding_proof_hash",
        "binding_proof",
    )
    for field in binding_fields:
        expected = identity.get(field)
        if not _same_canonical_json(resolved.get(field), expected):
            raise ValueError(f"dataset resolved-spec binding mismatch: {field}")
        if not _same_canonical_json(metadata.get(field), expected):
            raise ValueError(f"dataset metadata binding mismatch: {field}")
        if not _same_canonical_json(payload.get(field), expected):
            raise ValueError(f"dataset archive binding mismatch: {field}")
    if identity.get("binding_proof_hash") != binding_proof.get("proof_hash"):
        raise ValueError("dataset identity binding-proof hash mismatch")
    if identity.get("binding_kind") != binding_proof.get("binding_kind"):
        raise ValueError("dataset identity binding-kind mismatch")
    if identity.get("binding_status") != binding_proof.get("status"):
        raise ValueError("dataset identity binding-status mismatch")
    if identity.get("target_reference_validation_status") != binding_proof.get(
        "target_reference_validation_status"
    ):
        raise ValueError("dataset identity target-validation status mismatch")
    identity_spec = identity.get("spec")
    if not isinstance(identity_spec, dict):
        raise ValueError("dataset identity spec must be an object")
    configured_binding = identity_spec.get("binding")
    if (
        not isinstance(configured_binding, dict)
        or configured_binding.get("kind") != identity.get("binding_kind")
    ):
        raise ValueError("dataset configured binding does not match its proof")
    if identity.get("validation_artifact_id") != binding_proof.get(
        "certificate_artifact_id"
    ):
        raise ValueError("dataset proof certificate binding mismatch")
    proof_dataset_condition = binding_proof.get("dataset_condition")
    if not isinstance(proof_dataset_condition, dict):
        raise ValueError("dataset binding proof has no dataset condition")
    expected_hashes = metadata.get("tensor_hashes")
    if not isinstance(expected_hashes, dict):
        raise ValueError("dataset metadata has no tensor hashes")
    tensor_names = (
        "sample_ids",
        "inputs_reference",
        "targets_reference",
        "train_ids",
        "validation_ids",
        "test_ids",
    )
    for name in tensor_names:
        if name not in payload or name not in expected_hashes:
            raise ValueError(f"dataset tensor is missing: {name}")
        if tensor_sha256(payload[name]) != expected_hashes[name]:
            raise ValueError(f"dataset tensor hash mismatch: {name}")

    sample_ids = payload["sample_ids"]
    inputs = payload["inputs_reference"]
    targets = payload["targets_reference"]
    if sample_ids.ndim != 1 or sample_ids.dtype != torch.long:
        raise ValueError("dataset sample_ids must be a one-dimensional long tensor")
    if torch.unique(sample_ids).numel() != sample_ids.numel():
        raise ValueError("dataset sample_ids must be unique")
    n_samples = int(sample_ids.numel())
    reference_nx = int(payload["reference_nx"])
    if inputs.shape != (n_samples, reference_nx) or targets.shape != inputs.shape:
        raise ValueError("dataset reference tensor shapes are inconsistent")
    if inputs.dtype != targets.dtype or not torch.isfinite(inputs).all() or not torch.isfinite(targets).all():
        raise ValueError("dataset reference tensors must be finite and share dtype")

    split_tensors = [payload["train_ids"], payload["validation_ids"], payload["test_ids"]]
    if any(value.ndim != 1 or value.dtype != torch.long for value in split_tensors):
        raise ValueError("dataset splits must be one-dimensional long tensors")
    assigned = torch.cat(split_tensors)
    if assigned.numel() != n_samples or torch.unique(assigned).numel() != n_samples:
        raise ValueError("dataset splits are not a disjoint full cover")
    if set(assigned.tolist()) != set(sample_ids.tolist()):
        raise ValueError("dataset splits do not cover sample_ids exactly")
    split_payload = {
        "sample_ids": sample_ids.tolist(),
        "train_ids": payload["train_ids"].tolist(),
        "validation_ids": payload["validation_ids"].tolist(),
        "test_ids": payload["test_ids"].tolist(),
    }
    split_hash = stable_object_hash(split_payload)
    if split_hash != payload.get("split_hash") or split_hash != metadata.get("split_hash"):
        raise ValueError("dataset split hash mismatch")
    validation_artifact_id = payload.get("validation_artifact_id")
    if any(
        value != validation_artifact_id
        for value in (
            metadata.get("validation_artifact_id"),
            resolved.get("validation_artifact_id"),
            identity.get("validation_artifact_id"),
            binding_proof.get("certificate_artifact_id"),
        )
    ):
        raise ValueError("dataset validation binding mismatch")
    if any(
        not _same_canonical_json(value, reference_nx)
        for value in (
            metadata.get("reference_nx"),
            proof_dataset_condition.get("reference_nx"),
        )
    ):
        raise ValueError("dataset reference resolution binding mismatch")
    domain_length = float(payload["domain_length"])
    if (
        not _same_canonical_json(metadata.get("domain_length"), domain_length)
        or not _same_canonical_json(
            proof_dataset_condition.get("domain_length"), domain_length
        )
    ):
        raise ValueError("dataset domain binding mismatch")
    dtype_name = str(payload["dtype"])
    if (
        not _same_canonical_json(metadata.get("dtype"), dtype_name)
        or not _same_canonical_json(proof_dataset_condition.get("dtype"), dtype_name)
    ):
        raise ValueError("dataset dtype binding mismatch")
    if (
        not _same_canonical_json(
            metadata.get("target"), identity_spec.get("target")
        )
        or not _same_canonical_json(
            proof_dataset_condition.get("target"), identity_spec.get("target")
        )
    ):
        raise ValueError("dataset target-condition binding mismatch")
    return ReferenceDataset(
        artifact_id=str(payload["artifact_id"]),
        path=root,
        sample_ids=payload["sample_ids"],
        inputs_reference=payload["inputs_reference"],
        targets_reference=payload["targets_reference"],
        train_ids=payload["train_ids"],
        validation_ids=payload["validation_ids"],
        test_ids=payload["test_ids"],
        reference_nx=int(payload["reference_nx"]),
        domain_length=domain_length,
        dtype_name=dtype_name,
        target_metadata=dict(payload["target_metadata"]),
        split_hash=str(payload["split_hash"]),
        validation_artifact_id=str(validation_artifact_id),
        binding_kind=str(identity["binding_kind"]),
        binding_status=str(identity["binding_status"]),
        target_reference_validation_status=str(
            identity["target_reference_validation_status"]
        ),
        binding_proof=dict(binding_proof),
        binding_proof_hash=str(identity["binding_proof_hash"]),
    )
