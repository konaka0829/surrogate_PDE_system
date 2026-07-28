from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import torch

from pol.artifacts import ArtifactStore, verify_artifact
from pol.config.models import TrialSpec
from pol.data.dataset import ReferenceDataset
from pol.data.finite import build_feature_initial_state
from pol.math.periodic import spectral_resample_periodic
from pol.runtime.environment import numerical_environment_fingerprint
from pol.runtime.device import (
    execution_device_policy,
    require_cpu_tensor,
    require_cpu_tensors,
    verify_execution_device_policy,
)
from pol.runtime.hashing import stable_object_hash, tensor_sha256
from pol.runtime.io import atomic_torch_save, write_strict_json
from pol.systems.registry import evolve


@dataclass(frozen=True)
class FeatureState:
    sample_ids: torch.Tensor
    values: torch.Tensor
    metadata: dict[str, Any]
    cache_id: str
    cache_path: Path | None


class FeatureStateCache:
    def __init__(
        self,
        *,
        artifact_root: Path,
        enabled: bool = True,
        batch_size: int = 64,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.store = ArtifactStore(artifact_root)
        self.enabled = bool(enabled)
        self.batch_size = int(batch_size)
        self.memory: dict[str, FeatureState] = {}
        self.hits = 0
        self.misses = 0

    def identity(
        self,
        dataset: ReferenceDataset,
        sample_ids: torch.Tensor,
        trial: TrialSpec,
    ) -> dict[str, Any]:
        require_cpu_tensor(
            sample_ids,
            boundary="feature-state cache identity",
            name="sample_ids",
        )
        verify_execution_device_policy(
            dataset.__dict__,
            boundary="feature-state dataset input",
        )
        evolution = trial.feature.evolution
        return {
            "schema_version": "pol-feature-state-identity-v2",
            **execution_device_policy(),
            "environment": numerical_environment_fingerprint(),
            "dataset_artifact_id": dataset.artifact_id,
            "sample_ids": [int(value) for value in sample_ids.tolist()],
            "n_tar": int(trial.input.n_tar),
            "n_sur": int(trial.feature.n_sur),
            "feature_generator": {
                "kind": trial.feature.kind,
                "evolution": (
                    None if evolution is None else evolution.model_dump(mode="json")
                ),
            },
            "domain_length": dataset.domain_length,
            "dtype": dataset.dtype_name,
        }

    @torch.no_grad()
    def get_or_solve(
        self,
        dataset: ReferenceDataset,
        sample_ids: torch.Tensor,
        trial: TrialSpec,
    ) -> FeatureState:
        identity = self.identity(dataset, sample_ids, trial)
        ref = self.store.reference("feature_states", identity)
        if ref.artifact_id in self.memory:
            self.hits += 1
            return self.memory[ref.artifact_id]
        if self.enabled and self.store.exists(ref):
            manifest = verify_artifact(ref.path)
            stored_identity = manifest.get("identity")
            if not isinstance(stored_identity, dict):
                raise ValueError("cached feature-state identity must be an object")
            verify_execution_device_policy(
                stored_identity,
                boundary="feature-state artifact identity",
            )
            environment = stored_identity.get("environment")
            if not isinstance(environment, dict):
                raise ValueError(
                    "cached feature-state numerical environment is missing"
                )
            verify_execution_device_policy(
                environment,
                boundary="feature-state numerical environment",
            )
            if stable_object_hash(stored_identity) != stable_object_hash(identity):
                raise ValueError("cached feature-state request identity mismatch")
            identity_doc = json.loads(
                (ref.path / "identity.json").read_text(encoding="utf-8")
            )
            if stable_object_hash(identity_doc) != stable_object_hash(
                stored_identity
            ):
                raise ValueError(
                    "cached feature-state identity document mismatch"
                )
            payload = torch.load(
                ref.path / "state.pt", map_location="cpu", weights_only=True
            )
            if payload.get("schema_version") != "pol-feature-state-v2":
                raise ValueError("unsupported feature-state cache schema")
            metadata_doc = json.loads(
                (ref.path / "metadata.json").read_text(encoding="utf-8")
            )
            if metadata_doc.get("schema_version") != (
                "pol-feature-state-metadata-v2"
            ):
                raise ValueError("unsupported feature-state metadata schema")
            verify_execution_device_policy(
                payload,
                boundary="feature-state archive",
            )
            verify_execution_device_policy(
                metadata_doc,
                boundary="feature-state metadata",
            )
            require_cpu_tensors(
                payload,
                boundary="feature-state cache load",
                name="state",
            )
            if payload.get("cache_id") != ref.artifact_id:
                raise ValueError("cached feature-state identity mismatch")
            if metadata_doc.get("cache_id") != ref.artifact_id:
                raise ValueError("cached feature-state metadata identity mismatch")
            hashes = payload.get("tensor_hashes")
            if not isinstance(hashes, dict):
                raise ValueError("cached feature-state has no tensor hashes")
            if tensor_sha256(payload["sample_ids"]) != hashes.get("sample_ids"):
                raise ValueError("cached feature-state sample IDs are corrupted")
            if tensor_sha256(payload["values"]) != hashes.get("values"):
                raise ValueError("cached feature-state tensor is corrupted")
            expected_ids = sample_ids.detach().cpu().to(torch.long)
            if not torch.equal(payload["sample_ids"], expected_ids):
                raise ValueError("cached feature-state sample IDs do not match request")
            values = payload["values"]
            if values.shape != (expected_ids.numel(), int(trial.feature.n_sur)):
                raise ValueError("cached feature-state shape does not match request")
            if not torch.isfinite(values).all():
                raise ValueError("cached feature-state contains non-finite values")
            dtype_name = str(values.dtype).removeprefix("torch.")
            if dtype_name != dataset.dtype_name:
                raise ValueError("cached feature-state dtype does not match dataset")
            if metadata_doc.get("solver") != payload.get("metadata"):
                raise ValueError("cached feature-state solver metadata mismatch")
            if payload.get("metadata", {}).get("device") != "cpu":
                raise ValueError(
                    "cached feature-state solver metadata must report device=cpu"
                )
            if metadata_doc.get("tensor_hashes") != hashes:
                raise ValueError("cached feature-state metadata hash mismatch")
            state = FeatureState(
                sample_ids=payload["sample_ids"],
                values=values,
                metadata=dict(payload["metadata"]),
                cache_id=ref.artifact_id,
                cache_path=ref.path,
            )
            self.memory[ref.artifact_id] = state
            self.hits += 1
            return state

        batches: list[torch.Tensor] = []
        metadata: dict[str, Any] | None = None
        evolution = trial.feature.evolution
        for start in range(0, int(sample_ids.numel()), self.batch_size):
            batch_ids = sample_ids[start : start + self.batch_size]
            require_cpu_tensor(
                batch_ids,
                boundary="feature-state solve batch",
                name="sample_ids",
            )
            inputs_reference, _ = dataset.tensors_for(batch_ids)
            require_cpu_tensor(
                inputs_reference,
                boundary="feature-state solve dataset input",
                name="inputs_reference",
            )
            finite_inputs = spectral_resample_periodic(
                inputs_reference,
                int(trial.input.n_tar),
                domain_length=dataset.domain_length,
            )
            initial = build_feature_initial_state(
                finite_inputs,
                n_sur=int(trial.feature.n_sur),
                domain_length=dataset.domain_length,
            )
            require_cpu_tensor(
                initial,
                boundary="feature-state solve initial state",
                name="initial",
            )
            if trial.feature.kind == "static_input":
                batch_values = initial
                batch_metadata = {
                    "kind": "static_input",
                    "solver": "none",
                    "time": 0.0,
                    "domain_length": float(dataset.domain_length),
                    "dtype": str(initial.dtype).removeprefix("torch."),
                    "device": str(initial.device),
                }
            else:
                if evolution is None:
                    raise ValueError("pde_dynamics feature has no evolution")
                batch_values, batch_metadata = evolve(
                    initial,
                    evolution.model_dump(mode="json"),
                    domain_length=dataset.domain_length,
                )
            require_cpu_tensor(
                batch_values,
                boundary="feature-state solve result",
                name="values",
            )
            if batch_metadata.get("device") != "cpu":
                raise RuntimeError(
                    "feature-state solver metadata must report device=cpu"
                )
            batches.append(batch_values.detach().cpu())
            if metadata is None:
                metadata = batch_metadata
            elif metadata != batch_metadata:
                raise RuntimeError("feature solver metadata changed between batches")
        if not batches or metadata is None:
            raise ValueError("feature-state solve requires at least one sample")
        values = torch.cat(batches, dim=0)
        payload = {
            "schema_version": "pol-feature-state-v2",
            **execution_device_policy(),
            "cache_id": ref.artifact_id,
            "sample_ids": sample_ids.detach().cpu().clone(),
            "values": values.detach().cpu(),
            "metadata": metadata,
        }
        payload["tensor_hashes"] = {
            "sample_ids": tensor_sha256(payload["sample_ids"]),
            "values": tensor_sha256(payload["values"]),
        }
        require_cpu_tensors(
            payload,
            boundary="feature-state archive publication",
            name="state",
        )

        if self.enabled:
            def writer(root: Path) -> Iterable[str]:
                write_strict_json(root / "identity.json", identity)
                write_strict_json(
                    root / "metadata.json",
                    {
                        "schema_version": "pol-feature-state-metadata-v2",
                        **execution_device_policy(),
                        "cache_id": ref.artifact_id,
                        "solver": metadata,
                        "tensor_hashes": payload["tensor_hashes"],
                    },
                )
                atomic_torch_save(root / "state.pt", payload)
                return "identity.json", "metadata.json", "state.pt"

            self.store.publish(ref, identity=identity, writer=writer, force=False)
            verify_artifact(ref.path)
            path: Path | None = ref.path
        else:
            path = None
        state = FeatureState(
            sample_ids=payload["sample_ids"],
            values=payload["values"],
            metadata=metadata,
            cache_id=ref.artifact_id,
            cache_path=path,
        )
        self.memory[ref.artifact_id] = state
        self.misses += 1
        return state

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "memory_entries": len(self.memory)}
