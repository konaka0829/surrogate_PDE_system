from __future__ import annotations

import json
from typing import Any, Mapping

import torch

from pol.config.models import TrialSpec
from pol.runtime.device import require_cpu_tensor
from pol.runtime.hashing import stable_object_hash, tensor_sha256


TRAINING_SUBSET_SCHEMA_VERSION = "pol-training-subset-v1"
TRAINING_SUBSET_POLICY = "canonical_train_order_prefix_v1"


def resolve_training_subset(
    dataset: Any,
    trial: TrialSpec,
) -> tuple[torch.Tensor, dict[str, Any]]:
    parent = dataset.train_ids.to(torch.long)
    validation = dataset.validation_ids.to(torch.long)
    test = dataset.test_ids.to(torch.long)
    for name, values in (
        ("train_ids", parent),
        ("validation_ids", validation),
        ("test_ids", test),
    ):
        require_cpu_tensor(
            values,
            boundary="training-subset resolution",
            name=name,
        )
    requested = (
        int(parent.numel())
        if trial.training_subset is None
        else int(trial.training_subset.n_train)
    )
    if requested < 1 or requested > int(parent.numel()):
        raise ValueError(
            "nested training subset size must be between 1 and the canonical "
            f"train count {int(parent.numel())}; got {requested}"
        )
    ids = parent[:requested].clone()
    if torch.unique(ids).numel() != ids.numel():
        raise ValueError("canonical training IDs are not unique")
    if any(
        bool(torch.isin(ids, other).any())
        for other in (validation, test)
    ):
        raise ValueError(
            "nested training subset overlaps validation or test IDs"
        )
    unsigned = {
        "schema_version": TRAINING_SUBSET_SCHEMA_VERSION,
        "kind": "nested_train_prefix",
        "policy": TRAINING_SUBSET_POLICY,
        "policy_version": 1,
        "n_train": requested,
        "subset_ids": [int(value) for value in ids.tolist()],
        "subset_ids_hash": tensor_sha256(ids),
        "parent_train_ids_hash": tensor_sha256(parent),
        "parent_train_count": int(parent.numel()),
        "validation_ids_hash": tensor_sha256(validation),
    }
    return ids, {
        **unsigned,
        "training_subset_hash": stable_object_hash(unsigned),
    }


def training_subset_result_fields(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "n_train": int(record["n_train"]),
        "training_subset_kind": str(record["kind"]),
        "training_subset_policy": str(record["policy"]),
        "training_subset_policy_version": int(record["policy_version"]),
        "training_subset_ids": json.dumps(
            list(record["subset_ids"]),
            separators=(",", ":"),
        ),
        "training_subset_ids_hash": str(record["subset_ids_hash"]),
        "training_subset_hash": str(record["training_subset_hash"]),
        "parent_train_ids_hash": str(record["parent_train_ids_hash"]),
        "parent_train_count": int(record["parent_train_count"]),
        "validation_ids_hash": str(record["validation_ids_hash"]),
    }
