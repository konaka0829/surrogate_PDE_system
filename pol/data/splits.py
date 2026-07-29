"""Deterministic train/validation/test split ownership and provenance."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from pol.runtime.hashing import stable_object_hash


SPLIT_POLICY = "cpu_torch_randperm"
SPLIT_POLICY_VERSION = 1


def _require_int(name: str, value: object) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    return value


@dataclass(frozen=True)
class DataSplit:
    total_samples: int
    n_train: int
    n_validation: int
    n_test: int
    seed: int
    train_ids: torch.Tensor
    validation_ids: torch.Tensor
    test_ids: torch.Tensor

    def canonical_payload(self, sample_ids: torch.Tensor) -> dict[str, list[int]]:
        return canonical_split_payload(sample_ids, self)

    def split_hash(self, sample_ids: torch.Tensor) -> str:
        return stable_object_hash(self.canonical_payload(sample_ids))


def build_data_split(
    *,
    total_samples: int,
    n_train: int,
    n_validation: int,
    n_test: int,
    seed: int,
) -> DataSplit:
    """Reproduce the established CPU ``torch.randperm`` split exactly."""
    total_samples = _require_int("total_samples", total_samples)
    n_train = _require_int("n_train", n_train)
    n_validation = _require_int("n_validation", n_validation)
    n_test = _require_int("n_test", n_test)
    seed = _require_int("seed", seed)
    if total_samples <= 0:
        raise ValueError("total_samples must be positive")
    if n_train <= 0 or n_validation < 0 or n_test < 0:
        raise ValueError(
            "n_train must be positive and n_validation/n_test must be non-negative"
        )
    if total_samples != n_train + n_validation + n_test:
        raise ValueError(
            "total_samples must equal n_train + n_validation + n_test"
        )

    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(
        total_samples,
        generator=generator,
        dtype=torch.long,
    )
    train_ids = permutation[:n_train].clone()
    validation_ids = permutation[n_train : n_train + n_validation].clone()
    test_ids = permutation[n_train + n_validation :].clone()
    split_tensors = (train_ids, validation_ids, test_ids)
    if any(
        value.device != torch.device("cpu")
        or value.ndim != 1
        or value.dtype != torch.long
        for value in split_tensors
    ):
        raise RuntimeError("deterministic split must produce one-dimensional CPU longs")
    assigned = torch.cat(split_tensors)
    expected = torch.arange(total_samples, dtype=torch.long)
    if (
        assigned.numel() != total_samples
        or torch.unique(assigned).numel() != total_samples
        or not torch.equal(torch.sort(assigned).values, expected)
    ):
        raise RuntimeError("deterministic split is not a pairwise-disjoint full cover")

    return DataSplit(
        total_samples=total_samples,
        n_train=n_train,
        n_validation=n_validation,
        n_test=n_test,
        seed=seed,
        train_ids=train_ids,
        validation_ids=validation_ids,
        test_ids=test_ids,
    )


def canonical_split_payload(
    sample_ids: torch.Tensor,
    split: DataSplit,
) -> dict[str, list[int]]:
    if (
        not isinstance(sample_ids, torch.Tensor)
        or sample_ids.device != torch.device("cpu")
        or sample_ids.ndim != 1
        or sample_ids.dtype != torch.long
    ):
        raise ValueError("sample_ids must be a one-dimensional CPU long tensor")
    if sample_ids.numel() != split.total_samples:
        raise ValueError("sample_ids length does not match total_samples")
    if (
        torch.unique(sample_ids).numel() != split.total_samples
        or set(sample_ids.tolist()) != set(range(split.total_samples))
    ):
        raise ValueError("sample_ids must contain each deterministic sample ID once")
    return {
        "sample_ids": sample_ids.tolist(),
        "train_ids": split.train_ids.tolist(),
        "validation_ids": split.validation_ids.tolist(),
        "test_ids": split.test_ids.tolist(),
    }


def split_contract(
    split: DataSplit,
    *,
    sample_ids: torch.Tensor,
) -> dict[str, int | str]:
    return {
        "split_policy": SPLIT_POLICY,
        "split_policy_version": SPLIT_POLICY_VERSION,
        "total_samples": split.total_samples,
        "n_train": split.n_train,
        "n_validation": split.n_validation,
        "n_test": split.n_test,
        "seed": split.seed,
        "split_hash": split.split_hash(sample_ids),
    }


def calibration_split_provenance(
    split: DataSplit,
    calibration_sample_ids: Iterable[int],
    *,
    sample_ids: torch.Tensor,
) -> dict[str, object]:
    calibration_ids = tuple(calibration_sample_ids)
    if not calibration_ids:
        raise ValueError("calibration_sample_ids must not be empty")
    if any(type(value) is not int for value in calibration_ids):
        raise TypeError("calibration_sample_ids must contain integers")
    if len(set(calibration_ids)) != len(calibration_ids):
        raise ValueError("calibration_sample_ids must be unique")
    if any(value < 0 or value >= split.total_samples for value in calibration_ids):
        raise ValueError("calibration_sample_ids must lie in the dataset range")

    train = set(split.train_ids.tolist())
    validation = set(split.validation_ids.tolist())
    test = set(split.test_ids.tolist())
    overlap = sorted(set(calibration_ids) & test)
    if overlap:
        raise ValueError(
            "calibration_sample_ids overlap the deterministic test split: "
            f"offending calibration IDs={overlap}; "
            f"split seed={split.seed}; "
            "train/validation/test counts="
            f"{split.n_train}/{split.n_validation}/{split.n_test}; "
            "calibration IDs must belong to train or validation splits"
        )
    membership = {
        str(sample_id): (
            "train"
            if sample_id in train
            else "validation"
            if sample_id in validation
            else "test"
        )
        for sample_id in calibration_ids
    }
    if set(membership.values()) - {"train", "validation"}:
        raise RuntimeError("calibration split membership is incomplete")
    return {
        **split_contract(split, sample_ids=sample_ids),
        "calibration_sample_ids": list(calibration_ids),
        "calibration_split_membership": membership,
        "calibration_test_overlap_count": 0,
    }


__all__ = [
    "DataSplit",
    "SPLIT_POLICY",
    "SPLIT_POLICY_VERSION",
    "build_data_split",
    "calibration_split_provenance",
    "canonical_split_payload",
    "split_contract",
]
