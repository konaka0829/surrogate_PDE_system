from __future__ import annotations

import json
from pathlib import Path

import torch
import pytest

from pol.config.loader import load_dataset_spec, load_validation_spec
from pol.data.dataset import ensure_dataset, load_dataset
from pol.data.finite import build_feature_initial_state, derive_finite_view
from pol.math.periodic import periodic_grid, spectral_resample_periodic
from pol.validation.runner import ensure_validation
from tests.helpers import write_tiny_stack


def test_foundation_validation_publishes_passing_certificate(tmp_path: Path) -> None:
    validation_path, _, _ = write_tiny_stack(tmp_path)
    spec = load_validation_spec(validation_path, repo_root=tmp_path)
    outcome = ensure_validation(spec)
    assert outcome.certificate["status"] == "pass"
    checks = json.loads((outcome.reference.path / "checks.json").read_text(encoding="utf-8"))
    assert checks["finite_input_interface"]["dimension_independence"]["n_tar_le_J_exercised"]
    assert (outcome.reference.path / "master_initial_conditions.pt").is_file()


def test_reference_dataset_reuses_validation_and_has_disjoint_splits(tmp_path: Path) -> None:
    _, dataset_path, _ = write_tiny_stack(tmp_path)
    spec = load_dataset_spec(dataset_path, repo_root=tmp_path)
    dataset = ensure_dataset(spec, repo_root=tmp_path)
    reloaded = load_dataset(dataset.path)
    train = set(reloaded.train_ids.tolist())
    validation = set(reloaded.validation_ids.tolist())
    test = set(reloaded.test_ids.tolist())
    assert train.isdisjoint(validation) and train.isdisjoint(test) and validation.isdisjoint(test)
    assert train | validation | test == set(reloaded.sample_ids.tolist())
    assert reloaded.inputs_reference.shape == reloaded.targets_reference.shape == (12, 32)
    assert torch.isfinite(reloaded.targets_reference).all()


def test_finite_data_resolution_cannot_exceed_validated_reference() -> None:
    values = torch.zeros(2, 16, dtype=torch.float64)
    with pytest.raises(ValueError, match="must not exceed"):
        derive_finite_view(
            torch.tensor([0, 1], dtype=torch.long),
            values,
            values,
            n_tar=32,
            q=9,
            domain_length=1.0,
        )


def test_discarded_reference_modes_do_not_reach_feature_initial_state() -> None:
    x = periodic_grid(64, 1.0)
    low = 0.3 + torch.cos(4.0 * torch.pi * x)
    differing_reference = low + 0.25 * torch.cos(40.0 * torch.pi * x)
    finite = spectral_resample_periodic(
        torch.stack([low, differing_reference]), 16, domain_length=1.0
    )
    feature_initial = build_feature_initial_state(
        finite, n_sur=32, domain_length=1.0
    )
    assert torch.allclose(finite[0], finite[1], atol=1e-11, rtol=1e-11)
    assert torch.allclose(
        feature_initial[0], feature_initial[1], atol=1e-11, rtol=1e-11
    )
