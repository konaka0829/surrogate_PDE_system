from __future__ import annotations

import json
from pathlib import Path

import torch
import pytest

from pol.config.loader import load_dataset_spec, load_validation_spec
from pol.data.dataset import ensure_dataset, load_dataset
from pol.data.finite import build_feature_initial_state, derive_finite_view
from pol.data.initial_conditions import generate_grf_archive
from pol.math.periodic import periodic_grid, spectral_resample_periodic
from pol.numerics.initial_conditions import GRF_SAMPLER_SEMANTICS
from pol.validation.runner import ensure_validation
from tests.helpers import write_json, write_tiny_stack


def test_foundation_validation_publishes_passing_certificate(tmp_path: Path) -> None:
    validation_path, _, _ = write_tiny_stack(tmp_path)
    spec = load_validation_spec(validation_path, repo_root=tmp_path)
    outcome = ensure_validation(spec)
    assert outcome.certificate["status"] == "pass"
    checks = json.loads((outcome.reference.path / "checks.json").read_text(encoding="utf-8"))
    assert checks["finite_input_interface"]["dimension_independence"]["n_tar_le_J_exercised"]
    assert (outcome.reference.path / "master_initial_conditions.pt").is_file()


def test_grf_archive_records_the_sampler_domain_and_preserves_dtype_device() -> None:
    archive = generate_grf_archive(
        total_samples=2,
        nx=9,
        seed=23,
        gamma=2.0,
        tau=5.0,
        sigma=1.0,
        mean=0.0,
        domain_length=2.0,
        dtype="float32",
        device="cpu",
    )
    assert archive.domain_length == archive.metadata["domain_length"] == 2.0
    assert archive.metadata["sampler_semantics"] == GRF_SAMPLER_SEMANTICS
    assert archive.values.dtype == torch.float32
    assert archive.values.device == torch.device("cpu")
    assert torch.equal(
        archive.fourier,
        torch.fft.rfft(archive.values, dim=-1, norm="forward"),
    )


def test_nonunit_domain_is_bound_across_grf_archive_and_certificate(
    tmp_path: Path,
) -> None:
    validation_path, _, _ = write_tiny_stack(tmp_path)
    raw = json.loads(validation_path.read_text(encoding="utf-8"))
    raw["domain"]["length"] = 2.0
    write_json(validation_path, raw)
    spec = load_validation_spec(validation_path, repo_root=tmp_path)
    outcome = ensure_validation(spec)
    root = outcome.reference.path
    resolved = json.loads(
        (root / "resolved_spec.json").read_text(encoding="utf-8")
    )
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    master = torch.load(
        root / "master_initial_conditions.pt",
        map_location="cpu",
        weights_only=True,
    )
    foundation = outcome.certificate["foundation_contract"]
    master_binding = foundation["master_initial_conditions"]

    assert manifest["identity"]["schema_version"] == "pol-validation-identity-v3"
    assert master["schema_version"] == "pol-initial-condition-archive-v3"
    assert outcome.certificate["schema_version"] == "pol-validation-certificate-v3"
    assert foundation["schema_version"] == "pol-validation-foundation-contract-v2"
    assert master_binding["schema_version"] == (
        "pol-master-initial-condition-binding-v2"
    )
    assert resolved["domain"]["length"] == 2.0
    assert manifest["identity"]["spec"]["domain"]["length"] == 2.0
    assert master["domain_length"] == 2.0
    assert master["metadata"]["domain_length"] == 2.0
    assert foundation["domain_length"] == 2.0
    assert foundation["grf_sampler_domain_length"] == 2.0
    assert master_binding["domain_length"] == 2.0
    assert master_binding["metadata"]["domain_length"] == 2.0
    assert {
        manifest["identity"]["grf_sampler_semantics"],
        master["metadata"]["sampler_semantics"],
        foundation["grf_sampler_semantics"],
        master_binding["metadata"]["sampler_semantics"],
    } == {GRF_SAMPLER_SEMANTICS}


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
