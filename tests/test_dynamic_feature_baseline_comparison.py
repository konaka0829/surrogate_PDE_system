from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from pol.config.loader import load_study_spec
from pol.config.models import StudySpec, TrialSpec
from pol.data.dataset import ReferenceDataset
from pol.study.cache import FeatureStateCache
from pol.runtime.artifacts import manifest_records
from pol.runtime.io import write_csv, write_strict_json
from pol.study.runner import run_study, verify_study_run
from tests.helpers import write_tiny_stack


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "name",
    [
        "dynamic_feature_baseline_comparison_smoke.json",
        "dynamic_feature_baseline_comparison.json",
    ],
)
def test_four_feature_families_share_one_budget_and_readout_contract(
    name: str,
) -> None:
    spec = load_study_spec(ROOT / "studies" / name, repo_root=ROOT)
    assert spec.comparison is not None
    assert set(spec.comparison.feature_families.values()) == {
        "static_input",
        "heat",
        "burgers",
        "reaction_diffusion",
    }
    assert not spec.global_axes
    assert all(variant.search.kind == "static" for variant in spec.variants)
    assert all(
        variant.selection_source is not None
        for variant in spec.variants
        if variant.id != "static_input"
    )


def test_comparison_contract_rejects_budget_override_and_unknown_key() -> None:
    raw = json.loads(
        (
            ROOT
            / "studies"
            / "dynamic_feature_baseline_comparison_smoke.json"
        ).read_text(encoding="utf-8")
    )
    raw["variants"][0]["overrides"]["feature.observation.J"] = 8
    with pytest.raises(ValidationError, match="information budget"):
        StudySpec.model_validate(raw)

    clean = copy.deepcopy(raw)
    clean["variants"][0]["overrides"].pop("feature.observation.J")
    clean["comparison"]["unknown_scientific_key"] = True
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        StudySpec.model_validate(clean)


def _dataset(inputs: torch.Tensor) -> ReferenceDataset:
    ids = torch.tensor([0, 1, 2], dtype=torch.long)
    return ReferenceDataset(
        artifact_id="finite-boundary-dataset",
        path=Path("."),
        sample_ids=ids,
        inputs_reference=inputs,
        targets_reference=torch.zeros_like(inputs),
        train_ids=ids[:1],
        validation_ids=ids[1:2],
        test_ids=ids[2:],
        reference_nx=inputs.shape[-1],
        domain_length=1.0,
        dtype_name="float64",
        target_metadata={"kind": "synthetic"},
        split_hash="split",
        validation_artifact_id="validation",
        binding_kind="foundation_only",
        binding_status="pass",
        target_reference_validation_status="foundation_only",
        binding_proof={},
        binding_proof_hash="proof",
    )


def test_static_feature_cannot_observe_discarded_reference_modes(
    tmp_path: Path,
) -> None:
    trial = TrialSpec.model_validate(
        {
            "input": {"n_tar": 8},
            "feature": {
                "kind": "static_input",
                "evolution": None,
                "n_sur": 16,
                "observation": {"J": 8},
            },
            "output": {"q": 5},
            "readouts": [
                {"id": "direct", "kind": "direct_fourier_decoder"}
            ],
        }
    )
    grid = torch.arange(16, dtype=torch.float64) / 16
    low = torch.sin(2 * torch.pi * grid).repeat(3, 1)
    discarded = 0.7 * torch.cos(2 * torch.pi * 6 * grid).repeat(3, 1)
    ids = torch.tensor([0, 1, 2], dtype=torch.long)
    baseline = FeatureStateCache(
        artifact_root=tmp_path / "baseline",
        enabled=False,
        batch_size=3,
    ).get_or_solve(_dataset(low), ids, trial)
    perturbed = FeatureStateCache(
        artifact_root=tmp_path / "perturbed",
        enabled=False,
        batch_size=3,
    ).get_or_solve(_dataset(low + discarded), ids, trial)
    assert torch.allclose(baseline.values, perturbed.values, atol=1e-12, rtol=0)
    assert baseline.metadata["kind"] == "static_input"
    assert baseline.metadata["solver"] == "none"


def test_selected_comparison_table_is_exact_byte_and_semantic_verified(
    tmp_path: Path,
) -> None:
    _, _, study_path = write_tiny_stack(
        tmp_path,
        include_diagnostics=False,
    )
    result = run_study(
        load_study_spec(study_path, repo_root=tmp_path),
        repo_root=tmp_path,
    )
    table_path = result.path / "selected_comparison.csv"
    with table_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["test_field_relative_l2_mean"] = "123.0"
    write_csv(table_path, rows, fieldnames=rows[0].keys())
    manifest_path = result.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for index, record in enumerate(manifest["files"]):
        if record["relative_path"] == "selected_comparison.csv":
            manifest["files"][index] = manifest_records(
                result.path,
                ["selected_comparison.csv"],
            )[0]
            break
    write_strict_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="selected comparison table"):
        verify_study_run(result.path)
