from __future__ import annotations

import math
from pathlib import Path
from statistics import fmean, stdev
from types import SimpleNamespace

import pytest
from scipy.stats import t as student_t
import torch

from pol.config.models import MetricCurveReporterSpec, TrialSpec
from pol.data.dataset import ReferenceDataset
from pol.numerics.initial_conditions import GRF_SAMPLER_SEMANTICS
from pol.plotting.reporters import generate_reporters
from pol.study.trial import TrialEngine, summarize_independent_seed_metrics


@pytest.mark.parametrize(
    "values",
    [
        [1.0, 3.0],
        [0.2, 0.5, 0.9, 1.4, 2.1, 2.2, 3.0, 3.7, 4.1, 5.0],
    ],
)
def test_seed_metric_summary_matches_reference_student_t_calculation(
    values: list[float],
) -> None:
    summary = summarize_independent_seed_metrics(
        [{"test_metric": value} for value in values]
    )
    expected_mean = fmean(values)
    expected_std = stdev(values)
    critical = float(student_t.ppf(0.975, len(values) - 1))
    margin = critical * expected_std / math.sqrt(len(values))

    assert summary["test_metric"] == pytest.approx(expected_mean)
    assert summary["test_metric_seed_mean"] == pytest.approx(expected_mean)
    assert summary["test_metric_seed_std"] == pytest.approx(expected_std)
    assert summary["test_metric_seed_ci95_low"] == pytest.approx(
        expected_mean - margin
    )
    assert summary["test_metric_seed_ci95_high"] == pytest.approx(
        expected_mean + margin
    )
    if len(values) == 2:
        assert summary["test_metric_seed_ci95_low"] < 0.0


class _StaticCache:
    def get_or_solve(self, dataset, ids, trial):
        return SimpleNamespace(
            values=torch.zeros((ids.numel(), 4), dtype=torch.float64),
            cache_id="synthetic-feature-state",
        )


def _synthetic_trial_and_dataset() -> tuple[TrialSpec, ReferenceDataset]:
    trial = TrialSpec.model_validate(
        {
            "input": {"n_tar": 4},
            "feature": {
                "kind": "static_input",
                "n_sur": 4,
                "observation": {"J": 1},
            },
            "output": {"q": 3},
            "readouts": [
                {
                    "id": "random",
                    "kind": "random_feature_ridge",
                    "widths": [1],
                    "weight_scales": [0.0],
                    "bias_scales": [0.0],
                    "selection_seeds": [11],
                    "evaluation_seeds": [21, 22],
                    "zetas": [0.0],
                }
            ],
        }
    )
    values = torch.zeros((3, 4), dtype=torch.float64)
    dataset = ReferenceDataset(
        artifact_id="synthetic-dataset",
        path=Path("."),
        sample_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        inputs_reference=values,
        targets_reference=values,
        train_ids=torch.tensor([0], dtype=torch.long),
        validation_ids=torch.tensor([1], dtype=torch.long),
        test_ids=torch.tensor([2], dtype=torch.long),
        reference_nx=4,
        domain_length=1.0,
        dtype_name="float64",
        target_metadata={"kind": "heat"},
        split_hash="synthetic-split",
        validation_artifact_id="synthetic-validation",
        binding_kind="foundation_only",
        binding_status="pass",
        target_reference_validation_status="not_claimed",
        binding_proof={
            "schema_version": "pol-dataset-binding-proof-v2",
            "binding_kind": "foundation_only",
            "status": "pass",
            "target_reference_validation_status": "not_claimed",
            "certificate_artifact_id": "synthetic-validation",
            "grf_sampler_domain_length": 1.0,
            "grf_sampler_semantics": GRF_SAMPLER_SEMANTICS,
            "dataset_condition": {"domain_length": 1.0},
            "proof_hash": "synthetic-proof",
        },
        binding_proof_hash="synthetic-proof",
    )
    return trial, dataset


def test_primary_seed_metric_mean_differs_from_prediction_ensemble_metric() -> None:
    trial, dataset = _synthetic_trial_and_dataset()
    model = {
        "kind": "random_feature_ridge",
        "activation": "identity",
        "width": 1,
        "weight_scale": 0.0,
        "bias_scale": 0.0,
        "zeta": 0.0,
        "members": [
            {
                "seed": 21,
                "A": torch.zeros((1, 1), dtype=torch.float64),
                "c": torch.zeros(1, dtype=torch.float64),
                "W": torch.zeros((3, 2), dtype=torch.float64),
                "b": torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
            },
            {
                "seed": 22,
                "A": torch.zeros((1, 1), dtype=torch.float64),
                "c": torch.zeros(1, dtype=torch.float64),
                "W": torch.zeros((3, 2), dtype=torch.float64),
                "b": torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float64),
            },
        ],
    }
    evaluation = TrialEngine(
        dataset,
        SimpleNamespace(),
        _StaticCache(),
    ).evaluate_test(
        trial,
        model,
        readout_id="random",
        candidate_id="synthetic-candidate",
    )

    assert evaluation.primary_row["test_result_kind"] == (
        "independent_seed_metric_summary"
    )
    assert evaluation.primary_row["test_coefficient_mse"] == pytest.approx(1.0 / 3.0)
    assert evaluation.primary_row["test_coefficient_mse_seed_mean"] == pytest.approx(
        1.0 / 3.0
    )
    assert evaluation.ensemble_row is not None
    assert evaluation.ensemble_row["test_result_kind"] == "prediction_ensemble"
    assert evaluation.ensemble_row["test_ensemble_coefficient_mse"] == pytest.approx(
        0.0
    )
    assert {row["seed"] for row in evaluation.seed_rows} == {21, 22}


def test_test_reporter_reads_random_feature_canonical_seed_mean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_y: list[float] = []
    from matplotlib.axes import Axes

    original_plot = Axes.plot

    def capture_plot(self, x, y, *args, **kwargs):
        captured_y.extend(float(value) for value in y)
        return original_plot(self, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "plot", capture_plot)
    reporter = MetricCurveReporterSpec(
        filename="seed_mean",
        x="q",
        metric="test_field_relative_l2_mean",
        split="test",
        group_by=("variant_id", "readout_id"),
        formats=("png",),
    )
    row = {
        "variant_id": "synthetic",
        "readout_id": "random",
        "q": 1,
        "test_result_kind": "independent_seed_metric_summary",
        "test_field_relative_l2_mean": 2.5,
        "test_field_relative_l2_mean_seed_mean": 2.5,
    }
    created = generate_reporters(
        [reporter],
        validation_rows=[],
        test_rows=[row],
        noise_rows=[],
        output_dir=tmp_path,
    )

    assert created == ["seed_mean.png"]
    assert captured_y == [2.5]
