from __future__ import annotations

import math
from pathlib import Path
from statistics import fmean, stdev
from types import SimpleNamespace
from typing import Any, Mapping

import pytest
from scipy.stats import t as student_t
import torch

from pol.config.models import MetricCurveReporterSpec, TrialSpec
from pol.data.dataset import ReferenceDataset
from pol.numerics.initial_conditions import GRF_SAMPLER_SEMANTICS
from pol.plotting.reporters import generate_reporters
from pol.runtime.device import execution_device_policy
from pol.runtime.hashing import stable_object_hash
from pol.study.evaluation import summarize_independent_seed_metrics
from pol.study.trial import TrialEngine


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
        split_hash="d693c22500c07511a76bfb36f5b8227616c87692c8f5448be32b3538412ffc99",
        validation_artifact_id="synthetic-validation",
        binding_kind="foundation_only",
        binding_status="pass",
        target_reference_validation_status="not_claimed",
        binding_proof={
            "schema_version": "pol-dataset-binding-proof-v4",
            **execution_device_policy(),
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
        **execution_device_policy(),
    )
    return trial, dataset


def _readout_characterization_trial_and_dataset() -> tuple[
    TrialSpec, ReferenceDataset
]:
    trial = TrialSpec.model_validate(
        {
            "input": {"n_tar": 5},
            "feature": {
                "kind": "static_input",
                "n_sur": 4,
                "observation": {"J": 3},
            },
            "output": {"q": 5},
            "readouts": [
                {
                    "id": "direct",
                    "kind": "direct_fourier_decoder",
                },
                {
                    "id": "affine",
                    "kind": "affine_ridge",
                    "zetas": [0.0, 0.5],
                    "tie_break": "largest_zeta",
                },
                {
                    "id": "random",
                    "kind": "random_feature_ridge",
                    "widths": [1, 2],
                    "weight_scales": [0.0, 0.5],
                    "bias_scales": [0.0],
                    "selection_seeds": [11, 12],
                    "evaluation_seeds": [21, 22],
                    "zetas": [0.0, 0.5],
                },
            ],
        }
    )
    values = torch.zeros((5, 5), dtype=torch.float64)
    dataset = ReferenceDataset(
        artifact_id="readout-characterization-dataset",
        path=Path("."),
        sample_ids=torch.arange(5, dtype=torch.long),
        inputs_reference=values,
        targets_reference=values,
        train_ids=torch.tensor([0, 1], dtype=torch.long),
        validation_ids=torch.tensor([2], dtype=torch.long),
        test_ids=torch.tensor([3, 4], dtype=torch.long),
        reference_nx=5,
        domain_length=1.0,
        dtype_name="float64",
        target_metadata={"kind": "heat"},
        split_hash="readout-characterization-split",
        validation_artifact_id="readout-characterization-validation",
        binding_kind="foundation_only",
        binding_status="pass",
        target_reference_validation_status="not_claimed",
        binding_proof={},
        binding_proof_hash="readout-characterization-proof",
        **execution_device_policy(),
    )
    return trial, dataset


def _canonical_runtime_payload(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": value.tolist(),
        }
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_runtime_payload(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_runtime_payload(item) for item in value]
    return value


def test_trial_readout_validation_freeze_and_test_characterization() -> None:
    trial, dataset = _readout_characterization_trial_and_dataset()
    engine = TrialEngine(
        dataset,
        SimpleNamespace(
            selection=SimpleNamespace(
                metric="validation_field_relative_l2_mean"
            )
        ),
        _StaticCache(),
    )
    selected = engine.evaluate_selection(trial)
    evaluated = {
        readout_id: engine.evaluate_test(
            trial,
            model,
            readout_id=readout_id,
            candidate_id=selected.candidate_id,
        )
        for readout_id, model in selected.frozen_models.items()
    }

    snapshot = {
        "candidate_id": selected.candidate_id,
        "validation_rows": {
            key: stable_object_hash(value) for key, value in selected.rows.items()
        },
        "inner_selections": {
            key: stable_object_hash(value)
            for key, value in selected.inner_selections.items()
        },
        "frozen_models": {
            key: stable_object_hash(_canonical_runtime_payload(value))
            for key, value in selected.frozen_models.items()
        },
        "test_primary_rows": {
            key: stable_object_hash(value.primary_row)
            for key, value in evaluated.items()
        },
        "test_seed_rows": {
            key: stable_object_hash(value.seed_rows)
            for key, value in evaluated.items()
        },
        "test_ensemble_rows": {
            key: (
                None
                if value.ensemble_row is None
                else stable_object_hash(value.ensemble_row)
            )
            for key, value in evaluated.items()
        },
    }
    assert snapshot == {
        "candidate_id": (
            "809715dc4e19479b6756069d03be578603057200e00ebc3b4c03749e6166b5f1"
        ),
        "validation_rows": {
            "direct": (
                "5a0dc84a27b858f9fd9b294f85e73d5ebd01de9b4c44da1e5b716b4c30fa0c6d"
            ),
            "affine": (
                "c5f4f6a1c4df0ad87f5fa726bba1a57aa722673f15c9f659f2553ca26f405a53"
            ),
            "random": (
                "d51aea089443026512e03b5796861ada0d65273cf2c76a0f039f7cb90210971d"
            ),
        },
        "inner_selections": {
            "direct": (
                "4e0b060ca8045c7df0293e3e48b17120b8533d0f996b8de8807ae6e8ad3327c4"
            ),
            "affine": (
                "47877ad05be0395116e8753dcd4414fbf631086788a7e880180ca9eaee68de04"
            ),
            "random": (
                "0b38790f0cce002aadfe8e366a5137cb6585f0177d72dbd61485bad5bce51e5c"
            ),
        },
        "frozen_models": {
            "direct": (
                "a782f925021fa802b46f456aa462ffd07ad6dd84bd9e439a6e7e05f619119638"
            ),
            "affine": (
                "c9db7e4eb4918d5880f7f2a3f0525424e343bd5c331ab6e605c772647936a7aa"
            ),
            "random": (
                "944993ba74e142cbafa263073fd5eec211a49dffbc5550f14efdfc9349ee919b"
            ),
        },
        "test_primary_rows": {
            "direct": (
                "97969c0ef21c821e02504e41a0f397365caa5648b38a8611bef1a4bdb1df1784"
            ),
            "affine": (
                "4ef407ed2173b25e50798ceee19df62d64ba23dee4f3133128c6a3ec04eca1d4"
            ),
            "random": (
                "89eb44f2a2450f7a71c9f89fc44d785051374d6395b2ea0841efaab7f60a7cf0"
            ),
        },
        "test_seed_rows": {
            "direct": (
                "37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570"
            ),
            "affine": (
                "37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570"
            ),
            "random": (
                "cd11eea281548123cdbeb3e6244e1d7d74c5332fcc5144eb186c249c63226671"
            ),
        },
        "test_ensemble_rows": {
            "direct": None,
            "affine": None,
            "random": (
                "39ef9ff0c2923f215130fcff01fbc60dff4b212d71773fc502660a56d7fbee5d"
            ),
        },
    }

    direct_row = selected.rows["direct"]
    assert direct_row["decoder_observation_count"] == 3
    assert direct_row["decoder_requested_q"] == 5
    assert direct_row["decoder_zero_filled_coefficient_count"] == 2
    assert direct_row["decoder_zero_fill_applied"] is True
    assert selected.inner_selections["affine"]["zeta"] == 0.5
    assert selected.inner_selections["random"]["width"] == 1
    assert selected.inner_selections["random"]["weight_scale"] == 0.0
    assert selected.inner_selections["random"]["bias_scale"] == 0.0
    assert selected.inner_selections["random"]["zeta"] == 0.0
    assert [
        (
            item["width"],
            item["weight_scale"],
            item["bias_scale"],
            item["zeta"],
        )
        for item in selected.inner_selections["random"]["candidate_metrics"]
    ] == [
        (1, 0.0, 0.0, 0.0),
        (1, 0.0, 0.0, 0.5),
        (1, 0.5, 0.0, 0.0),
        (1, 0.5, 0.0, 0.5),
        (2, 0.0, 0.0, 0.0),
        (2, 0.0, 0.0, 0.5),
        (2, 0.5, 0.0, 0.0),
        (2, 0.5, 0.0, 0.5),
    ]
    assert {
        int(member["seed"])
        for member in selected.frozen_models["random"]["members"]
    } == {21, 22}
    for model in selected.frozen_models.values():
        for item in _iter_tensors(model):
            assert item.device.type == "cpu"
            assert torch.equal(item, torch.zeros_like(item))

    for readout_id in ("direct", "affine"):
        result = evaluated[readout_id]
        assert result.primary_row["test_result_kind"] == "single_model"
        assert result.seed_rows == ()
        assert result.ensemble_row is None
        assert not any(
            key.startswith("test_seed_") or "_seed_" in key
            for key in result.primary_row
        )
    random_result = evaluated["random"]
    assert random_result.primary_row["test_result_kind"] == (
        "independent_seed_metric_summary"
    )
    assert [row["seed"] for row in random_result.seed_rows] == [21, 22]
    assert random_result.ensemble_row is not None
    assert random_result.ensemble_row["test_result_kind"] == (
        "prediction_ensemble"
    )


def _iter_tensors(value: Any):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)


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
