from __future__ import annotations

import itertools
import math
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from pol.learning.random_features import RandomFeatureMap
from pol.learning.ridge import fit_centered_affine_ridge
from pol.study import readouts as readout_module
from pol.study.evaluation import (
    evaluate_coefficients,
    mean_metrics,
    prefix_metrics,
)
from pol.study.readouts import serialize_affine
from pol.study.trial import TrialEngine
from tests.test_random_feature_evaluation import (
    _readout_characterization_trial_and_dataset,
    _StaticCache,
)


def _engine() -> tuple[Any, Any, TrialEngine]:
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
    return trial, dataset, engine


def test_only_selected_candidate_evaluation_members_are_fit_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trial, _, engine = _engine()
    fit_count = 0
    evaluation_seed_creations = {21: 0, 22: 0}
    original_fit = readout_module.fit_centered_affine_ridge
    original_create = RandomFeatureMap.create

    def counted_fit(*args: Any, **kwargs: Any) -> Any:
        nonlocal fit_count
        fit_count += 1
        return original_fit(*args, **kwargs)

    def counted_create(*args: Any, **kwargs: Any) -> RandomFeatureMap:
        seed = int(kwargs["seed"])
        if seed in evaluation_seed_creations:
            evaluation_seed_creations[seed] += 1
        return original_create(*args, **kwargs)

    monkeypatch.setattr(
        readout_module,
        "fit_centered_affine_ridge",
        counted_fit,
    )
    monkeypatch.setattr(
        RandomFeatureMap,
        "create",
        staticmethod(counted_create),
    )
    first = engine.evaluate_selection(trial)
    second_trial = trial.model_copy(
        update={
            "feature": trial.feature.model_copy(update={"n_sur": 6}),
        }
    )
    second = engine.evaluate_selection(second_trial)
    selection_fit_count = fit_count

    assert evaluation_seed_creations == {21: 0, 22: 0}
    assert "members" not in first.selection_models["random"]
    assert "members" not in second.selection_models["random"]

    selected_model = engine.materialize_selected_readout(
        first,
        readout_id="random",
    )
    assert fit_count - selection_fit_count == 2
    assert evaluation_seed_creations == {21: 1, 22: 1}
    assert [member["seed"] for member in selected_model["members"]] == [
        21,
        22,
    ]

    assert (
        engine.materialize_selected_readout(first, readout_id="random")
        is selected_model
    )
    assert fit_count - selection_fit_count == 2
    assert evaluation_seed_creations == {21: 1, 22: 1}
    assert engine.materialization_stats[
        "selected_candidate_evaluation_member_fit_count"
    ] == 2
    assert engine.materialization_stats[
        "selected_readout_materialization_cache_hit_count"
    ] == 1


def test_random_maps_and_train_validation_lifts_are_cached_across_zetas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trial, _, engine = _engine()
    created_selection_seeds: list[int] = []
    call_count = 0
    original_create = RandomFeatureMap.create
    original_call = RandomFeatureMap.__call__

    def counted_create(*args: Any, **kwargs: Any) -> RandomFeatureMap:
        seed = int(kwargs["seed"])
        if seed in {11, 12}:
            created_selection_seeds.append(seed)
        return original_create(*args, **kwargs)

    def counted_call(
        self: RandomFeatureMap,
        values: torch.Tensor,
    ) -> torch.Tensor:
        nonlocal call_count
        if self.seed in {11, 12}:
            call_count += 1
        return original_call(self, values)

    monkeypatch.setattr(
        RandomFeatureMap,
        "create",
        staticmethod(counted_create),
    )
    monkeypatch.setattr(RandomFeatureMap, "__call__", counted_call)
    engine.evaluate_selection(trial)

    structure_count = 2 * 2 * 1
    expected_maps = structure_count * 2
    assert len(created_selection_seeds) == expected_maps
    assert call_count == 2 * expected_maps
    assert len(trial.readouts[2].zetas) == 2


def test_evaluation_seed_validation_diagnostics_cannot_change_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trial, _, engine = _engine()
    selected = engine.evaluate_selection(trial)
    before_row = dict(selected.rows["random"])
    before_inner = selected.inner_selections["random"]
    original_evaluate = readout_module.evaluate_coefficients

    def injected_metrics(*args: Any, **kwargs: Any) -> dict[str, float]:
        metrics = original_evaluate(*args, **kwargs)
        metrics["field_relative_l2_mean"] = -1.0e12
        return metrics

    monkeypatch.setattr(
        readout_module,
        "evaluate_coefficients",
        injected_metrics,
    )
    model = engine.materialize_selected_readout(
        selected,
        readout_id="random",
    )

    assert selected.rows["random"] == before_row
    assert selected.inner_selections["random"] is before_inner
    assert all(
        member["evaluation_seed_validation_metrics"][
            "field_relative_l2_mean"
        ]
        == -1.0e12
        for member in model["members"]
    )
    assert model["evaluation_seed_metrics_used_for_selection"] is False


def _legacy_eager_reference(
    readout: Any,
    inputs: Any,
) -> tuple[dict[str, Any], dict[str, float]]:
    structural: list[dict[str, Any]] = []
    for width, weight_scale, bias_scale, zeta in itertools.product(
        readout.widths,
        readout.weight_scales,
        readout.bias_scales,
        readout.zetas,
    ):
        seed_metrics: list[dict[str, float]] = []
        for seed in readout.selection_seeds:
            random_map = RandomFeatureMap.create(
                inputs.x_train.shape[1],
                int(width),
                activation=readout.activation,
                seed=int(seed),
                weight_scale=float(weight_scale),
                bias_scale=float(bias_scale),
                dtype=inputs.x_train.dtype,
                device=inputs.x_train.device,
            )
            fitted = fit_centered_affine_ridge(
                random_map(inputs.x_train),
                inputs.y_train,
                float(zeta),
                svd_rcond=readout.svd_rcond,
            )
            seed_metrics.append(
                prefix_metrics(
                    evaluate_coefficients(
                        fitted(random_map(inputs.x_validation)),
                        inputs.y_validation,
                        inputs.data_target_validation,
                        inputs.reference_target_validation,
                        n_tar=inputs.n_tar,
                        n_ref=inputs.n_ref,
                        domain_length=inputs.domain_length,
                    ),
                    "validation",
                )
            )
        structural.append(
            {
                "width": int(width),
                "weight_scale": float(weight_scale),
                "bias_scale": float(bias_scale),
                "zeta": float(zeta),
                "metrics": mean_metrics(seed_metrics),
            }
        )
    metric_values = [
        float(item["metrics"][inputs.selection_metric])
        for item in structural
    ]
    best = min(metric_values)
    chosen = next(
        item
        for item in structural
        if float(item["metrics"][inputs.selection_metric])
        <= best + readout.tie_tolerance
    )
    members: list[dict[str, Any]] = []
    for seed in readout.evaluation_seeds:
        random_map = RandomFeatureMap.create(
            inputs.x_train.shape[1],
            chosen["width"],
            activation=readout.activation,
            seed=int(seed),
            weight_scale=chosen["weight_scale"],
            bias_scale=chosen["bias_scale"],
            dtype=inputs.x_train.dtype,
            device=inputs.x_train.device,
        )
        fitted = fit_centered_affine_ridge(
            random_map(inputs.x_train),
            inputs.y_train,
            chosen["zeta"],
            svd_rcond=readout.svd_rcond,
        )
        metrics = evaluate_coefficients(
            fitted(random_map(inputs.x_validation)),
            inputs.y_validation,
            inputs.data_target_validation,
            inputs.reference_target_validation,
            n_tar=inputs.n_tar,
            n_ref=inputs.n_ref,
            domain_length=inputs.domain_length,
        )
        members.append(
            {
                **serialize_affine(fitted, zeta=chosen["zeta"]),
                "seed": int(seed),
                "A": random_map.A.detach().cpu(),
                "c": random_map.c.detach().cpu(),
                "evaluation_seed_validation_metrics": metrics,
            }
        )
    return {
        "kind": "random_feature_ridge",
        "activation": readout.activation,
        "width": chosen["width"],
        "weight_scale": chosen["weight_scale"],
        "bias_scale": chosen["bias_scale"],
        "zeta": chosen["zeta"],
        "members_materialized": True,
        "materialization_split": "train_validation_only",
        "evaluation_seed_metrics_used_for_selection": False,
        "members": members,
    }, chosen["metrics"]


def test_lazy_path_matches_same_runtime_legacy_eager_reference() -> None:
    trial, _, engine = _engine()
    selected = engine.evaluate_selection(trial)
    readout = trial.readouts[2]
    inputs = engine._selection_fit_inputs[selected.candidate_id]
    legacy, legacy_validation = _legacy_eager_reference(readout, inputs)
    lazy = engine.materialize_selected_readout(
        selected,
        readout_id="random",
    )

    structure_fields = (
        "width",
        "weight_scale",
        "bias_scale",
        "zeta",
    )
    assert {
        key: lazy[key] for key in structure_fields
    } == {
        key: legacy[key] for key in structure_fields
    }
    assert selected.rows["random"][
        "validation_field_relative_l2_mean"
    ] == pytest.approx(
        legacy_validation["validation_field_relative_l2_mean"],
        rel=0.0,
        abs=0.0,
    )
    for lazy_member, legacy_member in zip(
        lazy["members"],
        legacy["members"],
        strict=True,
    ):
        assert lazy_member.keys() == legacy_member.keys()
        for key in lazy_member:
            if isinstance(lazy_member[key], torch.Tensor):
                assert torch.equal(lazy_member[key], legacy_member[key])
            elif isinstance(lazy_member[key], float):
                assert math.isclose(
                    lazy_member[key],
                    legacy_member[key],
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
            else:
                assert lazy_member[key] == legacy_member[key]

    lazy_test = engine.evaluate_test(
        trial,
        lazy,
        readout_id="random",
        candidate_id=selected.candidate_id,
    )
    legacy_test = engine.evaluate_test(
        trial,
        legacy,
        readout_id="random",
        candidate_id=selected.candidate_id,
    )
    assert lazy_test.primary_row == legacy_test.primary_row
    assert lazy_test.seed_rows == legacy_test.seed_rows
    assert lazy_test.ensemble_row == legacy_test.ensemble_row
