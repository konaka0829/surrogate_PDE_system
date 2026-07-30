from __future__ import annotations

from typing import Any, Mapping

import torch

from pol.config.models import StudySpec, TrialSpec
from pol.data.dataset import ReferenceDataset
from pol.data.finite import FiniteDataView, derive_finite_view
from pol.learning.observations import observe_equispaced_periodic
from pol.runtime.device import (
    require_cpu_tensors,
    verify_execution_device_policy,
)
from pol.runtime.hashing import stable_object_hash
from .cache import FeatureStateCache
from .evaluation import (
    CandidateEvaluation,
    TestEvaluation,
    build_test_evaluation,
    build_validation_evaluation,
    representation_floor,
)
from .readouts import (
    ReadoutFitInputs,
    fit_readout,
    frozen_readout_diagnostic,
    materialize_random_feature_readout,
    predict_frozen,
)
from .training_subsets import resolve_training_subset


class TrialEngine:
    """Evaluate all readouts for one validated trial.

    Selection views are constructed from train and validation sample IDs only.
    Test IDs are accepted only by :meth:`evaluate_test`, which the study runner
    calls after the frozen plan has been written and read back.
    """

    def __init__(
        self,
        dataset: ReferenceDataset,
        study: StudySpec,
        cache: FeatureStateCache,
    ) -> None:
        self.dataset = dataset
        self.study = study
        self.cache = cache
        self.selection_ids = torch.cat(
            [dataset.train_ids, dataset.validation_ids]
        ).to(torch.long)
        self.canonical_n_train = int(dataset.train_ids.numel())
        self._finite_cache: dict[tuple[str, int, int], FiniteDataView] = {}
        self._candidate_cache: dict[str, CandidateEvaluation] = {}
        self._selection_fit_inputs: dict[str, ReadoutFitInputs] = {}
        self._materialized_readouts: dict[
            tuple[str, str],
            dict[str, Any],
        ] = {}
        self.materialization_stats = {
            "selected_readout_materialization_request_count": 0,
            "selected_readout_materialization_cache_hit_count": 0,
            "selected_readout_model_count": 0,
            "selected_random_feature_model_count": 0,
            "selected_candidate_evaluation_member_fit_count": 0,
        }
        verify_execution_device_policy(
            dataset.__dict__,
            boundary="trial-engine dataset",
        )
        require_cpu_tensors(
            {
                "sample_ids": dataset.sample_ids,
                "inputs_reference": dataset.inputs_reference,
                "targets_reference": dataset.targets_reference,
                "train_ids": dataset.train_ids,
                "validation_ids": dataset.validation_ids,
                "test_ids": dataset.test_ids,
            },
            boundary="trial-engine dataset",
            name="dataset",
        )

    def _finite(self, ids: torch.Tensor, trial: TrialSpec) -> FiniteDataView:
        key = (
            stable_object_hash([int(value) for value in ids.tolist()]),
            int(trial.input.n_tar),
            int(trial.output.q),
        )
        if key not in self._finite_cache:
            inputs, targets = self.dataset.tensors_for(ids)
            self._finite_cache[key] = derive_finite_view(
                ids,
                inputs,
                targets,
                n_tar=int(trial.input.n_tar),
                q=int(trial.output.q),
                domain_length=self.dataset.domain_length,
            )
            require_cpu_tensors(
                self._finite_cache[key].__dict__,
                boundary="finite data view",
                name="finite",
            )
        return self._finite_cache[key]

    def evaluate_selection(self, trial: TrialSpec) -> CandidateEvaluation:
        candidate_id = stable_object_hash(trial.model_dump(mode="json"))
        if candidate_id in self._candidate_cache:
            return self._candidate_cache[candidate_id]
        finite = self._finite(self.selection_ids, trial)
        state = self.cache.get_or_solve(self.dataset, self.selection_ids, trial)
        training_ids, training_subset = resolve_training_subset(
            self.dataset,
            trial,
        )
        if not torch.equal(
            training_ids,
            self.selection_ids[: int(training_ids.numel())],
        ):
            raise ValueError(
                "nested training subset does not match the canonical prefix"
            )
        n_train = int(training_ids.numel())
        features = observe_equispaced_periodic(
            state.values,
            int(trial.feature.observation.J),
            domain_length=self.dataset.domain_length,
            l2_scale=trial.feature.observation.l2_scale,
        )
        require_cpu_tensors(
            {
                "state": state.values,
                "features": features,
                "target_coefficients": finite.target_coefficients,
                "targets": finite.targets,
                "targets_reference": finite.targets_reference,
            },
            boundary="readout selection inputs",
            name="selection",
        )
        x_train = features[:n_train]
        x_validation = features[self.canonical_n_train :]
        y_train = finite.target_coefficients[:n_train]
        y_validation = finite.target_coefficients[self.canonical_n_train :]
        target_validation = finite.targets[self.canonical_n_train :]
        target_reference_validation = finite.targets_reference[
            self.canonical_n_train :
        ]
        floor = representation_floor(
            y_validation,
            target_validation,
            target_reference_validation,
            n_tar=int(trial.input.n_tar),
            n_ref=finite.n_ref,
            domain_length=self.dataset.domain_length,
        )
        fit_inputs = ReadoutFitInputs(
            x_train=x_train,
            y_train=y_train,
            x_validation=x_validation,
            y_validation=y_validation,
            data_target_validation=target_validation,
            reference_target_validation=target_reference_validation,
            observation_count=int(trial.feature.observation.J),
            q=int(trial.output.q),
            n_tar=int(trial.input.n_tar),
            n_ref=self.dataset.reference_nx,
            domain_length=self.dataset.domain_length,
            selection_metric=self.study.selection.metric,
        )
        self._selection_fit_inputs[candidate_id] = fit_inputs
        rows: dict[str, dict[str, Any]] = {}
        selection_models: dict[str, dict[str, Any]] = {}
        selections: dict[str, dict[str, Any]] = {}
        for readout in trial.readouts:
            fitted = fit_readout(readout, fit_inputs)
            validation = build_validation_evaluation(
                candidate_id=candidate_id,
                readout_id=readout.id,
                readout_kind=readout.kind,
                trial=trial,
                frozen_model=fitted.selection_model,
                readout_values=fitted.validation_values,
                inner_selection=fitted.inner_selection,
                floor_metrics=floor,
                training_subset=training_subset,
            )
            rows[readout.id] = validation.row
            selection_models[readout.id] = fitted.selection_model
            selections[readout.id] = validation.inner_selection
        result = CandidateEvaluation(
            candidate_id=candidate_id,
            trial=trial,
            rows=rows,
            selection_models=selection_models,
            inner_selections=selections,
            feature_cache_id=state.cache_id,
            training_subset=training_subset,
        )
        self._candidate_cache[candidate_id] = result
        return result

    def materialize_selected_readout(
        self,
        evaluation: CandidateEvaluation,
        *,
        readout_id: str,
    ) -> dict[str, Any]:
        """Materialize one selected readout using train/validation data only."""
        key = (evaluation.candidate_id, readout_id)
        self.materialization_stats[
            "selected_readout_materialization_request_count"
        ] += 1
        cached = self._materialized_readouts.get(key)
        if cached is not None:
            self.materialization_stats[
                "selected_readout_materialization_cache_hit_count"
            ] += 1
            return cached
        model = evaluation.selection_models.get(readout_id)
        if not isinstance(model, Mapping):
            raise ValueError("selected readout has no selection model")
        if model.get("kind") != "random_feature_ridge":
            materialized = dict(model)
        else:
            if bool(
                torch.isin(
                    self.selection_ids,
                    self.dataset.test_ids.to(torch.long),
                ).any()
            ):
                raise ValueError(
                    "random-feature materialization selection IDs include test IDs"
                )
            readout = next(
                (
                    item
                    for item in evaluation.trial.readouts
                    if item.id == readout_id
                ),
                None,
            )
            if readout is None or readout.kind != "random_feature_ridge":
                raise ValueError(
                    "selected random-feature recipe has no matching readout"
                )
            inputs = self._selection_fit_inputs.get(
                evaluation.candidate_id
            )
            if inputs is None:
                raise ValueError(
                    "selected random-feature recipe has no cached fit inputs"
                )
            materialized = materialize_random_feature_readout(
                readout,
                inputs,
                model,
            )
            self.materialization_stats[
                "selected_random_feature_model_count"
            ] += 1
            self.materialization_stats[
                "selected_candidate_evaluation_member_fit_count"
            ] += len(materialized["members"])
        self.materialization_stats["selected_readout_model_count"] += 1
        self._materialized_readouts[key] = materialized
        return materialized

    def evaluate_test(
        self,
        trial: TrialSpec,
        model: Mapping[str, Any],
        *,
        readout_id: str,
        candidate_id: str,
    ) -> TestEvaluation:
        ids = self.dataset.test_ids.to(torch.long)
        _, training_subset = resolve_training_subset(self.dataset, trial)
        finite = self._finite(ids, trial)
        state = self.cache.get_or_solve(self.dataset, ids, trial)
        features = observe_equispaced_periodic(
            state.values,
            int(trial.feature.observation.J),
            domain_length=self.dataset.domain_length,
            l2_scale=trial.feature.observation.l2_scale,
        )
        require_cpu_tensors(
            {
                "state": state.values,
                "features": features,
                "target_coefficients": finite.target_coefficients,
                "targets": finite.targets,
                "targets_reference": finite.targets_reference,
                "model": model,
            },
            boundary="test evaluation inputs",
            name="test",
        )
        predictions = predict_frozen(
            model,
            features,
            q=int(trial.output.q),
            domain_length=self.dataset.domain_length,
        )
        floor = representation_floor(
            finite.target_coefficients,
            finite.targets,
            finite.targets_reference,
            n_tar=int(trial.input.n_tar),
            n_ref=finite.n_ref,
            domain_length=self.dataset.domain_length,
        )
        direct_diagnostic = frozen_readout_diagnostic(
            model,
            observation_count=int(trial.feature.observation.J),
            q=int(trial.output.q),
            boundary="test-evaluation frozen direct model",
        )
        return build_test_evaluation(
            predictions,
            model=model,
            candidate_id=candidate_id,
            readout_id=readout_id,
            trial=trial,
            feature_cache_id=state.cache_id,
            target_coefficients=finite.target_coefficients,
            data_target_field=finite.targets,
            reference_target_field=finite.targets_reference,
            n_tar=int(trial.input.n_tar),
            n_ref=finite.n_ref,
            domain_length=self.dataset.domain_length,
            floor_metrics=floor,
            direct_diagnostic=direct_diagnostic,
            training_subset=training_subset,
        )
