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
    predict_frozen,
)


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
        self.n_train = int(dataset.train_ids.numel())
        self._finite_cache: dict[tuple[str, int, int], FiniteDataView] = {}
        self._candidate_cache: dict[str, CandidateEvaluation] = {}
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
        x_train = features[: self.n_train]
        x_validation = features[self.n_train :]
        y_train = finite.target_coefficients[: self.n_train]
        y_validation = finite.target_coefficients[self.n_train :]
        target_validation = finite.targets[self.n_train :]
        target_reference_validation = finite.targets_reference[self.n_train :]
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
        rows: dict[str, dict[str, Any]] = {}
        frozen: dict[str, dict[str, Any]] = {}
        selections: dict[str, dict[str, Any]] = {}
        for readout in trial.readouts:
            fitted = fit_readout(readout, fit_inputs)
            validation = build_validation_evaluation(
                candidate_id=candidate_id,
                readout_id=readout.id,
                readout_kind=readout.kind,
                trial=trial,
                readout_values=fitted.validation_values,
                inner_selection=fitted.inner_selection,
                floor_metrics=floor,
            )
            rows[readout.id] = validation.row
            frozen[readout.id] = fitted.frozen_model
            selections[readout.id] = validation.inner_selection
        result = CandidateEvaluation(
            candidate_id=candidate_id,
            trial=trial,
            rows=rows,
            frozen_models=frozen,
            inner_selections=selections,
            feature_cache_id=state.cache_id,
        )
        self._candidate_cache[candidate_id] = result
        return result

    def evaluate_test(
        self,
        trial: TrialSpec,
        model: Mapping[str, Any],
        *,
        readout_id: str,
        candidate_id: str,
    ) -> TestEvaluation:
        ids = self.dataset.test_ids.to(torch.long)
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
            model_kind=str(model["kind"]),
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
        )
