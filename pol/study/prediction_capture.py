from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from pol.config.models import PredictionCaptureSpec, StudySpec, TrialSpec
from pol.data.finite import derive_finite_view
from pol.learning.metrics import fourier_prediction_metrics
from pol.math.fourier import real_fourier_analysis, real_fourier_synthesis
from pol.runtime.device import require_cpu_tensors
from pol.runtime.hashing import stable_object_hash, tensor_sha256
from pol.runtime.io import atomic_torch_save
from .evaluation import (
    FrozenPredictions,
    TestEvaluation,
    feature_system_condition_hash,
    random_feature_member_parameter_hash,
)


PREDICTION_CAPTURE_SCHEMA_VERSION = "pol-prediction-capture-v1"
PREDICTION_CAPTURE_FILENAME = "prediction_capture.pt"


@dataclass(frozen=True)
class PredictionCaptureResult:
    payload: dict[str, Any]
    entry_count: int
    content_hash: str


def validate_prediction_capture_preflight(
    spec: StudySpec | PredictionCaptureSpec,
    dataset: Any,
) -> None:
    capture = (
        spec
        if isinstance(spec, PredictionCaptureSpec)
        else spec.prediction_capture
    )
    if capture is None:
        return
    test_ids = {int(value) for value in dataset.test_ids.tolist()}
    requested = set(capture.sample_ids)
    if not requested <= test_ids:
        missing = sorted(requested - test_ids)
        raise ValueError(
            "prediction capture sample_ids must be predeclared test IDs; "
            f"not in test split: {missing}"
        )


def _tensor_content_descriptor(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {"tensor_sha256": tensor_sha256(value)}
    if isinstance(value, Mapping):
        return {
            str(key): _tensor_content_descriptor(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_tensor_content_descriptor(item) for item in value]
    return value


def prediction_capture_content_hash(payload: Mapping[str, Any]) -> str:
    without_hash = {
        key: value
        for key, value in payload.items()
        if key != "capture_content_hash"
    }
    return stable_object_hash(_tensor_content_descriptor(without_hash))


def _mode_sum(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 1 or values.numel() % 2 != 1:
        raise ValueError("mode aggregation requires an odd coefficient vector")
    items = [values[0]]
    for mode in range(1, (values.numel() + 1) // 2):
        items.append(values[2 * mode - 1] + values[2 * mode])
    return torch.stack(items)


def _capture_entry(
    *,
    model_key: str,
    archive_entry: Mapping[str, Any],
    trial: TrialSpec,
    semantics: str,
    seed: int | None,
    member_parameter_hash: str | None,
    prediction: torch.Tensor,
    sample_positions: torch.Tensor,
    sample_ids: torch.Tensor,
    finite_all: Any,
    finite_capture: Any,
    domain_length: float,
) -> dict[str, Any]:
    q = int(trial.output.q)
    n_tar = int(trial.input.n_tar)
    n_ref = int(finite_all.n_ref)
    if tuple(prediction.shape) != (
        int(finite_all.sample_ids.numel()),
        q,
    ):
        raise ValueError("captured prediction does not cover the canonical test set")
    captured_prediction = prediction.index_select(0, sample_positions)
    prediction_n_tar = real_fourier_synthesis(
        captured_prediction,
        n_tar,
        domain_length=domain_length,
    )
    prediction_n_ref = real_fourier_synthesis(
        captured_prediction,
        n_ref,
        domain_length=domain_length,
    )
    captured_error = (
        captured_prediction - finite_capture.target_coefficients
    )
    all_error = prediction - finite_all.target_coefficients
    per_coefficient_mse = all_error.square().mean(dim=0)
    per_coefficient_target_energy = (
        finite_all.target_coefficients.square().mean(dim=0)
    )
    per_mode_mse = _mode_sum(per_coefficient_mse)
    per_mode_target_energy = _mode_sum(per_coefficient_target_energy)
    epsilon = torch.finfo(prediction.dtype).eps
    per_mode_relative = per_mode_mse / per_mode_target_energy.clamp_min(
        epsilon
    )
    mode_indices = torch.arange(
        per_mode_mse.numel(),
        dtype=torch.long,
    )
    physical_wavenumbers = (
        (2.0 * torch.pi / float(domain_length))
        * mode_indices.to(prediction.dtype)
    )
    metrics = fourier_prediction_metrics(
        prediction,
        finite_all.target_coefficients,
        finite_all.targets,
        finite_all.targets_reference,
        n_data=n_tar,
        n_reference=n_ref,
        domain_length=domain_length,
    )
    entry = {
        "model_key": model_key,
        "case_id": str(archive_entry["case_id"]),
        "variant_id": str(archive_entry["variant_id"]),
        "candidate_id": str(archive_entry["candidate_id"]),
        "readout_id": str(archive_entry["readout_id"]),
        "readout_kind": str(archive_entry["model"]["kind"]),
        "prediction_semantics": semantics,
        "seed": seed,
        "frozen_member_parameter_hash": member_parameter_hash,
        "feature_condition": trial.feature.model_dump(mode="json"),
        "feature_system_condition_hash": feature_system_condition_hash(trial),
        "domain_length": float(domain_length),
        "n_tar": n_tar,
        "n_ref": n_ref,
        "q": q,
        "sample_ids": sample_ids.clone(),
        "finite_input_n_tar": finite_capture.inputs.clone(),
        "target_field_n_tar": finite_capture.targets.clone(),
        "target_field_n_ref": finite_capture.targets_reference.clone(),
        "target_q_coefficients": finite_capture.target_coefficients.clone(),
        "prediction_q_coefficients": captured_prediction.clone(),
        "prediction_field_n_tar": prediction_n_tar,
        "prediction_field_n_ref": prediction_n_ref,
        "per_coefficient_error": captured_error,
        "per_coefficient_squared_error": captured_error.square(),
        "test_per_coefficient_squared_error_sample_mean": (
            per_coefficient_mse
        ),
        "test_per_coefficient_target_energy_sample_mean": (
            per_coefficient_target_energy
        ),
        "test_per_mode_squared_error_sample_mean": per_mode_mse,
        "test_per_mode_target_energy_sample_mean": per_mode_target_energy,
        "test_per_mode_relative_energy_error": per_mode_relative,
        "mode_indices": mode_indices,
        "physical_wavenumbers": physical_wavenumbers,
        "test_coefficient_mse": float(metrics["coefficient_mse"]),
    }
    entry["entry_content_hash"] = stable_object_hash(
        _tensor_content_descriptor(entry)
    )
    return entry


@torch.no_grad()
def build_prediction_capture(
    capture: PredictionCaptureSpec,
    *,
    dataset: Any,
    archive: Mapping[str, Any],
    evaluations: Mapping[str, TestEvaluation],
    selection_record_hash: str,
    frozen_plan_hash: str,
    frozen_model_archive_sha256: str,
) -> PredictionCaptureResult:
    validate_prediction_capture_preflight(
        capture,
        dataset,
    )
    test_ids = dataset.test_ids.to(torch.long)
    position_by_id = {
        int(sample_id): position
        for position, sample_id in enumerate(test_ids.tolist())
    }
    sample_ids = torch.tensor(capture.sample_ids, dtype=torch.long)
    sample_positions = torch.tensor(
        [position_by_id[int(sample_id)] for sample_id in capture.sample_ids],
        dtype=torch.long,
    )
    inputs_capture, targets_capture = dataset.tensors_for(sample_ids)
    inputs_all, targets_all = dataset.tensors_for(test_ids)
    entries: list[dict[str, Any]] = []
    models = archive.get("models")
    if not isinstance(models, Mapping):
        raise ValueError("frozen archive has no model mapping")
    for model_key, archive_entry in models.items():
        if not isinstance(archive_entry, Mapping):
            raise ValueError("frozen archive model entry is invalid")
        if archive_entry.get("readout_id") not in capture.readout_ids:
            continue
        trial = TrialSpec.model_validate(archive_entry["trial"])
        finite_all = derive_finite_view(
            test_ids,
            inputs_all,
            targets_all,
            n_tar=int(trial.input.n_tar),
            q=int(trial.output.q),
            domain_length=dataset.domain_length,
        )
        finite_capture = derive_finite_view(
            sample_ids,
            inputs_capture,
            targets_capture,
            n_tar=int(trial.input.n_tar),
            q=int(trial.output.q),
            domain_length=dataset.domain_length,
        )
        evaluated = evaluations.get(str(model_key))
        if evaluated is None:
            raise ValueError("prediction capture is missing a test evaluation")
        predictions: FrozenPredictions = evaluated.predictions
        model = archive_entry["model"]
        if model.get("kind") != "random_feature_ridge":
            if predictions.single_model_prediction is None:
                raise ValueError("deterministic capture has no prediction")
            entries.append(
                _capture_entry(
                    model_key=str(model_key),
                    archive_entry=archive_entry,
                    trial=trial,
                    semantics="single_model",
                    seed=None,
                    member_parameter_hash=None,
                    prediction=predictions.single_model_prediction,
                    sample_positions=sample_positions,
                    sample_ids=sample_ids,
                    finite_all=finite_all,
                    finite_capture=finite_capture,
                    domain_length=dataset.domain_length,
                )
            )
            continue
        prediction_by_seed = dict(predictions.per_seed_predictions)
        member_by_seed = {
            int(member["seed"]): member for member in model["members"]
        }
        for seed in capture.random_feature_members.seeds:
            prediction = prediction_by_seed.get(int(seed))
            member = member_by_seed.get(int(seed))
            if prediction is None or not isinstance(member, Mapping):
                raise ValueError(
                    "capture seed is absent from the frozen evaluation members"
                )
            entries.append(
                _capture_entry(
                    model_key=str(model_key),
                    archive_entry=archive_entry,
                    trial=trial,
                    semantics="independent_seed_realization",
                    seed=int(seed),
                    member_parameter_hash=random_feature_member_parameter_hash(
                        model,
                        member,
                    ),
                    prediction=prediction,
                    sample_positions=sample_positions,
                    sample_ids=sample_ids,
                    finite_all=finite_all,
                    finite_capture=finite_capture,
                    domain_length=dataset.domain_length,
                )
            )
        if capture.include_ensemble:
            entries.append(
                _capture_entry(
                    model_key=str(model_key),
                    archive_entry=archive_entry,
                    trial=trial,
                    semantics="prediction_ensemble",
                    seed=None,
                    member_parameter_hash=None,
                    prediction=predictions.prediction_ensemble(),
                    sample_positions=sample_positions,
                    sample_ids=sample_ids,
                    finite_all=finite_all,
                    finite_capture=finite_capture,
                    domain_length=dataset.domain_length,
                )
            )
    if not entries:
        raise ValueError("prediction capture selected no frozen models")
    payload: dict[str, Any] = {
        "schema_version": PREDICTION_CAPTURE_SCHEMA_VERSION,
        "capture_spec": capture.model_dump(mode="json"),
        "dataset_artifact_id": dataset.artifact_id,
        "dataset_split_hash": dataset.split_hash,
        "test_ids_hash": tensor_sha256(test_ids),
        "test_sample_count": int(test_ids.numel()),
        "sample_ids": sample_ids,
        "sample_ids_hash": tensor_sha256(sample_ids),
        "sample_test_positions": sample_positions,
        "selection_record_hash": selection_record_hash,
        "frozen_plan_hash": frozen_plan_hash,
        "frozen_model_archive_sha256": frozen_model_archive_sha256,
        "coefficient_ordering": "dc,cos(1),sin(1),cos(2),sin(2),...",
        "spectrum_definition": {
            "sample_aggregate": "arithmetic_mean_over_canonical_test_split",
            "per_mode_squared_error": (
                "dc_error_squared_or_cos_sin_pair_squared_sum"
            ),
            "per_mode_target_energy": (
                "dc_target_squared_or_cos_sin_pair_squared_sum"
            ),
            "relative_energy_error": (
                "per_mode_squared_error/per_mode_target_energy"
            ),
            "zero_denominator_policy": "dtype_machine_epsilon_clamp",
            "stored_prediction_policy": (
                "predeclared_samples_plus_all_test_per_coefficient_aggregates"
            ),
        },
        "entries": entries,
    }
    content_hash = prediction_capture_content_hash(payload)
    payload["capture_content_hash"] = content_hash
    require_cpu_tensors(
        payload,
        boundary="prediction capture publication",
        name="capture",
    )
    return PredictionCaptureResult(
        payload=payload,
        entry_count=len(entries),
        content_hash=content_hash,
    )


def write_prediction_capture(
    path: Path,
    result: PredictionCaptureResult,
) -> None:
    atomic_torch_save(path, result.payload)


def load_prediction_capture(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("prediction capture artifact must be an object")
    require_cpu_tensors(
        payload,
        boundary="prediction capture load",
        name="capture",
    )
    return payload


def verify_prediction_capture_payload(
    payload: Mapping[str, Any],
    *,
    capture_spec: Mapping[str, Any],
    dataset_artifact_id: str,
    dataset_split_hash: str,
    selection_record_hash: str,
    frozen_plan_hash: str,
    frozen_model_archive_sha256: str,
) -> None:
    if payload.get("schema_version") != PREDICTION_CAPTURE_SCHEMA_VERSION:
        raise ValueError("unsupported prediction capture schema")
    if payload.get("capture_spec") != dict(capture_spec):
        raise ValueError("prediction capture spec mismatch")
    for key, expected in {
        "dataset_artifact_id": dataset_artifact_id,
        "dataset_split_hash": dataset_split_hash,
        "selection_record_hash": selection_record_hash,
        "frozen_plan_hash": frozen_plan_hash,
        "frozen_model_archive_sha256": frozen_model_archive_sha256,
    }.items():
        if payload.get(key) != expected:
            raise ValueError(f"prediction capture {key} mismatch")
    stored_hash = payload.get("capture_content_hash")
    if (
        not isinstance(stored_hash, str)
        or prediction_capture_content_hash(payload) != stored_hash
    ):
        raise ValueError("prediction capture content hash mismatch")
    sample_ids = payload.get("sample_ids")
    positions = payload.get("sample_test_positions")
    if (
        not isinstance(sample_ids, torch.Tensor)
        or sample_ids.dtype != torch.long
        or sample_ids.ndim != 1
        or sample_ids.tolist() != list(capture_spec["sample_ids"])
        or payload.get("sample_ids_hash") != tensor_sha256(sample_ids)
        or not isinstance(positions, torch.Tensor)
        or positions.dtype != torch.long
        or positions.shape != sample_ids.shape
        or torch.unique(positions).numel() != positions.numel()
        or bool((positions < 0).any())
        or bool((positions >= int(payload["test_sample_count"])).any())
    ):
        raise ValueError("prediction capture sample binding mismatch")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("prediction capture has no entries")
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("prediction capture entry is invalid")
        stored_entry_hash = entry.get("entry_content_hash")
        entry_without_hash = {
            key: value
            for key, value in entry.items()
            if key != "entry_content_hash"
        }
        if stored_entry_hash != stable_object_hash(
            _tensor_content_descriptor(entry_without_hash)
        ):
            raise ValueError("prediction capture entry hash mismatch")
        q = int(entry["q"])
        n_tar = int(entry["n_tar"])
        n_ref = int(entry["n_ref"])
        domain_length = float(entry["domain_length"])
        prediction = entry["prediction_q_coefficients"]
        target_coefficients = entry["target_q_coefficients"]
        target_n_tar = entry["target_field_n_tar"]
        target_n_ref = entry["target_field_n_ref"]
        expected_shape = (sample_ids.numel(), q)
        if (
            tuple(prediction.shape) != expected_shape
            or tuple(target_coefficients.shape) != expected_shape
            or tuple(entry["finite_input_n_tar"].shape)
            != (sample_ids.numel(), n_tar)
            or tuple(target_n_tar.shape) != (sample_ids.numel(), n_tar)
            or tuple(target_n_ref.shape) != (sample_ids.numel(), n_ref)
            or not all(
                bool(torch.isfinite(value).all())
                for value in entry.values()
                if isinstance(value, torch.Tensor)
                and value.dtype.is_floating_point
            )
        ):
            raise ValueError("prediction capture tensor shape/value mismatch")
        if not torch.allclose(
            real_fourier_analysis(
                target_n_tar,
                q,
                domain_length=domain_length,
            ),
            target_coefficients,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError("prediction capture target coefficient mismatch")
        for grid, stored in (
            (n_tar, entry["prediction_field_n_tar"]),
            (n_ref, entry["prediction_field_n_ref"]),
        ):
            if not torch.allclose(
                real_fourier_synthesis(
                    prediction,
                    grid,
                    domain_length=domain_length,
                ),
                stored,
                rtol=1e-12,
                atol=1e-12,
            ):
                raise ValueError(
                    "prediction capture reconstructed field mismatch"
                )
        error = prediction - target_coefficients
        if (
            not torch.equal(entry["per_coefficient_error"], error)
            or not torch.equal(
                entry["per_coefficient_squared_error"],
                error.square(),
            )
        ):
            raise ValueError("prediction capture coefficient error mismatch")
        per_coefficient = entry[
            "test_per_coefficient_squared_error_sample_mean"
        ]
        target_energy = entry[
            "test_per_coefficient_target_energy_sample_mean"
        ]
        per_mode = _mode_sum(per_coefficient)
        per_mode_target = _mode_sum(target_energy)
        relative = per_mode / per_mode_target.clamp_min(
            torch.finfo(per_mode.dtype).eps
        )
        if (
            not torch.equal(
                entry["test_per_mode_squared_error_sample_mean"],
                per_mode,
            )
            or not torch.equal(
                entry["test_per_mode_target_energy_sample_mean"],
                per_mode_target,
            )
            or not torch.equal(
                entry["test_per_mode_relative_energy_error"],
                relative,
            )
            or not math.isclose(
                float(per_coefficient.mean()),
                float(entry["test_coefficient_mse"]),
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        ):
            raise ValueError("prediction capture spectrum aggregate mismatch")


__all__ = [
    "PREDICTION_CAPTURE_FILENAME",
    "PREDICTION_CAPTURE_SCHEMA_VERSION",
    "PredictionCaptureResult",
    "build_prediction_capture",
    "load_prediction_capture",
    "prediction_capture_content_hash",
    "validate_prediction_capture_preflight",
    "verify_prediction_capture_payload",
    "write_prediction_capture",
]
