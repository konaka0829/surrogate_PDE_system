from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Mapping

import torch
from scipy.stats import t as student_t

from pol.config.models import (
    HeatMultiplierDiagnosticSpec,
    ReadoutStabilityNoiseDiagnosticSpec,
    TrialSpec,
)
from pol.data.dataset import ReferenceDataset
from pol.data.finite import derive_finite_view
from pol.learning.direct import decode_point_observation_to_real_fourier
from pol.learning.metrics import fourier_prediction_metrics
from pol.learning.observations import observe_equispaced_periodic
from pol.learning.random_features import RandomFeatureMap
from pol.learning.ridge import l2_synthesis_matrix
from pol.runtime.device import require_cpu_tensor, require_cpu_tensors
from pol.runtime.hashing import stable_object_hash
from .cache import FeatureStateCache
from .evaluation import random_feature_map_parameter_hash
from .readouts import predict_affine
from .trial import predict_frozen


HEAT_MULTIPLIER_COEFFICIENT_SCHEMA_VERSION = (
    "pol-heat-multiplier-coefficient-v2"
)
HEAT_MULTIPLIER_SUMMARY_SCHEMA_VERSION = "pol-heat-multiplier-summary-v1"


@dataclass(frozen=True)
class HeatMultiplierDiagnosticResult:
    coefficient_rows: tuple[dict[str, Any], ...]
    summary_row: dict[str, Any]


READOUT_STABILITY_MODEL_SCHEMA_VERSION = "pol-readout-stability-model-v1"
READOUT_STABILITY_REPEAT_SCHEMA_VERSION = "pol-readout-stability-repeat-v1"
READOUT_STABILITY_SUMMARY_SCHEMA_VERSION = "pol-readout-stability-summary-v1"


@dataclass(frozen=True)
class ReadoutStabilityDiagnosticResult:
    model_rows: tuple[dict[str, Any], ...]
    repeat_rows: tuple[dict[str, Any], ...]
    summary_rows: tuple[dict[str, Any], ...]
    ensemble_repeat_rows: tuple[dict[str, Any], ...]
    ensemble_summary_rows: tuple[dict[str, Any], ...]


def _coefficient_descriptor(
    coefficient_index: int,
    *,
    observation_count: int,
) -> tuple[int, str, bool, str]:
    if coefficient_index == 0:
        return 0, "dc", True, "dc_identifiable"
    mode = (coefficient_index + 1) // 2
    coefficient_kind = "cosine" if coefficient_index % 2 else "sine"
    if mode < observation_count / 2:
        return (
            mode,
            coefficient_kind,
            True,
            "sine_cosine_pair_identifiable",
        )
    if observation_count % 2 == 0 and mode == observation_count // 2:
        status = (
            "even_grid_nyquist_cosine_not_pair_identifiable"
            if coefficient_kind == "cosine"
            else "even_grid_nyquist_sine_zero_on_observation_grid"
        )
        return mode, coefficient_kind, False, status
    return mode, coefficient_kind, False, "aliased_above_observable_band"


def _heat_condition(
    *,
    target_nu: float,
    target_time: float,
    surrogate_nu: float,
    surrogate_time: float,
) -> tuple[str, float, float, bool]:
    target_diffusion_time = target_nu * target_time
    surrogate_diffusion_time = surrogate_nu * surrogate_time
    if math.isclose(
        target_diffusion_time,
        surrogate_diffusion_time,
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        condition = "matched"
    elif surrogate_diffusion_time < target_diffusion_time:
        condition = "under_diffusive"
    else:
        condition = "more_diffusive"
    return (
        condition,
        target_diffusion_time,
        surrogate_diffusion_time,
        surrogate_diffusion_time > target_diffusion_time,
    )


def _common_heat_fields(
    *,
    dataset: ReferenceDataset,
    trial: TrialSpec,
    case_id: str,
    candidate_id: str,
    variant_id: str,
    readout_id: str,
    readout_kind: str,
) -> dict[str, Any]:
    evolution = trial.feature.evolution
    if evolution is None or evolution.system.kind != "heat":
        raise ValueError("heat multiplier diagnostic requires heat features")
    target_nu = float(dataset.target_metadata["nu"])
    target_time = float(dataset.target_metadata["time"])
    surrogate_nu = float(evolution.system.nu)
    surrogate_time = float(evolution.time)
    (
        condition,
        target_diffusion_time,
        surrogate_diffusion_time,
        inverse_amplification,
    ) = _heat_condition(
        target_nu=target_nu,
        target_time=target_time,
        surrogate_nu=surrogate_nu,
        surrogate_time=surrogate_time,
    )
    return {
        "case_id": case_id,
        "candidate_id": candidate_id,
        "variant_id": variant_id,
        "readout_id": readout_id,
        "readout_kind": readout_kind,
        "target_system": "heat",
        "target_nu": target_nu,
        "target_time": target_time,
        "target_diffusion_time": target_diffusion_time,
        "surrogate_system": "heat",
        "surrogate_nu": surrogate_nu,
        "surrogate_time": surrogate_time,
        "surrogate_diffusion_time": surrogate_diffusion_time,
        "diffusion_condition": condition,
        "inverse_amplification_required": inverse_amplification,
        "inverse_condition_note": (
            "more-diffusive features require a high-frequency-amplifying "
            "ideal inverse to recover a less-diffusive target"
            if inverse_amplification
            else "the ideal heat readout does not amplify high frequencies"
        ),
        "domain_length": float(dataset.domain_length),
        "n_tar": int(trial.input.n_tar),
        "n_sur": int(trial.feature.n_sur),
        "J": int(trial.feature.observation.J),
        "q": int(trial.output.q),
    }


def _effective_linear_map(
    model: Mapping[str, Any],
    *,
    q: int,
    J: int,
    domain_length: float,
) -> torch.Tensor | None:
    kind = model.get("kind")
    if kind == "affine_ridge":
        W = model.get("W")
        if not isinstance(W, torch.Tensor) or tuple(W.shape) != (q, J):
            raise ValueError("frozen affine readout has an invalid W shape")
        synthesis = l2_synthesis_matrix(
            q,
            J,
            domain_length=domain_length,
            dtype=W.dtype,
            device=W.device,
        )
        return W @ synthesis
    if kind == "direct_fourier_decoder":
        synthesis = l2_synthesis_matrix(
            q,
            J,
            domain_length=domain_length,
            dtype=torch.float64,
            device="cpu",
        )
        return decode_point_observation_to_real_fourier(
            synthesis.T,
            q,
            domain_length=domain_length,
        ).T
    return None


def heat_multiplier_diagnostic(
    diagnostic: HeatMultiplierDiagnosticSpec,
    *,
    dataset: ReferenceDataset,
    trial: TrialSpec,
    model: Mapping[str, Any],
    case_id: str,
    candidate_id: str,
    variant_id: str,
    readout_id: str,
) -> HeatMultiplierDiagnosticResult:
    readout_kind = str(model.get("kind"))
    base = {
        "schema_version": HEAT_MULTIPLIER_SUMMARY_SCHEMA_VERSION,
        "diagnostic_kind": "heat_multiplier",
        "case_id": case_id,
        "candidate_id": candidate_id,
        "variant_id": variant_id,
        "readout_id": readout_id,
        "readout_kind": readout_kind,
        "target_system": str(dataset.target_metadata.get("kind")),
        "domain_length": float(dataset.domain_length),
        "n_tar": int(trial.input.n_tar),
        "n_sur": int(trial.feature.n_sur),
        "J": int(trial.feature.observation.J),
        "q": int(trial.output.q),
        "selected_zeta": (
            None
            if model.get("zeta") is None
            else float(model["zeta"])
        ),
    }
    if dataset.target_metadata.get("kind") != "heat":
        return HeatMultiplierDiagnosticResult(
            coefficient_rows=(),
            summary_row={
                **base,
                "applicable": False,
                "diagnostic_status": "not_applicable_non_heat_target",
                "non_applicable_reason": (
                    "ideal heat multipliers require a heat target"
                ),
                "coefficient_row_count": 0,
                "identifiable_mode_count": 0,
                "identifiable_coefficient_count": 0,
                "diagonal_rmse": None,
                "diagonal_max_error": None,
                "off_diagonal_frobenius_norm": None,
                "max_ideal_amplification": None,
            },
        )
    evolution = trial.feature.evolution
    if trial.feature.kind != "pde_dynamics" or evolution is None:
        status = "not_applicable_non_heat_feature"
        reason = "ideal heat multipliers require a dynamic heat feature"
        return HeatMultiplierDiagnosticResult(
            coefficient_rows=(),
            summary_row={
                **base,
                "applicable": False,
                "diagnostic_status": status,
                "non_applicable_reason": reason,
                "coefficient_row_count": 0,
                "identifiable_mode_count": 0,
                "identifiable_coefficient_count": 0,
                "diagonal_rmse": None,
                "diagonal_max_error": None,
                "off_diagonal_frobenius_norm": None,
                "max_ideal_amplification": None,
            },
        )
    if evolution.system.kind != "heat":
        status = "not_applicable_non_heat_feature"
        reason = "ideal heat multipliers require a dynamic heat feature"
        return HeatMultiplierDiagnosticResult(
            coefficient_rows=(),
            summary_row={
                **base,
                "applicable": False,
                "diagnostic_status": status,
                "non_applicable_reason": reason,
                "coefficient_row_count": 0,
                "identifiable_mode_count": 0,
                "identifiable_coefficient_count": 0,
                "diagonal_rmse": None,
                "diagonal_max_error": None,
                "off_diagonal_frobenius_norm": None,
                "max_ideal_amplification": None,
            },
        )
    physical = _common_heat_fields(
        dataset=dataset,
        trial=trial,
        case_id=case_id,
        candidate_id=candidate_id,
        variant_id=variant_id,
        readout_id=readout_id,
        readout_kind=readout_kind,
    )
    effective = _effective_linear_map(
        model,
        q=int(trial.output.q),
        J=int(trial.feature.observation.J),
        domain_length=float(dataset.domain_length),
    )
    if effective is None:
        return HeatMultiplierDiagnosticResult(
            coefficient_rows=(),
            summary_row={
                **base,
                **physical,
                "applicable": False,
                "diagnostic_status": "not_applicable_nonlinear_readout",
                "non_applicable_reason": (
                    "random-feature readouts do not have one analytic "
                    "coefficient-wise linear multiplier"
                ),
                "coefficient_row_count": 0,
                "identifiable_mode_count": 0,
                "identifiable_coefficient_count": 0,
                "diagonal_rmse": None,
                "diagonal_max_error": None,
                "off_diagonal_frobenius_norm": None,
                "max_ideal_amplification": None,
            },
        )
    require_cpu_tensors(
        model,
        boundary="heat-multiplier diagnostic model",
        name="model",
    )
    q = int(trial.output.q)
    J = int(trial.feature.observation.J)
    require_cpu_tensor(
        effective,
        boundary="heat-multiplier diagnostic",
        name="effective",
    )
    if not torch.isfinite(effective).all():
        raise ValueError("effective heat readout map must be finite")
    target_nu = float(dataset.target_metadata["nu"])
    target_time = float(dataset.target_metadata["time"])
    surrogate_nu = float(evolution.system.nu)
    surrogate_time = float(evolution.time)
    domain_length = float(dataset.domain_length)
    maximum_log = math.log(torch.finfo(effective.dtype).max)
    minimum_log = math.log(torch.finfo(effective.dtype).tiny)
    minimum_subnormal_log = math.log(
        torch.nextafter(
            torch.zeros((), dtype=effective.dtype),
            torch.ones((), dtype=effective.dtype),
        ).item()
    )
    rows: list[dict[str, Any]] = []
    identifiable_modes: set[int] = set()
    diagonal_errors: list[float] = []
    ideal_amplifications: list[float] = []
    for index in range(q):
        mode, coefficient_kind, observation_identifiable, observation_status = (
            _coefficient_descriptor(index, observation_count=J)
        )
        physical_wavenumber = 2.0 * math.pi * mode / domain_length
        wavenumber_squared = physical_wavenumber ** 2
        target_log = -target_nu * target_time * wavenumber_squared
        surrogate_log = -surrogate_nu * surrogate_time * wavenumber_squared
        target_multiplier = (
            math.exp(target_log)
            if target_log >= minimum_subnormal_log
            else 0.0
        )
        surrogate_multiplier = (
            math.exp(surrogate_log)
            if surrogate_log >= minimum_subnormal_log
            else 0.0
        )
        multiplier_identifiable = (
            math.isfinite(surrogate_multiplier)
            and surrogate_multiplier
            > float(diagnostic.identifiable_multiplier_floor)
        )
        log_ideal = target_log - surrogate_log
        representable_ideal = minimum_log <= log_ideal <= maximum_log
        identifiable = (
            observation_identifiable
            and multiplier_identifiable
            and representable_ideal
        )
        if not observation_identifiable:
            status = observation_status
        elif not multiplier_identifiable:
            status = "surrogate_multiplier_below_identifiable_floor"
        elif log_ideal > maximum_log:
            status = "ideal_multiplier_overflow"
        elif log_ideal < minimum_log:
            status = "ideal_multiplier_underflow"
        else:
            status = "identifiable"
        ideal_multiplier = math.exp(log_ideal) if representable_ideal else None
        diagonal = float(effective[index, index])
        absolute_error = (
            None
            if not identifiable or ideal_multiplier is None
            else abs(diagonal - ideal_multiplier)
        )
        relative_error = None
        relative_error_status = "not_available_unidentifiable"
        if (
            absolute_error is not None
            and ideal_multiplier is not None
            and ideal_multiplier != 0.0
        ):
            candidate_relative_error = absolute_error / abs(ideal_multiplier)
            if math.isfinite(candidate_relative_error):
                relative_error = candidate_relative_error
                relative_error_status = "available"
            else:
                relative_error_status = "overflow"
        off = effective[index].clone()
        off[index] = 0
        amplification = (
            None
            if ideal_multiplier is None
            else bool(ideal_multiplier > 1.0)
        )
        amplification_magnitude = (
            None
            if ideal_multiplier is None
            else max(1.0, abs(ideal_multiplier))
        )
        if identifiable:
            identifiable_modes.add(mode)
            if absolute_error is not None:
                diagonal_errors.append(absolute_error)
            if amplification_magnitude is not None:
                ideal_amplifications.append(amplification_magnitude)
        rows.append(
            {
                "schema_version": (
                    HEAT_MULTIPLIER_COEFFICIENT_SCHEMA_VERSION
                ),
                "diagnostic_kind": "heat_multiplier",
                **physical,
                "coefficient_index": index,
                "mode_index": mode,
                "coefficient_kind": coefficient_kind,
                "physical_wavenumber": physical_wavenumber,
                "target_heat_multiplier": target_multiplier,
                "surrogate_heat_multiplier": surrogate_multiplier,
                "ideal_readout_multiplier": ideal_multiplier,
                "effective_learned_diagonal": diagonal,
                "absolute_diagonal_error": absolute_error,
                "relative_diagonal_error": relative_error,
                "relative_diagonal_error_status": relative_error_status,
                "off_diagonal_l2_contribution": float(
                    torch.linalg.vector_norm(off)
                ),
                "observation_identifiable": observation_identifiable,
                "observation_status": observation_status,
                "multiplier_identifiable": multiplier_identifiable,
                "identifiable_multiplier_floor": float(
                    diagnostic.identifiable_multiplier_floor
                ),
                "identifiable": identifiable,
                "diagnostic_status": status,
                "amplification": amplification,
                "amplification_magnitude": amplification_magnitude,
            }
        )
    off_diagonal = effective - torch.diag(torch.diagonal(effective))
    summary = {
        **base,
        **physical,
        "applicable": True,
        "diagnostic_status": "complete",
        "non_applicable_reason": None,
        "coefficient_row_count": len(rows),
        "identifiable_mode_count": len(identifiable_modes),
        "identifiable_coefficient_count": sum(
            bool(row["identifiable"]) for row in rows
        ),
        "diagonal_rmse": (
            None
            if not diagonal_errors
            else math.sqrt(
                math.fsum(error * error for error in diagonal_errors)
                / len(diagonal_errors)
            )
        ),
        "diagonal_max_error": (
            None if not diagonal_errors else max(diagonal_errors)
        ),
        "off_diagonal_frobenius_norm": float(
            torch.linalg.matrix_norm(off_diagonal)
        ),
        "max_ideal_amplification": (
            None
            if not ideal_amplifications
            else max(ideal_amplifications)
        ),
    }
    return HeatMultiplierDiagnosticResult(
        coefficient_rows=tuple(rows),
        summary_row=summary,
    )


def summarize_repeated_metrics(
    items: list[dict[str, float]],
    *,
    dimension: str,
) -> dict[str, float]:
    if len(items) < 2:
        raise ValueError("at least two repeated metric rows are required")
    keys = tuple(items[0])
    if any(tuple(item) != keys for item in items[1:]):
        raise ValueError("repeated metric rows must have identical keys")
    count = len(items)
    critical = float(student_t.ppf(0.975, count - 1))
    result: dict[str, float] = {}
    for key in keys:
        values = [float(item[key]) for item in items]
        mean = math.fsum(values) / count
        variance = math.fsum((value - mean) ** 2 for value in values) / (
            count - 1
        )
        std = math.sqrt(variance)
        margin = critical * std / math.sqrt(count)
        result[key] = mean
        result[f"{key}_{dimension}_mean"] = mean
        result[f"{key}_{dimension}_std"] = std
        result[f"{key}_{dimension}_ci95_low"] = mean - margin
        result[f"{key}_{dimension}_ci95_high"] = mean + margin
    return result


def covariance_diagnostics(
    values: torch.Tensor,
    *,
    rcond: float | None,
) -> dict[str, Any]:
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("covariance diagnostics require a 2D sample matrix")
    centered = values - values.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / (values.shape[0] - 1)
    singular = torch.linalg.svdvals(covariance)
    maximum = float(singular[0]) if singular.numel() else 0.0
    cutoff = (
        float(rcond) * maximum
        if rcond is not None
        else torch.finfo(values.dtype).eps
        * max(covariance.shape)
        * maximum
    )
    rank = int((singular > cutoff).sum())
    minimum = float(singular[-1]) if singular.numel() else 0.0
    raw_condition = (
        math.inf
        if rank < int(covariance.shape[0]) or minimum == 0.0
        else maximum / minimum
    )
    retained_condition = (
        None
        if rank == 0
        else maximum / float(singular[rank - 1])
    )
    return {
        "covariance_singular_values": json.dumps(
            [float(value) for value in singular],
            separators=(",", ":"),
        ),
        "covariance_rank": rank,
        "covariance_dimension": int(covariance.shape[0]),
        "covariance_rank_cutoff": cutoff,
        "covariance_raw_condition": raw_condition,
        "covariance_retained_rank_condition": retained_condition,
    }


def _model_norms(model: Mapping[str, Any]) -> dict[str, Any]:
    if model.get("kind") == "direct_fourier_decoder":
        return {
            "norm_status": "not_applicable_fixed_decoder",
            "weight_frobenius_norm": None,
            "weight_operator_norm": None,
            "bias_norm": None,
            "selected_ridge_zeta": None,
        }
    W = model.get("W")
    b = model.get("b")
    if not isinstance(W, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise ValueError("frozen learned readout has invalid W/b tensors")
    return {
        "norm_status": "available",
        "weight_frobenius_norm": float(torch.linalg.matrix_norm(W, ord="fro")),
        "weight_operator_norm": float(torch.linalg.matrix_norm(W, ord=2)),
        "bias_norm": float(torch.linalg.vector_norm(b)),
        "selected_ridge_zeta": float(model["zeta"]),
    }


def _random_member_design(
    parent: Mapping[str, Any],
    member: Mapping[str, Any],
    features: torch.Tensor,
) -> torch.Tensor:
    random_map = RandomFeatureMap(
        A=member["A"],
        c=member["c"],
        activation=str(parent["activation"]),
        seed=int(member["seed"]),
        weight_scale=float(parent["weight_scale"]),
        bias_scale=float(parent["bias_scale"]),
    )
    return random_map(features.to(member["A"].dtype))


def _member_prediction(
    model: Mapping[str, Any],
    features: torch.Tensor,
    *,
    member: Mapping[str, Any] | None,
    q: int,
    domain_length: float,
) -> torch.Tensor:
    if model.get("kind") != "random_feature_ridge":
        prediction = predict_frozen(
            model,
            features,
            q=q,
            domain_length=domain_length,
        ).single_model_prediction
        if prediction is None:
            raise ValueError("deterministic model returned no prediction")
        return prediction
    if member is None:
        raise ValueError("random-feature prediction requires a frozen member")
    return predict_affine(
        member,
        _random_member_design(model, member, features),
    )


@torch.no_grad()
def readout_stability_noise_diagnostic(
    diagnostic: ReadoutStabilityNoiseDiagnosticSpec,
    *,
    dataset: ReferenceDataset,
    cache: FeatureStateCache,
    trial: TrialSpec,
    model: Mapping[str, Any],
    model_key: str,
    case_id: str,
    candidate_id: str,
    variant_id: str,
    readout_id: str,
    selection_record_hash: str,
    frozen_plan_hash: str,
) -> ReadoutStabilityDiagnosticResult:
    require_cpu_tensors(
        model,
        boundary="readout stability frozen model",
        name="model",
    )
    ids = dataset.test_ids.to(torch.long)
    inputs, targets = dataset.tensors_for(ids)
    finite = derive_finite_view(
        ids,
        inputs,
        targets,
        n_tar=int(trial.input.n_tar),
        q=int(trial.output.q),
        domain_length=dataset.domain_length,
    )
    state = cache.get_or_solve(dataset, ids, trial)
    features = observe_equispaced_periodic(
        state.values,
        int(trial.feature.observation.J),
        domain_length=dataset.domain_length,
        l2_scale=trial.feature.observation.l2_scale,
    )
    require_cpu_tensors(
        {
            "inputs": inputs,
            "targets": targets,
            "state": state.values,
            "features": features,
            "finite": finite.__dict__,
        },
        boundary="noise diagnostic inputs",
        name="diagnostic",
    )
    rms = torch.sqrt(features.square().mean()).clamp_min(
        torch.finfo(features.dtype).eps
    )
    base_covariance = covariance_diagnostics(
        features,
        rcond=diagnostic.covariance_rcond,
    )
    common = {
        "diagnostic_kind": "readout_stability_noise",
        "case_id": case_id,
        "candidate_id": candidate_id,
        "variant_id": variant_id,
        "readout_id": readout_id,
        "readout_kind": str(model["kind"]),
        "model_key": model_key,
        "selection_record_hash": selection_record_hash,
        "frozen_plan_hash": frozen_plan_hash,
        "noise_scaling_kind": diagnostic.scaling.kind,
        "common_random_numbers": True,
        "sample_shape": json.dumps(list(features.shape), separators=(",", ":")),
        "feature_rms": float(rms),
    }
    members: list[Mapping[str, Any] | None]
    if model.get("kind") == "random_feature_ridge":
        members = list(model["members"])
    else:
        members = [None]
    model_rows: list[dict[str, Any]] = []
    for member in members:
        seed = None if member is None else int(member["seed"])
        active_model = model if member is None else member
        if member is None:
            design = features
            random_map_hash = None
        else:
            design = _random_member_design(model, member, features)
            random_map_hash = random_feature_map_parameter_hash(model, member)
        model_rows.append(
            {
                "schema_version": READOUT_STABILITY_MODEL_SCHEMA_VERSION,
                **common,
                "seed": seed,
                "random_map_parameter_hash": random_map_hash,
                **_model_norms(active_model),
                **{
                    f"base_feature_{key}": value
                    for key, value in base_covariance.items()
                },
                **{
                    f"readout_design_{key}": value
                    for key, value in covariance_diagnostics(
                        design,
                        rcond=diagnostic.covariance_rcond,
                    ).items()
                },
            }
        )
    model_row_by_seed = {row["seed"]: row for row in model_rows}
    repeat_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    ensemble_repeat_rows: list[dict[str, Any]] = []
    ensemble_summary_rows: list[dict[str, Any]] = []
    for level_index, level in enumerate(diagnostic.levels):
        metrics_by_seed: dict[int | None, list[dict[str, float]]] = {
            None if member is None else int(member["seed"]): []
            for member in members
        }
        ensemble_metrics: list[dict[str, float]] = []
        for repeat in range(diagnostic.repeats):
            noise_seed = int(diagnostic.seed) + 1009 * level_index + repeat
            generator = torch.Generator(device="cpu").manual_seed(
                noise_seed
            )
            noise = torch.randn(
                features.shape,
                generator=generator,
                dtype=features.dtype,
                device="cpu",
            )
            require_cpu_tensor(
                noise,
                boundary="noise diagnostic generation",
                name="noise",
            )
            noisy = features + float(level) * rms * noise
            member_predictions: list[torch.Tensor] = []
            for member in members:
                seed = None if member is None else int(member["seed"])
                prediction = _member_prediction(
                    model,
                    noisy,
                    member=member,
                    q=int(trial.output.q),
                    domain_length=dataset.domain_length,
                )
                member_predictions.append(prediction)
                metrics = fourier_prediction_metrics(
                    prediction,
                    finite.target_coefficients,
                    finite.targets,
                    finite.targets_reference,
                    n_data=finite.n_tar,
                    n_reference=finite.n_ref,
                    domain_length=dataset.domain_length,
                )
                metrics_by_seed[seed].append(metrics)
                repeat_rows.append(
                    {
                        "schema_version": READOUT_STABILITY_REPEAT_SCHEMA_VERSION,
                        **common,
                        "result_kind": (
                            "independent_seed_realization"
                            if seed is not None
                            else "single_model"
                        ),
                        "seed": seed,
                        "noise_level": float(level),
                        "noise_rms": float(level) * float(rms),
                        "noise_seed": noise_seed,
                        "repeat": repeat,
                        **metrics,
                    }
                )
            if (
                model.get("kind") == "random_feature_ridge"
                and diagnostic.include_prediction_ensemble
            ):
                ensemble_prediction = torch.stack(
                    member_predictions,
                    dim=0,
                ).mean(0)
                ensemble_metric = fourier_prediction_metrics(
                    ensemble_prediction,
                    finite.target_coefficients,
                    finite.targets,
                    finite.targets_reference,
                    n_data=finite.n_tar,
                    n_reference=finite.n_ref,
                    domain_length=dataset.domain_length,
                )
                ensemble_metrics.append(ensemble_metric)
                ensemble_repeat_rows.append(
                    {
                        "schema_version": READOUT_STABILITY_REPEAT_SCHEMA_VERSION,
                        **common,
                        "result_kind": "noise_prediction_ensemble",
                        "ensemble_member_count": len(members),
                        "noise_level": float(level),
                        "noise_rms": float(level) * float(rms),
                        "noise_seed": noise_seed,
                        "repeat": repeat,
                        **ensemble_metric,
                    }
                )
        per_seed_means: list[dict[str, float]] = []
        for seed, metrics in metrics_by_seed.items():
            repeated = summarize_repeated_metrics(
                metrics,
                dimension="repeat",
            )
            summary_rows.append(
                {
                    "schema_version": READOUT_STABILITY_SUMMARY_SCHEMA_VERSION,
                    **common,
                    "result_kind": (
                        "independent_seed_repeat_summary"
                        if seed is not None
                        else "single_model_repeat_summary"
                    ),
                    "seed": seed,
                    "noise_level": float(level),
                    "noise_rms": float(level) * float(rms),
                    "repeat_count": int(diagnostic.repeats),
                    "confidence_level": 0.95,
                    "confidence_interval_method": "student_t",
                    **{
                        key: value
                        for key, value in model_row_by_seed[seed].items()
                        if key.startswith("weight_")
                        or key == "bias_norm"
                        or key == "norm_status"
                        or key.startswith("readout_design_")
                        or key.startswith("base_feature_")
                    },
                    **repeated,
                }
            )
            per_seed_means.append(
                {key: repeated[key] for key in metrics[0]}
            )
        if model.get("kind") == "random_feature_ridge":
            summary_rows.append(
                {
                    "schema_version": READOUT_STABILITY_SUMMARY_SCHEMA_VERSION,
                    **common,
                    "result_kind": "independent_seed_primary_summary",
                    "seed_count": len(per_seed_means),
                    "noise_level": float(level),
                    "noise_rms": float(level) * float(rms),
                    "repeat_count": int(diagnostic.repeats),
                    "confidence_level": 0.95,
                    "confidence_interval_method": "student_t",
                    **summarize_repeated_metrics(
                        per_seed_means,
                        dimension="seed",
                    ),
                }
            )
        if ensemble_metrics:
            ensemble_summary_rows.append(
                {
                    "schema_version": READOUT_STABILITY_SUMMARY_SCHEMA_VERSION,
                    **common,
                    "result_kind": "noise_prediction_ensemble_summary",
                    "ensemble_member_count": len(members),
                    "noise_level": float(level),
                    "noise_rms": float(level) * float(rms),
                    "repeat_count": int(diagnostic.repeats),
                    "confidence_level": 0.95,
                    "confidence_interval_method": "student_t",
                    **summarize_repeated_metrics(
                        ensemble_metrics,
                        dimension="repeat",
                    ),
                }
            )
    return ReadoutStabilityDiagnosticResult(
        model_rows=tuple(model_rows),
        repeat_rows=tuple(repeat_rows),
        summary_rows=tuple(summary_rows),
        ensemble_repeat_rows=tuple(ensemble_repeat_rows),
        ensemble_summary_rows=tuple(ensemble_summary_rows),
    )
