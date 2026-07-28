from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import torch

from pol.artifacts import ArtifactRef, ArtifactStore, verify_artifact
from pol.config.models import BurgersTimeCandidateSpec, ValidationSpec
from pol.data.initial_conditions import (
    GRF_SAMPLER_SEMANTICS,
    InitialConditionArchive,
    generate_grf_archive,
)
from pol.learning.direct import decode_point_observation_to_real_fourier
from pol.learning.metrics import samplewise_l2_errors
from pol.learning.observations import observe_equispaced_periodic
from pol.math.fourier import real_fourier_analysis, real_fourier_synthesis
from pol.math.periodic import periodic_grid, spectral_resample_periodic
from pol.runtime.environment import numerical_environment_fingerprint
from pol.runtime.hashing import stable_object_hash, tensor_sha256
from pol.runtime.io import atomic_torch_save, write_csv, write_strict_json
from pol.systems.registry import evolve


@dataclass(frozen=True)
class ValidationOutcome:
    reference: ArtifactRef
    certificate: dict[str, Any]


def _scientific_identity(spec: ValidationSpec) -> dict[str, Any]:
    payload = spec.model_dump(mode="json")
    payload.pop("artifact_root", None)
    return {
        "schema_version": "pol-validation-identity-v3",
        "grf_sampler_semantics": GRF_SAMPLER_SEMANTICS,
        "environment": numerical_environment_fingerprint(),
        "spec": payload,
    }


def validation_reference(spec: ValidationSpec) -> ArtifactRef:
    identity = _scientific_identity(spec)
    return ArtifactStore(spec.artifact_root).reference("validations", identity)


def load_validation_certificate(path: Path | str) -> dict[str, Any]:
    root = Path(path).resolve()
    manifest = verify_artifact(root)
    if manifest.get("kind") != "validations":
        raise ValueError("artifact is not a passing validation artifact")
    certificate = json.loads((root / "certificate.json").read_text(encoding="utf-8"))
    if not isinstance(certificate, dict):
        raise ValueError("validation certificate payload must be an object")
    if certificate.get("schema_version") != "pol-validation-certificate-v3":
        raise ValueError(
            "unsupported validation certificate schema; P0-03 requires "
            "pol-validation-certificate-v3"
        )
    identity = manifest.get("identity")
    if not isinstance(identity, dict) or identity.get("schema_version") != (
        "pol-validation-identity-v3"
    ):
        raise ValueError("unsupported legacy validation artifact identity")
    if identity.get("grf_sampler_semantics") != GRF_SAMPLER_SEMANTICS:
        raise ValueError("validation artifact has unsupported GRF sampler semantics")
    resolved = json.loads((root / "resolved_spec.json").read_text(encoding="utf-8"))
    if not isinstance(resolved, dict):
        raise ValueError("validation resolved spec must be an object")
    if stable_object_hash(resolved) != stable_object_hash(identity.get("spec")):
        raise ValueError("validation resolved spec does not match artifact identity")
    try:
        spec = ValidationSpec.model_validate(resolved)
    except ValueError as exc:
        raise ValueError("validation artifact contains an invalid resolved spec") from exc
    checks = json.loads((root / "checks.json").read_text(encoding="utf-8"))
    if not isinstance(checks, dict):
        raise ValueError("validation checks payload must be an object")
    master = torch.load(
        root / "master_initial_conditions.pt",
        map_location="cpu",
        weights_only=True,
    )
    _validate_master_payload_against_spec(master, spec)
    try:
        expected = _certificate_payload(
            spec,
            artifact_id=str(manifest.get("artifact_id")),
            checks=checks,
            master_payload=master,
        )
    except (AttributeError, KeyError, TypeError) as exc:
        raise ValueError(
            "validation checks cannot reconstruct the required certificate contract"
        ) from exc
    _validate_target_reference_contract(expected["target_reference_contract"])
    if stable_object_hash(certificate) != stable_object_hash(expected):
        raise ValueError(
            "validation certificate contract is missing, contradictory, or "
            "inconsistent with the resolved spec/checks/master archive"
        )
    if certificate.get("artifact_id") != manifest.get("artifact_id"):
        raise ValueError("validation certificate identity mismatch")
    if certificate.get("status") != "pass":
        raise ValueError("validation certificate is not passing")
    check_statuses = certificate.get("checks")
    if not isinstance(check_statuses, dict) or not check_statuses or any(
        value != "pass" for value in check_statuses.values()
    ):
        raise ValueError("validation certificate contains a non-passing check")
    return certificate


def ensure_validation(spec: ValidationSpec, *, force: bool = False) -> ValidationOutcome:
    ref = validation_reference(spec)
    store = ArtifactStore(spec.artifact_root)
    if store.exists(ref) and not force:
        return ValidationOutcome(ref, load_validation_certificate(ref.path))
    return run_validation(spec, force=force)


def _allclose(a: torch.Tensor, b: torch.Tensor, spec: ValidationSpec) -> bool:
    if a.dtype == torch.float32:
        atol = spec.algebraic_tolerances.float32_atol
        rtol = spec.algebraic_tolerances.float32_rtol
    else:
        atol = spec.algebraic_tolerances.float64_atol
        rtol = spec.algebraic_tolerances.float64_rtol
    return bool(torch.allclose(a, b, atol=atol, rtol=rtol))


def _resampling_checks(spec: ValidationSpec) -> dict[str, Any]:
    dtype = torch.float64 if spec.samples.dtype == "float64" else torch.float32
    device = torch.device("cpu")
    L = float(spec.domain.length)
    rows: list[dict[str, Any]] = []
    for n_in, n_out, k in ((31, 48, 5), (48, 31, 5), (32, 64, 16), (64, 32, 7)):
        x_in = periodic_grid(n_in, L, dtype=dtype, device=device)
        x_out = periodic_grid(n_out, L, dtype=dtype, device=device)
        source = 0.3 + 0.7 * torch.cos(2 * torch.pi * k * x_in / L)
        expected = 0.3 + 0.7 * torch.cos(2 * torch.pi * k * x_out / L)
        actual = spectral_resample_periodic(source, n_out, domain_length=L)
        error = float((actual - expected).abs().max())
        rows.append(
            {
                "n_in": n_in,
                "n_out": n_out,
                "mode": k,
                "max_abs_error": error,
                "status": "pass" if _allclose(actual, expected, spec) else "fail",
            }
        )
    n_tar = int(spec.full_interface.n_tar)
    n_ref = max(2 * n_tar, n_tar + 8)
    if n_ref % 2:
        n_ref += 1
    x = periodic_grid(n_ref, L, dtype=dtype, device=device)
    low = 0.4 + torch.cos(4 * torch.pi * x / L)
    high_k = n_tar // 2 + 1
    high = low + 0.3 * torch.cos(2 * torch.pi * high_k * x / L)
    down = spectral_resample_periodic(torch.stack([low, high]), n_tar, domain_length=L)
    discarded = _allclose(down[0], down[1], spec)
    return {
        "status": "pass"
        if all(row["status"] == "pass" for row in rows) and discarded
        else "fail",
        "mode_transfer": rows,
        "high_frequency_discard": {
            "status": "pass" if discarded else "fail",
            "reference_nx": n_ref,
            "target_nx": n_tar,
            "high_mode": high_k,
            "max_abs_difference": float((down[0] - down[1]).abs().max()),
        },
    }


def _fourier_projector_check(spec: ValidationSpec) -> dict[str, Any]:
    dtype = torch.float64 if spec.samples.dtype == "float64" else torch.float32
    nx = max(int(spec.full_interface.n_tar), int(spec.full_interface.q) + 3)
    if nx % 2 and int(spec.full_interface.q) >= nx:
        nx += 1
    q = int(spec.full_interface.q)
    generator = torch.Generator(device="cpu").manual_seed(spec.samples.seed + 1949)
    coefficients = torch.randn((4, q), generator=generator, dtype=dtype)
    field = real_fourier_synthesis(coefficients, nx, domain_length=spec.domain.length)
    recovered = real_fourier_analysis(field, q, domain_length=spec.domain.length)
    passed = _allclose(recovered, coefficients, spec)
    return {
        "status": "pass" if passed else "fail",
        "nx": nx,
        "q": q,
        "max_abs_error": float((recovered - coefficients).abs().max()),
    }


def _finite_interface_checks(spec: ValidationSpec, archive: InitialConditionArchive) -> dict[str, Any]:
    dims = spec.full_interface
    L = float(spec.domain.length)
    ids = torch.tensor(spec.calibration_sample_ids, dtype=torch.long, device=archive.values.device)
    reference = archive.values.index_select(0, ids)
    finite = spectral_resample_periodic(reference, dims.n_tar, domain_length=L)
    feature_input = spectral_resample_periodic(finite, dims.n_sur, domain_length=L)
    shape_pass = finite.shape == (len(ids), dims.n_tar) and feature_input.shape == (
        len(ids),
        dims.n_sur,
    )

    n_ref = max(2 * dims.n_tar, dims.n_tar + 8)
    if n_ref % 2:
        n_ref += 1
    x = periodic_grid(n_ref, L, dtype=archive.values.dtype, device=archive.values.device)
    low = 0.25 + torch.cos(2 * torch.pi * 2 * x / L)
    high_k = dims.n_tar // 2 + 1
    pair = torch.stack(
        [low, low + 0.35 * torch.cos(2 * torch.pi * high_k * x / L)]
    )
    finite_pair = spectral_resample_periodic(pair, dims.n_tar, domain_length=L)
    feature_pair = spectral_resample_periodic(finite_pair, dims.n_sur, domain_length=L)
    no_leak = _allclose(finite_pair[0], finite_pair[1], spec) and _allclose(
        feature_pair[0], feature_pair[1], spec
    )

    # Deliberately exercise n_tar < J.  These dimensions are independent; only
    # q <= n_tar and J <= n_sur are mathematical interface constraints.
    independence_n_tar = max(4, min(dims.n_tar, dims.J // 2 or 1))
    if independence_n_tar % 2:
        independence_n_tar += 1
    independence_n_sur = max(dims.J, dims.n_sur)
    independent_finite = spectral_resample_periodic(
        reference, independence_n_tar, domain_length=L
    )
    independent_state = spectral_resample_periodic(
        independent_finite, independence_n_sur, domain_length=L
    )
    independent_features = observe_equispaced_periodic(
        independent_state, dims.J, domain_length=L, l2_scale=True
    )
    independence_pass = independent_features.shape[-1] == dims.J and independence_n_tar <= dims.J
    return {
        "status": "pass" if shape_pass and no_leak and independence_pass else "fail",
        "finite_shapes": {
            "status": "pass" if shape_pass else "fail",
            "n_tar": dims.n_tar,
            "n_sur": dims.n_sur,
        },
        "no_high_frequency_leak": {
            "status": "pass" if no_leak else "fail",
            "synthetic_reference_nx": n_ref,
            "high_mode": high_k,
            "max_finite_difference": float((finite_pair[0] - finite_pair[1]).abs().max()),
            "max_feature_input_difference": float(
                (feature_pair[0] - feature_pair[1]).abs().max()
            ),
        },
        "dimension_independence": {
            "status": "pass" if independence_pass else "fail",
            "n_tar": independence_n_tar,
            "n_sur": independence_n_sur,
            "J": dims.J,
            "n_tar_le_J_exercised": independence_n_tar <= dims.J,
        },
    }


def _decoder_checks(spec: ValidationSpec) -> dict[str, Any]:
    dtype = torch.float64 if spec.samples.dtype == "float64" else torch.float32
    L = float(spec.domain.length)
    full = spec.full_interface
    generator = torch.Generator(device="cpu").manual_seed(spec.samples.seed + 911)
    coeff = torch.randn((3, full.q), generator=generator, dtype=dtype)
    field = real_fourier_synthesis(coeff, full.n_sur, domain_length=L)
    features = observe_equispaced_periodic(
        field, full.J, domain_length=L, l2_scale=True
    )
    decoded = decode_point_observation_to_real_fourier(features, full.q, domain_length=L)
    full_pass = _allclose(decoded, coeff, spec)

    reduced = spec.reduced_observation
    rcoeff = torch.randn((2, reduced.q), generator=generator, dtype=dtype)
    rfield = real_fourier_synthesis(rcoeff, full.n_sur, domain_length=L)
    rfeatures = observe_equispaced_periodic(
        rfield, reduced.J, domain_length=L, l2_scale=True
    )
    rdecoded = decode_point_observation_to_real_fourier(
        rfeatures, reduced.q, domain_length=L
    )
    reduced_pass = _allclose(rdecoded, rcoeff, spec)

    kmax = (reduced.q - 1) // 2
    alias_k = reduced.J + max(1, kmax)
    alias_pass = False
    alias_error = float("nan")
    if alias_k < full.n_sur / 2:
        x = periodic_grid(full.n_sur, L, dtype=dtype)
        base = 0.2 + torch.cos(2 * torch.pi * max(1, kmax) * x / L)
        high = base + 0.4 * torch.cos(2 * torch.pi * alias_k * x / L)
        truth = real_fourier_analysis(base.unsqueeze(0), reduced.q, domain_length=L)
        aliased = decode_point_observation_to_real_fourier(
            observe_equispaced_periodic(
                high.unsqueeze(0), reduced.J, domain_length=L, l2_scale=True
            ),
            reduced.q,
            domain_length=L,
        )
        alias_error = float((aliased - truth).abs().max())
        alias_pass = not _allclose(aliased, truth, spec)
    return {
        "status": "pass" if full_pass and reduced_pass and alias_pass else "fail",
        "full_observation": {
            "status": "pass" if full_pass else "fail",
            "J": full.J,
            "q": full.q,
            "max_abs_error": float((decoded - coeff).abs().max()),
        },
        "reduced_bandlimited": {
            "status": "pass" if reduced_pass else "fail",
            "J": reduced.J,
            "q": reduced.q,
            "max_abs_error": float((rdecoded - rcoeff).abs().max()),
        },
        "aliasing_counterexample": {
            "status": "pass" if alias_pass else "fail",
            "high_mode": alias_k,
            "max_abs_difference": alias_error,
        },
    }


def _candidate_evolution(spec: ValidationSpec, candidate: BurgersTimeCandidateSpec) -> dict[str, Any]:
    base = spec.reference_evolution.model_dump(mode="json")
    system = dict(base["system"])
    system.update(
        {
            "solver": candidate.solver,
            "dt": candidate.dt,
            "fine_dt": candidate.fine_dt,
            "dealias": candidate.dealias,
        }
    )
    return {"system": system, "time": base["time"]}


def _pair_metrics(
    coarse: torch.Tensor,
    fine: torch.Tensor,
    *,
    q: int,
    domain_length: float,
) -> dict[str, float]:
    fine_common = spectral_resample_periodic(
        fine, coarse.shape[-1], domain_length=domain_length
    )
    _, relative = samplewise_l2_errors(
        coarse, fine_common, domain_length=domain_length
    )
    coarse_coeff = real_fourier_analysis(coarse, q, domain_length=domain_length)
    fine_coeff = real_fourier_analysis(fine_common, q, domain_length=domain_length)
    denominator = torch.linalg.vector_norm(fine_coeff, dim=-1).clamp_min(
        torch.finfo(fine_coeff.dtype).eps
    )
    low_relative = torch.linalg.vector_norm(
        coarse_coeff - fine_coeff, dim=-1
    ) / denominator
    return {
        "mean_relative_l2": float(relative.mean()),
        "max_relative_l2": float(relative.max()),
        "low_mode_relative_l2": float(low_relative.mean()),
    }


def _passes(metrics: dict[str, float], spec: ValidationSpec) -> bool:
    tol = spec.reference_tolerances
    return (
        metrics["mean_relative_l2"] <= tol.mean_relative_l2
        and metrics["max_relative_l2"] <= tol.max_relative_l2
        and metrics["low_mode_relative_l2"] <= tol.low_mode_relative_l2
    )


def _coarsest_stable_index(rows: list[dict[str, Any]]) -> int | None:
    # A candidate is accepted only if every refinement after it is also within
    # tolerance.  This prevents selecting an accidentally good nonmonotone pair.
    for index in range(len(rows)):
        if all(row["status"] == "pass" for row in rows[index:]):
            return index
    return None


def _reference_convergence(
    spec: ValidationSpec,
    archive: InitialConditionArchive,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    L = float(spec.domain.length)
    ids = torch.tensor(spec.calibration_sample_ids, dtype=torch.long, device=archive.values.device)
    initial_master = archive.values.index_select(0, ids)
    nx_values = [int(value) for value in spec.reference_nx_candidates]
    candidates = list(spec.time_candidates)
    finest_candidate = candidates[-1]
    spatial_solutions: dict[int, torch.Tensor] = {}
    metadata: dict[str, Any] = {}
    for nx in nx_values:
        initial = spectral_resample_periodic(initial_master, nx, domain_length=L)
        solution, meta = evolve(
            initial,
            _candidate_evolution(spec, finest_candidate),
            domain_length=L,
        )
        spatial_solutions[nx] = solution
        metadata[f"spatial_{nx}"] = meta
    rows: list[dict[str, Any]] = []
    spatial_rows: list[dict[str, Any]] = []
    for coarse_nx, fine_nx in zip(nx_values[:-1], nx_values[1:]):
        metrics = _pair_metrics(
            spatial_solutions[coarse_nx],
            spatial_solutions[fine_nx],
            q=int(spec.q_reference_check),
            domain_length=L,
        )
        row = {
            "check": "spatial",
            "coarse": coarse_nx,
            "fine": fine_nx,
            **metrics,
            "status": "pass" if _passes(metrics, spec) else "fail",
        }
        spatial_rows.append(row)
        rows.append(row)
    spatial_index = _coarsest_stable_index(spatial_rows)
    selected_nx = None if spatial_index is None else int(spatial_rows[spatial_index]["coarse"])

    finest_nx = nx_values[-1]
    initial_finest = spectral_resample_periodic(initial_master, finest_nx, domain_length=L)
    temporal_solutions: list[torch.Tensor] = []
    temporal_metadata: list[dict[str, Any]] = []
    for candidate in candidates:
        solution, meta = evolve(
            initial_finest,
            _candidate_evolution(spec, candidate),
            domain_length=L,
        )
        temporal_solutions.append(solution)
        temporal_metadata.append(meta)
    temporal_rows: list[dict[str, Any]] = []
    for index in range(len(candidates) - 1):
        metrics = _pair_metrics(
            temporal_solutions[index],
            temporal_solutions[index + 1],
            q=int(spec.q_reference_check),
            domain_length=L,
        )
        row = {
            "check": "temporal",
            "coarse": index,
            "fine": index + 1,
            "coarse_dt": candidates[index].dt,
            "coarse_fine_dt": candidates[index].fine_dt,
            "fine_dt": candidates[index + 1].dt,
            "fine_fine_dt": candidates[index + 1].fine_dt,
            **metrics,
            "status": "pass" if _passes(metrics, spec) else "fail",
        }
        temporal_rows.append(row)
        rows.append(row)
    temporal_index = _coarsest_stable_index(temporal_rows)
    selected_time_index = temporal_index

    joint_status = "fail"
    joint_row: dict[str, Any] | None = None
    if selected_nx is not None and selected_time_index is not None:
        selected_initial = spectral_resample_periodic(
            initial_master, selected_nx, domain_length=L
        )
        selected_solution, selected_meta = evolve(
            selected_initial,
            _candidate_evolution(spec, candidates[selected_time_index]),
            domain_length=L,
        )
        joint_metrics = _pair_metrics(
            selected_solution,
            temporal_solutions[-1],
            q=int(spec.q_reference_check),
            domain_length=L,
        )
        joint_status = "pass" if _passes(joint_metrics, spec) else "fail"
        joint_row = {
            "check": "joint",
            "coarse": selected_nx,
            "fine": finest_nx,
            "time_candidate": selected_time_index,
            **joint_metrics,
            "status": joint_status,
        }
        rows.append(joint_row)
        metadata["joint_selected"] = selected_meta

    result = {
        "status": "pass"
        if selected_nx is not None
        and selected_time_index is not None
        and joint_status == "pass"
        else "fail",
        "spatial_status": "pass" if selected_nx is not None else "fail",
        "temporal_status": "pass" if selected_time_index is not None else "fail",
        "joint_status": joint_status,
        "selected_reference_nx": selected_nx,
        "selected_reference_candidate_index": spatial_index,
        "selected_time_candidate_index": selected_time_index,
        "selected_time_candidate": (
            None
            if selected_time_index is None
            else candidates[selected_time_index].model_dump(mode="json")
        ),
        "finest_reference_nx": finest_nx,
        "finest_time_candidate": candidates[-1].model_dump(mode="json"),
        "solver_metadata": {
            **metadata,
            "temporal": temporal_metadata,
        },
        "joint_row": joint_row,
    }
    return result, rows


def _master_payload(archive: InitialConditionArchive) -> dict[str, Any]:
    return {
        "schema_version": "pol-initial-condition-archive-v3",
        "sample_ids": archive.sample_ids.detach().cpu(),
        "values": archive.values.detach().cpu(),
        "fourier": archive.fourier.detach().cpu(),
        "nx": archive.nx,
        "domain_length": archive.domain_length,
        "seed": archive.seed,
        "metadata": archive.metadata,
    }


def _validate_master_payload_against_spec(
    payload: dict[str, Any], spec: ValidationSpec
) -> None:
    if not isinstance(payload, dict):
        raise ValueError("initial-condition archive payload must be an object")
    if payload.get("schema_version") != "pol-initial-condition-archive-v3":
        raise ValueError(
            "unsupported initial-condition archive schema; P0-03 requires v3"
        )
    sample_ids = payload.get("sample_ids")
    values = payload.get("values")
    fourier = payload.get("fourier")
    if not all(isinstance(value, torch.Tensor) for value in (sample_ids, values, fourier)):
        raise ValueError("initial-condition archive tensors are missing")
    total = int(spec.samples.total_samples)
    nx = max(int(value) for value in spec.reference_nx_candidates)
    if sample_ids.shape != (total,) or sample_ids.dtype != torch.long:
        raise ValueError("initial-condition archive sample_ids are inconsistent")
    if not torch.equal(sample_ids, torch.arange(total, dtype=torch.long)):
        raise ValueError("initial-condition archive sample_ids are not canonical")
    if values.shape != (total, nx) or fourier.shape != (total, nx // 2 + 1):
        raise ValueError("initial-condition archive tensor shapes are inconsistent")
    expected_dtype = torch.float64 if spec.samples.dtype == "float64" else torch.float32
    if values.dtype != expected_dtype:
        raise ValueError("initial-condition archive dtype mismatch")
    if not torch.isfinite(values).all() or not torch.isfinite(fourier).all():
        raise ValueError("initial-condition archive tensors must be finite")
    if stable_object_hash(payload.get("nx")) != stable_object_hash(nx):
        raise ValueError("initial-condition archive resolution mismatch")
    if stable_object_hash(payload.get("domain_length")) != stable_object_hash(
        float(spec.domain.length)
    ):
        raise ValueError("initial-condition archive domain mismatch")
    if stable_object_hash(payload.get("seed")) != stable_object_hash(
        int(spec.samples.seed)
    ):
        raise ValueError("initial-condition archive seed mismatch")
    metadata = payload.get("metadata")
    expected_ic = spec.samples.initial_condition.model_dump(mode="json")
    if not isinstance(metadata, dict):
        raise ValueError("initial-condition archive metadata is missing")
    for name in ("kind", "gamma", "tau", "sigma", "mean"):
        if stable_object_hash(metadata.get(name)) != stable_object_hash(
            expected_ic.get(name)
        ):
            raise ValueError(
                f"initial-condition archive specification mismatch: {name}"
            )
    if metadata.get("dtype") != spec.samples.dtype:
        raise ValueError("initial-condition archive metadata dtype mismatch")
    if stable_object_hash(metadata.get("domain_length")) != stable_object_hash(
        float(spec.domain.length)
    ):
        raise ValueError("initial-condition archive sampler domain mismatch")
    if metadata.get("sampler_semantics") != GRF_SAMPLER_SEMANTICS:
        raise ValueError("initial-condition archive sampler semantics mismatch")


def _master_archive_binding(payload: dict[str, Any]) -> dict[str, Any]:
    tensor_hashes = {
        name: tensor_sha256(payload[name])
        for name in ("sample_ids", "values", "fourier")
    }
    identity = {
        "schema_version": "pol-master-initial-condition-binding-v2",
        "archive_schema_version": payload["schema_version"],
        "nx": int(payload["nx"]),
        "total_samples": int(payload["sample_ids"].numel()),
        "domain_length": float(payload["domain_length"]),
        "seed": int(payload["seed"]),
        "metadata": dict(payload["metadata"]),
        "tensor_hashes": tensor_hashes,
    }
    return {
        **identity,
        "archive_identity_hash": stable_object_hash(identity),
    }


def _foundation_contract(
    spec: ValidationSpec,
    *,
    checks: dict[str, Any],
    master_payload: dict[str, Any],
) -> dict[str, Any]:
    general_names = (
        "periodic_resampling",
        "real_fourier_projector",
        "finite_input_interface",
        "fixed_decoder",
    )
    statuses = {
        name: checks[name]["status"]
        for name in general_names
    }
    samples = spec.samples
    return {
        "schema_version": "pol-validation-foundation-contract-v2",
        "domain_length": float(spec.domain.length),
        "grf_sampler_domain_length": float(
            master_payload["metadata"]["domain_length"]
        ),
        "grf_sampler_semantics": master_payload["metadata"]["sampler_semantics"],
        "dtype": samples.dtype,
        "samples": {
            "total_samples": int(samples.total_samples),
            "n_train": int(samples.n_train),
            "n_validation": int(samples.n_validation),
            "n_test": int(samples.n_test),
            "seed": int(samples.seed),
            "device": samples.device,
            "preprocessing": samples.preprocessing,
        },
        "initial_condition": samples.initial_condition.model_dump(mode="json"),
        "master_initial_conditions": _master_archive_binding(master_payload),
        "general_foundation_checks": {
            "status": (
                "pass"
                if all(value == "pass" for value in statuses.values())
                else "fail"
            ),
            "checks": statuses,
        },
    }


def _target_reference_contract(
    spec: ValidationSpec,
    convergence: dict[str, Any],
) -> dict[str, Any]:
    nx_candidates = [int(value) for value in spec.reference_nx_candidates]
    time_candidates = [
        candidate.model_dump(mode="json") for candidate in spec.time_candidates
    ]
    reference_index = convergence.get("selected_reference_candidate_index")
    time_index = convergence.get("selected_time_candidate_index")
    reference_allowed_indices = (
        []
        if not isinstance(reference_index, int)
        else list(range(reference_index, len(nx_candidates)))
    )
    time_allowed_indices = (
        []
        if not isinstance(time_index, int)
        else list(range(time_index, len(time_candidates)))
    )
    system = spec.reference_evolution.system
    return {
        "schema_version": "pol-target-reference-contract-v1",
        "system_kind": system.kind,
        "invariant_parameters": {
            "nu": float(system.nu),
            "advection_coefficient": float(system.advection_coefficient),
        },
        "evolution_time": float(spec.reference_evolution.time),
        "dtype": spec.samples.dtype,
        "domain_length": float(spec.domain.length),
        "reference_resolution": {
            "selected_value": convergence.get("selected_reference_nx"),
            "selected_candidate_index": reference_index,
            "candidates": nx_candidates,
        },
        "time_discretization": {
            "selected_candidate": convergence.get("selected_time_candidate"),
            "selected_candidate_index": time_index,
            "candidates": time_candidates,
        },
        "selection_policy": spec.selection_policy,
        "allowed_refinement_relation": {
            "kind": "validated_candidate_suffix_exact_membership",
            "reference_nx_allowed_indices": reference_allowed_indices,
            "reference_nx_allowed_values": [
                nx_candidates[index] for index in reference_allowed_indices
            ],
            "time_candidate_allowed_indices": time_allowed_indices,
            "time_candidate_allowed_values": [
                time_candidates[index] for index in time_allowed_indices
            ],
        },
    }


def _validate_target_reference_contract(contract: dict[str, Any]) -> None:
    reference = contract.get("reference_resolution")
    time_discretization = contract.get("time_discretization")
    relation = contract.get("allowed_refinement_relation")
    if not all(
        isinstance(value, dict)
        for value in (reference, time_discretization, relation)
    ):
        raise ValueError("validation target-reference contract is incomplete")
    nx_candidates = reference.get("candidates")
    time_candidates = time_discretization.get("candidates")
    reference_index = reference.get("selected_candidate_index")
    time_index = time_discretization.get("selected_candidate_index")
    if (
        not isinstance(nx_candidates, list)
        or not isinstance(time_candidates, list)
        or type(reference_index) is not int
        or type(time_index) is not int
        or not (0 <= reference_index < len(nx_candidates) - 1)
        or not (0 <= time_index < len(time_candidates) - 1)
    ):
        raise ValueError(
            "validation target-reference selected candidate indices are invalid"
        )
    if stable_object_hash(reference.get("selected_value")) != stable_object_hash(
        nx_candidates[reference_index]
    ):
        raise ValueError(
            "validation selected reference value/index is self-inconsistent"
        )
    if stable_object_hash(
        time_discretization.get("selected_candidate")
    ) != stable_object_hash(time_candidates[time_index]):
        raise ValueError(
            "validation selected time candidate/index is self-inconsistent"
        )
    expected_reference_indices = list(range(reference_index, len(nx_candidates)))
    expected_time_indices = list(range(time_index, len(time_candidates)))
    expected_relation = {
        "kind": "validated_candidate_suffix_exact_membership",
        "reference_nx_allowed_indices": expected_reference_indices,
        "reference_nx_allowed_values": [
            nx_candidates[index] for index in expected_reference_indices
        ],
        "time_candidate_allowed_indices": expected_time_indices,
        "time_candidate_allowed_values": [
            time_candidates[index] for index in expected_time_indices
        ],
    }
    if stable_object_hash(relation) != stable_object_hash(expected_relation):
        raise ValueError(
            "validation allowed refinement relation is not the selected "
            "candidate suffix"
        )


def _certificate_payload(
    spec: ValidationSpec,
    *,
    artifact_id: str,
    checks: dict[str, Any],
    master_payload: dict[str, Any],
) -> dict[str, Any]:
    statuses = {name: value["status"] for name, value in checks.items()}
    overall = "pass" if all(value == "pass" for value in statuses.values()) else "fail"
    foundation = _foundation_contract(
        spec, checks=checks, master_payload=master_payload
    )
    target_reference = _target_reference_contract(
        spec, checks["reference_convergence"]
    )
    if overall == "pass":
        _validate_target_reference_contract(target_reference)
    return {
        "schema_version": "pol-validation-certificate-v3",
        "status": overall,
        "name": spec.name,
        "profile": spec.profile,
        "artifact_id": artifact_id,
        "checks": statuses,
        "foundation_contract": foundation,
        "foundation_contract_hash": stable_object_hash(foundation),
        "target_reference_contract": target_reference,
        "target_reference_contract_hash": stable_object_hash(target_reference),
    }


def run_validation(spec: ValidationSpec, *, force: bool = False) -> ValidationOutcome:
    if spec.samples.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested by validation spec but unavailable")
    max_nx = max(int(value) for value in spec.reference_nx_candidates)
    ic = spec.samples.initial_condition
    archive = generate_grf_archive(
        total_samples=spec.samples.total_samples,
        nx=max_nx,
        seed=spec.samples.seed,
        gamma=ic.gamma,
        tau=ic.tau,
        sigma=ic.sigma,
        mean=ic.mean,
        domain_length=spec.domain.length,
        dtype=spec.samples.dtype,
        device=spec.samples.device,
    )
    checks = {
        "periodic_resampling": _resampling_checks(spec),
        "real_fourier_projector": _fourier_projector_check(spec),
        "finite_input_interface": _finite_interface_checks(spec, archive),
        "fixed_decoder": _decoder_checks(spec),
    }
    convergence, convergence_rows = _reference_convergence(spec, archive)
    checks["reference_convergence"] = convergence
    identity = _scientific_identity(spec)
    store = ArtifactStore(spec.artifact_root)
    ref = store.reference("validations", identity)
    master_payload = _master_payload(archive)
    _validate_master_payload_against_spec(master_payload, spec)
    certificate = _certificate_payload(
        spec,
        artifact_id=ref.artifact_id,
        checks=checks,
        master_payload=master_payload,
    )
    statuses = certificate["checks"]
    overall = certificate["status"]

    def writer(root: Path) -> Iterable[str]:
        write_strict_json(root / "resolved_spec.json", identity["spec"])
        write_strict_json(root / "checks.json", checks)
        write_strict_json(root / "certificate.json", certificate)
        write_csv(
            root / "reference_convergence.csv",
            convergence_rows,
            fieldnames=[
                "check",
                "coarse",
                "fine",
                "coarse_dt",
                "coarse_fine_dt",
                "fine_dt",
                "fine_fine_dt",
                "time_candidate",
                "mean_relative_l2",
                "max_relative_l2",
                "low_mode_relative_l2",
                "status",
            ],
        )
        atomic_torch_save(root / "master_initial_conditions.pt", master_payload)
        return (
            "resolved_spec.json",
            "checks.json",
            "certificate.json",
            "reference_convergence.csv",
            "master_initial_conditions.pt",
        )

    if overall != "pass":
        failure_ref = store.reference("validation_failures", identity)
        store.publish(failure_ref, identity=identity, writer=writer, force=force)
        raise RuntimeError(
            f"validation failed ({statuses}); diagnostics: {failure_ref.path}"
        )
    store.publish(ref, identity=identity, writer=writer, force=force)
    return ValidationOutcome(ref, load_validation_certificate(ref.path))
