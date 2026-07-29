from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch

from pol.artifacts import ArtifactRef, ArtifactStore, verify_artifact
from pol.config.models import (
    BurgersConvergenceReferenceSpec,
    BurgersTimeCandidateSpec,
    EnabledBurgersCrossSolverValidationSpec,
    HeatAnalyticReferenceSpec,
    ReactionDiffusionConvergenceReferenceSpec,
    ReactionDiffusionTimeCandidateSpec,
    ReferenceToleranceSpec,
    ValidationSpec,
)
from pol.data.initial_conditions import (
    GRF_SAMPLER_SEMANTICS,
    InitialConditionArchive,
    generate_grf_archive,
)
from pol.data.splits import build_data_split, calibration_split_provenance
from pol.learning.direct import (
    decode_point_observation_to_real_fourier,
    fixed_fourier_decoder_bandwidth,
)
from pol.learning.metrics import (
    samplewise_l2_errors,
    symmetric_field_discrepancy,
)
from pol.learning.observations import observe_equispaced_periodic
from pol.math.fourier import real_fourier_analysis, real_fourier_synthesis
from pol.math.periodic import periodic_grid, spectral_resample_periodic
from pol.runtime.environment import numerical_environment_fingerprint
from pol.runtime.device import (
    execution_device_policy,
    require_cpu_tensor,
    require_cpu_tensors,
    verify_execution_device_policy,
)
from pol.runtime.hashing import stable_object_hash, tensor_sha256
from pol.runtime.io import atomic_torch_save, write_csv, write_strict_json
from pol.systems.heat import solve_heat_exact
from pol.systems.reaction_diffusion import solve_reaction_diffusion
from pol.systems.registry import evolve
from .conditions import (
    CONVERGENCE_CSV_SCHEMA_VERSION,
    CONVERGENCE_ROW_FIELDS,
    CONVERGENCE_ROW_SCHEMA_VERSION,
    CROSS_SOLVER_CHECK_SCHEMA_VERSION,
    CROSS_SOLVER_METRIC_DEFINITION,
    burgers_refinement_proof,
    canonical_invariant_parameters,
    canonical_numerical_condition,
    cross_solver_discrepancy_evidence_hash,
    make_convergence_row,
    reaction_diffusion_refinement_proof,
    reference_refinement_proof,
    validate_cross_solver_validation_block,
    validate_target_reference_contract,
)
from .model1_consistency import (
    MODEL1_CONSISTENCY_CHECK_SCHEMA_VERSION,
    model1_foundation_summary,
    run_matched_model1_pipeline_check,
    validate_matched_model1_pipeline_check,
)
from .quadrature import (
    FIELD_QUADRATURE_CHECK_SCHEMA_VERSION,
    field_quadrature_foundation_summary,
    run_field_quadrature_check,
    validate_field_quadrature_check,
)


@dataclass(frozen=True)
class ValidationOutcome:
    reference: ArtifactRef
    certificate: dict[str, Any]


@dataclass(frozen=True)
class _TimeSequenceResult:
    refinement_proof: dict[str, Any]
    conditions: list[dict[str, Any]]
    solutions: list[torch.Tensor]
    runtime_metadata: list[dict[str, Any]]
    rows: list[dict[str, Any]]
    selected_index: int | None


class _ValidationSolveFailure(RuntimeError):
    def __init__(self, diagnostic: dict[str, Any]) -> None:
        super().__init__(str(diagnostic.get("message", "validation solve failed")))
        self.diagnostic = diagnostic


def _calibration_provenance(
    spec: ValidationSpec,
    *,
    sample_ids: torch.Tensor | None = None,
) -> dict[str, object]:
    samples = spec.samples
    split = build_data_split(
        total_samples=int(samples.total_samples),
        n_train=int(samples.n_train),
        n_validation=int(samples.n_validation),
        n_test=int(samples.n_test),
        seed=int(samples.seed),
    )
    if sample_ids is None:
        sample_ids = torch.arange(split.total_samples, dtype=torch.long)
    return calibration_split_provenance(
        split,
        spec.target_reference.calibration_sample_ids,
        sample_ids=sample_ids,
    )


def _scientific_identity(spec: ValidationSpec) -> dict[str, Any]:
    payload = spec.model_dump(mode="json")
    payload.pop("artifact_root", None)
    return {
        "schema_version": "pol-validation-identity-v12",
        "reference_convergence_csv_schema_version": (
            CONVERGENCE_CSV_SCHEMA_VERSION
        ),
        "cross_solver_check_schema_version": (
            CROSS_SOLVER_CHECK_SCHEMA_VERSION
        ),
        "matched_model1_pipeline_check_schema_version": (
            MODEL1_CONSISTENCY_CHECK_SCHEMA_VERSION
        ),
        "field_quadrature_check_schema_version": (
            FIELD_QUADRATURE_CHECK_SCHEMA_VERSION
        ),
        **execution_device_policy(),
        "grf_sampler_semantics": GRF_SAMPLER_SEMANTICS,
        "calibration_provenance": _calibration_provenance(spec),
        "environment": numerical_environment_fingerprint(),
        "spec": payload,
    }


def validation_reference(spec: ValidationSpec) -> ArtifactRef:
    identity = _scientific_identity(spec)
    return ArtifactStore(spec.artifact_root).reference("validations", identity)


def _validate_convergence_csv(
    path: Path,
    expected_rows: list[dict[str, Any]],
) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if reader.fieldnames != list(CONVERGENCE_ROW_FIELDS):
        raise ValueError("reference convergence CSV schema/header mismatch")
    expected = [
        {
            name: "" if row[name] is None else str(row[name])
            for name in CONVERGENCE_ROW_FIELDS
        }
        for row in expected_rows
    ]
    if rows != expected:
        raise ValueError(
            "reference convergence CSV rows disagree with certificate evidence"
        )


def load_validation_certificate(path: Path | str) -> dict[str, Any]:
    root = Path(path).resolve()
    manifest = verify_artifact(root)
    if manifest.get("kind") != "validations":
        raise ValueError("artifact is not a passing validation artifact")
    certificate = json.loads((root / "certificate.json").read_text(encoding="utf-8"))
    if not isinstance(certificate, dict):
        raise ValueError("validation certificate payload must be an object")
    if certificate.get("schema_version") != "pol-validation-certificate-v12":
        raise ValueError(
            "unsupported validation certificate schema; Phase 2-05B requires "
            "pol-validation-certificate-v12"
        )
    identity = manifest.get("identity")
    if not isinstance(identity, dict) or identity.get("schema_version") != (
        "pol-validation-identity-v12"
    ):
        raise ValueError("unsupported legacy validation artifact identity")
    if identity.get("reference_convergence_csv_schema_version") != (
        CONVERGENCE_CSV_SCHEMA_VERSION
    ):
        raise ValueError("unsupported reference convergence CSV schema")
    if identity.get("cross_solver_check_schema_version") != (
        CROSS_SOLVER_CHECK_SCHEMA_VERSION
    ):
        raise ValueError("unsupported cross-solver check schema")
    if identity.get("matched_model1_pipeline_check_schema_version") != (
        MODEL1_CONSISTENCY_CHECK_SCHEMA_VERSION
    ):
        raise ValueError("unsupported matched Model 1 pipeline check schema")
    if identity.get("field_quadrature_check_schema_version") != (
        FIELD_QUADRATURE_CHECK_SCHEMA_VERSION
    ):
        raise ValueError("unsupported field quadrature check schema")
    verify_execution_device_policy(
        identity,
        boundary="validation artifact identity",
    )
    environment = identity.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("validation artifact numerical environment is missing")
    verify_execution_device_policy(
        environment,
        boundary="validation numerical environment",
    )
    verify_execution_device_policy(
        certificate,
        boundary="validation certificate",
    )
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
    fixed_decoder_check = checks.get("fixed_decoder")
    if not isinstance(fixed_decoder_check, dict):
        raise ValueError("validation fixed-decoder check is missing")
    _validate_decoder_characterization(fixed_decoder_check)
    matched_model1_check = checks.get("matched_model1_pipeline")
    if not isinstance(matched_model1_check, dict):
        raise ValueError("matched Model 1 pipeline check is missing")
    validate_matched_model1_pipeline_check(
        matched_model1_check,
        domain_length=float(spec.domain.length),
    )
    field_quadrature_check = checks.get("field_quadrature")
    if not isinstance(field_quadrature_check, dict):
        raise ValueError("field quadrature check is missing")
    validate_field_quadrature_check(
        field_quadrature_check,
        domain_length=float(spec.domain.length),
    )
    if isinstance(spec.target_reference, HeatAnalyticReferenceSpec):
        analytic_check = checks.get("heat_analytic")
        if not isinstance(analytic_check, dict) or stable_object_hash(
            analytic_check
        ) != stable_object_hash(_heat_analytic_checks(spec)):
            raise ValueError(
                "validation heat analytic check is missing or inconsistent"
            )
    if isinstance(
        spec.target_reference,
        ReactionDiffusionConvergenceReferenceSpec,
    ):
        characterization = checks.get(
            "reaction_diffusion_characterization"
        )
        if not isinstance(characterization, dict) or stable_object_hash(
            characterization
        ) != stable_object_hash(
            _reaction_diffusion_characterization(spec)
        ):
            raise ValueError(
                "validation reaction-diffusion characterization is missing "
                "or inconsistent"
            )
    cross_enabled = (
        isinstance(
            spec.target_reference,
            BurgersConvergenceReferenceSpec,
        )
        and spec.target_reference.cross_solver_validation.enabled
    )
    if cross_enabled:
        cross_check = checks.get("cross_solver_validation")
        if not isinstance(cross_check, dict):
            raise ValueError(
                "enabled cross-solver validation evidence is missing"
            )
        validate_cross_solver_validation_block(cross_check)
        _validate_cross_solver_check_against_spec(spec, cross_check)
    elif "cross_solver_validation" in checks:
        raise ValueError(
            "disabled cross-solver validation must not contain evidence"
        )
    master = torch.load(
        root / "master_initial_conditions.pt",
        map_location="cpu",
        weights_only=True,
    )
    require_cpu_tensors(
        master,
        boundary="master initial-condition archive load",
        name="master",
    )
    _validate_master_payload_against_spec(master, spec)
    expected_calibration = _calibration_provenance(
        spec,
        sample_ids=master["sample_ids"],
    )
    if stable_object_hash(identity.get("calibration_provenance")) != (
        stable_object_hash(expected_calibration)
    ):
        raise ValueError(
            "validation identity calibration provenance does not match "
            "the deterministic split"
        )
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
    expected_rows = expected["target_reference_contract"][
        "convergence_evidence"
    ]["rows"]
    _validate_convergence_csv(
        root / "reference_convergence.csv",
        expected_rows,
    )
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


def _validate_decoder_characterization(check: dict[str, Any]) -> None:
    characterization = check.get("zero_fill_characterization")
    if not isinstance(characterization, dict):
        raise ValueError("fixed-decoder zero-fill characterization is missing")
    try:
        diagnostic = fixed_fourier_decoder_bandwidth(
            characterization["observation_count"],
            characterization["requested_q"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "fixed-decoder zero-fill characterization has invalid J/q"
        ) from exc
    for field, expected in diagnostic.as_dict().items():
        if characterization.get(field) != expected:
            raise ValueError(
                "fixed-decoder zero-fill characterization does not match "
                f"the bandwidth formula: {field}"
            )
    expected_coefficient_range = {
        "start_inclusive": diagnostic.retained_q,
        "stop_exclusive": diagnostic.requested_q,
    }
    if characterization.get(
        "zero_filled_coefficient_index_range"
    ) != expected_coefficient_range:
        raise ValueError("fixed-decoder zero-filled coefficient range mismatch")
    expected_mode_range = {
        "start_inclusive": diagnostic.observable_max_mode + 1,
        "stop_inclusive": diagnostic.requested_max_mode,
    }
    if characterization.get("zero_filled_mode_range") != expected_mode_range:
        raise ValueError("fixed-decoder zero-filled mode range mismatch")
    observable_part = characterization.get("observable_part")
    zero_filled_part = characterization.get("zero_filled_part")
    if (
        characterization.get("status") != "pass"
        or not isinstance(observable_part, dict)
        or observable_part.get("status") != "pass"
        or not isinstance(zero_filled_part, dict)
        or zero_filled_part.get("status") != "pass"
        or zero_filled_part.get("exact_zero") is not True
        or zero_filled_part.get("coefficient_count")
        != diagnostic.zero_filled_coefficient_count
    ):
        raise ValueError(
            "fixed-decoder zero-fill characterization is not passing"
        )


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
        require_cpu_tensor(
            actual,
            boundary="validation periodic-resampling check",
            name="actual",
        )
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
    require_cpu_tensors(
        archive.__dict__,
        boundary="validation finite-interface check",
        name="archive",
    )
    dims = spec.full_interface
    L = float(spec.domain.length)
    ids = torch.tensor(
        spec.target_reference.calibration_sample_ids,
        dtype=torch.long,
        device=archive.values.device,
    )
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

    zero_fill_J = 4
    zero_fill_q = 7
    bandwidth = fixed_fourier_decoder_bandwidth(zero_fill_J, zero_fill_q)
    observable_coefficients = torch.randn(
        (2, bandwidth.retained_q),
        generator=generator,
        dtype=dtype,
    )
    requested_coefficients = torch.zeros(
        (2, zero_fill_q),
        dtype=dtype,
    )
    requested_coefficients[:, : bandwidth.retained_q] = observable_coefficients
    zero_fill_source_nx = max(8, int(full.n_sur))
    zero_fill_field = real_fourier_synthesis(
        requested_coefficients,
        zero_fill_source_nx,
        domain_length=L,
    )
    zero_fill_features = observe_equispaced_periodic(
        zero_fill_field,
        zero_fill_J,
        domain_length=L,
        l2_scale=True,
    )
    zero_fill_decoded = decode_point_observation_to_real_fourier(
        zero_fill_features,
        zero_fill_q,
        domain_length=L,
    )
    observable_part_pass = _allclose(
        zero_fill_decoded[:, : bandwidth.retained_q],
        observable_coefficients,
        spec,
    )
    zero_filled_part = zero_fill_decoded[:, bandwidth.retained_q :]
    zero_filled_part_pass = torch.equal(
        zero_filled_part,
        torch.zeros_like(zero_filled_part),
    )
    zero_fill_pass = (
        bandwidth.zero_fill_applied
        and observable_part_pass
        and zero_filled_part_pass
    )
    return {
        "status": (
            "pass"
            if full_pass and reduced_pass and alias_pass and zero_fill_pass
            else "fail"
        ),
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
        "zero_fill_characterization": {
            "status": "pass" if zero_fill_pass else "fail",
            **bandwidth.as_dict(),
            "source_nx": zero_fill_source_nx,
            "zero_filled_coefficient_index_range": {
                "start_inclusive": bandwidth.retained_q,
                "stop_exclusive": bandwidth.requested_q,
            },
            "zero_filled_mode_range": {
                "start_inclusive": bandwidth.observable_max_mode + 1,
                "stop_inclusive": bandwidth.requested_max_mode,
            },
            "observable_part": {
                "status": "pass" if observable_part_pass else "fail",
                "max_abs_error": float(
                    (
                        zero_fill_decoded[:, : bandwidth.retained_q]
                        - observable_coefficients
                    )
                    .abs()
                    .max()
                ),
            },
            "zero_filled_part": {
                "status": "pass" if zero_filled_part_pass else "fail",
                "exact_zero": bool(zero_filled_part_pass),
                "coefficient_count": bandwidth.zero_filled_coefficient_count,
            },
        },
    }


def _heat_analytic_case(
    spec: ValidationSpec,
    *,
    case_id: str,
    nx: int,
    domain_length: float,
    dtype: torch.dtype,
    basis: str,
    modes: tuple[tuple[int, float, float], ...],
    constant: float,
) -> dict[str, Any]:
    target = spec.target_reference
    if not isinstance(target, HeatAnalyticReferenceSpec):
        raise TypeError("heat analytic checks require heat_analytic")
    system = target.reference_evolution.system
    nu = float(system.nu)
    time = float(target.reference_evolution.time)
    x = periodic_grid(nx, domain_length, dtype=dtype, device="cpu")
    values = torch.full((nx,), constant, dtype=dtype)
    expected = torch.full((nx,), constant, dtype=dtype)
    mode_values: list[int] = []
    multipliers: list[float] = []
    for mode, cosine_amplitude, sine_amplitude in modes:
        angular_wavenumber = (
            2.0 * math.pi * float(mode) / float(domain_length)
        )
        phase = angular_wavenumber * x
        multiplier = math.exp(
            -nu * angular_wavenumber * angular_wavenumber * time
        )
        values = values + (
            cosine_amplitude * torch.cos(phase)
            + sine_amplitude * torch.sin(phase)
        )
        expected = expected + multiplier * (
            cosine_amplitude * torch.cos(phase)
            + sine_amplitude * torch.sin(phase)
        )
        mode_values.append(mode)
        multipliers.append(multiplier)
    actual = solve_heat_exact(
        values,
        nu=nu,
        time=time,
        domain_length=domain_length,
    )
    actual_coefficients = torch.fft.rfft(actual, dim=-1, norm="forward")
    expected_coefficients = torch.fft.rfft(
        expected,
        dim=-1,
        norm="forward",
    )
    max_abs_error = float((actual - expected).abs().max())
    max_coefficient_abs_error = float(
        (actual_coefficients - expected_coefficients).abs().max()
    )
    if dtype == torch.float32:
        atol = float(spec.algebraic_tolerances.float32_atol)
        rtol = float(spec.algebraic_tolerances.float32_rtol)
    else:
        atol = float(spec.algebraic_tolerances.float64_atol)
        rtol = float(spec.algebraic_tolerances.float64_rtol)
    tolerance = atol + rtol * float(expected.abs().max())
    value_pass = bool(
        torch.allclose(actual, expected, atol=atol, rtol=rtol)
    )
    coefficient_pass = bool(
        torch.allclose(
            actual_coefficients,
            expected_coefficients,
            atol=atol,
            rtol=rtol,
        )
    )
    shape_pass = actual.shape == values.shape
    dtype_pass = actual.dtype == dtype
    device_pass = actual.device == torch.device("cpu")
    finite_pass = bool(torch.isfinite(actual).all())
    passed = (
        value_pass
        and coefficient_pass
        and shape_pass
        and dtype_pass
        and device_pass
        and finite_pass
    )
    return {
        "case_id": case_id,
        "basis": basis,
        "nx": nx,
        "domain_length": float(domain_length),
        "dtype": str(dtype).removeprefix("torch."),
        "mode": (
            0
            if not mode_values
            else mode_values[0]
            if len(mode_values) == 1
            else mode_values
        ),
        "expected_multiplier": (
            1.0
            if not multipliers
            else multipliers[0]
            if len(multipliers) == 1
            else multipliers
        ),
        "max_abs_error": max_abs_error,
        "max_coefficient_abs_error": max_coefficient_abs_error,
        "tolerance": tolerance,
        "shape_status": "pass" if shape_pass else "fail",
        "dtype_status": "pass" if dtype_pass else "fail",
        "device_status": "pass" if device_pass else "fail",
        "finite_status": "pass" if finite_pass else "fail",
        "status": "pass" if passed else "fail",
    }


def _heat_analytic_checks(spec: ValidationSpec) -> dict[str, Any]:
    cases = [
        _heat_analytic_case(
            spec,
            case_id="constant_odd_float64",
            nx=15,
            domain_length=1.0,
            dtype=torch.float64,
            basis="constant",
            modes=(),
            constant=0.375,
        ),
        _heat_analytic_case(
            spec,
            case_id="cosine_even_float64",
            nx=16,
            domain_length=1.0,
            dtype=torch.float64,
            basis="cosine",
            modes=((3, 0.7, 0.0),),
            constant=0.0,
        ),
        _heat_analytic_case(
            spec,
            case_id="sine_odd_nonunit_float64",
            nx=15,
            domain_length=2.5,
            dtype=torch.float64,
            basis="sine",
            modes=((2, 0.0, -0.55),),
            constant=0.0,
        ),
        _heat_analytic_case(
            spec,
            case_id="multimode_even_nonunit_float64",
            nx=16,
            domain_length=1.7,
            dtype=torch.float64,
            basis="constant_plus_sine_cosine",
            modes=((1, 0.7, -0.2), (3, -0.15, 0.4)),
            constant=0.2,
        ),
        _heat_analytic_case(
            spec,
            case_id="cosine_odd_nonunit_float32",
            nx=15,
            domain_length=2.0,
            dtype=torch.float32,
            basis="cosine",
            modes=((2, 0.4, 0.0),),
            constant=-0.1,
        ),
        _heat_analytic_case(
            spec,
            case_id="sine_even_nonunit_float32",
            nx=16,
            domain_length=2.2,
            dtype=torch.float32,
            basis="sine",
            modes=((3, 0.0, 0.35),),
            constant=0.1,
        ),
        _heat_analytic_case(
            spec,
            case_id="nyquist_cosine_unpaired",
            nx=16,
            domain_length=1.3,
            dtype=torch.float64,
            basis="nyquist_cosine_unpaired",
            modes=((8, 0.3, 0.0),),
            constant=0.25,
        ),
    ]
    return {
        "status": (
            "pass"
            if all(case["status"] == "pass" for case in cases)
            else "fail"
        ),
        "temporal_status": "analytic_exact",
        "cases": cases,
    }


def _reaction_diffusion_actual(
    values: torch.Tensor,
    *,
    case_id: str,
    nu: float,
    alpha: float,
    beta: float,
    time: float,
    dt: float,
    domain_length: float,
    nonlinear_filter: str,
) -> torch.Tensor:
    try:
        result = solve_reaction_diffusion(
            values,
            nu=nu,
            alpha=alpha,
            beta=beta,
            time=time,
            dt=dt,
            domain_length=domain_length,
            nonlinear_filter=nonlinear_filter,
        )
    except FloatingPointError as exc:
        raise _ValidationSolveFailure(
            {
                "status": "fail",
                "failure_kind": "nonfinite_reaction_diffusion_solve",
                "stage": "analytic_characterization",
                "case_id": case_id,
                "message": str(exc),
            }
        ) from exc
    if not bool(torch.isfinite(result.values).all()):
        raise _ValidationSolveFailure(
            {
                "status": "fail",
                "failure_kind": "nonfinite_reaction_diffusion_solve",
                "stage": "analytic_characterization",
                "case_id": case_id,
                "message": (
                    "reaction-diffusion characterization produced NaN/Inf"
                ),
            }
        )
    return result.values


def _reaction_diffusion_constant_case(
    spec: ValidationSpec,
    *,
    case_id: str,
    value: float,
    nx: int,
    domain_length: float,
    nonlinear_filter: str,
    steps: int,
) -> dict[str, Any]:
    target = spec.target_reference
    if not isinstance(
        target,
        ReactionDiffusionConvergenceReferenceSpec,
    ):
        raise TypeError(
            "reaction-diffusion characterization requires "
            "reaction_diffusion_convergence"
        )
    system = target.reference_evolution.system
    dt = float(system.dt)
    alpha = float(system.alpha)
    beta = float(system.beta)
    scalar = float(value)
    for _ in range(steps):
        scalar = scalar + dt * alpha * scalar - dt * beta * scalar**3
    initial = torch.full(
        (2, nx),
        float(value),
        dtype=torch.float64,
        device="cpu",
    )
    actual = _reaction_diffusion_actual(
        initial,
        case_id=case_id,
        nu=float(system.nu),
        alpha=alpha,
        beta=beta,
        time=steps * dt,
        dt=dt,
        domain_length=domain_length,
        nonlinear_filter=nonlinear_filter,
    )
    expected = torch.full_like(actual, scalar)
    passed = _allclose(actual, expected, spec)
    return {
        "case_id": case_id,
        "characterization": "independent_scalar_recurrence",
        "initial_constant": float(value),
        "expected_final_constant": scalar,
        "nx": nx,
        "grid_parity": "even" if nx % 2 == 0 else "odd",
        "domain_length": float(domain_length),
        "dt": dt,
        "step_count": steps,
        "nonlinear_filter": nonlinear_filter,
        "max_abs_error": float((actual - expected).abs().max()),
        "finite_status": "pass",
        "status": "pass" if passed else "fail",
    }


def _reaction_diffusion_equilibrium_case(
    spec: ValidationSpec,
    *,
    sign: int,
    nonlinear_filter: str,
) -> dict[str, Any]:
    target = spec.target_reference
    if not isinstance(
        target,
        ReactionDiffusionConvergenceReferenceSpec,
    ):
        raise TypeError(
            "reaction-diffusion characterization requires "
            "reaction_diffusion_convergence"
        )
    system = target.reference_evolution.system
    alpha = float(system.alpha)
    beta = float(system.beta)
    case_id = (
        f"equilibrium_{'positive' if sign > 0 else 'negative'}_"
        f"{nonlinear_filter}"
    )
    if alpha <= 0.0 or beta <= 0.0:
        return {
            "case_id": case_id,
            "characterization": "nonzero_constant_equilibrium",
            "applicable": False,
            "reason": "requires alpha > 0 and beta > 0",
            "status": "not_applicable",
        }
    value = float(sign) * math.sqrt(alpha / beta)
    dt = float(system.dt)
    nx = 15 if sign > 0 else 16
    initial = torch.full((1, nx), value, dtype=torch.float64)
    actual = _reaction_diffusion_actual(
        initial,
        case_id=case_id,
        nu=float(system.nu),
        alpha=alpha,
        beta=beta,
        time=3 * dt,
        dt=dt,
        domain_length=2.3,
        nonlinear_filter=nonlinear_filter,
    )
    expected = torch.full_like(actual, value)
    passed = _allclose(actual, expected, spec)
    return {
        "case_id": case_id,
        "characterization": "nonzero_constant_equilibrium",
        "applicable": True,
        "equilibrium": value,
        "nx": nx,
        "domain_length": 2.3,
        "dt": dt,
        "step_count": 3,
        "nonlinear_filter": nonlinear_filter,
        "max_abs_error": float((actual - expected).abs().max()),
        "finite_status": "pass",
        "status": "pass" if passed else "fail",
    }


def _reaction_diffusion_linear_mode_case(
    spec: ValidationSpec,
    *,
    case_id: str,
    nx: int,
    domain_length: float,
    mode: int,
    basis: str,
    nonlinear_filter: str,
) -> dict[str, Any]:
    target = spec.target_reference
    if not isinstance(
        target,
        ReactionDiffusionConvergenceReferenceSpec,
    ):
        raise TypeError(
            "reaction-diffusion characterization requires "
            "reaction_diffusion_convergence"
        )
    system = target.reference_evolution.system
    dt = float(system.dt)
    nu = float(system.nu)
    alpha = float(system.alpha)
    x = periodic_grid(
        nx,
        domain_length,
        dtype=torch.float64,
        device="cpu",
    )
    angular_wavenumber = 2.0 * math.pi * mode / domain_length
    if basis == "cosine":
        initial_one = 0.4 * torch.cos(angular_wavenumber * x)
    elif basis == "sine":
        initial_one = -0.35 * torch.sin(angular_wavenumber * x)
    else:
        raise ValueError(f"unsupported linear-mode basis: {basis}")
    initial = initial_one.unsqueeze(0)
    multiplier = (1.0 + dt * alpha) / (
        1.0 + dt * nu * angular_wavenumber**2
    )
    expected = multiplier * initial
    actual = _reaction_diffusion_actual(
        initial,
        case_id=case_id,
        nu=nu,
        alpha=alpha,
        beta=0.0,
        time=dt,
        dt=dt,
        domain_length=domain_length,
        nonlinear_filter=nonlinear_filter,
    )
    passed = _allclose(actual, expected, spec)
    return {
        "case_id": case_id,
        "characterization": "beta_zero_linear_mode_one_step",
        "beta": 0.0,
        "basis": basis,
        "mode": mode,
        "physical_angular_wavenumber": angular_wavenumber,
        "expected_multiplier": multiplier,
        "nx": nx,
        "grid_parity": "even" if nx % 2 == 0 else "odd",
        "domain_length": float(domain_length),
        "dt": dt,
        "step_count": 1,
        "nonlinear_filter": nonlinear_filter,
        "max_abs_error": float((actual - expected).abs().max()),
        "finite_status": "pass",
        "status": "pass" if passed else "fail",
    }


def _reaction_diffusion_characterization(
    spec: ValidationSpec,
) -> dict[str, Any]:
    target = spec.target_reference
    if not isinstance(
        target,
        ReactionDiffusionConvergenceReferenceSpec,
    ):
        raise TypeError(
            "reaction-diffusion characterization requires "
            "reaction_diffusion_convergence"
        )
    system = target.reference_evolution.system
    dt = float(system.dt)
    zero_initial = torch.zeros((2, 15), dtype=torch.float64)
    zero_actual = _reaction_diffusion_actual(
        zero_initial,
        case_id="zero_equilibrium",
        nu=float(system.nu),
        alpha=float(system.alpha),
        beta=float(system.beta),
        time=4 * dt,
        dt=dt,
        domain_length=1.9,
        nonlinear_filter="two_thirds",
    )
    zero_exact = bool(torch.equal(zero_actual, zero_initial))
    zero_case = {
        "case_id": "zero_equilibrium",
        "characterization": "zero_equilibrium",
        "nx": 15,
        "domain_length": 1.9,
        "dt": dt,
        "step_count": 4,
        "nonlinear_filter": "two_thirds",
        "exact_zero": zero_exact,
        "max_abs_error": float(zero_actual.abs().max()),
        "finite_status": "pass",
        "status": "pass" if zero_exact else "fail",
    }
    constant_cases = [
        _reaction_diffusion_constant_case(
            spec,
            case_id="positive_odd_none",
            value=0.25,
            nx=15,
            domain_length=2.5,
            nonlinear_filter="none",
            steps=4,
        ),
        _reaction_diffusion_constant_case(
            spec,
            case_id="negative_even_two_thirds",
            value=-0.4,
            nx=16,
            domain_length=1.7,
            nonlinear_filter="two_thirds",
            steps=5,
        ),
    ]
    equilibrium_cases = [
        _reaction_diffusion_equilibrium_case(
            spec,
            sign=1,
            nonlinear_filter="none",
        ),
        _reaction_diffusion_equilibrium_case(
            spec,
            sign=-1,
            nonlinear_filter="two_thirds",
        ),
    ]
    linear_mode_cases = [
        _reaction_diffusion_linear_mode_case(
            spec,
            case_id="linear_cosine_odd_nonunit_none",
            nx=15,
            domain_length=2.5,
            mode=2,
            basis="cosine",
            nonlinear_filter="none",
        ),
        _reaction_diffusion_linear_mode_case(
            spec,
            case_id="linear_sine_even_nonunit_two_thirds",
            nx=16,
            domain_length=1.7,
            mode=3,
            basis="sine",
            nonlinear_filter="two_thirds",
        ),
    ]
    required_statuses = [
        zero_case["status"],
        *(case["status"] for case in constant_cases),
        *(
            case["status"]
            for case in equilibrium_cases
            if case["status"] != "not_applicable"
        ),
        *(case["status"] for case in linear_mode_cases),
    ]
    return {
        "schema_version": (
            "pol-reaction-diffusion-characterization-v1"
        ),
        "expected_value_construction": (
            "independent_scalar_and_fourier_mode_algebra"
        ),
        "status": (
            "pass"
            if all(value == "pass" for value in required_statuses)
            else "fail"
        ),
        "zero_equilibrium": zero_case,
        "constant_scalar_recurrence": constant_cases,
        "nonzero_equilibria": equilibrium_cases,
        "beta_zero_linear_modes": linear_mode_cases,
    }


TimeCandidateSpec = (
    BurgersTimeCandidateSpec | ReactionDiffusionTimeCandidateSpec
)


def _candidate_evolution(
    spec: ValidationSpec,
    candidate: TimeCandidateSpec,
) -> dict[str, Any]:
    target = spec.target_reference
    if not isinstance(
        target,
        (
            BurgersConvergenceReferenceSpec,
            ReactionDiffusionConvergenceReferenceSpec,
        ),
    ):
        raise TypeError(
            "candidate evolution requires a time-refined target reference"
        )
    base = target.reference_evolution.model_dump(mode="json")
    system = dict(base["system"])
    if isinstance(candidate, BurgersTimeCandidateSpec):
        system.update(
            {
                "solver": candidate.solver,
                "dt": candidate.dt,
                "fine_dt": candidate.fine_dt,
                "dealias": candidate.dealias,
            }
        )
    elif isinstance(candidate, ReactionDiffusionTimeCandidateSpec):
        system.update(
            {
                "solver": candidate.solver,
                "dt": candidate.dt,
                "nonlinear_filter": candidate.nonlinear_filter,
            }
        )
    else:
        raise TypeError(f"unsupported time candidate: {type(candidate).__name__}")
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


def _passes(
    metrics: dict[str, float],
    tolerances: ReferenceToleranceSpec,
) -> bool:
    return (
        metrics["mean_relative_l2"] <= tolerances.mean_relative_l2
        and metrics["max_relative_l2"] <= tolerances.max_relative_l2
        and metrics["low_mode_relative_l2"]
        <= tolerances.low_mode_relative_l2
    )


def _coarsest_stable_index(rows: list[dict[str, Any]]) -> int | None:
    # A candidate is accepted only if every refinement after it is also within
    # tolerance.  This prevents selecting an accidentally good nonmonotone pair.
    for index in range(len(rows)):
        if all(row["status"] == "pass" for row in rows[index:]):
            return index
    return None


def _target_time_refinement_proof(
    target: (
        BurgersConvergenceReferenceSpec
        | ReactionDiffusionConvergenceReferenceSpec
    ),
    candidates: list[TimeCandidateSpec],
) -> dict[str, Any]:
    values = [
        candidate.model_dump(mode="json")
        for candidate in candidates
    ]
    evolution_time = float(target.reference_evolution.time)
    if isinstance(target, BurgersConvergenceReferenceSpec):
        return burgers_refinement_proof(
            values,
            evolution_time=evolution_time,
        )
    return reaction_diffusion_refinement_proof(
        values,
        evolution_time=evolution_time,
    )


def _verify_solver_metadata(
    system_kind: str,
    metadata: dict[str, Any],
    condition: dict[str, Any],
) -> None:
    if system_kind == "burgers":
        actual = {
            name: metadata.get(name)
            for name in (
                "solver",
                "requested_outer_dt",
                "requested_fine_dt",
                "outer_step_count",
                "effective_substep",
                "substeps_per_outer",
                "dealias",
            )
        }
    elif system_kind == "reaction_diffusion":
        actual = {
            "solver": metadata.get("solver"),
            "dt": metadata.get("requested_dt"),
            "nonlinear_filter": metadata.get("nonlinear_filter"),
        }
    else:
        raise ValueError(
            f"unsupported runtime numerical condition: {system_kind}"
        )
    if stable_object_hash(actual) != stable_object_hash(condition):
        raise ValueError(
            f"{system_kind} runtime metadata disagrees with the canonical "
            "numerical condition"
        )


def _checked_evolve(
    initial: torch.Tensor,
    evolution: dict[str, Any],
    *,
    domain_length: float,
    stage: str,
    candidate_index: int,
    nx: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    system = evolution.get("system")
    system_kind = (
        str(system.get("kind"))
        if isinstance(system, dict)
        else "unknown"
    )
    try:
        solution, metadata = evolve(
            initial,
            evolution,
            domain_length=domain_length,
        )
    except FloatingPointError as exc:
        raise _ValidationSolveFailure(
            {
                "status": "fail",
                "failure_kind": "nonfinite_solver_state",
                "stage": stage,
                "system_kind": system_kind,
                "candidate_index": candidate_index,
                "nx": nx,
                "numerical_condition": (
                    dict(system) if isinstance(system, dict) else system
                ),
                "message": str(exc),
            }
        ) from exc
    if not bool(torch.isfinite(solution).all()):
        raise _ValidationSolveFailure(
            {
                "status": "fail",
                "failure_kind": "nonfinite_solver_state",
                "stage": stage,
                "system_kind": system_kind,
                "candidate_index": candidate_index,
                "nx": nx,
                "numerical_condition": (
                    dict(system) if isinstance(system, dict) else system
                ),
                "message": "solver returned a state containing NaN/Inf",
            }
        )
    return solution, metadata


def _time_sequence_convergence(
    spec: ValidationSpec,
    *,
    initial: torch.Tensor,
    candidates: list[TimeCandidateSpec],
    nx: int,
    reference_candidate_index: int,
    tolerances: ReferenceToleranceSpec,
    boundary: str,
) -> _TimeSequenceResult:
    """Solve and score one already-validated fixed-method sequence."""
    target = spec.target_reference
    if not isinstance(
        target,
        (
            BurgersConvergenceReferenceSpec,
            ReactionDiffusionConvergenceReferenceSpec,
        ),
    ):
        raise TypeError(
            "time convergence requires a time-refined target reference"
        )
    proof = _target_time_refinement_proof(
        target,
        candidates,
    )
    conditions = proof["ordered_candidates"]
    system_kind = target.reference_evolution.system.kind
    solutions: list[torch.Tensor] = []
    metadata_rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        solution, metadata = _checked_evolve(
            initial,
            _candidate_evolution(spec, candidate),
            domain_length=float(spec.domain.length),
            stage=boundary,
            candidate_index=index,
            nx=nx,
        )
        require_cpu_tensor(
            solution,
            boundary=boundary,
            name=f"solution_candidate_{index}",
        )
        _verify_solver_metadata(
            system_kind,
            metadata,
            conditions[index],
        )
        solutions.append(solution)
        metadata_rows.append(metadata)
    rows: list[dict[str, Any]] = []
    for index in range(len(candidates) - 1):
        metrics = _pair_metrics(
            solutions[index],
            solutions[index + 1],
            q=int(target.q_reference_check),
            domain_length=float(spec.domain.length),
        )
        rows.append(
            make_convergence_row(
                check_kind="temporal",
                candidate_axis="numerical_condition",
                coarse_candidate_index=index,
                fine_candidate_index=index + 1,
                coarse_reference_candidate_index=(
                    reference_candidate_index
                ),
                fine_reference_candidate_index=(
                    reference_candidate_index
                ),
                coarse_nx=nx,
                fine_nx=nx,
                coarse_condition_index=index,
                fine_condition_index=index + 1,
                coarse_condition=conditions[index],
                fine_condition=conditions[index + 1],
                common_nx=nx,
                metrics=metrics,
                status=(
                    "pass"
                    if _passes(metrics, tolerances)
                    else "fail"
                ),
            )
        )
    return _TimeSequenceResult(
        refinement_proof=proof,
        conditions=conditions,
        solutions=solutions,
        runtime_metadata=metadata_rows,
        rows=rows,
        selected_index=_coarsest_stable_index(rows),
    )


def _time_refined_reference_convergence(
    spec: ValidationSpec,
    archive: InitialConditionArchive,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    target = spec.target_reference
    if not isinstance(
        target,
        (
            BurgersConvergenceReferenceSpec,
            ReactionDiffusionConvergenceReferenceSpec,
        ),
    ):
        raise TypeError(
            "time-refined reference convergence requires Burgers or "
            "reaction-diffusion"
        )
    require_cpu_tensors(
        archive.__dict__,
        boundary="validation reference-convergence input",
        name="archive",
    )
    L = float(spec.domain.length)
    ids = torch.tensor(
        target.calibration_sample_ids,
        dtype=torch.long,
        device=archive.values.device,
    )
    initial_master = archive.values.index_select(0, ids)
    nx_values = [int(value) for value in target.reference_nx_candidates]
    candidates = list(target.time_candidates)
    refinement_proof = _target_time_refinement_proof(
        target,
        candidates,
    )
    conditions = refinement_proof["ordered_candidates"]
    system_kind = target.reference_evolution.system.kind
    finest_candidate = candidates[-1]
    finest_condition_index = len(conditions) - 1
    finest_condition = conditions[finest_condition_index]
    spatial_solutions: dict[int, torch.Tensor] = {}
    metadata: dict[str, Any] = {}
    for reference_index, nx in enumerate(nx_values):
        initial = spectral_resample_periodic(initial_master, nx, domain_length=L)
        solution, meta = _checked_evolve(
            initial,
            _candidate_evolution(spec, finest_candidate),
            domain_length=L,
            stage="spatial_reference_convergence",
            candidate_index=finest_condition_index,
            nx=nx,
        )
        require_cpu_tensor(
            solution,
            boundary="validation spatial reference-convergence solve",
            name=f"solution_nx_{nx}",
        )
        _verify_solver_metadata(system_kind, meta, finest_condition)
        spatial_solutions[nx] = solution
        metadata[f"spatial_{reference_index}"] = meta
    rows: list[dict[str, Any]] = []
    spatial_rows: list[dict[str, Any]] = []
    for index, (coarse_nx, fine_nx) in enumerate(
        zip(nx_values[:-1], nx_values[1:])
    ):
        metrics = _pair_metrics(
            spatial_solutions[coarse_nx],
            spatial_solutions[fine_nx],
            q=int(target.q_reference_check),
            domain_length=L,
        )
        row = make_convergence_row(
            check_kind="spatial",
            candidate_axis="reference_resolution",
            coarse_candidate_index=index,
            fine_candidate_index=index + 1,
            coarse_reference_candidate_index=index,
            fine_reference_candidate_index=index + 1,
            coarse_nx=coarse_nx,
            fine_nx=fine_nx,
            coarse_condition_index=finest_condition_index,
            fine_condition_index=finest_condition_index,
            coarse_condition=finest_condition,
            fine_condition=finest_condition,
            common_nx=coarse_nx,
            metrics=metrics,
            status=(
                "pass"
                if _passes(metrics, target.reference_tolerances)
                else "fail"
            ),
        )
        spatial_rows.append(row)
        rows.append(row)
    spatial_index = _coarsest_stable_index(spatial_rows)
    selected_nx = (
        None if spatial_index is None else nx_values[spatial_index]
    )

    finest_nx = nx_values[-1]
    finest_reference_index = len(nx_values) - 1
    initial_finest = spectral_resample_periodic(
        initial_master,
        finest_nx,
        domain_length=L,
    )
    temporal = _time_sequence_convergence(
        spec,
        initial=initial_finest,
        candidates=candidates,
        nx=finest_nx,
        reference_candidate_index=finest_reference_index,
        tolerances=target.reference_tolerances,
        boundary="validation temporal reference-convergence solve",
    )
    if stable_object_hash(temporal.conditions) != stable_object_hash(
        conditions
    ):
        raise ValueError(
            "primary time convergence conditions changed during reuse"
        )
    temporal_solutions = temporal.solutions
    temporal_metadata = temporal.runtime_metadata
    temporal_rows = temporal.rows
    rows.extend(temporal_rows)
    selected_time_index = temporal.selected_index

    joint_status = "fail"
    joint_row: dict[str, Any] | None = None
    if selected_nx is not None and selected_time_index is not None:
        selected_initial = spectral_resample_periodic(
            initial_master, selected_nx, domain_length=L
        )
        selected_solution, selected_meta = _checked_evolve(
            selected_initial,
            _candidate_evolution(spec, candidates[selected_time_index]),
            domain_length=L,
            stage="joint_reference_convergence",
            candidate_index=selected_time_index,
            nx=selected_nx,
        )
        require_cpu_tensor(
            selected_solution,
            boundary="validation joint reference-convergence solve",
            name="selected_solution",
        )
        _verify_solver_metadata(
            system_kind,
            selected_meta,
            conditions[selected_time_index],
        )
        joint_metrics = _pair_metrics(
            selected_solution,
            temporal_solutions[-1],
            q=int(target.q_reference_check),
            domain_length=L,
        )
        joint_status = (
            "pass"
            if _passes(joint_metrics, target.reference_tolerances)
            else "fail"
        )
        joint_row = make_convergence_row(
            check_kind="joint",
            candidate_axis="coupled",
            coarse_candidate_index=selected_time_index,
            fine_candidate_index=finest_condition_index,
            coarse_reference_candidate_index=spatial_index,
            fine_reference_candidate_index=finest_reference_index,
            coarse_nx=selected_nx,
            fine_nx=finest_nx,
            coarse_condition_index=selected_time_index,
            fine_condition_index=finest_condition_index,
            coarse_condition=conditions[selected_time_index],
            fine_condition=finest_condition,
            common_nx=selected_nx,
            metrics=joint_metrics,
            status=joint_status,
        )
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
            else conditions[selected_time_index]
        ),
        "finest_reference_nx": finest_nx,
        "finest_time_candidate": finest_condition,
        "candidate_refinement_proof": refinement_proof,
        "solver_metadata": {
            **metadata,
            "temporal": temporal_metadata,
        },
        "joint_row": joint_row,
        "rows": rows,
    }
    return result, rows


def _cross_solver_self_evidence(
    result: _TimeSequenceResult,
) -> dict[str, Any]:
    finest_index = len(result.conditions) - 1
    selected_index = result.selected_index
    return {
        "status": "pass" if selected_index is not None else "fail",
        "ordered_candidates": result.conditions,
        "candidate_refinement_proof": result.refinement_proof,
        "rows": result.rows,
        "rows_hash": stable_object_hash(result.rows),
        "pairwise_row_hashes": [
            row["row_hash"] for row in result.rows
        ],
        "selected_candidate_index": selected_index,
        "selected_condition": (
            None
            if selected_index is None
            else result.conditions[selected_index]
        ),
        "finest_candidate_index": finest_index,
        "finest_condition": result.conditions[finest_index],
        "runtime_solver_metadata": result.runtime_metadata,
    }


def _validate_cross_solver_check_against_spec(
    spec: ValidationSpec,
    block: dict[str, Any],
) -> None:
    target = spec.target_reference
    if not isinstance(target, BurgersConvergenceReferenceSpec):
        raise ValueError("cross-solver evidence requires a Burgers spec")
    diagnostic = target.cross_solver_validation
    if not isinstance(
        diagnostic,
        EnabledBurgersCrossSolverValidationSpec,
    ):
        raise ValueError("cross-solver evidence is disabled in the spec")
    system = target.reference_evolution.system
    expected_context = {
        "system_kind": "burgers",
        "invariant_parameters": canonical_invariant_parameters(
            "burgers",
            system.model_dump(mode="json"),
        ),
        "evolution_time": float(target.reference_evolution.time),
        "domain_length": float(spec.domain.length),
        "dtype": spec.samples.dtype,
        "dealias": diagnostic.context.dealias,
        "common_nx": int(target.reference_nx_candidates[-1]),
        "reference_candidate_index": (
            len(target.reference_nx_candidates) - 1
        ),
        "sample_ids": [
            int(value) for value in target.calibration_sample_ids
        ],
    }
    if stable_object_hash(block.get("context")) != stable_object_hash(
        expected_context
    ):
        raise ValueError(
            "cross-solver validation context disagrees with the resolved spec"
        )
    if stable_object_hash(block.get("tolerances")) != stable_object_hash(
        diagnostic.tolerances.model_dump(mode="json")
    ):
        raise ValueError(
            "cross-solver tolerances disagree with the resolved spec"
        )
    self_convergence = block.get("self_convergence")
    if not isinstance(self_convergence, dict):
        raise ValueError("cross-solver self-convergence evidence is missing")
    for family in ("split_step", "etdrk4"):
        expected_proof = burgers_refinement_proof(
            [
                candidate.model_dump(mode="json")
                for candidate in getattr(
                    diagnostic.solvers,
                    family,
                ).candidates
            ],
            evolution_time=float(target.reference_evolution.time),
        )
        evidence = self_convergence.get(family)
        if (
            not isinstance(evidence, dict)
            or stable_object_hash(evidence.get("ordered_candidates"))
            != stable_object_hash(expected_proof["ordered_candidates"])
            or stable_object_hash(
                evidence.get("candidate_refinement_proof")
            )
            != stable_object_hash(expected_proof)
        ):
            raise ValueError(
                f"cross-solver {family} candidates disagree with the "
                "resolved spec"
            )


def _burgers_cross_solver_validation(
    spec: ValidationSpec,
    archive: InitialConditionArchive,
) -> dict[str, Any]:
    """Build supporting, symmetric split-step/ETDRK4 evidence."""
    target = spec.target_reference
    if not isinstance(target, BurgersConvergenceReferenceSpec):
        raise TypeError("cross-solver validation requires burgers_convergence")
    diagnostic = target.cross_solver_validation
    if not isinstance(
        diagnostic,
        EnabledBurgersCrossSolverValidationSpec,
    ):
        raise TypeError("cross-solver validation is not enabled")
    require_cpu_tensors(
        archive.__dict__,
        boundary="cross-solver validation input",
        name="archive",
    )
    domain_length = float(spec.domain.length)
    sample_ids = [
        int(value) for value in target.calibration_sample_ids
    ]
    ids = torch.tensor(
        sample_ids,
        dtype=torch.long,
        device=archive.values.device,
    )
    finest_nx = int(target.reference_nx_candidates[-1])
    reference_index = len(target.reference_nx_candidates) - 1
    initial_master = archive.values.index_select(0, ids)
    initial_finest = spectral_resample_periodic(
        initial_master,
        finest_nx,
        domain_length=domain_length,
    )

    sequence_results: dict[str, _TimeSequenceResult] = {}
    for family in ("split_step", "etdrk4"):
        family_spec = getattr(diagnostic.solvers, family)
        sequence_results[family] = _time_sequence_convergence(
            spec,
            initial=initial_finest,
            candidates=list(family_spec.candidates),
            nx=finest_nx,
            reference_candidate_index=reference_index,
            tolerances=diagnostic.tolerances,
            boundary=f"cross-solver {family} self-convergence solve",
        )
    self_convergence = {
        family: _cross_solver_self_evidence(sequence_results[family])
        for family in ("split_step", "etdrk4")
    }
    finest_conditions = {
        family: sequence_results[family].conditions[-1]
        for family in ("split_step", "etdrk4")
    }
    self_pass = all(
        evidence["status"] == "pass"
        for evidence in self_convergence.values()
    )
    if self_pass:
        discrepancy_metrics = symmetric_field_discrepancy(
            sequence_results["split_step"].solutions[-1],
            sequence_results["etdrk4"].solutions[-1],
            q=int(target.q_reference_check),
            domain_length=domain_length,
        )
        discrepancy_status = (
            "pass"
            if _passes(discrepancy_metrics, diagnostic.tolerances)
            else "fail"
        )
        not_evaluated_reason = None
    else:
        discrepancy_metrics = None
        discrepancy_status = "not_evaluated"
        not_evaluated_reason = (
            "self_convergence_must_pass_before_cross_comparison"
        )
    system = target.reference_evolution.system
    context = {
        "system_kind": "burgers",
        "invariant_parameters": canonical_invariant_parameters(
            "burgers",
            system.model_dump(mode="json"),
        ),
        "evolution_time": float(target.reference_evolution.time),
        "domain_length": domain_length,
        "dtype": spec.samples.dtype,
        "dealias": diagnostic.context.dealias,
        "common_nx": finest_nx,
        "reference_candidate_index": reference_index,
        "sample_ids": sample_ids,
    }
    block = {
        "schema_version": CROSS_SOLVER_CHECK_SCHEMA_VERSION,
        "enabled": True,
        "status": (
            "pass"
            if self_pass and discrepancy_status == "pass"
            else "fail"
        ),
        "role": "supporting_evidence_not_primary_allowed_refinement",
        "context": context,
        "tolerances": diagnostic.tolerances.model_dump(mode="json"),
        "self_convergence": self_convergence,
        "finest_conditions": finest_conditions,
        "discrepancy_definition": CROSS_SOLVER_METRIC_DEFINITION,
        "discrepancy_metrics": discrepancy_metrics,
        "discrepancy_status": discrepancy_status,
        "discrepancy_not_evaluated_reason": not_evaluated_reason,
        "discrepancy_evidence_hash": (
            cross_solver_discrepancy_evidence_hash(
                finest_conditions=finest_conditions,
                common_nx=finest_nx,
                sample_ids=sample_ids,
                metrics=discrepancy_metrics,
                not_evaluated_reason=not_evaluated_reason,
            )
        ),
    }
    validate_cross_solver_validation_block(block)
    _validate_cross_solver_check_against_spec(spec, block)
    return block


def _heat_pair_metrics(
    coarse: torch.Tensor,
    fine: torch.Tensor,
    *,
    q: int,
    domain_length: float,
) -> dict[str, float]:
    coarse_common = spectral_resample_periodic(
        coarse,
        int(fine.shape[-1]),
        domain_length=domain_length,
    )
    _, relative = samplewise_l2_errors(
        coarse_common,
        fine,
        domain_length=domain_length,
    )
    coarse_coeff = real_fourier_analysis(
        coarse_common,
        q,
        domain_length=domain_length,
    )
    fine_coeff = real_fourier_analysis(
        fine,
        q,
        domain_length=domain_length,
    )
    denominator = torch.linalg.vector_norm(fine_coeff, dim=-1).clamp_min(
        torch.finfo(fine_coeff.dtype).eps
    )
    low_relative = torch.linalg.vector_norm(
        coarse_coeff - fine_coeff,
        dim=-1,
    ) / denominator
    return {
        "mean_relative_l2": float(relative.mean()),
        "max_relative_l2": float(relative.max()),
        "low_mode_relative_l2": float(low_relative.mean()),
    }


def _heat_reference_convergence(
    spec: ValidationSpec,
    archive: InitialConditionArchive,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    target = spec.target_reference
    if not isinstance(target, HeatAnalyticReferenceSpec):
        raise TypeError("heat reference convergence requires heat_analytic")
    require_cpu_tensors(
        archive.__dict__,
        boundary="validation heat reference-convergence input",
        name="archive",
    )
    L = float(spec.domain.length)
    ids = torch.tensor(
        target.calibration_sample_ids,
        dtype=torch.long,
        device=archive.values.device,
    )
    initial_master = archive.values.index_select(0, ids)
    nx_values = [int(value) for value in target.reference_nx_candidates]
    evolution = target.reference_evolution.model_dump(mode="json")
    condition = canonical_numerical_condition(
        "heat",
        target.reference_evolution.system.model_dump(mode="json"),
    )
    solutions: dict[int, torch.Tensor] = {}
    metadata: dict[str, Any] = {}
    for nx in nx_values:
        initial = spectral_resample_periodic(
            initial_master,
            nx,
            domain_length=L,
        )
        solution, solver_metadata = _checked_evolve(
            initial,
            evolution,
            domain_length=L,
            stage="heat_spatial_reference_convergence",
            candidate_index=0,
            nx=nx,
        )
        require_cpu_tensor(
            solution,
            boundary="validation heat spatial reference-convergence solve",
            name=f"solution_nx_{nx}",
        )
        solutions[nx] = solution
        metadata[f"spatial_{nx}"] = solver_metadata

    rows: list[dict[str, Any]] = []
    spatial_rows: list[dict[str, Any]] = []
    for index, (coarse_nx, fine_nx) in enumerate(
        zip(nx_values[:-1], nx_values[1:])
    ):
        metrics = _heat_pair_metrics(
            solutions[coarse_nx],
            solutions[fine_nx],
            q=int(target.q_reference_check),
            domain_length=L,
        )
        row = make_convergence_row(
            check_kind="spatial",
            candidate_axis="reference_resolution",
            coarse_candidate_index=index,
            fine_candidate_index=index + 1,
            coarse_reference_candidate_index=index,
            fine_reference_candidate_index=index + 1,
            coarse_nx=coarse_nx,
            fine_nx=fine_nx,
            coarse_condition_index=0,
            fine_condition_index=0,
            coarse_condition=condition,
            fine_condition=condition,
            common_nx=fine_nx,
            metrics=metrics,
            status=(
                "pass"
                if _passes(
                    metrics,
                    target.reference_tolerances,
                )
                else "fail"
            ),
        )
        spatial_rows.append(row)
        rows.append(row)
    spatial_index = _coarsest_stable_index(spatial_rows)
    selected_nx = (
        None if spatial_index is None else nx_values[spatial_index]
    )
    finest_nx = nx_values[-1]
    finest_reference_index = len(nx_values) - 1
    joint_row: dict[str, Any] | None = None
    joint_status = "fail"
    if selected_nx is not None:
        joint_metrics = _heat_pair_metrics(
            solutions[selected_nx],
            solutions[finest_nx],
            q=int(target.q_reference_check),
            domain_length=L,
        )
        joint_status = (
            "pass"
            if _passes(
                joint_metrics,
                target.reference_tolerances,
            )
            else "fail"
        )
        joint_row = make_convergence_row(
            check_kind="joint",
            candidate_axis="coupled",
            coarse_candidate_index=spatial_index,
            fine_candidate_index=finest_reference_index,
            coarse_reference_candidate_index=spatial_index,
            fine_reference_candidate_index=finest_reference_index,
            coarse_nx=selected_nx,
            fine_nx=finest_nx,
            coarse_condition_index=0,
            fine_condition_index=0,
            coarse_condition=condition,
            fine_condition=condition,
            common_nx=finest_nx,
            metrics=joint_metrics,
            status=joint_status,
        )
        rows.append(joint_row)
    return (
        {
            "status": (
                "pass"
                if selected_nx is not None and joint_status == "pass"
                else "fail"
            ),
            "spatial_status": (
                "pass" if selected_nx is not None else "fail"
            ),
            "temporal_status": "analytic_exact",
            "joint_status": joint_status,
            "selected_reference_nx": selected_nx,
            "selected_reference_candidate_index": spatial_index,
            "selected_numerical_condition": condition,
            "selected_numerical_condition_index": 0,
            "finest_reference_nx": finest_nx,
            "solver_metadata": metadata,
            "joint_row": joint_row,
            "rows": rows,
        },
        rows,
    )


def _reference_convergence(
    spec: ValidationSpec,
    archive: InitialConditionArchive,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    target = spec.target_reference
    if isinstance(
        target,
        (
            BurgersConvergenceReferenceSpec,
            ReactionDiffusionConvergenceReferenceSpec,
        ),
    ):
        return _time_refined_reference_convergence(spec, archive)
    if isinstance(target, HeatAnalyticReferenceSpec):
        return _heat_reference_convergence(spec, archive)
    raise TypeError(
        f"unsupported target-reference validation: {type(target).__name__}"
    )


def _master_payload(archive: InitialConditionArchive) -> dict[str, Any]:
    require_cpu_tensors(
        archive.__dict__,
        boundary="master initial-condition archive publication",
        name="archive",
    )
    return {
        "schema_version": "pol-initial-condition-archive-v4",
        **execution_device_policy(),
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
    if payload.get("schema_version") != "pol-initial-condition-archive-v4":
        raise ValueError(
            "unsupported initial-condition archive schema; P0-04 requires v4"
        )
    verify_execution_device_policy(
        payload,
        boundary="master initial-condition archive",
    )
    require_cpu_tensors(
        payload,
        boundary="master initial-condition archive verification",
        name="master",
    )
    sample_ids = payload.get("sample_ids")
    values = payload.get("values")
    fourier = payload.get("fourier")
    if not all(isinstance(value, torch.Tensor) for value in (sample_ids, values, fourier)):
        raise ValueError("initial-condition archive tensors are missing")
    total = int(spec.samples.total_samples)
    nx = max(
        int(value)
        for value in spec.target_reference.reference_nx_candidates
    )
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
    if metadata.get("device") != "cpu":
        raise ValueError("initial-condition archive metadata device must be cpu")
    verify_execution_device_policy(
        metadata,
        boundary="master initial-condition archive metadata",
    )
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
        "schema_version": "pol-master-initial-condition-binding-v3",
        **execution_device_policy(),
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
        "matched_model1_pipeline",
        "field_quadrature",
    )
    statuses = {
        name: checks[name]["status"]
        for name in general_names
    }
    samples = spec.samples
    return {
        "schema_version": "pol-validation-foundation-contract-v8",
        **execution_device_policy(),
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
        "calibration_provenance": _calibration_provenance(
            spec,
            sample_ids=master_payload["sample_ids"],
        ),
        "master_initial_conditions": _master_archive_binding(master_payload),
        "general_foundation_checks": {
            "status": (
                "pass"
                if all(value == "pass" for value in statuses.values())
                else "fail"
            ),
            "checks": statuses,
        },
        "fixed_decoder_bandwidth_contract": checks["fixed_decoder"][
            "zero_fill_characterization"
        ],
        "matched_model1_pipeline_contract": model1_foundation_summary(
            checks["matched_model1_pipeline"]
        ),
        "field_quadrature_contract": field_quadrature_foundation_summary(
            checks["field_quadrature"]
        ),
    }


def _target_reference_contract(
    spec: ValidationSpec,
    convergence: dict[str, Any],
) -> dict[str, Any]:
    target = spec.target_reference
    nx_candidates = [
        int(value) for value in target.reference_nx_candidates
    ]
    reference_index = convergence.get("selected_reference_candidate_index")
    reference_allowed_indices = (
        []
        if not isinstance(reference_index, int)
        else list(range(reference_index, len(nx_candidates)))
    )
    system = target.reference_evolution.system
    system_values = system.model_dump(mode="json")
    evolution_time = float(target.reference_evolution.time)
    if isinstance(
        target,
        (
            BurgersConvergenceReferenceSpec,
            ReactionDiffusionConvergenceReferenceSpec,
        ),
    ):
        refinement_proof = _target_time_refinement_proof(
            target,
            list(target.time_candidates),
        )
        conditions = refinement_proof["ordered_candidates"]
        condition_index = convergence.get("selected_time_candidate_index")
        method_kind = "candidate_refinement"
        temporal_status = (
            "converged"
            if convergence.get("temporal_status") == "pass"
            else "failed"
        )
    elif isinstance(target, HeatAnalyticReferenceSpec):
        conditions = [
            canonical_numerical_condition("heat", system_values)
        ]
        refinement_proof = None
        condition_index = convergence.get(
            "selected_numerical_condition_index"
        )
        method_kind = "analytic_exact"
        temporal_status = "analytic_exact"
    else:
        raise TypeError(
            f"unsupported target-reference spec: {type(target).__name__}"
        )
    condition_allowed_indices = (
        []
        if not isinstance(condition_index, int)
        else list(range(condition_index, len(conditions)))
    )
    finest_reference_index = len(nx_candidates) - 1
    finest_condition_index = len(conditions) - 1
    convergence_rows = convergence.get("rows")
    if not isinstance(convergence_rows, list):
        raise ValueError("reference convergence did not return canonical rows")
    pairwise_row_hashes = [
        row["row_hash"]
        for row in convergence_rows
        if row.get("check_kind") in {"spatial", "temporal"}
    ]
    joint_rows = [
        row
        for row in convergence_rows
        if row.get("check_kind") == "joint"
    ]
    contract = {
        "schema_version": "pol-target-reference-contract-v4",
        "system_kind": system.kind,
        "invariant_parameters": canonical_invariant_parameters(
            system.kind,
            system_values,
        ),
        "evolution_time": evolution_time,
        "dtype": spec.samples.dtype,
        "domain_length": float(spec.domain.length),
        "reference_resolution": {
            "selected_value": convergence.get("selected_reference_nx"),
            "selected_candidate_index": reference_index,
            "finest_value": nx_candidates[finest_reference_index],
            "finest_candidate_index": finest_reference_index,
            "candidates": nx_candidates,
            "candidate_refinement_proof": reference_refinement_proof(
                nx_candidates
            ),
        },
        "numerical_method_validation": {
            "kind": method_kind,
            "selected_condition": (
                None
                if not isinstance(condition_index, int)
                else conditions[condition_index]
            ),
            "selected_candidate_index": condition_index,
            "finest_condition": conditions[finest_condition_index],
            "finest_candidate_index": finest_condition_index,
            "candidates": conditions,
            "candidate_refinement_proof": refinement_proof,
            "temporal_status": temporal_status,
        },
        "convergence_evidence": {
            "schema_version": CONVERGENCE_CSV_SCHEMA_VERSION,
            "row_schema_version": CONVERGENCE_ROW_SCHEMA_VERSION,
            "row_semantics": (
                "adjacent_candidate_pairs_plus_selected_vs_finest"
            ),
            "blank_field_semantics": {
                "requested_fine_dt": (
                    "null/blank exactly for ETDRK4, reaction-diffusion, "
                    "and analytic heat conditions"
                ),
                "burgers_step_metadata": (
                    "null/blank for analytic heat and reaction-diffusion "
                    "conditions"
                ),
                "reaction_diffusion_method_fields": (
                    "null/blank for analytic heat and Burgers conditions"
                ),
            },
            "tolerances": target.reference_tolerances.model_dump(
                mode="json"
            ),
            "rows": convergence_rows,
            "rows_hash": stable_object_hash(convergence_rows),
            "pairwise_row_hashes": pairwise_row_hashes,
            "joint_row_hash": (
                None if not joint_rows else joint_rows[0]["row_hash"]
            ),
            "observed_order_diagnostics": [],
        },
        "selection_policy": target.selection_policy,
        "allowed_refinement_relation": {
            "kind": "validated_candidate_suffix_exact_membership",
            "reference_nx_allowed_indices": reference_allowed_indices,
            "reference_nx_allowed_values": [
                nx_candidates[index] for index in reference_allowed_indices
            ],
            "numerical_condition_allowed_indices": (
                condition_allowed_indices
            ),
            "numerical_condition_allowed_values": [
                conditions[index] for index in condition_allowed_indices
            ],
        },
    }
    return contract


def _validate_target_reference_contract(contract: dict[str, Any]) -> None:
    validate_target_reference_contract(contract)


def _certificate_payload(
    spec: ValidationSpec,
    *,
    artifact_id: str,
    checks: dict[str, Any],
    master_payload: dict[str, Any],
) -> dict[str, Any]:
    cross_enabled = (
        isinstance(
            spec.target_reference,
            BurgersConvergenceReferenceSpec,
        )
        and spec.target_reference.cross_solver_validation.enabled
    )
    cross_solver_validation: dict[str, Any] | None = None
    if cross_enabled:
        cross_value = checks.get("cross_solver_validation")
        if not isinstance(cross_value, dict):
            raise ValueError(
                "enabled cross-solver validation evidence is missing"
            )
        validate_cross_solver_validation_block(cross_value)
        _validate_cross_solver_check_against_spec(spec, cross_value)
        cross_solver_validation = cross_value
    elif "cross_solver_validation" in checks:
        raise ValueError(
            "disabled cross-solver validation must not contain evidence"
        )
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
        "schema_version": "pol-validation-certificate-v12",
        "reference_convergence_csv_schema_version": (
            CONVERGENCE_CSV_SCHEMA_VERSION
        ),
        "reference_convergence_row_schema_version": (
            CONVERGENCE_ROW_SCHEMA_VERSION
        ),
        "reference_convergence_rows_hash": target_reference[
            "convergence_evidence"
        ]["rows_hash"],
        "cross_solver_check_schema_version": (
            CROSS_SOLVER_CHECK_SCHEMA_VERSION
        ),
        "matched_model1_pipeline_check_schema_version": (
            MODEL1_CONSISTENCY_CHECK_SCHEMA_VERSION
        ),
        "field_quadrature_check_schema_version": (
            FIELD_QUADRATURE_CHECK_SCHEMA_VERSION
        ),
        **execution_device_policy(),
        "status": overall,
        "name": spec.name,
        "profile": spec.profile,
        "artifact_id": artifact_id,
        "checks": statuses,
        "foundation_contract": foundation,
        "foundation_contract_hash": stable_object_hash(foundation),
        "target_reference_contract": target_reference,
        "target_reference_contract_hash": stable_object_hash(target_reference),
        "cross_solver_validation": cross_solver_validation,
        "cross_solver_validation_hash": (
            None
            if cross_solver_validation is None
            else stable_object_hash(cross_solver_validation)
        ),
    }


def _publish_solve_failure(
    spec: ValidationSpec,
    *,
    identity: dict[str, Any],
    master_payload: dict[str, Any],
    diagnostic: dict[str, Any],
    force: bool,
) -> ArtifactRef:
    store = ArtifactStore(spec.artifact_root)
    diagnostic_hash = stable_object_hash(diagnostic)
    failure_identity = {
        **identity,
        "failure_artifact_schema_version": "pol-validation-failure-v1",
        "failure_diagnostic_hash": diagnostic_hash,
    }
    failure_ref = store.reference(
        "validation_failures",
        failure_identity,
    )
    failure = {
        "schema_version": "pol-validation-failure-v1",
        "status": "fail",
        "name": spec.name,
        "profile": spec.profile,
        "diagnostic": diagnostic,
        "diagnostic_hash": diagnostic_hash,
    }

    def writer(root: Path) -> Iterable[str]:
        write_strict_json(root / "resolved_spec.json", identity["spec"])
        write_strict_json(root / "failure.json", failure)
        atomic_torch_save(
            root / "master_initial_conditions.pt",
            master_payload,
        )
        return (
            "resolved_spec.json",
            "failure.json",
            "master_initial_conditions.pt",
        )

    store.publish(
        failure_ref,
        identity=failure_identity,
        writer=writer,
        force=force,
    )
    verify_artifact(failure_ref.path)
    return failure_ref


def run_validation(spec: ValidationSpec, *, force: bool = False) -> ValidationOutcome:
    _calibration_provenance(spec)
    max_nx = max(
        int(value)
        for value in spec.target_reference.reference_nx_candidates
    )
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
    require_cpu_tensors(
        archive.__dict__,
        boundary="validation workflow input",
        name="initial_conditions",
    )
    identity = _scientific_identity(spec)
    store = ArtifactStore(spec.artifact_root)
    ref = store.reference("validations", identity)
    master_payload = _master_payload(archive)
    _validate_master_payload_against_spec(master_payload, spec)
    try:
        checks = {
            "periodic_resampling": _resampling_checks(spec),
            "real_fourier_projector": _fourier_projector_check(spec),
            "finite_input_interface": _finite_interface_checks(spec, archive),
            "fixed_decoder": _decoder_checks(spec),
            "matched_model1_pipeline": run_matched_model1_pipeline_check(
                domain_length=float(spec.domain.length),
            ),
            "field_quadrature": run_field_quadrature_check(
                domain_length=float(spec.domain.length),
            ),
        }
        _validate_decoder_characterization(checks["fixed_decoder"])
        validate_matched_model1_pipeline_check(
            checks["matched_model1_pipeline"],
            domain_length=float(spec.domain.length),
        )
        validate_field_quadrature_check(
            checks["field_quadrature"],
            domain_length=float(spec.domain.length),
        )
        if isinstance(spec.target_reference, HeatAnalyticReferenceSpec):
            checks["heat_analytic"] = _heat_analytic_checks(spec)
        if isinstance(
            spec.target_reference,
            ReactionDiffusionConvergenceReferenceSpec,
        ):
            checks["reaction_diffusion_characterization"] = (
                _reaction_diffusion_characterization(spec)
            )
        convergence, convergence_rows = _reference_convergence(
            spec,
            archive,
        )
        checks["reference_convergence"] = convergence
        if (
            isinstance(
                spec.target_reference,
                BurgersConvergenceReferenceSpec,
            )
            and spec.target_reference.cross_solver_validation.enabled
        ):
            checks["cross_solver_validation"] = (
                _burgers_cross_solver_validation(spec, archive)
            )
    except _ValidationSolveFailure as exc:
        failure_ref = _publish_solve_failure(
            spec,
            identity=identity,
            master_payload=master_payload,
            diagnostic=exc.diagnostic,
            force=force,
        )
        raise RuntimeError(
            "validation failed because a numerical solve produced a "
            f"non-finite state; diagnostics: {failure_ref.path}"
        ) from exc
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
            fieldnames=CONVERGENCE_ROW_FIELDS,
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
