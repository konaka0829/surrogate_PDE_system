from __future__ import annotations

from typing import Any

import torch

from pol.config.models import ValidationSpec
from pol.data.initial_conditions import InitialConditionArchive
from pol.learning.direct import (
    decode_point_observation_to_real_fourier,
    fixed_fourier_decoder_bandwidth,
)
from pol.learning.observations import observe_equispaced_periodic
from pol.math.fourier import real_fourier_analysis, real_fourier_synthesis
from pol.math.periodic import periodic_grid, spectral_resample_periodic
from pol.runtime.device import require_cpu_tensor, require_cpu_tensors
from .check_utils import algebraic_allclose
from .model1_consistency import (
    run_matched_model1_pipeline_check,
    validate_matched_model1_pipeline_check,
)
from .quadrature import (
    run_field_quadrature_check,
    validate_field_quadrature_check,
)


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
                "status": "pass" if algebraic_allclose(actual, expected, spec) else "fail",
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
    discarded = algebraic_allclose(down[0], down[1], spec)
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
    passed = algebraic_allclose(recovered, coefficients, spec)
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
    no_leak = algebraic_allclose(finite_pair[0], finite_pair[1], spec) and algebraic_allclose(
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
    full_pass = algebraic_allclose(decoded, coeff, spec)

    reduced = spec.reduced_observation
    kmax = (reduced.q - 1) // 2
    if reduced.J < 2 or kmax < 1:
        raise ValueError(
            "fixed-decoder aliasing counterexample requires reduced J >= 2 "
            "and reduced q >= 3"
        )
    reduced_bandwidth = fixed_fourier_decoder_bandwidth(
        reduced.J,
        reduced.q,
    )
    base_mode = kmax
    if base_mode > reduced_bandwidth.observable_max_mode:
        raise ValueError(
            "fixed-decoder aliasing counterexample requires the base mode "
            "to lie in the observable Fourier band"
        )
    alias_k = reduced.J + base_mode
    alias_source_nx = max(
        int(full.n_sur),
        2 * int(alias_k) + 2,
    )
    if alias_source_nx % 2:
        alias_source_nx += 1
    if (
        reduced.J > alias_source_nx
        or base_mode >= alias_source_nx / 2
        or alias_k >= alias_source_nx / 2
    ):
        raise ValueError(
            "fixed-decoder aliasing source grid cannot represent the "
            "synthetic base and high modes strictly below Nyquist"
        )

    rcoeff = torch.randn((2, reduced.q), generator=generator, dtype=dtype)
    rfield = real_fourier_synthesis(rcoeff, full.n_sur, domain_length=L)
    rfeatures = observe_equispaced_periodic(
        rfield, reduced.J, domain_length=L, l2_scale=True
    )
    rdecoded = decode_point_observation_to_real_fourier(
        rfeatures, reduced.q, domain_length=L
    )
    reduced_pass = algebraic_allclose(rdecoded, rcoeff, spec)

    x = periodic_grid(alias_source_nx, L, dtype=dtype)
    base = 0.2 + torch.cos(2 * torch.pi * base_mode * x / L)
    high = base + 0.4 * torch.cos(2 * torch.pi * alias_k * x / L)
    truth = real_fourier_analysis(
        base.unsqueeze(0),
        reduced.q,
        domain_length=L,
    )
    aliased = decode_point_observation_to_real_fourier(
        observe_equispaced_periodic(
            high.unsqueeze(0),
            reduced.J,
            domain_length=L,
            l2_scale=True,
        ),
        reduced.q,
        domain_length=L,
    )
    alias_difference = (aliased - truth).abs().max()
    if not (
        bool(torch.isfinite(truth).all())
        and bool(torch.isfinite(aliased).all())
        and bool(torch.isfinite(alias_difference))
    ):
        raise ValueError(
            "fixed-decoder aliasing counterexample produced a non-finite result"
        )
    alias_error = float(alias_difference)
    alias_pass = not algebraic_allclose(aliased, truth, spec)

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
    observable_part_pass = algebraic_allclose(
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
            "source_nx": alias_source_nx,
            "observation_count": reduced.J,
            "base_mode": base_mode,
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


def run_foundation_checks(
    spec: ValidationSpec,
    archive: InitialConditionArchive,
) -> dict[str, Any]:
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
    validate_foundation_checks(spec, checks)
    return checks


def validate_foundation_checks(
    spec: ValidationSpec,
    checks: dict[str, Any],
) -> None:
    fixed_decoder = checks.get("fixed_decoder")
    if not isinstance(fixed_decoder, dict):
        raise ValueError("validation fixed-decoder check is missing")
    _validate_decoder_characterization(fixed_decoder)
    matched_model1 = checks.get("matched_model1_pipeline")
    if not isinstance(matched_model1, dict):
        raise ValueError("matched Model 1 pipeline check is missing")
    validate_matched_model1_pipeline_check(
        matched_model1,
        domain_length=float(spec.domain.length),
    )
    field_quadrature = checks.get("field_quadrature")
    if not isinstance(field_quadrature, dict):
        raise ValueError("field quadrature check is missing")
    validate_field_quadrature_check(
        field_quadrature,
        domain_length=float(spec.domain.length),
    )
