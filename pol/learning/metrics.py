from __future__ import annotations

import torch

from pol.math.fourier import real_fourier_synthesis
from pol.math.periodic import spectral_resample_periodic


def periodic_l2_norm(values: torch.Tensor, *, domain_length: float) -> torch.Tensor:
    if values.ndim < 1:
        raise ValueError("values must have a spatial axis")
    return torch.sqrt(
        (float(domain_length) / float(values.shape[-1]))
        * values.square().sum(dim=-1)
    )


def samplewise_l2_errors(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    domain_length: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    absolute = periodic_l2_norm(prediction - target, domain_length=domain_length)
    denominator = periodic_l2_norm(target, domain_length=domain_length).clamp_min(
        torch.finfo(target.dtype).eps
    )
    return absolute, absolute / denominator


def aggregate(values: torch.Tensor, prefix: str) -> dict[str, float]:
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("metric aggregation expects a nonempty vector")
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_median": float(values.median()),
        f"{prefix}_max": float(values.max()),
    }


def prediction_metrics(
    coefficient_prediction: torch.Tensor,
    coefficient_target: torch.Tensor,
    field_prediction: torch.Tensor,
    field_target: torch.Tensor,
    *,
    domain_length: float,
) -> dict[str, float]:
    coefficient_error = coefficient_prediction - coefficient_target
    coefficient_mse = float(coefficient_error.square().mean())
    coefficient_den = torch.linalg.vector_norm(
        coefficient_target, dim=-1
    ).clamp_min(torch.finfo(coefficient_target.dtype).eps)
    coefficient_rel = torch.linalg.vector_norm(
        coefficient_error, dim=-1
    ) / coefficient_den
    field_abs, field_rel = samplewise_l2_errors(
        field_prediction, field_target, domain_length=domain_length
    )
    return {
        "coefficient_mse": coefficient_mse,
        **aggregate(coefficient_rel, "coefficient_relative_l2"),
        **aggregate(field_abs, "field_absolute_l2"),
        **aggregate(field_rel, "field_relative_l2"),
    }


def fourier_prediction_metrics(
    coefficient_prediction: torch.Tensor,
    coefficient_target: torch.Tensor,
    data_field_target: torch.Tensor,
    reference_field_target: torch.Tensor,
    *,
    n_data: int,
    n_reference: int,
    domain_length: float,
) -> dict[str, float]:
    """Evaluate a Fourier prediction in coefficient, data, and field space.

    ``field_*`` metrics use the dataset reference grid as a quadrature rule
    for the continuous periodic :math:`L^2` norm.  Whether target-reference
    convergence is claimed is carried separately by the dataset validation
    binding.  The additional ``data_field_*`` metrics compare against the
    finite ``n_data`` target array supplied to the learning problem.  Keeping
    both is essential when different ``n_tar`` values are compared:
    data-space error alone does not see error between the finite target
    samples.
    """
    if data_field_target.shape[-1] != int(n_data):
        raise ValueError("data_field_target does not match n_data")
    if reference_field_target.shape[-1] != int(n_reference):
        raise ValueError("reference_field_target does not match n_reference")
    if data_field_target.shape[:-1] != reference_field_target.shape[:-1]:
        raise ValueError("data and reference targets must share sample axes")

    reference_prediction = real_fourier_synthesis(
        coefficient_prediction,
        int(n_reference),
        domain_length=domain_length,
    )
    metrics = prediction_metrics(
        coefficient_prediction,
        coefficient_target,
        reference_prediction,
        reference_field_target,
        domain_length=domain_length,
    )
    data_prediction = real_fourier_synthesis(
        coefficient_prediction,
        int(n_data),
        domain_length=domain_length,
    )
    data_absolute, data_relative = samplewise_l2_errors(
        data_prediction,
        data_field_target,
        domain_length=domain_length,
    )
    return {
        **metrics,
        **aggregate(data_absolute, "data_field_absolute_l2"),
        **aggregate(data_relative, "data_field_relative_l2"),
    }


def fourier_representation_floor(
    coefficient_target: torch.Tensor,
    data_field_target: torch.Tensor,
    reference_field_target: torch.Tensor,
    *,
    n_data: int,
    n_reference: int,
    domain_length: float,
) -> dict[str, float]:
    """Return the unavoidable error of the configured ``q``-mode output.

    The unqualified representation-floor values are reference-field errors;
    the ``data_`` values are their finite-data counterparts.
    """
    reference_projection = real_fourier_synthesis(
        coefficient_target,
        int(n_reference),
        domain_length=domain_length,
    )
    _, reference_relative = samplewise_l2_errors(
        reference_projection,
        reference_field_target,
        domain_length=domain_length,
    )
    data_projection = real_fourier_synthesis(
        coefficient_target,
        int(n_data),
        domain_length=domain_length,
    )
    _, data_relative = samplewise_l2_errors(
        data_projection,
        data_field_target,
        domain_length=domain_length,
    )
    return {
        "representation_floor_relative_l2_mean": float(reference_relative.mean()),
        "representation_floor_relative_l2_median": float(reference_relative.median()),
        "representation_floor_relative_l2_max": float(reference_relative.max()),
        "data_representation_floor_relative_l2_mean": float(data_relative.mean()),
        "data_representation_floor_relative_l2_median": float(data_relative.median()),
        "data_representation_floor_relative_l2_max": float(data_relative.max()),
    }


def compare_fields_on_common_grid(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    domain_length: float,
    n_common: int | None = None,
) -> dict[str, float]:
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0]:
        raise ValueError("fields must have shape (samples, nx)")
    n = n_common or min(a.shape[-1], b.shape[-1])
    aa = spectral_resample_periodic(a, n, domain_length=domain_length)
    bb = spectral_resample_periodic(b, n, domain_length=domain_length)
    absolute, relative = samplewise_l2_errors(
        aa, bb, domain_length=domain_length
    )
    return {
        **aggregate(absolute, "absolute_l2"),
        **aggregate(relative, "relative_l2"),
    }
