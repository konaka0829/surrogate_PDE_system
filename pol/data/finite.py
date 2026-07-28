from __future__ import annotations

from dataclasses import dataclass

import torch

from pol.math.fourier import real_fourier_analysis
from pol.math.periodic import spectral_resample_periodic


@dataclass(frozen=True)
class FiniteDataView:
    sample_ids: torch.Tensor
    inputs: torch.Tensor
    targets: torch.Tensor
    targets_reference: torch.Tensor
    target_coefficients: torch.Tensor
    n_ref: int
    n_tar: int
    q: int


@torch.no_grad()
def derive_finite_view(
    sample_ids: torch.Tensor,
    inputs_reference: torch.Tensor,
    targets_reference: torch.Tensor,
    *,
    n_tar: int,
    q: int,
    domain_length: float,
) -> FiniteDataView:
    if inputs_reference.shape != targets_reference.shape or inputs_reference.ndim != 2:
        raise ValueError("reference input/target must share shape (samples, n_ref)")
    if sample_ids.ndim != 1 or sample_ids.numel() != inputs_reference.shape[0]:
        raise ValueError("sample_ids must match the sample axis")
    n_ref = int(inputs_reference.shape[-1])
    if n_tar > n_ref:
        raise ValueError(
            f"n_tar={n_tar} must not exceed the dataset reference resolution "
            f"n_ref={n_ref}"
        )
    inputs = spectral_resample_periodic(
        inputs_reference, n_tar, domain_length=domain_length
    )
    targets = spectral_resample_periodic(
        targets_reference, n_tar, domain_length=domain_length
    )
    coefficients = real_fourier_analysis(targets, q, domain_length=domain_length)
    return FiniteDataView(
        sample_ids=sample_ids.clone(),
        inputs=inputs,
        targets=targets,
        targets_reference=targets_reference,
        target_coefficients=coefficients,
        n_ref=n_ref,
        n_tar=int(n_tar),
        q=int(q),
    )


@torch.no_grad()
def build_feature_initial_state(
    finite_inputs: torch.Tensor,
    *,
    n_sur: int,
    domain_length: float,
) -> torch.Tensor:
    """Build the dynamic feature input from finite data only."""
    return spectral_resample_periodic(
        finite_inputs, n_sur, domain_length=domain_length
    )
