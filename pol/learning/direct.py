from __future__ import annotations

import math

import torch

from pol.math.fourier import real_fourier_analysis


def decode_point_observation_to_real_fourier(
    features: torch.Tensor,
    q: int,
    *,
    domain_length: float,
) -> torch.Tensor:
    """Apply the fixed Fourier decoder to L2-scaled equispaced observations."""
    if features.ndim < 1 or features.shape[-1] < 2:
        raise ValueError("features must have shape (..., J) with J >= 2")
    J = int(features.shape[-1])
    raw = features * math.sqrt(float(J) / float(domain_length))
    q_observable = J if J % 2 else J - 1
    retained = min(q, q_observable)
    decoded = real_fourier_analysis(raw, retained, domain_length=domain_length)
    if retained == q:
        return decoded
    return torch.cat(
        [
            decoded,
            torch.zeros(
                (*decoded.shape[:-1], q - retained),
                dtype=decoded.dtype,
                device=decoded.device,
            ),
        ],
        dim=-1,
    )
