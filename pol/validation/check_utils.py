from __future__ import annotations

import torch

from pol.config.models import ValidationSpec


def algebraic_allclose(
    a: torch.Tensor,
    b: torch.Tensor,
    spec: ValidationSpec,
) -> bool:
    if a.dtype == torch.float32:
        atol = spec.algebraic_tolerances.float32_atol
        rtol = spec.algebraic_tolerances.float32_rtol
    else:
        atol = spec.algebraic_tolerances.float64_atol
        rtol = spec.algebraic_tolerances.float64_rtol
    return bool(torch.allclose(a, b, atol=atol, rtol=rtol))
