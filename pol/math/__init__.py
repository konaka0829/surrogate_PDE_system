"""Numerical representations on periodic domains."""

from .periodic import periodic_grid, spectral_resample_periodic
from .fourier import (
    real_fourier_analysis,
    real_fourier_synthesis,
    validate_real_fourier_dim,
)

__all__ = [
    "periodic_grid",
    "spectral_resample_periodic",
    "real_fourier_analysis",
    "real_fourier_synthesis",
    "validate_real_fourier_dim",
]
