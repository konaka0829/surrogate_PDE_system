"""A small, explicit one-dimensional Fourier neural operator."""
from __future__ import annotations

import torch
from torch import nn

from .protocol import FNO1dCandidateSpec


class SpectralConvolution1d(nn.Module):
    """Low-mode complex multiplication with real-valued trainable parameters."""

    def __init__(self, width: int, modes: int) -> None:
        super().__init__()
        scale = 1.0 / max(1, width)
        self.width = int(width)
        self.modes = int(modes)
        self.weight_real = nn.Parameter(
            scale * torch.randn(width, width, modes)
        )
        self.weight_imag = nn.Parameter(
            scale * torch.randn(width, width, modes)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        transformed = torch.fft.rfft(values, dim=-1)
        used = min(self.modes, int(transformed.shape[-1]))
        weights = torch.complex(
            self.weight_real[..., :used],
            self.weight_imag[..., :used],
        )
        output = torch.zeros(
            values.shape[0],
            self.width,
            transformed.shape[-1],
            device=values.device,
            dtype=transformed.dtype,
        )
        output[..., :used] = torch.einsum(
            "bim,iom->bom",
            transformed[..., :used],
            weights,
        )
        return torch.fft.irfft(output, n=values.shape[-1], dim=-1)


class FNO1d(nn.Module):
    """Map one finite periodic field to one finite output field."""

    def __init__(self, candidate: FNO1dCandidateSpec, *, n_tar: int) -> None:
        super().__init__()
        self.n_tar = int(n_tar)
        width = int(candidate.width)
        self.lifting = nn.Linear(2, width)
        self.spectral_layers = nn.ModuleList(
            SpectralConvolution1d(width, int(candidate.modes))
            for _ in range(int(candidate.depth))
        )
        self.local_layers = nn.ModuleList(
            nn.Conv1d(width, width, kernel_size=1)
            for _ in range(int(candidate.depth))
        )
        self.projection_hidden = nn.Linear(width, width)
        self.projection_output = nn.Linear(width, 1)
        self.activation = nn.GELU()

    def forward(self, finite_input: torch.Tensor) -> torch.Tensor:
        if finite_input.ndim != 2 or finite_input.shape[-1] != self.n_tar:
            raise ValueError(
                "FNO1d input must have shape (samples, configured n_tar)"
            )
        coordinate = torch.arange(
            self.n_tar,
            dtype=finite_input.dtype,
            device=finite_input.device,
        ) / float(self.n_tar)
        coordinate = coordinate.expand(finite_input.shape[0], -1)
        values = torch.stack((finite_input, coordinate), dim=-1)
        values = self.lifting(values).transpose(1, 2)
        for spectral, local in zip(
            self.spectral_layers,
            self.local_layers,
            strict=True,
        ):
            values = self.activation(spectral(values) + local(values))
        values = values.transpose(1, 2)
        values = self.activation(self.projection_hidden(values))
        return self.projection_output(values).squeeze(-1)


def parameter_count(model: nn.Module) -> int:
    """Count trainable real scalar parameters."""
    return sum(
        int(parameter.numel()) * (2 if parameter.is_complex() else 1)
        for parameter in model.parameters()
        if parameter.requires_grad
    )


__all__ = ["FNO1d", "SpectralConvolution1d", "parameter_count"]
