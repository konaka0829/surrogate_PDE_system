"""A small, explicit one-dimensional Fourier neural operator."""
from __future__ import annotations

import math

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

    def __init__(
        self,
        candidate: FNO1dCandidateSpec,
        *,
        n_tar: int,
        coordinate_channel: str,
        domain_length: float,
    ) -> None:
        super().__init__()
        self.n_tar = int(n_tar)
        if coordinate_channel not in {"none", "periodic_sin_cos"}:
            raise ValueError("unsupported FNO coordinate-channel policy")
        if not math.isfinite(float(domain_length)) or domain_length <= 0:
            raise ValueError("FNO domain length must be finite and positive")
        self.coordinate_channel = str(coordinate_channel)
        self.domain_length = float(domain_length)
        self.lifting_input_channels = (
            1 if coordinate_channel == "none" else 3
        )
        width = int(candidate.width)
        self.lifting = nn.Linear(self.lifting_input_channels, width)
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
        values = finite_input.unsqueeze(-1)
        if self.coordinate_channel == "periodic_sin_cos":
            coordinate = periodic_sin_cos_coordinates(
                self.n_tar,
                domain_length=self.domain_length,
                dtype=finite_input.dtype,
                device=finite_input.device,
            )
            coordinate = coordinate.unsqueeze(0).expand(
                finite_input.shape[0],
                -1,
                -1,
            )
            values = torch.cat((values, coordinate), dim=-1)
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


def periodic_sin_cos_coordinates(
    n: int,
    *,
    domain_length: float,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Return endpoint-free sin/cos channels from physical coordinates."""
    count = int(n)
    length = float(domain_length)
    if count <= 0:
        raise ValueError("periodic coordinate count must be positive")
    if not math.isfinite(length) or length <= 0:
        raise ValueError("periodic coordinate domain length must be positive")
    indices = torch.arange(count, dtype=dtype, device=device)
    physical_coordinate = length * indices / float(count)
    angular_phase = 2.0 * torch.pi * physical_coordinate / length
    return torch.stack(
        (torch.sin(angular_phase), torch.cos(angular_phase)),
        dim=-1,
    )


def parameter_count(model: nn.Module) -> int:
    """Count trainable real scalar parameters."""
    return sum(
        int(parameter.numel()) * (2 if parameter.is_complex() else 1)
        for parameter in model.parameters()
        if parameter.requires_grad
    )


__all__ = [
    "FNO1d",
    "SpectralConvolution1d",
    "parameter_count",
    "periodic_sin_cos_coordinates",
]
