from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RandomFeatureMap:
    """Map ``phi`` to ``[phi, rho(phi A^T+c)/sqrt(M)]``."""

    A: torch.Tensor
    c: torch.Tensor
    activation: str
    seed: int
    weight_scale: float
    bias_scale: float

    @classmethod
    def create(
        cls,
        J: int,
        width: int,
        *,
        activation: str,
        seed: int,
        weight_scale: float,
        bias_scale: float,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> "RandomFeatureMap":
        if J <= 0 or width <= 0:
            raise ValueError("J and width must be positive")
        if activation not in {"tanh", "relu", "identity"}:
            raise ValueError("activation must be tanh, relu, or identity")
        if weight_scale < 0 or bias_scale < 0:
            raise ValueError("random-feature scales must be nonnegative")
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        A = weight_scale * torch.randn(
            width, J, generator=generator, dtype=dtype
        )
        c = bias_scale * torch.randn(width, generator=generator, dtype=dtype)
        return cls(
            A.to(device),
            c.to(device),
            activation,
            int(seed),
            float(weight_scale),
            float(bias_scale),
        )

    def __call__(self, phi: torch.Tensor) -> torch.Tensor:
        hidden = phi @ self.A.T + self.c
        if self.activation == "tanh":
            hidden = torch.tanh(hidden)
        elif self.activation == "relu":
            hidden = F.relu(hidden)
        return torch.cat([phi, hidden / math.sqrt(self.A.shape[0])], dim=-1)
