from __future__ import annotations

from dataclasses import dataclass

import torch

from pol.numerics.initial_conditions import (
    GRF_SAMPLER_SEMANTICS,
    sample_gaussian_random_field_initial_conditions,
)


_DTYPE_MAP = {"float32": torch.float32, "float64": torch.float64}


@dataclass(frozen=True)
class InitialConditionArchive:
    sample_ids: torch.Tensor
    values: torch.Tensor
    fourier: torch.Tensor
    nx: int
    domain_length: float
    seed: int
    metadata: dict[str, object]


def resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unsupported device: {name}")


def torch_dtype(name: str) -> torch.dtype:
    try:
        return _DTYPE_MAP[name]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {name}") from exc


def generate_grf_archive(
    *,
    total_samples: int,
    nx: int,
    seed: int,
    gamma: float,
    tau: float,
    sigma: float,
    mean: float,
    domain_length: float,
    dtype: str,
    device: str,
) -> InitialConditionArchive:
    resolved_device = resolve_device(device)
    resolved_dtype = torch_dtype(dtype)
    values = sample_gaussian_random_field_initial_conditions(
        total_samples,
        nx,
        domain_length=domain_length,
        seed=seed,
        gamma=gamma,
        tau=tau,
        sigma=sigma,
        mean=mean,
        device=resolved_device,
        dtype=resolved_dtype,
    )
    return InitialConditionArchive(
        sample_ids=torch.arange(total_samples, dtype=torch.long, device=resolved_device),
        values=values,
        fourier=torch.fft.rfft(values, dim=-1, norm="forward"),
        nx=nx,
        domain_length=float(domain_length),
        seed=int(seed),
        metadata={
            "kind": "periodic_grf",
            "gamma": float(gamma),
            "tau": float(tau),
            "sigma": float(sigma),
            "mean": float(mean),
            "domain_length": float(domain_length),
            "sampler_semantics": GRF_SAMPLER_SEMANTICS,
            "dtype": dtype,
            "device": str(resolved_device),
        },
    )
