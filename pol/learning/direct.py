from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Any, Mapping

import torch

from pol.math.fourier import real_fourier_analysis


DIRECT_DECODER_POLICY = "equispaced_real_fourier_observable_prefix_zero_fill_v1"


@dataclass(frozen=True)
class FixedFourierDecoderBandwidth:
    """Observable-bandwidth contract for the fixed real-Fourier decoder."""

    observation_count: int
    requested_q: int
    observable_q: int
    retained_q: int
    requested_max_mode: int
    observable_max_mode: int
    zero_filled_mode_count: int
    zero_filled_coefficient_count: int
    zero_fill_applied: bool
    decoder_policy: str = DIRECT_DECODER_POLICY

    def as_dict(self) -> dict[str, Any]:
        return {
            "observation_count": self.observation_count,
            "requested_q": self.requested_q,
            "observable_q": self.observable_q,
            "retained_q": self.retained_q,
            "requested_max_mode": self.requested_max_mode,
            "observable_max_mode": self.observable_max_mode,
            "zero_filled_mode_count": self.zero_filled_mode_count,
            "zero_filled_coefficient_count": self.zero_filled_coefficient_count,
            "zero_fill_applied": self.zero_fill_applied,
            "decoder_policy": self.decoder_policy,
        }

    def as_artifact_fields(self) -> dict[str, Any]:
        return {
            f"decoder_{key}" if key != "decoder_policy" else key: value
            for key, value in self.as_dict().items()
        }


DIRECT_DECODER_DIAGNOSTIC_FIELDS = tuple(
    (
        field.name
        if field.name == "decoder_policy"
        else f"decoder_{field.name}"
    )
    for field in fields(FixedFourierDecoderBandwidth)
)


def fixed_fourier_decoder_bandwidth(
    observation_count: int,
    requested_q: int,
) -> FixedFourierDecoderBandwidth:
    """Return the fixed decoder's directly observable and zero-filled bandwidth."""
    if (
        not isinstance(observation_count, int)
        or isinstance(observation_count, bool)
        or observation_count < 2
    ):
        raise ValueError("observation_count must be an integer with J >= 2")
    if (
        not isinstance(requested_q, int)
        or isinstance(requested_q, bool)
        or requested_q <= 0
    ):
        raise ValueError("requested_q must be a positive integer")
    if requested_q % 2 == 0:
        raise ValueError("requested_q must be odd")
    observable_q = (
        observation_count if observation_count % 2 else observation_count - 1
    )
    retained_q = min(requested_q, observable_q)
    requested_max_mode = (requested_q - 1) // 2
    observable_max_mode = (observable_q - 1) // 2
    return FixedFourierDecoderBandwidth(
        observation_count=observation_count,
        requested_q=requested_q,
        observable_q=observable_q,
        retained_q=retained_q,
        requested_max_mode=requested_max_mode,
        observable_max_mode=observable_max_mode,
        zero_filled_mode_count=max(0, requested_max_mode - observable_max_mode),
        zero_filled_coefficient_count=requested_q - retained_q,
        zero_fill_applied=requested_q > observable_q,
    )


def _artifact_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError(f"{field} must be a boolean")


def _artifact_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if str(value).strip() != str(parsed):
        raise ValueError(f"{field} must be an integer")
    return parsed


def verify_fixed_fourier_decoder_diagnostic(
    values: Mapping[str, Any],
    *,
    observation_count: int,
    requested_q: int,
    boundary: str,
) -> FixedFourierDecoderBandwidth:
    """Verify stored ``decoder_*`` fields against the canonical J/q contract."""
    diagnostic = fixed_fourier_decoder_bandwidth(observation_count, requested_q)
    expected = diagnostic.as_artifact_fields()
    for field, expected_value in expected.items():
        if field not in values or values.get(field) in ("", None):
            raise ValueError(f"{boundary} is missing {field}")
        actual = values[field]
        if isinstance(expected_value, bool):
            actual = _artifact_bool(actual, field=field)
        elif isinstance(expected_value, int):
            actual = _artifact_int(actual, field=field)
        elif not isinstance(actual, str):
            raise ValueError(f"{field} must be a string")
        if actual != expected_value:
            raise ValueError(
                f"{boundary} {field} does not match the fixed decoder bandwidth"
            )
    return diagnostic


def has_fixed_fourier_decoder_diagnostic(values: Mapping[str, Any]) -> bool:
    """Return whether any nonempty direct-decoder diagnostic field is present."""
    return any(
        values.get(field) not in ("", None)
        for field in DIRECT_DECODER_DIAGNOSTIC_FIELDS
    )


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
    bandwidth = fixed_fourier_decoder_bandwidth(J, q)
    raw = features * math.sqrt(float(J) / float(domain_length))
    decoded = real_fourier_analysis(
        raw,
        bandwidth.retained_q,
        domain_length=domain_length,
    )
    if not bandwidth.zero_fill_applied:
        return decoded
    return torch.cat(
        [
            decoded,
            torch.zeros(
                (*decoded.shape[:-1], bandwidth.zero_filled_coefficient_count),
                dtype=decoded.dtype,
                device=decoded.device,
            ),
        ],
        dim=-1,
    )
