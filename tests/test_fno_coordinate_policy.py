from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from pol.config.loader import load_digital_baseline_spec
from pol.digital_baselines.evaluation import (
    build_fno,
    load_fno_checkpoint,
    state_dict_content_hash,
)
from pol.digital_baselines.fno1d import (
    parameter_count,
    periodic_sin_cos_coordinates,
)
from pol.digital_baselines.protocol import (
    FNO1dCandidateSpec,
    FNO1dModelSpec,
)


ROOT = Path(__file__).resolve().parents[1]


def _candidate() -> FNO1dCandidateSpec:
    return FNO1dCandidateSpec(
        id="coordinate_test",
        modes=3,
        width=4,
        depth=2,
    )


def test_legacy_unit_periodic_ramp_has_explicit_migration_error() -> None:
    with pytest.raises(ValueError, match="legacy unit_periodic ramp"):
        FNO1dModelSpec.model_validate(
            {
                "kind": "fno1d",
                "activation": "gelu",
                "coordinate_channel": "unit_periodic",
                "candidates": [
                    {
                        "id": "candidate",
                        "modes": 2,
                        "width": 3,
                        "depth": 1,
                    }
                ],
            }
        )


@pytest.mark.parametrize("name", ["fno1d_smoke.json", "fno1d.json"])
def test_checked_in_periodic_burgers_baselines_use_no_coordinate_channel(
    name: str,
) -> None:
    raw = json.loads(
        (ROOT / "digital_baselines" / name).read_text(encoding="utf-8")
    )
    assert raw["model"]["coordinate_channel"] == "none"
    spec = load_digital_baseline_spec(
        ROOT / "digital_baselines" / name,
        repo_root=ROOT,
    )
    assert spec.model.coordinate_channel == "none"


def test_lifting_shape_and_parameter_count_follow_coordinate_policy() -> None:
    candidate = _candidate()
    none = build_fno(
        candidate,
        n_tar=9,
        dtype=torch.float64,
        seed=17,
        coordinate_channel="none",
        domain_length=2.5,
    )
    periodic = build_fno(
        candidate,
        n_tar=9,
        dtype=torch.float64,
        seed=17,
        coordinate_channel="periodic_sin_cos",
        domain_length=2.5,
    )
    assert tuple(none.state_dict()["lifting.weight"].shape) == (4, 1)
    assert tuple(periodic.state_dict()["lifting.weight"].shape) == (4, 3)
    assert parameter_count(periodic) - parameter_count(none) == 2 * 4
    assert state_dict_content_hash(none.state_dict()) != (
        state_dict_content_hash(periodic.state_dict())
    )


@pytest.mark.parametrize(
    ("dtype", "atol"),
    [(torch.float32, 2e-5), (torch.float64, 2e-12)],
)
def test_no_coordinate_fno_is_circular_shift_equivariant(
    dtype: torch.dtype,
    atol: float,
) -> None:
    model = build_fno(
        _candidate(),
        n_tar=15,
        dtype=dtype,
        seed=23,
        coordinate_channel="none",
        domain_length=3.0,
    )
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(29)
    values = torch.randn(3, 15, dtype=dtype, generator=generator)
    shift = 4
    with torch.no_grad():
        expected = model(values).roll(shift, dims=-1)
        actual = model(values.roll(shift, dims=-1))
    assert torch.allclose(actual, expected, rtol=0.0, atol=atol)


@pytest.mark.parametrize("n", [7, 8])
@pytest.mark.parametrize("domain_length", [0.75, 3.5])
def test_periodic_sin_cos_is_one_endpoint_free_physical_period(
    n: int,
    domain_length: float,
) -> None:
    channels = periodic_sin_cos_coordinates(
        n,
        domain_length=domain_length,
        dtype=torch.float64,
        device="cpu",
    )
    unit_channels = periodic_sin_cos_coordinates(
        n,
        domain_length=1.0,
        dtype=torch.float64,
        device="cpu",
    )
    indices = torch.arange(n, dtype=torch.float64)
    physical = domain_length * indices / n
    phase = 2.0 * torch.pi * physical / domain_length
    expected = torch.stack((torch.sin(phase), torch.cos(phase)), dim=-1)
    assert torch.allclose(channels, expected, rtol=0.0, atol=1e-15)
    assert torch.allclose(channels, unit_channels, rtol=0.0, atol=1e-15)
    assert torch.allclose(
        channels.square().sum(dim=-1),
        torch.ones(n, dtype=torch.float64),
        rtol=0.0,
        atol=2e-15,
    )
    assert torch.equal(
        channels[0],
        torch.tensor([0.0, 1.0], dtype=torch.float64),
    )
    ramp = indices / n
    assert not torch.allclose(channels[:, 0], ramp)
    assert not torch.allclose(channels[:, 1], ramp)


def test_checkpoint_coordinate_policy_mismatch_fails_at_model_load() -> None:
    candidate = _candidate()
    none = build_fno(
        candidate,
        n_tar=9,
        dtype=torch.float64,
        seed=31,
        coordinate_channel="none",
        domain_length=2.0,
    )
    state = {
        name: value.detach().clone()
        for name, value in none.state_dict().items()
    }
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_fno_checkpoint(
            candidate,
            n_tar=9,
            dtype=torch.float64,
            state_dict=state,
            expected_hash=state_dict_content_hash(state),
            coordinate_channel="periodic_sin_cos",
            domain_length=2.0,
        )
