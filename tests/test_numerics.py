from __future__ import annotations

import pytest
import torch

from pol.numerics.burgers import simulate_burgers_split_step
from pol.numerics.etdrk4 import (
    cox_matthews_coefficients,
    cox_matthews_etdrk4_step,
    simulate_burgers_etdrk4,
    simulate_burgers_etdrk4_trajectory,
)
from pol.numerics.initial_conditions import sample_gaussian_random_field_initial_conditions


def test_etdrk4_zero_linear_part_matches_classical_rk4() -> None:
    y0 = torch.tensor([0.2], dtype=torch.float64)
    dt = 0.05
    nonlinear = lambda y: y * y
    got = cox_matthews_etdrk4_step(y0, L=torch.zeros_like(y0), dt=dt, nonlinear=nonlinear)
    k1 = nonlinear(y0)
    k2 = nonlinear(y0 + 0.5 * dt * k1)
    k3 = nonlinear(y0 + 0.5 * dt * k2)
    k4 = nonlinear(y0 + dt * k3)
    expected = y0 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    assert torch.allclose(got, expected, atol=1e-14, rtol=1e-14)


def test_etdrk4_coefficients_have_rk4_limits() -> None:
    L = torch.zeros(3, dtype=torch.float64)
    dt = 0.125
    _, _, Q, f1, f2, f3 = cox_matthews_coefficients(L, dt)
    assert torch.allclose(Q, torch.full_like(Q, dt / 2.0))
    assert torch.allclose(f1, torch.full_like(f1, dt / 6.0))
    assert torch.allclose(2.0 * f2, torch.full_like(f2, dt / 3.0))
    assert torch.allclose(f3, torch.full_like(f3, dt / 6.0))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("nx", [15, 16])
@pytest.mark.parametrize("dealias", [False, True])
def test_split_step_is_deterministic_for_even_and_odd_grids(
    dtype: torch.dtype, nx: int, dealias: bool
) -> None:
    x = torch.arange(nx, dtype=dtype) / nx
    u0 = (0.2 * torch.sin(2.0 * torch.pi * x))[None]
    kwargs = dict(
        dt=0.01,
        Tr=0.03,
        obs_steps=[1, 3],
        nu=0.05,
        fine_dt=0.005,
        dealias=dealias,
    )
    first = simulate_burgers_split_step(u0, **kwargs)
    second = simulate_burgers_split_step(u0, **kwargs)
    assert [tuple(value.shape) for value in first] == [(1, nx), (1, nx)]
    assert all(value.dtype == dtype and torch.isfinite(value).all() for value in first)
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("nx", [15, 16])
def test_etdrk4_trajectory_is_deterministic_for_even_and_odd_grids(
    dtype: torch.dtype, nx: int
) -> None:
    x = torch.arange(nx, dtype=dtype) / nx
    u0 = (0.1 * torch.cos(2.0 * torch.pi * x))[None]
    kwargs = dict(nu=0.05, T=0.02, dt=0.01, obs_steps=[1, 2])
    first = simulate_burgers_etdrk4_trajectory(u0, **kwargs)
    second = simulate_burgers_etdrk4_trajectory(u0, **kwargs)
    assert [tuple(value.shape) for value in first] == [(1, nx), (1, nx)]
    assert all(value.dtype == dtype and torch.isfinite(value).all() for value in first)
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("nx", [15, 16, 31])
def test_grf_sampling_is_seed_deterministic(
    dtype: torch.dtype, nx: int
) -> None:
    kwargs = dict(
        num_samples=3,
        nx=nx,
        seed=19,
        gamma=2.25,
        tau=4.0,
        sigma=7.0,
        mean=0.5,
        device=torch.device("cpu"),
        dtype=dtype,
    )
    first = sample_gaussian_random_field_initial_conditions(**kwargs)
    second = sample_gaussian_random_field_initial_conditions(**kwargs)
    assert first.shape == (3, nx)
    assert first.dtype == dtype
    assert torch.equal(first, second)


def test_burgers_etdrk4_terminal_state_is_finite() -> None:
    x = torch.arange(32, dtype=torch.float64) / 32
    u0 = torch.sin(2.0 * torch.pi * x).unsqueeze(0)
    out = simulate_burgers_etdrk4(u0, nu=0.01, T=0.02, dt=0.005)
    assert out.shape == u0.shape and torch.isfinite(out).all()


@pytest.mark.parametrize(
    "call",
    [
        lambda: simulate_burgers_split_step(
            torch.zeros(4),
            dt=0.1,
            Tr=0.1,
            obs_steps=[1],
            nu=0.1,
            fine_dt=0.01,
        ),
        lambda: sample_gaussian_random_field_initial_conditions(
            0, 16, seed=0, device=torch.device("cpu")
        ),
    ],
)
def test_numerics_reject_invalid_inputs(call) -> None:
    with pytest.raises(ValueError):
        call()
