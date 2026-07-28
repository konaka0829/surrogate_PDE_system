from __future__ import annotations

import math

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
        domain_length=1.0,
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
    assert first.device == torch.device("cpu")
    assert torch.equal(first, second)


def test_grf_unit_domain_output_matches_pre_p0_03_regression() -> None:
    expected = torch.tensor(
        [
            [
                0.6221994492645122,
                0.5328401381391653,
                0.40316923108059954,
                0.4106692656292553,
                0.4592239638514888,
                0.45642908574241686,
                0.5143068144184001,
                0.6011620518741617,
            ],
            [
                0.5318873353281477,
                0.548394373198575,
                0.6236231162540826,
                0.5003658580916053,
                0.4446934415500668,
                0.3904409815684965,
                0.44678015037122315,
                0.5138147436378027,
            ],
        ],
        dtype=torch.float64,
    )
    actual = sample_gaussian_random_field_initial_conditions(
        2,
        8,
        domain_length=1.0,
        seed=2024,
        gamma=2.25,
        tau=4.0,
        sigma=7.0,
        mean=0.5,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("nx", [15, 16])
def test_grf_mode_amplitudes_follow_physical_domain_scaling(nx: int) -> None:
    gamma = 1.75
    tau = 3.0
    sigma = 4.0
    kwargs = {
        "num_samples": 4,
        "nx": nx,
        "seed": 731,
        "gamma": gamma,
        "tau": tau,
        "sigma": sigma,
        "mean": 0.75,
        "device": torch.device("cpu"),
        "dtype": torch.float64,
    }
    unit = sample_gaussian_random_field_initial_conditions(
        **kwargs, domain_length=1.0
    )
    doubled = sample_gaussian_random_field_initial_conditions(
        **kwargs, domain_length=2.0
    )
    unit_coefficients = torch.fft.rfft(unit, dim=-1, norm="forward")
    doubled_coefficients = torch.fft.rfft(doubled, dim=-1, norm="forward")

    modes = torch.arange(1, nx // 2 + 1, dtype=torch.float64)
    lambda_unit = sigma**2 * (
        (2.0 * torch.pi * modes / 1.0) ** 2 + tau**2
    ).pow(-gamma)
    lambda_doubled = sigma**2 * (
        (2.0 * torch.pi * modes / 2.0) ** 2 + tau**2
    ).pow(-gamma)
    expected_ratio = torch.sqrt(lambda_doubled / lambda_unit)
    actual_ratio = (
        doubled_coefficients[:, 1:].abs() / unit_coefficients[:, 1:].abs()
    )

    assert torch.allclose(
        actual_ratio,
        expected_ratio.unsqueeze(0),
        atol=5e-12,
        rtol=5e-12,
    )
    assert torch.allclose(
        unit_coefficients[:, 0].real,
        torch.full((4,), 0.75, dtype=torch.float64),
        atol=2e-16,
        rtol=0.0,
    )
    assert torch.allclose(
        doubled_coefficients[:, 0],
        unit_coefficients[:, 0],
        atol=2e-16,
        rtol=0.0,
    )


def test_grf_even_grid_nyquist_uses_physical_wavenumber() -> None:
    nx = 16
    domain_length = 2.0
    mode = nx // 2
    from_mode_formula = 2.0 * math.pi * mode / domain_length
    from_nyquist_formula = math.pi * nx / domain_length
    assert math.isclose(
        from_mode_formula,
        from_nyquist_formula,
        rel_tol=0.0,
        abs_tol=0.0,
    )

    kwargs = {
        "num_samples": 3,
        "nx": nx,
        "seed": 937,
        "gamma": 2.25,
        "tau": 4.0,
        "sigma": 5.0,
        "mean": -0.25,
        "device": torch.device("cpu"),
        "dtype": torch.float64,
    }
    unit = sample_gaussian_random_field_initial_conditions(
        **kwargs, domain_length=1.0
    )
    physical = sample_gaussian_random_field_initial_conditions(
        **kwargs, domain_length=domain_length
    )
    unit_nyquist = torch.fft.rfft(unit, norm="forward")[:, -1].abs()
    physical_nyquist = torch.fft.rfft(physical, norm="forward")[:, -1].abs()
    lambda_unit = kwargs["sigma"] ** 2 * (
        (math.pi * nx) ** 2 + kwargs["tau"] ** 2
    ) ** (-kwargs["gamma"])
    lambda_physical = kwargs["sigma"] ** 2 * (
        from_nyquist_formula**2 + kwargs["tau"] ** 2
    ) ** (-kwargs["gamma"])
    assert torch.allclose(
        physical_nyquist / unit_nyquist,
        torch.full_like(unit_nyquist, math.sqrt(lambda_physical / lambda_unit)),
        atol=5e-12,
        rtol=5e-12,
    )


@pytest.mark.parametrize(
    "domain_length",
    [float("nan"), float("inf"), float("-inf"), 0.0, -1.0],
)
def test_grf_rejects_non_finite_or_non_positive_domain_length(
    domain_length: float,
) -> None:
    with pytest.raises(ValueError, match="domain_length must be finite and positive"):
        sample_gaussian_random_field_initial_conditions(
            1,
            8,
            domain_length=domain_length,
            seed=0,
            device=torch.device("cpu"),
        )


def test_grf_domain_length_is_a_required_keyword_argument() -> None:
    with pytest.raises(TypeError, match="domain_length"):
        sample_gaussian_random_field_initial_conditions(
            1,
            8,
            seed=0,
            device=torch.device("cpu"),
        )


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
            0,
            16,
            domain_length=1.0,
            seed=0,
            device=torch.device("cpu"),
        ),
    ],
)
def test_numerics_reject_invalid_inputs(call) -> None:
    with pytest.raises(ValueError):
        call()
