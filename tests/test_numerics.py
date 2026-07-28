from __future__ import annotations

import math

import pytest
import torch

from pol.numerics.burgers import (
    burgers_nonlinear_hat,
    burgers_split_step_outer,
    make_dealias_mask,
    make_wavenumbers,
    simulate_burgers_split_step,
)
from pol.numerics.etdrk4 import (
    burgers_nonlinear_hat as etdrk4_burgers_nonlinear_hat,
    cox_matthews_coefficients,
    cox_matthews_etdrk4_step,
    rfft_wavenumbers,
    simulate_burgers_etdrk4,
    simulate_burgers_etdrk4_trajectory,
)
from pol.numerics.initial_conditions import sample_gaussian_random_field_initial_conditions
from pol.runtime.hashing import tensor_sha256


def _mixed_mode_field(nx: int) -> torch.Tensor:
    x = torch.arange(nx, dtype=torch.float64) / nx
    highest_mode = (nx - 1) // 2
    return (
        0.35
        + 0.4 * torch.cos(2.0 * torch.pi * 2.0 * x)
        + 0.25 * torch.sin(2.0 * torch.pi * highest_mode * x)
    ).unsqueeze(0)


def _explicit_split_step_reference(
    u0: torch.Tensor,
    *,
    dt: float,
    obs_steps: list[int],
    nu: float,
    fine_dt: float,
    b: float,
    dealias: bool,
) -> list[torch.Tensor]:
    nx = u0.shape[-1]
    frequencies = torch.fft.rfftfreq(
        nx,
        d=1.0 / nx,
        device=u0.device,
    )
    k = (2.0 * torch.pi * frequencies).to(dtype=u0.dtype)
    mask = (
        torch.arange(nx // 2 + 1, device=u0.device) <= nx // 3
    ).to(dtype=u0.dtype)
    u_hat = torch.fft.rfft(u0, n=nx, dim=-1, norm="forward")
    if dealias:
        u_hat = u_hat * mask

    n_sub = max(1, math.ceil(dt / fine_dt)) if fine_dt > 0.0 else 1
    h = dt / n_sub
    heat = torch.exp(-nu * k.pow(2) * h).to(dtype=u0.dtype)
    observed: list[torch.Tensor] = []
    obs_set = set(obs_steps)
    for step in range(1, max(obs_steps) + 1):
        for _ in range(n_sub):
            u_hat = u_hat * heat
            u = torch.fft.irfft(
                u_hat,
                n=nx,
                dim=-1,
                norm="forward",
            )
            nonlinear_hat = (
                -0.5j
                * b
                * k
                * torch.fft.rfft(
                    u * u,
                    n=nx,
                    dim=-1,
                    norm="forward",
                )
            )
            if dealias:
                nonlinear_hat = nonlinear_hat * mask
            u_hat = u_hat + h * nonlinear_hat
            if dealias:
                u_hat = u_hat * mask
        if step in obs_set:
            observed.append(
                torch.fft.irfft(
                    u_hat,
                    n=nx,
                    dim=-1,
                    norm="forward",
                ).clone()
            )
    return observed


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


def test_split_step_odd_grid_nonlinearity_matches_explicit_nx_reference() -> None:
    nx = 15
    b = 0.7
    u = _mixed_mode_field(nx)
    u_hat = torch.fft.rfft(u, n=nx, dim=-1, norm="forward")
    k = make_wavenumbers(nx, u.device, u.dtype)

    reconstructed = torch.fft.irfft(
        u_hat,
        n=nx,
        dim=-1,
        norm="forward",
    )
    expected = (
        -0.5j
        * b
        * k
        * torch.fft.rfft(
            reconstructed * reconstructed,
            n=nx,
            dim=-1,
            norm="forward",
        )
    )
    actual = burgers_nonlinear_hat(u_hat, k, nx=nx, b=b)

    assert torch.allclose(actual, expected, atol=1e-13, rtol=1e-13)


def test_rfft_width_does_not_determine_real_grid_length_and_nx_is_required() -> None:
    even_nx = 14
    odd_nx = 15
    even_hat = torch.fft.rfft(torch.zeros(even_nx, dtype=torch.float64))
    odd_hat = torch.fft.rfft(torch.zeros(odd_nx, dtype=torch.float64))
    assert even_hat.shape == odd_hat.shape == (8,)
    assert even_nx != odd_nx

    k = make_wavenumbers(odd_nx, odd_hat.device, odd_hat.real.dtype)
    with pytest.raises(TypeError, match="nx"):
        burgers_nonlinear_hat(odd_hat, k)


@pytest.mark.parametrize("nx", [15, 16])
@pytest.mark.parametrize("dealias", [False, True])
def test_split_step_nonlinearity_matches_independent_explicit_nx_reference(
    nx: int,
    dealias: bool,
) -> None:
    b = 0.7
    u = _mixed_mode_field(nx)
    u_hat = torch.fft.rfft(u, n=nx, dim=-1, norm="forward")
    k = make_wavenumbers(nx, u.device, u.dtype)
    mask = make_dealias_mask(nx, u.device, u.dtype) if dealias else None

    reconstructed = torch.fft.irfft(
        u_hat,
        n=nx,
        dim=-1,
        norm="forward",
    )
    expected = (
        -0.5j
        * b
        * k
        * torch.fft.rfft(
            reconstructed * reconstructed,
            n=nx,
            dim=-1,
            norm="forward",
        )
    )
    if dealias:
        expected = expected * (
            torch.arange(nx // 2 + 1, device=u.device) <= nx // 3
        ).to(dtype=u.dtype)

    actual = burgers_nonlinear_hat(
        u_hat,
        k,
        nx=nx,
        b=b,
        dealias=dealias,
        mask=mask,
    )
    assert actual.shape == u_hat.shape
    assert torch.allclose(actual, expected, atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize("nx", [15, 16])
@pytest.mark.parametrize("dealias", [False, True])
def test_split_step_short_trajectory_matches_independent_reference(
    nx: int,
    dealias: bool,
) -> None:
    u0 = _mixed_mode_field(nx)
    kwargs = {
        "dt": 0.01,
        "Tr": 0.02,
        "obs_steps": [1, 2],
        "nu": 0.05,
        "fine_dt": 0.005,
        "b": 0.7,
        "dealias": dealias,
    }
    expected = _explicit_split_step_reference(
        u0,
        dt=kwargs["dt"],
        obs_steps=kwargs["obs_steps"],
        nu=kwargs["nu"],
        fine_dt=kwargs["fine_dt"],
        b=kwargs["b"],
        dealias=dealias,
    )
    actual = simulate_burgers_split_step(u0, **kwargs)

    assert [value.shape for value in actual] == [u0.shape, u0.shape]
    for actual_step, expected_step in zip(actual, expected, strict=True):
        assert torch.allclose(
            actual_step,
            expected_step,
            atol=1e-13,
            rtol=1e-13,
        )


@pytest.mark.parametrize(
    ("dealias", "nonlinear_hash", "trajectory_hashes"),
    [
        (
            False,
            "d2e26a3545d88d85940bf60cc196b4f0c8f6aa706e3fb2fc7223847fc2a826a2",
            [
                "961f4a8a516c6e4ece14eeabb2e8b8b2c61bd82a0a06605f314e073671fd40e3",
                "df716d79d542fabeb809cf9878ff737def54eb78b2d47878d23d699f74cf5c89",
            ],
        ),
        (
            True,
            "58ffd9295e680931d22d19ead40f5596ec73fd67d8b297a9897709b228b3c28e",
            [
                "ff3f11110107fc14c9c143ddf6fdf234fc4a00b6a9bba9efd0a0bca3e687a943",
                "c3163f74b2b6d7a4e2597a2d509454f2c54491fb2b21b6a6fce89a20fd396bb3",
            ],
        ),
    ],
)
def test_split_step_even_grid_matches_pre_correction_exact_regression(
    dealias: bool,
    nonlinear_hash: str,
    trajectory_hashes: list[str],
) -> None:
    nx = 16
    u0 = _mixed_mode_field(nx)
    u_hat = torch.fft.rfft(u0, n=nx, dim=-1, norm="forward")
    k = make_wavenumbers(nx, u0.device, u0.dtype)
    mask = make_dealias_mask(nx, u0.device, u0.dtype) if dealias else None
    nonlinear = burgers_nonlinear_hat(
        u_hat,
        k,
        nx=nx,
        b=0.7,
        dealias=dealias,
        mask=mask,
    )
    trajectory = simulate_burgers_split_step(
        u0,
        dt=0.01,
        Tr=0.02,
        obs_steps=[1, 2],
        nu=0.05,
        fine_dt=0.005,
        b=0.7,
        dealias=dealias,
    )

    assert tensor_sha256(nonlinear) == nonlinear_hash
    assert [tensor_sha256(value) for value in trajectory] == trajectory_hashes


def test_split_step_rejects_spectral_contract_mismatches() -> None:
    nx = 16
    width = nx // 2 + 1
    u_hat = torch.zeros((2, width), dtype=torch.complex128)
    k = torch.arange(width, dtype=torch.float64)
    mask = torch.ones(width, dtype=torch.float64)

    with pytest.raises(ValueError, match="nx must be >= 2"):
        burgers_nonlinear_hat(
            torch.zeros((2, 1), dtype=torch.complex128),
            torch.zeros(1, dtype=torch.float64),
            nx=1,
        )
    with pytest.raises(ValueError, match="u_hat Fourier width"):
        burgers_nonlinear_hat(u_hat[..., :-1], k, nx=nx)
    with pytest.raises(ValueError, match="k Fourier width"):
        burgers_nonlinear_hat(u_hat, k[:-1], nx=nx)
    with pytest.raises(ValueError, match="mask Fourier width"):
        burgers_nonlinear_hat(
            u_hat,
            k,
            nx=nx,
            dealias=True,
            mask=mask[:-1],
        )
    with pytest.raises(ValueError, match="forcing_hat must have shape"):
        burgers_split_step_outer(
            u_hat,
            dt=0.01,
            nu=0.05,
            k=k,
            fine_dt=0.005,
            nx=nx,
            forcing_hat=torch.zeros((1, width), dtype=torch.complex128),
        )


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
@pytest.mark.parametrize("dealias", [False, True])
def test_etdrk4_trajectory_is_deterministic_for_even_and_odd_grids(
    dtype: torch.dtype,
    nx: int,
    dealias: bool,
) -> None:
    x = torch.arange(nx, dtype=dtype) / nx
    u0 = (0.1 * torch.cos(2.0 * torch.pi * x))[None]
    kwargs = dict(
        nu=0.05,
        T=0.02,
        dt=0.01,
        obs_steps=[1, 2],
        dealias=dealias,
    )
    first = simulate_burgers_etdrk4_trajectory(u0, **kwargs)
    second = simulate_burgers_etdrk4_trajectory(u0, **kwargs)
    assert [tuple(value.shape) for value in first] == [(1, nx), (1, nx)]
    assert all(value.dtype == dtype and torch.isfinite(value).all() for value in first)
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))


@pytest.mark.parametrize("nx", [15, 16])
@pytest.mark.parametrize("dealias", [False, True])
def test_etdrk4_nonlinearity_has_parity_consistent_real_and_fourier_lengths(
    nx: int,
    dealias: bool,
) -> None:
    u = _mixed_mode_field(nx)
    k = rfft_wavenumbers(nx, device=u.device, dtype=u.dtype)
    expected = (
        -0.5j
        * k
        * torch.fft.rfft(
            u * u,
            n=nx,
            dim=-1,
        )
    )
    if dealias:
        expected = expected * (
            torch.arange(nx // 2 + 1, device=u.device) <= nx // 3
        ).to(dtype=expected.dtype)
    actual = etdrk4_burgers_nonlinear_hat(
        u,
        k,
        dealias=dealias,
    )

    assert u.shape[-1] == nx
    assert actual.shape[-1] == nx // 2 + 1
    assert k.shape[-1] == nx // 2 + 1
    assert torch.allclose(actual, expected, atol=1e-13, rtol=1e-13)


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
