from __future__ import annotations

import math

import pytest
import torch

from pol.learning.metrics import symmetric_field_discrepancy
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
from pol.systems.burgers import step_metadata
from pol.systems.reaction_diffusion import solve_reaction_diffusion


def _mixed_mode_field(nx: int) -> torch.Tensor:
    x = torch.arange(nx, dtype=torch.float64) / nx
    highest_mode = (nx - 1) // 2
    return (
        0.35
        + 0.4 * torch.cos(2.0 * torch.pi * 2.0 * x)
        + 0.25 * torch.sin(2.0 * torch.pi * highest_mode * x)
    ).unsqueeze(0)


def test_burgers_step_metadata_distinguishes_requested_and_effective_steps() -> None:
    metadata = step_metadata(
        solver="semi_implicit",
        dt=0.01,
        fine_dt=0.003,
        final_time=0.02,
    )
    assert metadata == {
        "solver": "split_step",
        "requested_outer_dt": 0.01,
        "requested_fine_dt": 0.003,
        "outer_step_count": 2,
        "effective_substep": 0.0025,
        "substeps_per_outer": 4,
    }


def test_etdrk4_step_metadata_uses_requested_dt_as_canonical_step() -> None:
    metadata = step_metadata(
        solver="fourier_pseudospectral_etdrk4",
        dt=0.005,
        fine_dt=None,
        final_time=0.02,
    )
    assert metadata == {
        "solver": "etdrk4",
        "requested_outer_dt": 0.005,
        "requested_fine_dt": None,
        "outer_step_count": 4,
        "effective_substep": 0.005,
        "substeps_per_outer": 1,
    }


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


def _pre_correction_even_split_step_reference(
    u0: torch.Tensor,
    *,
    dt: float,
    obs_steps: list[int],
    nu: float,
    fine_dt: float,
    b: float,
    dealias: bool,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Reproduce the old width-inferred algorithm where even nx=2*(width-1)."""
    nx = u0.shape[-1]
    if nx % 2 != 0:
        raise ValueError("pre-correction reference is valid only for an even grid")

    frequencies = torch.fft.rfftfreq(
        nx,
        d=1.0 / nx,
        device=u0.device,
    )
    k = (2.0 * torch.pi * frequencies).to(dtype=u0.dtype)
    mask = (
        torch.arange(nx // 2 + 1, device=u0.device) <= nx // 3
    ).to(dtype=u0.dtype)

    def apply_dealias(value: torch.Tensor) -> torch.Tensor:
        if not dealias:
            return value
        return value * mask.to(
            device=value.device,
            dtype=value.real.dtype,
        ).unsqueeze(0)

    def nonlinear(value: torch.Tensor) -> torch.Tensor:
        rfft_width = value.shape[-1]
        inferred_nx = 2 * (rfft_width - 1)
        assert inferred_nx == nx
        real_values = torch.fft.irfft(
            value,
            n=inferred_nx,
            dim=-1,
            norm="forward",
        )
        nonlinear_hat = (
            -0.5j
            * b
            * k
            * torch.fft.rfft(
                real_values * real_values,
                dim=-1,
                norm="forward",
            )
        )
        return apply_dealias(nonlinear_hat)

    initial_hat = torch.fft.rfft(u0, dim=-1, norm="forward")
    rfft_width = initial_hat.shape[-1]
    inferred_nx = 2 * (rfft_width - 1)
    assert inferred_nx == nx
    initial_nonlinear_hat = nonlinear(initial_hat)

    n_sub = max(1, int(math.ceil(dt / fine_dt))) if fine_dt > 0.0 else 1
    h = dt / float(n_sub)
    heat = torch.exp(-nu * k.pow(2) * h).to(dtype=initial_hat.real.dtype)
    u_hat = apply_dealias(initial_hat)

    observed: list[torch.Tensor] = []
    obs_sorted = sorted(set(int(value) for value in obs_steps))
    obs_ptr = 0
    for step in range(1, obs_sorted[-1] + 1):
        for _ in range(n_sub):
            u_hat = u_hat * heat
            u_hat = u_hat + h * nonlinear(u_hat)
            u_hat = apply_dealias(u_hat)
        while obs_ptr < len(obs_sorted) and step == obs_sorted[obs_ptr]:
            observed.append(
                torch.fft.irfft(
                    u_hat,
                    n=nx,
                    dim=-1,
                    norm="forward",
                ).clone()
            )
            obs_ptr += 1

    return initial_nonlinear_hat, observed


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


@pytest.mark.parametrize("dealias", [False, True])
def test_split_step_even_grid_matches_pre_correction_exact_regression(
    dealias: bool,
) -> None:
    nx = 16
    u0 = _mixed_mode_field(nx)
    assert u0.dtype == torch.float64

    expected_nonlinear, expected_trajectory = (
        _pre_correction_even_split_step_reference(
            u0,
            dt=0.01,
            obs_steps=[1, 2],
            nu=0.05,
            fine_dt=0.005,
            b=0.7,
            dealias=dealias,
        )
    )

    u_hat = torch.fft.rfft(u0, n=nx, dim=-1, norm="forward")
    k = make_wavenumbers(nx, u0.device, u0.dtype)
    mask = make_dealias_mask(nx, u0.device, u0.dtype) if dealias else None
    actual_nonlinear = burgers_nonlinear_hat(
        u_hat,
        k,
        nx=nx,
        b=0.7,
        dealias=dealias,
        mask=mask,
    )
    actual_trajectory = simulate_burgers_split_step(
        u0,
        dt=0.01,
        Tr=0.02,
        obs_steps=[1, 2],
        nu=0.05,
        fine_dt=0.005,
        b=0.7,
        dealias=dealias,
    )

    assert torch.equal(actual_nonlinear, expected_nonlinear)
    assert len(actual_trajectory) == len(expected_trajectory) == 2
    for actual_step, expected_step in zip(
        actual_trajectory,
        expected_trajectory,
        strict=True,
    ):
        assert torch.equal(actual_step, expected_step)


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


@pytest.mark.parametrize("nx", [15, 16])
def test_symmetric_field_discrepancy_is_swap_invariant_with_low_modes(
    nx: int,
) -> None:
    x = torch.arange(nx, dtype=torch.float64) / nx
    a = (
        0.2
        + 0.3 * torch.sin(2.0 * torch.pi * x)
        + 0.1 * torch.cos(4.0 * torch.pi * x)
    ).unsqueeze(0)
    b = (
        0.2
        + 0.29 * torch.sin(2.0 * torch.pi * x)
        + 0.08 * torch.cos(4.0 * torch.pi * x)
    ).unsqueeze(0)
    forward = symmetric_field_discrepancy(
        a,
        b,
        q=5,
        domain_length=1.0,
    )
    reverse = symmetric_field_discrepancy(
        b,
        a,
        q=5,
        domain_length=1.0,
    )
    assert forward == reverse
    assert forward["mean_absolute_l2"] > 0.0
    assert forward["mean_relative_l2"] > 0.0
    assert forward["low_mode_relative_l2"] > 0.0
    assert all(math.isfinite(value) for value in forward.values())


@pytest.mark.parametrize("nx", [15, 16])
@pytest.mark.parametrize("nonlinear_filter", ["none", "two_thirds"])
def test_reaction_diffusion_zero_equilibrium_is_exact(
    nx: int,
    nonlinear_filter: str,
) -> None:
    initial = torch.zeros((2, nx), dtype=torch.float64)
    result = solve_reaction_diffusion(
        initial,
        nu=0.07,
        alpha=1.2,
        beta=0.9,
        time=0.05,
        dt=0.01,
        domain_length=2.3,
        nonlinear_filter=nonlinear_filter,
    )
    assert torch.equal(result.values, initial)
    assert result.metadata["step_count"] == 5


@pytest.mark.parametrize(
    ("constant", "nx", "domain_length", "nonlinear_filter"),
    [
        (0.25, 15, 2.5, "none"),
        (-0.4, 16, 1.7, "two_thirds"),
    ],
)
def test_reaction_diffusion_constant_field_matches_independent_scalar_recurrence(
    constant: float,
    nx: int,
    domain_length: float,
    nonlinear_filter: str,
) -> None:
    dt = 0.01
    steps = 6
    alpha = 0.8
    beta = 1.1
    expected_scalar = constant
    for _ in range(steps):
        expected_scalar = (
            expected_scalar
            + dt * alpha * expected_scalar
            - dt * beta * expected_scalar**3
        )
    initial = torch.full((2, nx), constant, dtype=torch.float64)
    actual = solve_reaction_diffusion(
        initial,
        nu=0.07,
        alpha=alpha,
        beta=beta,
        time=steps * dt,
        dt=dt,
        domain_length=domain_length,
        nonlinear_filter=nonlinear_filter,
    ).values
    expected = torch.full_like(actual, expected_scalar)
    assert torch.allclose(actual, expected, atol=2e-14, rtol=2e-14)


@pytest.mark.parametrize(
    ("sign", "nx", "nonlinear_filter"),
    [(1.0, 15, "none"), (-1.0, 16, "two_thirds")],
)
def test_reaction_diffusion_nonzero_constant_equilibria_are_preserved(
    sign: float,
    nx: int,
    nonlinear_filter: str,
) -> None:
    alpha = 2.0
    beta = 0.5
    equilibrium = sign * math.sqrt(alpha / beta)
    initial = torch.full((1, nx), equilibrium, dtype=torch.float64)
    actual = solve_reaction_diffusion(
        initial,
        nu=0.04,
        alpha=alpha,
        beta=beta,
        time=0.05,
        dt=0.01,
        domain_length=1.8,
        nonlinear_filter=nonlinear_filter,
    ).values
    assert torch.allclose(actual, initial, atol=2e-14, rtol=2e-14)


@pytest.mark.parametrize(
    ("nx", "domain_length", "mode", "basis", "nonlinear_filter"),
    [
        (15, 2.5, 2, "cosine", "none"),
        (16, 1.7, 3, "sine", "two_thirds"),
    ],
)
def test_reaction_diffusion_beta_zero_one_step_has_analytic_multiplier(
    nx: int,
    domain_length: float,
    mode: int,
    basis: str,
    nonlinear_filter: str,
) -> None:
    dt = 0.005
    nu = 0.07
    alpha = 0.8
    x = torch.arange(nx, dtype=torch.float64) * domain_length / nx
    k = 2.0 * math.pi * mode / domain_length
    values = torch.cos(k * x) if basis == "cosine" else torch.sin(k * x)
    initial = values.unsqueeze(0)
    expected = ((1.0 + dt * alpha) / (1.0 + dt * nu * k**2)) * initial
    actual = solve_reaction_diffusion(
        initial,
        nu=nu,
        alpha=alpha,
        beta=0.0,
        time=dt,
        dt=dt,
        domain_length=domain_length,
        nonlinear_filter=nonlinear_filter,
    ).values
    assert torch.allclose(actual, expected, atol=2e-14, rtol=2e-14)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("nu", float("nan"), "finite"),
        ("alpha", float("nan"), "finite"),
        ("beta", float("nan"), "finite"),
        ("time", float("nan"), "finite"),
        ("dt", float("nan"), "finite"),
        ("domain_length", float("nan"), "finite"),
        ("dt", 0.0, "positive"),
        ("time", 0.0, "positive"),
        ("nonlinear_filter", "invalid", "nonlinear_filter"),
    ],
)
def test_reaction_diffusion_rejects_invalid_parameters(
    field: str,
    value: object,
    message: str,
) -> None:
    kwargs: dict[str, object] = {
        "nu": 0.1,
        "alpha": 1.0,
        "beta": 1.0,
        "time": 0.02,
        "dt": 0.01,
        "domain_length": 1.0,
        "nonlinear_filter": "two_thirds",
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=message):
        solve_reaction_diffusion(
            torch.zeros((1, 15), dtype=torch.float64),
            **kwargs,
        )


def test_reaction_diffusion_rejects_time_misalignment_and_nonfinite_state() -> None:
    with pytest.raises(ValueError, match="must align"):
        solve_reaction_diffusion(
            torch.zeros((1, 15), dtype=torch.float64),
            nu=0.1,
            alpha=1.0,
            beta=1.0,
            time=0.02,
            dt=0.006,
            domain_length=1.0,
        )
    initial = torch.zeros((1, 15), dtype=torch.float64)
    initial[0, 0] = torch.nan
    with pytest.raises(FloatingPointError, match="NaN/Inf"):
        solve_reaction_diffusion(
            initial,
            nu=0.1,
            alpha=1.0,
            beta=1.0,
            time=0.02,
            dt=0.01,
            domain_length=1.0,
        )


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
