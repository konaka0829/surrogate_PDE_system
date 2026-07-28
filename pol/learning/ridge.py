from __future__ import annotations

from dataclasses import dataclass

import torch

from pol.math.fourier import real_fourier_synthesis


def l2_synthesis_matrix(
    q: int,
    J: int,
    *,
    domain_length: float,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    eye = torch.eye(q, dtype=dtype, device=device)
    points = real_fourier_synthesis(eye, J, domain_length=domain_length).T
    return points * (domain_length / J) ** 0.5


@dataclass(frozen=True)
class AffineReadout:
    """Affine row-sample prediction ``y = x @ W.T + b``."""

    W: torch.Tensor
    b: torch.Tensor
    solver: str
    svd_rcond: float | None = None
    singular_value_cutoff: float | None = None
    numerical_rank: int | None = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.T + self.b


def fit_centered_affine_ridge(
    x: torch.Tensor,
    y: torch.Tensor,
    zeta: float,
    *,
    svd_rcond: float | None = None,
) -> AffineReadout:
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] == 0:
        raise ValueError("ridge expects x=(N,J), y=(N,q) with shared positive N")
    if x.dtype != y.dtype or x.device != y.device or zeta < 0:
        raise ValueError("ridge dtype/device mismatch or negative zeta")
    xm, ym = x.mean(0), y.mean(0)
    xc, yc = x - xm, y - ym
    if zeta == 0:
        rcond = (
            float(svd_rcond)
            if svd_rcond is not None
            else torch.finfo(x.dtype).eps * max(xc.shape)
        )
        if not 0.0 < rcond < 1.0:
            raise ValueError("svd_rcond must lie strictly between zero and one")
        u, singular_values, vh = torch.linalg.svd(xc, full_matrices=False)
        cutoff = (
            rcond * float(singular_values.max())
            if singular_values.numel()
            else 0.0
        )
        retained = singular_values > cutoff
        if bool(retained.any()):
            projected = u[:, retained].T @ yc
            beta = vh[retained].T @ (
                projected / singular_values[retained, None]
            )
        else:
            beta = torch.zeros(
                (x.shape[1], y.shape[1]), dtype=x.dtype, device=x.device
            )
        solver = "svd_minimum_norm"
        rank = int(retained.sum())
    elif x.shape[1] <= x.shape[0]:
        gram = xc.T @ xc / x.shape[0]
        rhs = xc.T @ yc / x.shape[0]
        beta = torch.linalg.solve(
            gram + zeta * torch.eye(x.shape[1], dtype=x.dtype, device=x.device),
            rhs,
        )
        solver, rcond, cutoff, rank = "primal_ridge_solve", None, None, None
    else:
        gram = xc @ xc.T / x.shape[0]
        dual = torch.linalg.solve(
            gram + zeta * torch.eye(x.shape[0], dtype=x.dtype, device=x.device),
            yc / x.shape[0],
        )
        beta = xc.T @ dual
        solver, rcond, cutoff, rank = "dual_ridge_solve", None, None, None
    W = beta.T.contiguous()
    return AffineReadout(W, ym - xm @ beta, solver, rcond, cutoff, rank)
