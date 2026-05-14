"""TorchDiffEqSolver: adaptive methods must honor the same time grid as fixed-step solvers."""

from datetime import datetime, timedelta

import pytest
import torch

from tumortwin.models.base import TumorGrowthModel3D
from tumortwin.solvers.torch_solver import (
    ADAPTIVE_TORCHDIFFEQ_METHODS,
    TorchDiffEqSolver,
    TorchDiffEqSolverOptions,
)


def test_torchdiffeq_solver_exports_adaptive_method_set():
    assert "dopri5" in ADAPTIVE_TORCHDIFFEQ_METHODS


class _ZeroDynamicsWithRtCallback(TumorGrowthModel3D):
    """du/dt = 0; callback_step scales state if integration hits exactly ``rt_hit_day``."""

    def __init__(self, rt_hit_day: float = 1.0):
        super().__init__()
        self.device = torch.device("cpu")
        self.rt_hit_day = float(rt_hit_day)

    def forward(self, t: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(u)

    def callback_step(self, t, u, dt):
        if float(t) == self.rt_hit_day:
            u.mul_(0.1)
        return u


class _ThreeChannelZeroDynamicsRt(TumorGrowthModel3D):
    """
    Stacked state ``(C, D, H, W)`` like ``HemoInvasion3D``: du/dt = 0; RT scales channels 0 and 1.

    ``torchdiffeq`` discards the return value of ``callback_step``; updates must be in-place
    (``u[0].copy_(...)`` / ``mul_``), not only ``return torch.stack(...)``.
    """

    def __init__(self, rt_hit_day: float = 1.0):
        super().__init__()
        self.device = torch.device("cpu")
        self.rt_hit_day = float(rt_hit_day)

    def forward(self, t: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(u)

    def callback_step(self, t, u, dt):
        if float(t) == self.rt_hit_day:
            u[0].mul_(0.1)
            u[1].mul_(0.1)
        return u


@pytest.mark.parametrize("method", ("rk4", "dopri5"))
def test_adaptive_solver_hits_rt_callback_grid(method):
    """
    Fixed solvers use ``grid_constructor`` so substeps land on treatment days.

    Adaptive solvers ignore ``grid_constructor``; they need ``step_t`` built from the same grid
    or ``callback_step`` (e.g. radiotherapy) never runs at the scheduled day.
    """
    model = _ZeroDynamicsWithRtCallback(rt_hit_day=1.0)
    solver = TorchDiffEqSolver(
        model,
        TorchDiffEqSolverOptions(
            step_size=timedelta(days=0.5),
            method=method,
            device=torch.device("cpu"),
            use_adjoint=False,
        ),
    )
    t0 = datetime(2020, 1, 1)
    timepoints = [t0, t0 + timedelta(days=3.0)]
    u0 = torch.ones((4, 4, 4), dtype=torch.float32)
    _, u_traj = solver.solve(timepoints, u0)
    # Final state should reflect RT at day 1 while du/dt = 0.
    assert torch.allclose(u_traj[-1], torch.full_like(u0, 0.1), rtol=0, atol=1e-5)


@pytest.mark.parametrize("method", ("rk4", "dopri5"))
def test_multichannel_state_rt_callback_applies_inplace(method):
    """Regression: stacked PDE state must mutate ``u`` in-place so RT survives integration."""
    model = _ThreeChannelZeroDynamicsRt(rt_hit_day=1.0)
    solver = TorchDiffEqSolver(
        model,
        TorchDiffEqSolverOptions(
            step_size=timedelta(days=0.5),
            method=method,
            device=torch.device("cpu"),
            use_adjoint=False,
        ),
    )
    t0 = datetime(2020, 1, 1)
    timepoints = [t0, t0 + timedelta(days=3.0)]
    u0 = torch.ones((3, 4, 4, 4), dtype=torch.float32)
    _, u_traj = solver.solve(timepoints, u0)
    want = torch.ones_like(u0)
    want[0].fill_(0.1)
    want[1].fill_(0.1)
    assert torch.allclose(u_traj[-1], want, rtol=0, atol=1e-5)


def test_dopri5_accepts_rtol_atol_and_ode_options():
    model = _ZeroDynamicsWithRtCallback(rt_hit_day=1.0)
    solver = TorchDiffEqSolver(
        model,
        TorchDiffEqSolverOptions(
            step_size=timedelta(days=0.5),
            method="dopri5",
            device=torch.device("cpu"),
            use_adjoint=False,
            rtol=1e-6,
            atol=1e-8,
            ode_options={"max_num_steps": 10_000},
        ),
    )
    t0 = datetime(2020, 1, 1)
    timepoints = [t0, t0 + timedelta(days=2.0)]
    u0 = torch.ones((2, 2, 2), dtype=torch.float32)
    _, u_traj = solver.solve(timepoints, u0)
    assert torch.allclose(u_traj[-1], torch.full_like(u0, 0.1), rtol=0, atol=1e-5)


def test_dopri5_short_integration_finite():
    model = _ZeroDynamicsWithRtCallback(rt_hit_day=1.0)
    solver = TorchDiffEqSolver(
        model,
        TorchDiffEqSolverOptions(
            step_size=timedelta(days=1.0),
            method="dopri5",
            device=torch.device("cpu"),
            use_adjoint=False,
        ),
    )
    t0 = datetime(2020, 1, 1)
    timepoints = [t0, t0 + timedelta(days=1.0)]
    u0 = torch.randn((2, 2, 2), dtype=torch.float32)
    _, u_traj = solver.solve(timepoints, u0)
    assert u_traj.shape == (2,) + u0.shape
    assert torch.isfinite(u_traj).all()
