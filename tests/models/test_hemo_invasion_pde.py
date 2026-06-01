"""Integration checks for HemoInvasion3D model."""

from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from tumortwin.models.hemo_invasion_3d import HemoInvasion3D
from tumortwin.models.base import TumorGrowthModel3D
from tumortwin.models.pde_system import extract_trajectory_component, extract_trajectory_sum_components
from tumortwin.solvers.torch_solver import TorchDiffEqSolver, TorchDiffEqSolverOptions
from tumortwin.types.imaging import NibabelNifti


def _patient_with_brain_mask(mask_np: np.ndarray):
    return SimpleNamespace(brainmask_image=NibabelNifti.from_array(mask_np))


@pytest.fixture
def tiny_hemo_model():
    shape = (8, 8, 8)
    mask = np.ones(shape, dtype=np.float32)
    patient = _patient_with_brain_mask(mask)
    initial_n = torch.rand(shape, dtype=torch.float32) * 0.1
    vessel_mask = torch.zeros(shape, dtype=torch.bool)
    vessel_mask[2:4, 2:4, 2:4] = True
    model = HemoInvasion3D(
        B=torch.tensor(0.05, dtype=torch.float32),
        Dn=torch.tensor(0.01, dtype=torch.float32),
        Ds=torch.tensor(0.05, dtype=torch.float32),
        k_s=torch.tensor(0.2, dtype=torch.float32),
        s_star=torch.tensor(0.1, dtype=torch.float32),
        patient_data=patient,
        initial_n=initial_n,
        s_crit=torch.tensor(0.5, dtype=torch.float32),
        s_smooth=torch.tensor(0.1, dtype=torch.float32),
        vessel_mask=vessel_mask,
        s_vessel=1.0,
        s_outside=0.0,
        require_grad=False,
        device=torch.device("cpu"),
    )
    return model, shape


def test_hemo_forward_shape_and_finite(tiny_hemo_model):
    model, shape = tiny_hemo_model
    u0 = model.get_initial_state()
    assert u0.shape == (3,) + shape
    model.validate_state_shape(u0)
    du = model.forward(torch.tensor(0.0), u0)
    assert du.shape == u0.shape
    assert torch.all(torch.isfinite(du))


def test_hemo_total_cellularity_equals_n_plus_m(tiny_hemo_model):
    model, shape = tiny_hemo_model
    solver = TorchDiffEqSolver(
        model,
        TorchDiffEqSolverOptions(
            step_size=timedelta(days=1.0),
            method="rk4",
            device=torch.device("cpu"),
            use_adjoint=False,
        ),
    )
    t0 = datetime(2020, 1, 1)
    timepoints = [t0, t0 + timedelta(days=1.0)]
    _, u_traj = solver.solve(timepoints, model.get_initial_state())
    n = extract_trajectory_component(u_traj, 0)
    m = extract_trajectory_component(u_traj, 1)
    tot = extract_trajectory_sum_components(u_traj, HemoInvasion3D.total_cellularity_component_indices)
    assert tot.shape == n.shape
    assert torch.allclose(tot, n + m)


def test_torch_solver_short_run_hemo(tiny_hemo_model):
    model, shape = tiny_hemo_model
    solver = TorchDiffEqSolver(
        model,
        TorchDiffEqSolverOptions(
            step_size=timedelta(days=1.0),
            method="rk4",
            device=torch.device("cpu"),
            use_adjoint=False,
        ),
    )
    t0 = datetime(2020, 1, 1)
    timepoints = [t0, t0 + timedelta(days=1.0)]
    _, u_traj = solver.solve(timepoints, model.get_initial_state())
    assert u_traj.shape == (2, 3) + shape
    n_series = extract_trajectory_component(u_traj, 0)
    assert n_series.shape == (2,) + shape
    assert torch.isfinite(n_series).all()


class _NoTreatmentModel(TumorGrowthModel3D):
    """Minimal model intentionally missing treatment metadata attrs."""

    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")
        self.rate = nn.Parameter(torch.tensor(0.1, dtype=torch.float32), requires_grad=False)

    def forward(self, t: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return -self.rate * u


def test_solver_grid_constructor_without_treatment_attributes():
    model = _NoTreatmentModel()
    solver = TorchDiffEqSolver(
        model,
        TorchDiffEqSolverOptions(
            step_size=timedelta(days=0.5),
            method="rk4",
            device=torch.device("cpu"),
            use_adjoint=False,
        ),
    )
    t0 = datetime(2020, 1, 1)
    timepoints = [t0, t0 + timedelta(days=2.0)]
    u0 = torch.ones((4, 4, 4), dtype=torch.float32)
    _, u_traj = solver.solve(timepoints, u0)
    assert u_traj.shape[0] == 2
    assert torch.isfinite(u_traj).all()
