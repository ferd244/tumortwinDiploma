"""Integration checks for HemoInvasion3D model."""

from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tumortwin.models.hemo_invasion_3d import HemoInvasion3D
from tumortwin.models.pde_system import extract_trajectory_component
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
