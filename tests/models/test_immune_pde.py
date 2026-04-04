"""PDE-system integration: ImmuneResponse3D, solver, and trajectory helpers."""

from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tumortwin.models.immune_3d import ImmuneResponse3D
from tumortwin.models.pde_system import (
    extract_state_component,
    extract_trajectory_component,
)
from tumortwin.solvers.torch_solver import TorchDiffEqSolver, TorchDiffEqSolverOptions
from tumortwin.types.imaging import NibabelNifti


def _patient_with_brain_mask(mask_np: np.ndarray):
    return SimpleNamespace(brainmask_image=NibabelNifti.from_array(mask_np))


@pytest.fixture
def tiny_immune_model():
    shape = (8, 8, 8)
    mask = np.ones(shape, dtype=np.float32)
    patient = _patient_with_brain_mask(mask)
    initial_u1 = torch.rand(shape, dtype=torch.float32) * 0.3
    model = ImmuneResponse3D(
        D1=torch.tensor(0.01, dtype=torch.float32),
        mu1=torch.tensor(0.05, dtype=torch.float32),
        gamma12=torch.tensor(0.01, dtype=torch.float32),
        D4=torch.tensor(0.02, dtype=torch.float32),
        gamma21=torch.tensor(0.01, dtype=torch.float32),
        v=[0.0, 0.0, 0.0],
        patient_data=patient,
        initial_time=datetime(2020, 1, 1),
        initial_u1=initial_u1,
        radiotherapy_specification=None,
        chemotherapy_specifications=None,
        require_grad=False,
        device=torch.device("cpu"),
    )
    return model, shape


def test_immune_has_spatial_fd_and_forward_matches_state(tiny_immune_model):
    model, shape = tiny_immune_model
    assert hasattr(model, "spatial_fd")
    u0 = model.get_initial_state()
    assert u0.shape == (2,) + shape
    model.validate_state_shape(u0)
    t = torch.tensor(0.0)
    du = model.forward(t, u0)
    assert du.shape == u0.shape


def test_extract_state_and_trajectory_components():
    t, c, d, h, w = 4, 2, 5, 5, 5
    traj = torch.zeros(t, c, d, h, w)
    tum_series = extract_trajectory_component(traj, 0)
    assert tum_series.shape == (t, d, h, w)
    u_step = torch.zeros(c, d, h, w)
    assert extract_state_component(u_step, 0).shape == (d, h, w)
    assert extract_state_component(u_step, 1).shape == (d, h, w)
    single = torch.zeros(d, h, w)
    assert extract_state_component(single, 0).shape == (d, h, w)


def test_torch_solver_short_run_immune_pde(tiny_immune_model):
    model, shape = tiny_immune_model
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
    timepoints = [t0, t0 + timedelta(days=2.0)]
    t_tensor, u_traj = solver.solve(timepoints, model.get_initial_state())
    assert t_tensor.numel() == 2
    assert u_traj.shape[0] == 2
    assert u_traj.shape[1] == 2
    assert u_traj.shape[2:] == shape
    tum_only = extract_trajectory_component(u_traj, 0)
    assert tum_only.shape == (2,) + shape
    assert torch.all(torch.isfinite(tum_only))
