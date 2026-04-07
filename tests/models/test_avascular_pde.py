"""Avascular multi-field tumor model (n, m, h, s) + ψ Poisson helper."""

from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tumortwin.models.avascular_tumor_3d import AvascularTumorGrowth3D
from tumortwin.models.pde_system import extract_trajectory_component
from tumortwin.solvers.torch_solver import TorchDiffEqSolver, TorchDiffEqSolverOptions
from tumortwin.types.imaging import NibabelNifti


def _patient_with_brain_mask(mask_np: np.ndarray):
    return SimpleNamespace(brainmask_image=NibabelNifti.from_array(mask_np))


@pytest.fixture
def tiny_avascular_model():
    shape = (8, 8, 8)
    mask = np.ones(shape, dtype=np.float32)
    patient = _patient_with_brain_mask(mask)
    initial_n = torch.rand(shape, dtype=torch.float32) * 0.2
    model = AvascularTumorGrowth3D(
        B=torch.tensor(0.05, dtype=torch.float32),
        L=torch.tensor(0.02, dtype=torch.float32),
        Dn=torch.tensor(0.01, dtype=torch.float32),
        Ds=torch.tensor(0.05, dtype=torch.float32),
        mu=torch.tensor(0.01, dtype=torch.float32),
        q_s=torch.tensor(0.5, dtype=torch.float32),
        s_0=torch.tensor(0.1, dtype=torch.float32),
        s_x=torch.tensor(0.5, dtype=torch.float32),
        s_K=torch.tensor(0.5, dtype=torch.float32),
        g_0=torch.tensor(1.0, dtype=torch.float32),
        patient_data=patient,
        initial_n=initial_n,
        poisson_iterations=16,
        require_grad=False,
        device=torch.device("cpu"),
    )
    return model, shape


def test_avascular_forward_shape_and_finite(tiny_avascular_model):
    model, shape = tiny_avascular_model
    u0 = model.get_initial_state()
    assert u0.shape == (4,) + shape
    model.validate_state_shape(u0)
    du = model.forward(torch.tensor(0.0), u0)
    assert du.shape == u0.shape
    assert torch.all(torch.isfinite(du))


def test_avascular_fd_divergence_runs(tiny_avascular_model):
    model, shape = tiny_avascular_model
    x = torch.randn(shape, dtype=torch.float32, device=model.device)
    g = model.spatial_fd.gradient(x)
    div = model.spatial_fd.divergence(g)
    assert div.shape == shape
    assert torch.isfinite(div).all()


def test_torch_solver_short_run_avascular(tiny_avascular_model):
    model, shape = tiny_avascular_model
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
    assert u_traj.shape == (2, 4) + shape
    n_series = extract_trajectory_component(u_traj, 0)
    assert torch.all(torch.isfinite(n_series))
