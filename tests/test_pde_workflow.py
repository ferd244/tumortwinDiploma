import torch

from tumortwin.pde_workflow import (
    fields_at_times_from_trajectory,
    initial_pde_state_from_tumor_field,
    select_timepoint_indices,
    spatiotemporal_residual_vector,
    squared_error_loss,
    trajectory_to_map_list,
    trajectory_component_timeseries,
)


def test_trajectory_to_map_list_and_component():
    t, c, d, h, w = 5, 2, 4, 4, 4
    traj = torch.arange(t * c * d * h * w, dtype=torch.float32).reshape(t, c, d, h, w)
    lst = trajectory_to_map_list(traj, 0)
    assert len(lst) == t
    assert lst[0].shape == (d, h, w)
    ts = trajectory_component_timeseries(traj, 0)
    assert ts.shape == (t, d, h, w)


def test_initial_pde_state_from_tumor_field():
    tumor = torch.ones(3, 4, 5) * 0.2
    u = initial_pde_state_from_tumor_field(tumor, num_components=2, other_fill=0.5)
    assert u.shape == (2, 3, 4, 5)
    assert torch.allclose(u[0], tumor)
    assert torch.allclose(u[1], torch.full_like(tumor, 0.5))


def test_select_timepoint_indices():
    t = torch.tensor([0.0, 1.0, 2.0, 3.0])
    idx = select_timepoint_indices(t, [0.0, 2.0], atol=0.01)
    assert idx == [0, 2]


def test_fields_at_times_from_trajectory():
    traj = torch.zeros(4, 2, 3, 3, 3)
    traj[:, 0] = 1.0
    traj[:, 1] = 2.0
    fs = fields_at_times_from_trajectory(traj, [1, 3], component_idx=1)
    assert len(fs) == 2
    assert torch.all(fs[0] == 2.0)


def test_spatiotemporal_residual_and_loss():
    p = [torch.ones(2, 2, 2), torch.zeros(2, 2, 2)]
    m = [torch.ones(2, 2, 2), torch.ones(2, 2, 2)]
    r = spatiotemporal_residual_vector(p, m)
    assert r.numel() == 2 * 2 * 2 * 2
    assert squared_error_loss(r).item() == float((r * r).sum())
