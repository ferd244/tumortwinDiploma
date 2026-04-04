"""
End-to-end helpers for coupled PDE systems (stacked state ``(C, D, H, W)``).

The HGG/TNBC tutorials use a single tumor field ``(D, H, W)`` everywhere: initial
condition, ``solver.solve`` output, and LM residuals. Multi-component models
(``ImmuneResponse3D``, other ``PDESystemModel3D`` subclasses) return
``(T, C, D, H, W)``. These functions bridge that gap so the same postprocessing
and calibration patterns apply to the **tumor component** (or any chosen channel).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import torch

from tumortwin.models.pde_system import extract_trajectory_component


def trajectory_component_timeseries(
    trajectory: torch.Tensor,
    component_idx: int = 0,
) -> torch.Tensor:
    """
    Drop the component axis from a solver trajectory, keeping one scalar field over time.

    Same as :func:`tumortwin.models.pde_system.extract_trajectory_component`; kept under
    this module so tutorials can import one namespace for the full PDE workflow.

    Args:
        trajectory: ``(T, D, H, W)`` or ``(T, C, D, H, W)`` from ``TorchDiffEqSolver.solve``.
        component_idx: Which PDE unknown (``0`` = tumor in ``ImmuneResponse3D``).

    Returns:
        Tensor of shape ``(T, D, H, W)``.
    """
    return extract_trajectory_component(trajectory, component_idx)


def trajectory_to_map_list(
    trajectory: torch.Tensor,
    component_idx: int = 0,
) -> List[torch.Tensor]:
    """
    Convert a trajectory to a **list** of per-time spatial maps ``(D, H, W)``.

    This matches what :func:`tumortwin.postprocessing.prediction_summary.plot_predicted_TCC`
    expects: ``List[torch.Tensor]`` with one ``(D, H, W)`` tensor per output time.

    Args:
        trajectory: Output ``u`` from the solver (``T`` leading dimension).
        component_idx: Component to extract when ``trajectory.ndim == 5``.
    """
    u = extract_trajectory_component(trajectory, component_idx)
    return [u[i].contiguous() for i in range(int(u.shape[0]))]


def initial_pde_state_from_tumor_field(
    tumor_u0: torch.Tensor,
    *,
    num_components: int = 2,
    other_fill: Union[float, torch.Tensor, None] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Build a stacked initial state ``(C, D, H, W)`` when imaging only constrains the tumor.

    Component ``0`` is ``tumor_u0``. Remaining components are filled with a scalar
    (e.g. blood lymphocyte level for ``ImmuneResponse3D``), a matching ``(D, H, W)``
    tensor, or zeros if ``other_fill is None``.

    Args:
        tumor_u0: Tumor field ``(D, H, W)``.
        num_components: Total number of scalar PDE unknowns (``C``).
        other_fill: Value(s) for components ``1 .. C-1``.
        device: Optional device for the result (defaults to ``tumor_u0.device``).
    """
    if tumor_u0.dim() != 3:
        raise ValueError(
            f"tumor_u0 must have shape (D, H, W); got {tuple(tumor_u0.shape)}"
        )
    if num_components < 1:
        raise ValueError("num_components must be >= 1")
    d, h, w = tumor_u0.shape
    dev = device if device is not None else tumor_u0.device
    tumor_u0 = tumor_u0.to(dev)
    if num_components == 1:
        return tumor_u0.unsqueeze(0)
    rest: List[torch.Tensor] = []
    for _ in range(1, num_components):
        if isinstance(other_fill, torch.Tensor):
            if tuple(other_fill.shape) != (d, h, w):
                raise ValueError(
                    "other_fill tensor must have shape (D, H, W) matching tumor_u0"
                )
            rest.append(other_fill.to(dev, dtype=tumor_u0.dtype))
        elif other_fill is not None:
            rest.append(
                torch.full(
                    (d, h, w),
                    float(other_fill),
                    device=dev,
                    dtype=tumor_u0.dtype,
                )
            )
        else:
            rest.append(torch.zeros((d, h, w), device=dev, dtype=tumor_u0.dtype))
    return torch.stack([tumor_u0, *rest], dim=0)


def select_timepoint_indices(
    integration_times_days: torch.Tensor,
    target_times_days: Sequence[float],
    *,
    atol: float = 1e-3,
) -> List[int]:
    """
    Map each target time (days) to the nearest row in ``integration_times_days``.

    Use after ``t, u = solver.solve(...)`` to pick states at visit times for calibration.

    Raises:
        ValueError: If the nearest neighbor is farther than ``atol`` from the target.
    """
    t = integration_times_days.detach().float().reshape(-1).cpu()
    out: List[int] = []
    for target in target_times_days:
        dist = torch.abs(t - float(target))
        i = int(torch.argmin(dist).item())
        if float(dist[i]) > atol:
            raise ValueError(
                f"No integration time within atol={atol} of {target} days "
                f"(closest is {float(t[i])} days)."
            )
        out.append(i)
    return out


def fields_at_times_from_trajectory(
    trajectory: torch.Tensor,
    time_indices: Sequence[int],
    component_idx: int = 0,
) -> List[torch.Tensor]:
    """
    Extract a list of spatial maps at selected time indices (tumor or any component).

    Args:
        trajectory: Solver output ``u``, shape ``(T, ...)``.
        time_indices: Indices along the time dimension.
        component_idx: Component index when ``trajectory.ndim == 5``.
    """
    u = extract_trajectory_component(trajectory, component_idx)
    return [u[int(i)].contiguous() for i in time_indices]


def spatiotemporal_residual_vector(
    predicted_maps: Sequence[torch.Tensor],
    measured_maps: Sequence[torch.Tensor],
    *,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Flatten ``pred - meas`` across visits for least-squares calibration (LM / scipy).

    Aligns with the HGG demo objective: sum of squared differences over voxels at
    each calibration visit. Supply **tumor-only** maps of identical shape.

    Args:
        predicted_maps: One tensor per visit, each ``(D, H, W)``.
        measured_maps: Observed maps, same length and shapes as ``predicted_maps``.
        mask: Optional ``(D, H, W)`` boolean or 0/1 mask; only voxels where ``mask`` is
            True/nonzero are included in the residual.
    """
    if len(predicted_maps) != len(measured_maps):
        raise ValueError(
            f"Length mismatch: {len(predicted_maps)} predicted vs {len(measured_maps)} measured"
        )
    parts: List[torch.Tensor] = []
    for p_raw, m_raw in zip(predicted_maps, measured_maps):
        p = p_raw if isinstance(p_raw, torch.Tensor) else torch.as_tensor(p_raw)
        m = m_raw if isinstance(m_raw, torch.Tensor) else torch.as_tensor(m_raw)
        p = p.float()
        m = m.float()
        if p.shape != m.shape:
            raise ValueError(f"Shape mismatch: pred {p.shape} vs meas {m.shape}")
        diff = (p - m).reshape(-1)
        if mask is not None:
            mv = mask
            if not isinstance(mv, torch.Tensor):
                mv = torch.as_tensor(mv)
            mv = mv.reshape(-1).bool().to(diff.device)
            diff = diff[mv]
        parts.append(diff)
    if not parts:
        return torch.zeros(0)
    return torch.cat(parts, dim=0)


def squared_error_loss(residual_vector: torch.Tensor) -> torch.Tensor:
    """Scalar ``r @ r`` for monitoring calibration (matches LM internal metric style)."""
    r = residual_vector.reshape(-1).float()
    return r @ r
