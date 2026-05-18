"""
Adam-based refinement for :class:`~tumortwin.models.hemo_invasion_3d.HemoInvasion3D`
total cellularity (``n + m``) with optional radiotherapy ``alpha`` in the graph.

Typical use: warm-start from LM, fix diffusion ``Dn``, and fine-tune ``B``, ``K``, and
linear–radiosensitivity ``alpha`` with gradient clipping and box constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Sequence, Tuple

import torch

from tumortwin.models.hemo_invasion_3d import HemoInvasion3D
from tumortwin.models.pde_system import extract_trajectory_component

__all__ = ["AdamHemoInvasionRecord", "adam_refine_hemo_total_cellularity"]


@dataclass
class AdamHemoInvasionRecord:
    """One optimization step (loss and scalars for logging)."""

    step: int
    loss: float
    B: float
    K: float
    alpha_rt: float


def adam_refine_hemo_total_cellularity(
    model: HemoInvasion3D,
    solver,
    y_target: torch.Tensor,
    timepoints: Sequence[datetime],
    *,
    num_steps: int = 50,
    lr: float = 1e-3,
    grad_clip_max_norm: float = 1.0,
    b_bounds: Tuple[float, float] = (3.5, 4.5),
    k_bounds: Tuple[float, float] = (0.8, 1.5),
    alpha_bounds: Tuple[float, float] = (0.03, 0.09),
    initial_B: Optional[float] = None,
    initial_K: Optional[float] = None,
    initial_alpha_rt: Optional[float] = None,
    fix_Dn: bool = True,
    log_every: int = 5,
    verbose: bool = True,
) -> List[AdamHemoInvasionRecord]:
    """
    Minimize sum of squared errors between predicted total cellularity ``clamp(n+m)``
    and ``y_target`` at ``timepoints``.

    Freezes all :class:`torch.nn.Parameter` modules on ``model`` except ``B`` and ``K``.
    If ``model.radiotherapy_specification`` is set, ``alpha`` is promoted to a scalar
    tensor on the same device/dtype as ``B`` and included in the optimization; otherwise
    ``initial_alpha_rt`` is ignored.

    Args:
        model: Hemodynamic invasion model.
        solver: Object with ``solve(timepoints=..., u_initial=...) -> (_, trajectory)``.
        y_target: Tensor shaped like the predicted maps (e.g. ``(T, D, H, W)``).
        timepoints: Wall-clock times, same length as leading dim of ``y_target``.
        num_steps: Adam iterations.
        lr: Learning rate.
        grad_clip_max_norm: Global L2 norm clip for ``B``, ``K``, and ``alpha``.
        b_bounds / k_bounds / alpha_bounds: Box constraints applied after each step.
        initial_B / initial_K / initial_alpha_rt: Warm-start values; ``None`` keeps
            the tensor values already in the model.
        fix_Dn: If True, ``Dn.requires_grad_(False)`` (in addition to other frozen params).
        log_every: Print every this many steps if ``verbose`` (also step 0).
        verbose: Print progress lines.

    Returns:
        List of :class:`AdamHemoInvasionRecord`, one per iteration.
    """
    for p in model.parameters():
        p.requires_grad_(False)

    model.B.requires_grad_(True)
    model.K.requires_grad_(True)
    if fix_Dn:
        model.Dn.requires_grad_(False)

    rt = model.radiotherapy_specification
    alpha_t: Optional[torch.Tensor] = None
    if rt is not None:
        a = rt.alpha
        if isinstance(a, torch.Tensor):
            alpha_t = a.detach().to(device=model.B.device, dtype=model.B.dtype).reshape(())
        else:
            alpha_t = torch.tensor(
                float(a), device=model.B.device, dtype=model.B.dtype
            )
        alpha_t.requires_grad_(True)
        rt.alpha = alpha_t

    params = [model.B, model.K]
    if alpha_t is not None:
        params.append(alpha_t)

    with torch.no_grad():
        if initial_B is not None:
            model.B.data.fill_(float(initial_B))
        if initial_K is not None:
            model.K.data.fill_(float(initial_K))
        if alpha_t is not None and initial_alpha_rt is not None:
            alpha_t.fill_(float(initial_alpha_rt))

    optimizer = torch.optim.Adam(params, lr=lr)
    history: List[AdamHemoInvasionRecord] = []

    for step in range(num_steps):
        optimizer.zero_grad(set_to_none=True)
        _, traj = solver.solve(
            timepoints=list(timepoints),
            u_initial=model.get_initial_state(),
        )
        n = extract_trajectory_component(traj, 0)
        m = extract_trajectory_component(traj, 1)
        pred = torch.clamp(n + m, 0.0, 1.0)

        loss = ((pred - y_target) ** 2).sum()
        loss_val = float(loss.detach())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip_max_norm)
        optimizer.step()

        with torch.no_grad():
            model.B.clamp_(b_bounds[0], b_bounds[1])
            model.K.clamp_(k_bounds[0], k_bounds[1])
            if alpha_t is not None:
                alpha_t.clamp_(alpha_bounds[0], alpha_bounds[1])
            if rt is not None and alpha_t is not None:
                rt.alpha = alpha_t

        alpha_scalar = (
            float(alpha_t.detach().item())
            if alpha_t is not None
            else float("nan")
        )
        history.append(
            AdamHemoInvasionRecord(
                step=step,
                loss=loss_val,
                B=float(model.B.detach().item()),
                K=float(model.K.detach().item()),
                alpha_rt=alpha_scalar,
            )
        )
        if verbose and log_every > 0 and step % log_every == 0:
            msg = (
                f"step {step:03d}: SSE={loss_val:.4e} | "
                f"B={model.B.item():.4f} K={model.K.item():.4f}"
            )
            if alpha_t is not None:
                msg += f" aRT={alpha_t.item():.5f}"
            print(msg)

    return history
