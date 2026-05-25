"""
Adam-based refinement for :class:`~tumortwin.models.hemo_invasion_3d.HemoInvasion3D`
total cellularity (``n + m``) with optional radiotherapy ``alpha`` in the graph.

Typical use: warm-start from LM, fix diffusion ``Dn``, and fine-tune ``B``, ``K``, and
linear–radiosensitivity ``alpha`` with gradient clipping and box constraints.

The objective defaults to pure SSE on total cellularity; optional soft Dice and
per-timestep volume MSE can be mixed in via ``loss_w_dice`` / ``loss_w_vol``.

Use ``calibrate_params`` to optimize any subset of scalar model parameters (see
``HEMO_INVASION_ADAM_PARAM_NAMES``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.parametrize import (
    is_parametrized,
    register_parametrization,
    remove_parametrizations,
)

from tumortwin.models.hemo_invasion_3d import HemoInvasion3D
from tumortwin.models.pde_system import extract_trajectory_component

__all__ = [
    "AdamHemoInvasionRecord",
    "HEMO_INVASION_ADAM_PARAM_NAMES",
    "HEMO_INVASION_POSITIVE_SCALAR_NAMES",
    "adam_refine_hemo_total_cellularity",
]

# Scalar ``nn.Parameter`` names on :class:`HemoInvasion3D` plus ``alpha_rt`` from RT spec.
HEMO_INVASION_ADAM_PARAM_NAMES: FrozenSet[str] = frozenset(
    {"B", "Dn", "Ds", "k_s", "s_star", "K", "s_crit", "s_smooth", "alpha_rt"}
)

# Strictly-positive scalars under :class:`HemoInvasion3D` suitable for θ = exp(u) calibration.
HEMO_INVASION_POSITIVE_SCALAR_NAMES: FrozenSet[str] = frozenset(
    {"B", "Dn", "Ds", "k_s", "s_star", "K", "s_crit", "s_smooth"}
)


def _soft_dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.05,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    Differentiable soft Dice versus a binary mask derived from ``target``.

    ``pred`` — typically ``clamp(n + m, 0, 1)``; gradients flow only through ``pred``.
    ``target`` is thresholded without gradients (constants are fine).
    """
    tgt_mask = (target > threshold).to(dtype=pred.dtype)
    intersection = (pred * tgt_mask).sum()
    denom = pred.sum() + tgt_mask.sum()
    return 1.0 - (2.0 * intersection + smooth) / (denom + smooth)


def _soft_volume_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.05,
) -> torch.Tensor:
    """MSE of predicted spatial sums vs voxel counts in the masked target, per timestep."""
    pred_vols = pred.sum(dim=(-3, -2, -1))
    tgt_vols = (target > threshold).to(dtype=pred.dtype).sum(dim=(-3, -2, -1))
    return torch.nn.functional.mse_loss(pred_vols, tgt_vols)


def _combined_hemo_loss(
    pred: torch.Tensor,
    y_target: torch.Tensor,
    *,
    w_sse: float = 1.0,
    w_dice: float = 0.0,
    w_vol: float = 0.0,
    mask_threshold: float = 0.05,
) -> torch.Tensor:
    """L = w_sse * SSE + w_dice * SoftDice + w_vol * VolumeMSE (whole calibration stack)."""
    pc = torch.clamp(pred, 0.0, 1.0)
    loss = w_sse * ((pc - y_target) ** 2).sum()
    if w_dice != 0.0:
        loss = loss + w_dice * _soft_dice_loss(
            pc, y_target, threshold=mask_threshold
        )
    if w_vol != 0.0:
        loss = loss + w_vol * _soft_volume_loss(
            pc, y_target, threshold=mask_threshold
        )
    return loss


class _PositiveExpScalarParam(nn.Module):
    """Maps unconstrained u to exp(u); exposes u to the optimizer."""

    eps: float = 1e-12

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return torch.exp(u)

    def right_inverse(self, y: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.clamp(y, min=self.eps))


@dataclass
class AdamHemoInvasionRecord:
    """One optimization step (loss and scalars for logging)."""

    step: int
    loss: float
    B: float
    K: float
    alpha_rt: float
    #: Values of all parameters that were marked trainable this run (name → scalar).
    snapshot: Dict[str, float] = field(default_factory=dict)


def _param_tensor(model: HemoInvasion3D, name: str) -> torch.nn.Parameter:
    if name == "alpha_rt":
        raise KeyError("alpha_rt is stored on radiotherapy_specification")
    p = getattr(model, name, None)
    # After register_parametrization, getattr returns a Tensor forward of exp(original).
    if p is None or not isinstance(p, torch.nn.Parameter):
        raise KeyError(
            f"{name!r} is not a plain nn.Parameter on HemoInvasion3D "
            f"(possibly parametrized: use underlying leaf accessors)."
        )
    return p


def _underlying_train_leaf(
    model: HemoInvasion3D, name: str
) -> torch.nn.Parameter:
    """Optimizer leaf tensor (log-space unconstrained scalar if parametrized)."""
    if is_parametrized(model, name):
        leaf = model.parametrizations[name].original  # type: ignore[index]
        if not isinstance(leaf, torch.nn.Parameter):
            raise TypeError(f"parametrizations[{name!r}].original is not a Parameter")
        return leaf
    return _param_tensor(model, name)


def _assign_trainable_physical(
    model: HemoInvasion3D, name: str, val: float
) -> None:
    """Set a trained scalar parameter to ``val`` in physical units (handles log-space)."""
    eps = float(_PositiveExpScalarParam.eps)
    leaf = _underlying_train_leaf(model, name)
    phys = torch.tensor(
        float(val), device=leaf.device, dtype=torch.float64
    ).clamp_min(eps)
    with torch.no_grad():
        if is_parametrized(model, name):
            leaf.copy_(phys.log().to(dtype=leaf.dtype))
        else:
            leaf.copy_(phys.to(dtype=leaf.dtype))


def _register_positive_log_space(model: HemoInvasion3D, name: str) -> None:
    if name not in HEMO_INVASION_POSITIVE_SCALAR_NAMES:
        raise ValueError(f"Cannot use log-space for {name!r}.")
    if is_parametrized(model, name):
        raise ValueError(
            f"{name!r} is already parametrized; remove parametrizations first."
        )
    register_parametrization(model, name, _PositiveExpScalarParam(), unsafe=False)


def _snapshot_trainable(
    model: HemoInvasion3D,
    names: Sequence[str],
    alpha_t: Optional[torch.Tensor],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for n in names:
        if n == "alpha_rt":
            if alpha_t is not None:
                out[n] = float(alpha_t.detach().item())
        else:
            # Works for plain Parameters and parametrized exp(·) views.
            out[n] = float(getattr(model, n).detach().reshape(()).cpu().item())
    return out


def _resolve_bounds(
    name: str,
    *,
    param_bounds: Optional[Dict[str, Tuple[float, float]]],
    b_bounds: Tuple[float, float],
    k_bounds: Tuple[float, float],
    alpha_bounds: Tuple[float, float],
    rt_active: bool,
) -> Optional[Tuple[float, float]]:
    if param_bounds is not None and name in param_bounds:
        return param_bounds[name]
    if name == "B":
        return b_bounds
    if name == "K":
        return k_bounds
    if name == "alpha_rt":
        return alpha_bounds if rt_active else None
    return None


def _clamp_trainable(
    model: HemoInvasion3D,
    names: Sequence[str],
    alpha_t: Optional[torch.Tensor],
    rt,
    *,
    param_bounds: Optional[Dict[str, Tuple[float, float]]],
    b_bounds: Tuple[float, float],
    k_bounds: Tuple[float, float],
    alpha_bounds: Tuple[float, float],
) -> None:
    eps = float(_PositiveExpScalarParam.eps)
    rt_active = rt is not None and alpha_t is not None
    for name in names:
        bounds = _resolve_bounds(
            name,
            param_bounds=param_bounds,
            b_bounds=b_bounds,
            k_bounds=k_bounds,
            alpha_bounds=alpha_bounds,
            rt_active=rt_active,
        )
        if bounds is None:
            continue
        lo, hi = bounds
        if name == "alpha_rt":
            if alpha_t is not None:
                alpha_t.clamp_(lo, hi)
                rt.alpha = alpha_t
        elif is_parametrized(model, name):
            phys = getattr(model, name).detach().reshape(())
            clamped = phys.clamp(float(lo), float(hi)).clamp_min(eps)
            leaf = _underlying_train_leaf(model, name)
            with torch.no_grad():
                leaf.copy_(torch.log(clamped.to(device=leaf.device, dtype=leaf.dtype)))
        else:
            _param_tensor(model, name).clamp_(lo, hi)


def _normalize_train_names(
    calibrate_params: Optional[Sequence[str]],
    rt,
    *,
    fix_Dn: bool,
) -> List[str]:
    """Resolve ordered unique trainable parameter names."""
    if calibrate_params is None:
        names = ["B", "K"]
        if rt is not None:
            names.append("alpha_rt")
        return names

    seen = set()
    out: List[str] = []
    for raw in calibrate_params:
        n = str(raw).strip()
        if n not in HEMO_INVASION_ADAM_PARAM_NAMES:
            allowed = ", ".join(sorted(HEMO_INVASION_ADAM_PARAM_NAMES))
            raise ValueError(
                f"Unknown calibrate_params entry {raw!r}. "
                f"Allowed names: {allowed}"
            )
        if n in seen:
            continue
        seen.add(n)
        out.append(n)
    if not out:
        raise ValueError("calibrate_params must name at least one parameter.")
    if "alpha_rt" in out and rt is None:
        raise ValueError(
            "alpha_rt was requested but model.radiotherapy_specification is None."
        )
    if fix_Dn and "Dn" in out:
        raise ValueError(
            "Dn is listed in calibrate_params but fix_Dn=True; "
            "set fix_Dn=False when calibrating Dn."
        )
    return out


def adam_refine_hemo_total_cellularity(
    model: HemoInvasion3D,
    solver,
    y_target: torch.Tensor,
    timepoints: Sequence[datetime],
    *,
    calibrate_params: Optional[Sequence[str]] = None,
    param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    initial_values: Optional[Dict[str, float]] = None,
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
    calibrate_log_space: bool = False,
    adam_amsgrad: bool = False,
    cosine_scheduler_T_max: Optional[int] = None,
    cosine_scheduler_eta_min: float = 1e-3,
    log_every: int = 5,
    verbose: bool = True,
    loss_w_sse: float = 1.0,
    loss_w_dice: float = 0.0,
    loss_w_vol: float = 0.0,
    loss_mask_threshold: float = 0.05,
) -> List[AdamHemoInvasionRecord]:
    """
    Minimize a weighted loss between predicted total cellularity ``clamp(n+m)``
    and ``y_target`` at ``timepoints``: SSE plus optional soft Dice / volume penalties.

    Default (``calibrate_params is None``): train ``B`` and ``K``; if radiotherapy is set,
    also train ``alpha`` as a scalar leaf on ``radiotherapy_specification`` (same as
    before). ``fix_Dn`` keeps ``Dn`` frozen in this mode only.

    Custom selection: pass ``calibrate_params`` as any non-empty subset of
    ``HEMO_INVASION_ADAM_PARAM_NAMES`` (e.g. ``("B", "Dn", "k_s", "s_crit")``).
    ``fix_Dn`` is ignored unless it conflicts with listing ``Dn`` (then an error is
    raised). Optional ``param_bounds`` overrides box constraints per name; for ``B``,
    ``K``, and ``alpha_rt`` the legacy ``*_bounds`` arguments still apply when a name
    is omitted from ``param_bounds``.

    If ``calibrate_log_space=True``, each calibrated positive scalar in
    ``HEMO_INVASION_POSITIVE_SCALAR_NAMES`` is internally re-parameterized as
    ``physical = exp(u)``. Adam updates the unconstrained ``u`` while box constraints and
    ``param_bounds`` stay in physical units (the same semantics as plain calibration).
    ``alpha_rt`` is always optimized in linear space. Exp parametrizations are removed
    when this function returns so the model exposes plain ``nn.Parameter`` fields again.

    Args:
        model: Hemodynamic invasion model.
        solver: Object with ``solve(timepoints=..., u_initial=...) -> (_, trajectory)``.
        y_target: Tensor shaped like the predicted maps (e.g. ``(T, D, H, W)``).
        timepoints: Wall-clock times, same length as leading dim of ``y_target``.
        calibrate_params: Optional explicit list of scalar parameters to train.
        param_bounds: Optional per-name ``(low, high)`` clamps after each Adam step.
        initial_values: Optional warm-start scalars for trained parameters (by name).
        num_steps: Adam iterations.
        lr: Learning rate.
        grad_clip_max_norm: Global L2 norm clip for optimized tensors.
        b_bounds / k_bounds / alpha_bounds: Default box constraints for those names.
        initial_B / initial_K / initial_alpha_rt: Legacy warm-start (merged into
            ``initial_values`` when the corresponding parameter is trained).
        fix_Dn: Legacy: when ``calibrate_params is None``, freeze ``Dn``.
        calibrate_log_space: Reparametrize positive scalars as ``exp(u)`` for Adam on ``u``.
        adam_amsgrad: Pass-through to ``torch.optim.Adam(..., amsgrad=...)``.
        cosine_scheduler_T_max: If set, wraps Adam in
            :class:`~torch.optim.lr_scheduler.CosineAnnealingLR` with this ``T_max``.
        cosine_scheduler_eta_min: Minimum LR for cosine annealing.
        log_every: Print every this many steps if ``verbose`` (also step 0).
        verbose: Print progress lines.
        loss_w_sse: Weight on summed squared error (same as legacy pure SSE when dice/vol are 0).
        loss_w_dice: Weight on :func:`_soft_dice_loss` (overlap with mask from ``y_target``).
        loss_w_vol: Weight on :func:`_soft_volume_loss` (per-frame volume MSE).
        loss_mask_threshold: Threshold applied to ``y_target`` for Dice / volume terms.

    Returns:
        List of :class:`AdamHemoInvasionRecord`, one per iteration.
    """
    rt = model.radiotherapy_specification
    train_names = _normalize_train_names(
        calibrate_params, rt, fix_Dn=fix_Dn
    )

    log_space_registered: List[str] = []
    try:
        if calibrate_log_space:
            for name in train_names:
                if name in HEMO_INVASION_POSITIVE_SCALAR_NAMES:
                    _register_positive_log_space(model, name)
                    log_space_registered.append(name)

        for p in model.parameters():
            p.requires_grad_(False)

        alpha_t: Optional[torch.Tensor] = None
        if rt is not None:
            a = rt.alpha
            if isinstance(a, torch.Tensor):
                alpha_t = a.detach().to(
                    device=model.B.device, dtype=model.B.dtype
                ).reshape(())
            else:
                alpha_t = torch.tensor(
                    float(a), device=model.B.device, dtype=model.B.dtype
                )
            if "alpha_rt" in train_names:
                alpha_t.requires_grad_(True)
            else:
                alpha_t.requires_grad_(False)
            rt.alpha = alpha_t

        for name in train_names:
            if name == "alpha_rt":
                continue
            _underlying_train_leaf(model, name).requires_grad_(True)

        if calibrate_params is None and fix_Dn:
            _underlying_train_leaf(model, "Dn").requires_grad_(False)

        params: List[torch.Tensor] = []
        for name in train_names:
            if name == "alpha_rt":
                if alpha_t is not None:
                    params.append(alpha_t)
            else:
                params.append(_underlying_train_leaf(model, name))

        init_map: Dict[str, float] = dict(initial_values or {})
        if initial_B is not None:
            init_map.setdefault("B", float(initial_B))
        if initial_K is not None:
            init_map.setdefault("K", float(initial_K))
        if initial_alpha_rt is not None:
            init_map.setdefault("alpha_rt", float(initial_alpha_rt))

        with torch.no_grad():
            for key, val in init_map.items():
                if key not in train_names:
                    continue
                if key == "alpha_rt":
                    if alpha_t is not None:
                        alpha_t.fill_(float(val))
                        if rt is not None:
                            rt.alpha = alpha_t
                else:
                    _assign_trainable_physical(model, key, float(val))

        optimizer = torch.optim.Adam(params, lr=lr, amsgrad=adam_amsgrad)
        scheduler = None
        if cosine_scheduler_T_max is not None:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(cosine_scheduler_T_max),
                eta_min=float(cosine_scheduler_eta_min),
            )
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

            loss = _combined_hemo_loss(
                pred,
                y_target,
                w_sse=loss_w_sse,
                w_dice=loss_w_dice,
                w_vol=loss_w_vol,
                mask_threshold=loss_mask_threshold,
            )
            loss_val = float(loss.detach())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip_max_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            with torch.no_grad():
                _clamp_trainable(
                    model,
                    train_names,
                    alpha_t,
                    rt,
                    param_bounds=param_bounds,
                    b_bounds=b_bounds,
                    k_bounds=k_bounds,
                    alpha_bounds=alpha_bounds,
                )

            alpha_scalar = (
                float(alpha_t.detach().item())
                if alpha_t is not None
                else float("nan")
            )
            snap = _snapshot_trainable(model, train_names, alpha_t)
            history.append(
                AdamHemoInvasionRecord(
                    step=step,
                    loss=loss_val,
                    B=float(model.B.detach().reshape(()).cpu().item()),
                    K=float(model.K.detach().reshape(()).cpu().item()),
                    alpha_rt=alpha_scalar,
                    snapshot=snap,
                )
            )
            if verbose and log_every > 0 and step % log_every == 0:
                parts = [f"{k}={snap[k]:.5g}" for k in sorted(snap.keys())]
                msg = f"step {step:03d}: loss={loss_val:.4e} | " + " ".join(parts)
                print(msg)

        return history
    finally:
        for name in reversed(log_space_registered):
            if is_parametrized(model, name):
                remove_parametrizations(
                    model, name, leave_parametrized=False
                )
