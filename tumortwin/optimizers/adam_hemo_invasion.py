"""
Adam-based refinement for :class:`~tumortwin.models.hemo_invasion_3d.HemoInvasion3D`
total cellularity (``n + m``) with optional radiotherapy ``alpha`` in the graph.

Typical use: warm-start from LM, fix diffusion ``Dn``, and fine-tune ``B``, ``K``, and
linear–radiosensitivity ``alpha`` with gradient clipping and box constraints.

Use ``calibrate_params`` to optimize any subset of scalar model parameters (see
``HEMO_INVASION_ADAM_PARAM_NAMES``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import torch

from tumortwin.models.hemo_invasion_3d import HemoInvasion3D
from tumortwin.models.pde_system import extract_trajectory_component

__all__ = [
    "AdamHemoInvasionRecord",
    "HEMO_INVASION_ADAM_PARAM_NAMES",
    "adam_refine_hemo_total_cellularity",
]

# Scalar ``nn.Parameter`` names on :class:`HemoInvasion3D` plus ``alpha_rt`` from RT spec.
HEMO_INVASION_ADAM_PARAM_NAMES: FrozenSet[str] = frozenset(
    {"B", "Dn", "Ds", "k_s", "s_star", "K", "s_crit", "s_smooth", "alpha_rt"}
)


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


def _param_tensor(model: HemoInvasion3D, name: str) -> torch.Tensor:
    if name == "alpha_rt":
        raise KeyError("alpha_rt is stored on radiotherapy_specification")
    p = getattr(model, name, None)
    if p is None or not isinstance(p, torch.nn.Parameter):
        raise KeyError(f"{name!r} is not an nn.Parameter on HemoInvasion3D")
    return p


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
            out[n] = float(_param_tensor(model, n).detach().item())
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
    log_every: int = 5,
    verbose: bool = True,
) -> List[AdamHemoInvasionRecord]:
    """
    Minimize sum of squared errors between predicted total cellularity ``clamp(n+m)``
    and ``y_target`` at ``timepoints``.

    Default (``calibrate_params is None``): train ``B`` and ``K``; if radiotherapy is set,
    also train ``alpha`` as a scalar leaf on ``radiotherapy_specification`` (same as
    before). ``fix_Dn`` keeps ``Dn`` frozen in this mode only.

    Custom selection: pass ``calibrate_params`` as any non-empty subset of
    ``HEMO_INVASION_ADAM_PARAM_NAMES`` (e.g. ``("B", "Dn", "k_s", "s_crit")``).
    ``fix_Dn`` is ignored unless it conflicts with listing ``Dn`` (then an error is
    raised). Optional ``param_bounds`` overrides box constraints per name; for ``B``,
    ``K``, and ``alpha_rt`` the legacy ``*_bounds`` arguments still apply when a name
    is omitted from ``param_bounds``.

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
        log_every: Print every this many steps if ``verbose`` (also step 0).
        verbose: Print progress lines.

    Returns:
        List of :class:`AdamHemoInvasionRecord`, one per iteration.
    """
    rt = model.radiotherapy_specification
    train_names = _normalize_train_names(
        calibrate_params, rt, fix_Dn=fix_Dn
    )

    for p in model.parameters():
        p.requires_grad_(False)

    alpha_t: Optional[torch.Tensor] = None
    if rt is not None:
        a = rt.alpha
        if isinstance(a, torch.Tensor):
            alpha_t = a.detach().to(device=model.B.device, dtype=model.B.dtype).reshape(())
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
        _param_tensor(model, name).requires_grad_(True)

    if calibrate_params is None and fix_Dn:
        model.Dn.requires_grad_(False)

    params: List[torch.Tensor] = []
    for name in train_names:
        if name == "alpha_rt":
            if alpha_t is not None:
                params.append(alpha_t)
        else:
            params.append(_param_tensor(model, name))

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
                _param_tensor(model, key).fill_(float(val))

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
                B=float(model.B.detach().item()),
                K=float(model.K.detach().item()),
                alpha_rt=alpha_scalar,
                snapshot=snap,
            )
        )
        if verbose and log_every > 0 and step % log_every == 0:
            parts = [f"{k}={snap[k]:.5g}" for k in sorted(snap.keys())]
            msg = f"step {step:03d}: SSE={loss_val:.4e} | " + " ".join(parts)
            print(msg)

    return history
