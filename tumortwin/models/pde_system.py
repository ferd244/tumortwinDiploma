"""
Building blocks for coupled PDE systems on 3D grids (torchdiffeq state tensors).

Convention (no batch):
    ``u`` shape ``(C, D, H, W)`` — ``C`` coupled components, then spatial axes.

With batch (optional):
    ``(B, C, D, H, W)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Tuple

import torch

from tumortwin.models.base import TumorGrowthModel3D


@dataclass(frozen=True)
class PDEStateLayout:
    """
    Describes how component fields are packed into the ODE state tensor.

    Attributes:
        num_components: Number of scalar PDE unknowns stacked along ``component_dim``.
        component_dim: Index of the component axis in the **non-batched** layout.
        spatial_ndim: Number of spatial axes (3 for volumetric grids).
    """

    num_components: int
    component_dim: int = 0
    spatial_ndim: int = 3

    def state_ndim(self, batched: bool) -> int:
        return (1 if batched else 0) + 1 + self.spatial_ndim

    def component_axis(self, batched: bool) -> int:
        return (1 if batched else 0) + self.component_dim


def stack_pde_components(
    *components: torch.Tensor, dim: int = 0
) -> torch.Tensor:
    """Stack scalar fields ``(*spatial)`` into ``(C, *spatial)`` (default ``dim=0``)."""
    return torch.stack(components, dim=dim)


def unbind_components(u: torch.Tensor, component_axis: int = 0) -> Tuple[torch.Tensor, ...]:
    """Split ``(C, *spatial)`` into a tuple of ``C`` tensors."""
    return torch.unbind(u, dim=component_axis)


def expand_mask_for_components(
    mask_spatial: torch.Tensor, num_components: int, component_axis: int = 0
) -> torch.Tensor:
    """
    Broadcast a spatial mask to all components.

    Args:
        mask_spatial: ``(*spatial)`` boolean or 0/1 mask.
        num_components: ``C``.
        component_axis: Where the component axis will sit in the state tensor.

    Returns:
        View/expanded tensor that broadcasts against ``(C, *spatial)`` when
        ``component_axis == 0``: shape ``(C, *spatial)``.
    """
    if mask_spatial.dim() != 3:
        return mask_spatial
    if component_axis != 0:
        raise NotImplementedError("Only component_axis=0 is implemented for 3D spatial masks.")
    shape = (num_components,) + mask_spatial.shape
    return mask_spatial.unsqueeze(0).expand(shape)


def apply_spatial_mask_to_state(
    u: torch.Tensor,
    mask_spatial: torch.Tensor,
) -> torch.Tensor:
    """Broadcast a ``(D,H,W)`` mask onto ``u`` shaped ``(C,D,H,W)`` or ``(B,C,D,H,W)``."""
    m = mask_spatial.to(device=u.device, dtype=u.dtype)
    while m.dim() < u.dim():
        m = m.unsqueeze(0)
    return u * m


def extract_trajectory_component(
    trajectory: torch.Tensor, component_idx: int = 0
) -> torch.Tensor:
    """
    Select one PDE field from a ``torchdiffeq`` trajectory (second return of ``TorchDiffEqSolver.solve``).

    Shapes:
        - Single-field model: ``(T, D, H, W)`` — returned unchanged (``component_idx`` must be ``0``).
        - Coupled system: ``(T, C, D, H, W)`` — returns ``(T, D, H, W)`` for ``trajectory[:, component_idx]``.

    Downstream code that iterates ``for u_t in trajectory`` assumes a single spatial field per step; use this
    first when ``C > 1`` (e.g. tumor channel ``0`` in ``ImmuneResponse3D``).
    """
    if trajectory.dim() == 4:
        if component_idx != 0:
            raise ValueError(
                "Trajectory has shape (T, D, H, W); component_idx must be 0."
            )
        return trajectory
    if trajectory.dim() == 5:
        return trajectory[:, component_idx].contiguous()
    raise ValueError(
        f"Expected trajectory with ndim 4 or 5, got {trajectory.dim()} {tuple(trajectory.shape)}."
    )


class PDESystemModel3D(TumorGrowthModel3D):
    """
    Base class for 3D PDE systems integrated as a single torchdiffeq state tensor.

    Subclasses should set ``layout`` and implement ``forward(t, u)`` returning
    ``du/dt`` with the same shape as ``u``.
    """

    layout: ClassVar[PDEStateLayout] = PDEStateLayout(num_components=1)

    @property
    def num_state_components(self) -> int:
        return self.layout.num_components

    @property
    def num_state_components(self) -> int:
        return self.layout.num_components

    def validate_state_shape(
        self, u: torch.Tensor, *, allow_batch: bool = True
    ) -> None:
        """Raise ``ValueError`` if ``u`` does not match ``layout``."""
        if allow_batch and u.dim() == 5:
            expected_c = u.shape[1]
            ndim_ok = u.dim() == self.layout.state_ndim(batched=True)
        elif u.dim() == 4:
            expected_c = u.shape[0]
            ndim_ok = u.dim() == self.layout.state_ndim(batched=False)
        else:
            raise ValueError(
                f"Expected state of ndim 4 (C,D,H,W) or 5 (B,C,D,H,W); got shape {tuple(u.shape)}."
            )
        if not ndim_ok:
            raise ValueError(f"Unexpected state shape {tuple(u.shape)} for layout {self.layout}.")
        if expected_c != self.layout.num_components:
            raise ValueError(
                f"Expected {self.layout.num_components} components, got {expected_c}."
            )
