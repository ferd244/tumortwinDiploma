"""
Reusable 3D finite-difference operators on a structured grid with voxel BC tags.

Used by single-field and coupled PDE models so Laplacian / gradient stencils stay
consistent across the codebase.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn

from tumortwin.types.utility import Boundary


class FiniteDifferenceOperator3D(nn.Module):
    """
    Laplacian and gradient on a 3D scalar field using the same BC-aware stencils
    as ``ReactionDiffusion3D`` / ``ImmuneResponse3D``.

    ``bcs`` has shape ``(D, H, W, 3)`` with boundary tags per axis from
    ``bound_condition_maker``. ``spacing`` is ``(dx, dy, dz)`` in mm (or the same
    units as the PDE coefficients).
    """

    def __init__(
        self,
        bcs: torch.Tensor,
        spacing_xyz: Sequence[float],
    ):
        super().__init__()
        if len(spacing_xyz) != 3:
            raise ValueError("spacing_xyz must have length 3 (dx, dy, dz).")
        self.register_buffer("bcs", bcs.long())
        sp = torch.tensor(list(spacing_xyz), dtype=torch.float32)
        self.register_buffer("spacing", sp)

        for ax in (0, 1, 2):
            back_mask = self.bcs[:, :, :, ax] == Boundary.BACKWARD.value
            interior_mask = self.bcs[:, :, :, ax] == Boundary.INTERIOR.value
            forward_mask = self.bcs[:, :, :, ax] == Boundary.FORWARD.value
            inv_dx2 = 1.0 / (sp[ax] * sp[ax])
            self.register_buffer(
                f"lap_back_{ax}",
                self._central_slice_bool(back_mask, ax) * (2.0 * inv_dx2),
            )
            self.register_buffer(
                f"lap_cent_{ax}",
                self._central_slice_bool(interior_mask, ax) * (1.0 * inv_dx2),
            )
            self.register_buffer(
                f"lap_forw_{ax}",
                self._central_slice_bool(forward_mask, ax) * (2.0 * inv_dx2),
            )

    @staticmethod
    def _backward_slice(x: torch.Tensor, ax: int) -> torch.Tensor:
        return torch.narrow(x, ax, 0, x.shape[ax] - 2)

    @staticmethod
    def _central_slice(x: torch.Tensor, ax: int) -> torch.Tensor:
        return torch.narrow(x, ax, 1, x.shape[ax] - 2)

    @staticmethod
    def _forward_slice(x: torch.Tensor, ax: int) -> torch.Tensor:
        return torch.narrow(x, ax, 2, x.shape[ax] - 2)

    @classmethod
    def _central_slice_bool(cls, m: torch.Tensor, ax: int) -> torch.Tensor:
        return cls._central_slice(m, ax).to(dtype=torch.float32)

    def laplacian(self, field: torch.Tensor) -> torch.Tensor:
        """∇² field on the interior stencil; same shape as ``field``."""
        laplacian = torch.zeros_like(field)
        for ax in (0, 1, 2):
            back_c = getattr(self, f"lap_back_{ax}").to(field.device)
            cent_c = getattr(self, f"lap_cent_{ax}").to(field.device)
            forw_c = getattr(self, f"lap_forw_{ax}").to(field.device)
            back = self._backward_slice(field, ax)
            cent = self._central_slice(field, ax)
            forw = self._forward_slice(field, ax)
            contrib = (
                back_c * (back - cent)
                + cent_c * (back - 2.0 * cent + forw)
                + forw_c * (forw - cent)
            )
            self._central_slice(laplacian, ax).add_(contrib)
        return laplacian

    def gradient(self, field: torch.Tensor) -> torch.Tensor:
        """
        Gradient ``(∂/∂x, ∂/∂y, ∂/∂z)`` with shape ``(3,) + field.shape``,
        using central differences in the interior and one-sided differences on BC faces.
        """
        grad = torch.zeros((3,) + field.shape, device=field.device, dtype=field.dtype)
        for ax in (0, 1, 2):
            dx = self.spacing[ax].to(field.device)
            back_mask = self.bcs[:, :, :, ax] == Boundary.BACKWARD.value
            interior_mask = self.bcs[:, :, :, ax] == Boundary.INTERIOR.value
            forward_mask = self.bcs[:, :, :, ax] == Boundary.FORWARD.value

            if interior_mask.any():
                cent = self._central_slice(field, ax)
                back = self._backward_slice(field, ax)
                forw = self._forward_slice(field, ax)
                grad_cent = (forw - back) / (2.0 * dx)
                grad_slice = self._central_slice(grad[ax], ax)
                mask_slice = self._central_slice(interior_mask, ax)
                grad_slice[mask_slice] = grad_cent[mask_slice]

            if back_mask.any():
                cent = self._central_slice(field, ax)
                forw = self._forward_slice(field, ax)
                grad_back = (forw - cent) / dx
                grad_slice = self._central_slice(grad[ax], ax)
                mask_slice = self._central_slice(back_mask, ax)
                grad_slice[mask_slice] = grad_back[mask_slice]

            if forward_mask.any():
                cent = self._central_slice(field, ax)
                back = self._backward_slice(field, ax)
                grad_forw = (cent - back) / dx
                grad_slice = self._central_slice(grad[ax], ax)
                mask_slice = self._central_slice(forward_mask, ax)
                grad_slice[mask_slice] = grad_forw[mask_slice]

        return grad

    @staticmethod
    def laplacian_per_component(
        operator: "FiniteDifferenceOperator3D", *fields: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """Apply the same Laplacian operator to multiple scalar fields."""
        return tuple(operator.laplacian(f) for f in fields)
