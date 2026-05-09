"""
Directed tumor growth without chemotaxis (PDE system).

State fields:
    n : proliferating tumor cells
    m : quiescent tumor cells
    s : nutrient / oxygen

Auxiliary field:
    phi : velocity potential solved from Poisson equation each RHS call.

Implemented equations:
    dn/dt = B*n*(1-n) - P(s)*n + div(Dn*(1-n)*grad(n)) - <grad(n), grad(phi)>
    dm/dt = P(s)*n - m*B*n - div(Dn*m*grad(n)) - <grad(m), grad(phi)>
    ds/dt = -k_s*n*s/(s+s_star) + Ds*Δs
    Δphi  = B*n

Default smooth transition:
    P(s) = B*K*0.5*(1 - tanh((s - s_crit)/s_smooth))
"""

from __future__ import annotations

import warnings
from typing import ClassVar, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from tumortwin.models.pde_system import PDEStateLayout, PDESystemModel3D
from tumortwin.preprocessing import bound_condition_maker
from tumortwin.spatial import FiniteDifferenceOperator3D
from tumortwin.types import HGGPatientData, TNBCPatientData


class HemoInvasion3D(PDESystemModel3D):
    """Three-field PDE model (n, m, s) with convective transport from phi."""

    layout: ClassVar[PDEStateLayout] = PDEStateLayout(num_components=3)

    def __init__(
        self,
        B: torch.Tensor,
        Dn: torch.Tensor,
        Ds: torch.Tensor,
        k_s: torch.Tensor,
        s_star: torch.Tensor,
        patient_data: Union[HGGPatientData, TNBCPatientData],
        *,
        initial_n: torch.Tensor,
        initial_m: Optional[torch.Tensor] = None,
        initial_s: Optional[torch.Tensor] = None,
        K: Optional[torch.Tensor] = None,
        s_crit: Optional[torch.Tensor] = None,
        s_smooth: Optional[torch.Tensor] = None,
        s_x: Optional[torch.Tensor] = None,
        s_outside: float = 0.0,
        s_vessel: float = 1.0,
        vessel_mask: Optional[torch.Tensor] = None,
        time_scale_days: float = 30.0,
        poisson_iterations: int = 32,
        poisson_relaxation: Optional[float] = None,
        require_grad: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.device = device
        self.time_scale_days = float(time_scale_days)
        if self.time_scale_days <= 0:
            raise ValueError("time_scale_days must be > 0.")
        self.poisson_iterations = int(poisson_iterations)

        mask_image = (
            patient_data.breastmask_image
            if hasattr(patient_data, "breastmask_image")
            else patient_data.brainmask_image
        )
        h_min = float(min(mask_image.spacing.x, mask_image.spacing.y, mask_image.spacing.z))
        default_relax = 0.35 * (h_min**2) / 6.0
        self.poisson_relaxation = float(
            poisson_relaxation if poisson_relaxation is not None else default_relax
        )

        self.B = nn.Parameter(B.to(device), requires_grad=require_grad)
        self.Dn = nn.Parameter(Dn.to(device), requires_grad=require_grad)
        self.Ds = nn.Parameter(Ds.to(device), requires_grad=require_grad)
        self.k_s = nn.Parameter(k_s.to(device), requires_grad=require_grad)
        self.s_star = nn.Parameter(s_star.to(device), requires_grad=require_grad)
        if K is None:
            K = torch.tensor(1.0, dtype=torch.float32, device=device)
        self.K = nn.Parameter(K.to(device), requires_grad=require_grad)
        if s_crit is None:
            s_crit = torch.tensor(0.5, dtype=torch.float32, device=device)
        if s_smooth is None:
            if s_x is not None:
                s_smooth = s_x
            else:
                s_smooth = torch.tensor(0.1, dtype=torch.float32, device=device)
        self.s_crit = nn.Parameter(s_crit.to(device), requires_grad=require_grad)
        self.s_smooth = nn.Parameter(s_smooth.to(device), requires_grad=require_grad)
        self.register_buffer(
            "s_outside", torch.tensor(float(s_outside), dtype=torch.float32, device=device)
        )
        self.register_buffer(
            "s_vessel", torch.tensor(float(s_vessel), dtype=torch.float32, device=device)
        )

        self.register_buffer("n_initial", initial_n.to(device).float())
        if initial_m is None:
            self.register_buffer("m_initial", torch.zeros_like(self.n_initial))
        else:
            if initial_m.shape != self.n_initial.shape:
                raise ValueError(
                    "initial_m must have the same shape as initial_n: "
                    f"{tuple(initial_m.shape)} vs {tuple(self.n_initial.shape)}"
                )
            self.register_buffer("m_initial", initial_m.to(device).float())
        if initial_s is None:
            self.register_buffer("s_initial", torch.ones_like(self.n_initial))
        else:
            if initial_s.shape != self.n_initial.shape:
                raise ValueError(
                    "initial_s must have the same shape as initial_n: "
                    f"{tuple(initial_s.shape)} vs {tuple(self.n_initial.shape)}"
                )
            self.register_buffer("s_initial", initial_s.to(device).float())

        mask_np = np.asarray(mask_image.array, dtype=float)
        if not (mask_np > 0).any():
            raise ValueError("Anatomical mask is empty (all zeros).")
        self.bcs = torch.from_numpy(bound_condition_maker(mask_image).array).to(device)
        comp = torch.as_tensor(mask_np, dtype=torch.float32).clamp(0.0, 1.0)
        self.register_buffer("comp_mask", (comp > 0).float().to(device))
        self.spacing = mask_image.spacing
        _nin = float((self.n_initial * self.comp_mask).sum())
        _nsum = float(self.n_initial.sum())
        if _nsum > 1e-6 and _nin < 1e-4 * _nsum:
            warnings.warn(
                "Initial tumor mass is almost entirely outside the anatomical mask "
                "(n_initial * comp_mask is tiny vs n_initial sum). "
                "Check mask / cellularity alignment after crop.",
                UserWarning,
                stacklevel=2,
            )
        self._prepare_fd_stencils()

        if vessel_mask is not None:
            vessel_mask_bool = vessel_mask.to(device).bool()
            if vessel_mask_bool.shape != self.n_initial.shape:
                raise ValueError(
                    "vessel_mask must have the same shape as initial_n: "
                    f"{tuple(vessel_mask_bool.shape)} vs {tuple(self.n_initial.shape)}"
                )
            self.register_buffer("vessel_mask", vessel_mask_bool)
        else:
            self.vessel_mask = None

        # Keep compatibility with treatment-aware solver grid constructor.
        self.t_initial = None
        self.radiotherapy_specification = None
        self.chemotherapy_specifications = None
        self.progress_bar = None

    def _prepare_fd_stencils(self) -> None:
        sp = self.spacing
        spacing_xyz = [sp.x, sp.y, sp.z]
        self.spatial_fd = FiniteDifferenceOperator3D(self.bcs, spacing_xyz)

    def get_initial_state(self) -> torch.Tensor:
        m = self.comp_mask
        return torch.stack(
            [
                self.n_initial * m,
                self.m_initial * m,
                self.s_initial * m,
            ],
            dim=0,
        )

    def _P_transition(self, s: torch.Tensor) -> torch.Tensor:
        smooth = torch.clamp(self.s_smooth.to(s.device), min=1e-6)
        z = (s - self.s_crit.to(s.device)) / smooth
        return 0.5 * self.B.to(s.device) * self.K.to(s.device) * (1.0 - torch.tanh(z))

    @staticmethod
    def _dot_grad(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.sum(a * b, dim=0)

    def _solve_phi(self, rhs: torch.Tensor) -> torch.Tensor:
        phi = torch.zeros_like(rhs)
        alpha = self.poisson_relaxation
        mask = self.comp_mask.to(device=rhs.device, dtype=rhs.dtype)
        rhs_m = torch.nan_to_num(rhs * mask, nan=0.0, posinf=1e6, neginf=-1e6)
        for _ in range(self.poisson_iterations):
            lap = self.spatial_fd.laplacian(phi)
            phi = phi - alpha * (rhs_m - lap)
            phi = torch.nan_to_num(phi * mask, nan=0.0, posinf=1e6, neginf=-1e6)
        return phi

    @torch.enable_grad()
    def forward(self, t: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        self.validate_state_shape(u, allow_batch=False)
        n, m, s = u[0], u[1], u[2]
        n = torch.clamp(torch.nan_to_num(n, nan=0.0, posinf=1.0, neginf=0.0), min=0.0, max=1.0)
        m = torch.clamp(torch.nan_to_num(m, nan=0.0, posinf=1.0, neginf=0.0), min=0.0, max=1.0)
        s = torch.clamp(torch.nan_to_num(s, nan=0.0, posinf=1.0, neginf=0.0), min=0.0)

        grad_n = self.spatial_fd.gradient(n)
        grad_m = self.spatial_fd.gradient(m)
        lap_s = self.spatial_fd.laplacian(s)
        rhs_phi = self.B.to(n.device) * n
        phi = self._solve_phi(rhs_phi)
        grad_phi = self.spatial_fd.gradient(phi)

        p_s = self._P_transition(s)
        diffusion_n = self.spatial_fd.divergence(
            self.Dn.to(n.device) * (1.0 - n).unsqueeze(0) * grad_n
        )
        diffusion_m = self.spatial_fd.divergence(
            self.Dn.to(n.device) * m.unsqueeze(0) * grad_n
        )
        conv_n = self._dot_grad(grad_n, grad_phi)
        conv_m = self._dot_grad(grad_m, grad_phi)

        dn_dt = (
            self.B.to(n.device) * n * (1.0 - n)
            - p_s * n
            + diffusion_n
            - conv_n
        )
        dm_dt = p_s * n - m * self.B.to(n.device) * n - diffusion_m - conv_m
        denom = s + torch.clamp(self.s_star.to(s.device), min=1e-6)
        ds_dt = self.Ds.to(s.device) * lap_s - self.k_s.to(s.device) * n * s / denom

        out = torch.nan_to_num(
            torch.stack([dn_dt, dm_dt, ds_dt]),
            nan=0.0,
            posinf=1e6,
            neginf=-1e6,
        )
        return out / self.time_scale_days

    def callback_step(self, t, u, dt):
        if self.progress_bar is not None:
            try:
                self.progress_bar.update(dt.item())
            except (TypeError, AttributeError):
                pass

        self.validate_state_shape(u, allow_batch=False)

        tissue = self.comp_mask.to(device=u.device, dtype=u.dtype)
        n, m, s = u[0], u[1], u[2]
        n = torch.nan_to_num(n * tissue, nan=0.0, posinf=1.0, neginf=0.0)
        m = torch.nan_to_num(m * tissue, nan=0.0, posinf=1.0, neginf=0.0)
        s = torch.nan_to_num(s * tissue, nan=0.0, posinf=1.0, neginf=0.0)

        outside = tissue <= 0
        s[outside] = self.s_outside.to(s.device)
        if self.vessel_mask is not None:
            vm = self.vessel_mask.to(device=s.device)
            s[vm] = self.s_vessel.to(s.device)

        n = torch.clamp(n, min=0.0, max=1.0)
        m = torch.clamp(m, min=0.0, max=1.0)
        s = torch.clamp(s, min=0.0)
        occ = n + m
        over = occ > 1.0
        if over.any():
            n[over] = n[over] / occ[over]
            m[over] = m[over] / occ[over]

        return torch.stack([n, m, s])

    def callback_step_adjoint(self, t, u, dt):
        mask = self.comp_mask.to(device=u[2].device, dtype=torch.bool)
        adj_y = u[2]
        if adj_y.dim() == 4:
            for c in range(adj_y.shape[0]):
                adj_y[c].mul_(mask)
        elif adj_y.dim() == 5:
            for c in range(adj_y.shape[1]):
                adj_y[:, c].mul_(mask)
        return u
