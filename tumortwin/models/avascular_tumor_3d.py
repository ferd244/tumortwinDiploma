"""
Four-field avascular tumor model (n, m, h, s) with nutrient s and auxiliary velocity potential ψ.

Coupled PDEs (3D, conservative flux form used in code):

    ∂n/∂t = B n − P(s) n + ∇·((1−n)(D_n ∇n − μ n ∇g(s))) − ∇·(n ∇ψ),
    ∂m/∂t = P(s) n − ∇·(m ∇ψ),
    ∂h/∂t = L n h − ∇·(h ∇ψ),
    ∂s/∂t = D_s Δs + Q(n,s),
    Δψ     = B n + L n h,

with

    P(s)   = B K (1 − tanh(s / s_x)),
    Q(n,s) = − q_s n s / (s + s_0),
    g(s)   = g_0 arctan(s / s_K).

ψ is recovered each RHS evaluation by fixed-point (Richardson) iterations on the same
Laplacian stencil as :class:`~tumortwin.spatial.FiniteDifferenceOperator3D`.

**Note:** :math:`P(s)` is high when nutrient ``s`` is low and low when ``s`` is high; it
feeds conversion into the quiescent compartment ``m``. If you need growth limited by
nutrient instead, swap to e.g. :math:`P_{\\mathrm{growth}}(s)=\\tanh(s/s_x)` for the
:math:`B \\, n` term in a fork or added option.

Field packing in the ODE state: ``[n, m, h, s]`` with shape ``(4, D, H, W)``.
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


class AvascularTumorGrowth3D(PDESystemModel3D):
    """
    Avascular tumor growth with proliferating ``n``, quiescent ``m``, host ``h``, nutrient ``s``.
    """

    layout: ClassVar[PDEStateLayout] = PDEStateLayout(num_components=4)

    def __init__(
        self,
        B: torch.Tensor,
        L: torch.Tensor,
        Dn: torch.Tensor,
        Ds: torch.Tensor,
        mu: torch.Tensor,
        q_s: torch.Tensor,
        s_0: torch.Tensor,
        s_x: torch.Tensor,
        s_K: torch.Tensor,
        g_0: torch.Tensor,
        patient_data: Union[HGGPatientData, TNBCPatientData],
        *,
        initial_n: torch.Tensor,
        initial_m: Optional[torch.Tensor] = None,
        initial_h: Optional[torch.Tensor] = None,
        initial_s: Optional[torch.Tensor] = None,
        K: Optional[torch.Tensor] = None,
        time_scale_days: float = 30.0,
        poisson_iterations: int = 48,
        poisson_relaxation: Optional[float] = None,
        require_grad: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.device = device
        self.poisson_iterations = int(poisson_iterations)
        self.time_scale_days = float(time_scale_days)
        if self.time_scale_days <= 0:
            raise ValueError("time_scale_days must be > 0.")

        mask_image = (
            patient_data.breastmask_image
            if hasattr(patient_data, "breastmask_image")
            else patient_data.brainmask_image
        )
        # Stable Richardson step ~ h^2 / (2d) for d=3 (scaled by 0.35 for margin)
        h_min = float(min(mask_image.spacing.x, mask_image.spacing.y, mask_image.spacing.z))
        default_relax = 0.35 * (h_min**2) / 6.0
        self.poisson_relaxation = float(
            poisson_relaxation if poisson_relaxation is not None else default_relax
        )

        self.B = nn.Parameter(B.to(device), requires_grad=require_grad)
        self.L = nn.Parameter(L.to(device), requires_grad=require_grad)
        self.Dn = nn.Parameter(Dn.to(device), requires_grad=require_grad)
        self.Ds = nn.Parameter(Ds.to(device), requires_grad=require_grad)
        self.mu = nn.Parameter(mu.to(device), requires_grad=require_grad)
        self.q_s = nn.Parameter(q_s.to(device), requires_grad=require_grad)
        if K is None:
            K = torch.tensor(1.0, dtype=torch.float32, device=device)
        self.K = nn.Parameter(K.to(device), requires_grad=require_grad)
        self.s_0 = nn.Parameter(s_0.to(device), requires_grad=require_grad)
        self.s_x = nn.Parameter(s_x.to(device), requires_grad=require_grad)
        self.s_K = nn.Parameter(s_K.to(device), requires_grad=require_grad)
        self.g_0 = nn.Parameter(g_0.to(device), requires_grad=require_grad)
        if torch.any(self.L.detach() >= 0):
            raise ValueError(
                "Strict article model requires L < 0 (tumor-healthy interaction suppression)."
            )

        self.register_buffer("n_initial", initial_n.to(device).float())
        if initial_m is None:
            self.register_buffer("m_initial", torch.zeros_like(self.n_initial))
        else:
            self.register_buffer("m_initial", initial_m.to(device).float())
        if initial_h is None:
            h0 = torch.clamp(1.0 - self.n_initial, min=0.0, max=1.0)
            self.register_buffer("h_initial", h0)
        else:
            self.register_buffer("h_initial", initial_h.to(device).float())
        if initial_s is None:
            self.register_buffer(
                "s_initial",
                torch.ones_like(self.n_initial),
            )
        else:
            self.register_buffer("s_initial", initial_s.to(device).float())

        # Binary mask: 1 inside tissue / ROI, 0 outside (same convention as ReactionDiffusion3D).
        mask_np = np.asarray(mask_image.array, dtype=float)
        if not (mask_np > 0).any():
            raise ValueError(
                "Anatomical mask is empty (all zeros). Check patient mask / crop (CropSettings)."
            )
        self.bcs = torch.from_numpy(bound_condition_maker(mask_image).array).to(device)
        comp = torch.as_tensor(mask_np, dtype=torch.float32).clamp(0.0, 1.0)
        comp = (comp > 0).float()
        # Must be a buffer so ``model.to(device)`` keeps mask aligned with integrated state.
        self.register_buffer("comp_mask", comp.to(device))
        self.spacing = mask_image.spacing
        _nin = float((self.n_initial * self.comp_mask).sum())
        _nsum = float(self.n_initial.sum())
        if _nsum > 1e-6 and _nin < 1e-4 * _nsum:
            warnings.warn(
                "Initial tumor mass is almost entirely outside the anatomical mask "
                "(n_initial * comp_mask is tiny vs n_initial sum). "
                "Check brainmask vs cellularity grid alignment after crop.",
                UserWarning,
                stacklevel=2,
            )
        self._prepare_fd_stencils()

        # Keep solver compatibility with treatment-aware grid constructor.
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
                self.h_initial * m,
                self.s_initial * m,
            ],
            dim=0,
        )

    def _P_transition(self, s: torch.Tensor) -> torch.Tensor:
        """P(s) = B K (1 - tanh(s / s_x)), strict article form for n -> m conversion."""
        sx = torch.clamp(self.s_x.to(s.device), min=1e-6)
        return self.B.to(s.device) * self.K.to(s.device) * (1.0 - torch.tanh(s / sx))

    def _dg_ds(self, s: torch.Tensor) -> torch.Tensor:
        sk = torch.clamp(self.s_K.to(s.device), min=1e-6)
        g0 = self.g_0.to(s.device)
        return g0 * sk / (sk * sk + s * s)

    def _Q(self, n: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        denom = s + torch.clamp(self.s_0.to(s.device), min=1e-6)
        return -(self.q_s.to(s.device) * n * s / denom)

    def _solve_psi(self, rhs: torch.Tensor) -> torch.Tensor:
        """Approximate Δψ = rhs via Richardson iteration (ψ ← ψ + α (rhs - Δψ))."""
        psi = torch.zeros_like(rhs)
        alpha = self.poisson_relaxation
        mask = self.comp_mask.to(device=rhs.device, dtype=rhs.dtype)
        rhs_m = torch.nan_to_num(rhs * mask, nan=0.0, posinf=1e6, neginf=-1e6)
        for _ in range(self.poisson_iterations):
            lap = self.spatial_fd.laplacian(psi)
            psi = psi + alpha * (rhs_m - lap)
            psi = torch.nan_to_num(psi * mask, nan=0.0, posinf=1e6, neginf=-1e6)
        return psi

    @torch.enable_grad()
    def forward(self, t: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        device = self.device
        u = u.to(device)
        B = self.B.to(device)
        L = self.L.to(device)
        Dn = self.Dn.to(device)
        Ds = self.Ds.to(device)
        mu = self.mu.to(device)

        if u.dim() == 5:
            raise NotImplementedError(
                "AvascularTumorGrowth3D supports state shape (4, D, H, W) only, not batched (B, 4, ...)."
            )
        if u.dim() != 4:
            raise ValueError(f"Expected u of shape (4, D, H, W); got {tuple(u.shape)}")
        n, m, h, s = u[0], u[1], u[2], u[3]
        n = torch.clamp(torch.nan_to_num(n, nan=0.0, posinf=1.0, neginf=0.0), min=0.0, max=1.0)
        # Treat m and h as local volume fractions (same as n): enforce [0, 1] each RHS call.
        # This prevents explosive advection/reaction feedback from transient overshoots within RK stages.
        m = torch.clamp(torch.nan_to_num(m, nan=0.0, posinf=1.0, neginf=0.0), min=0.0, max=1.0)
        h = torch.clamp(torch.nan_to_num(h, nan=0.0, posinf=1.0, neginf=0.0), min=0.0, max=1.0)
        s = torch.clamp(torch.nan_to_num(s, nan=0.0, posinf=1.0, neginf=0.0), min=0.0)

        P_s = self._P_transition(s)
        rhs_psi = B * n + L * n * h
        psi = self._solve_psi(rhs_psi)
        grad_psi = self.spatial_fd.gradient(psi)
        grad_s = self.spatial_fd.gradient(s)
        dg_ds = self._dg_ds(s)
        grad_g = grad_s * dg_ds.unsqueeze(0)

        free_space = torch.clamp(1.0 - (n + m + h), min=0.0, max=1.0)

        flux_tumor_mvmt = free_space.unsqueeze(0) * (
            Dn * self.spatial_fd.gradient(n) - mu * n.unsqueeze(0) * grad_g
        )
        div_tumor_mvmt = self.spatial_fd.divergence(flux_tumor_mvmt)
        div_n_psi = self.spatial_fd.divergence(n.unsqueeze(0) * grad_psi)

        dn_dt = B * n * free_space - P_s * n + div_tumor_mvmt - div_n_psi
        dm_dt = P_s * n - self.spatial_fd.divergence(m.unsqueeze(0) * grad_psi)
        dh_dt = L * n * h - self.spatial_fd.divergence(h.unsqueeze(0) * grad_psi)
        ds_dt = Ds * self.spatial_fd.laplacian(s) + self._Q(n, s)
        out = torch.nan_to_num(
            torch.stack([dn_dt, dm_dt, dh_dt, ds_dt]),
            nan=0.0,
            posinf=1e6,
            neginf=-1e6,
        )
        out = out / self.time_scale_days
        return out

    def callback_step(self, t, u, dt):
        if self.progress_bar is not None:
            self.progress_bar.update(dt.item())
        tissue = self.comp_mask.to(device=u.device, dtype=u.dtype)
        if u.dim() != 4:
            raise ValueError(
                f"callback_step expects shape (4, D, H, W); got {tuple(u.shape)}"
            )
        n, m, h, s = u[0], u[1], u[2], u[3]

        n = torch.nan_to_num(n * tissue, nan=0.0, posinf=1.0, neginf=0.0)
        m = torch.nan_to_num(m * tissue, nan=0.0, posinf=1.0, neginf=0.0)
        h = torch.nan_to_num(h * tissue, nan=0.0, posinf=1.0, neginf=0.0)
        s = torch.nan_to_num(s * tissue, nan=0.0, posinf=1.0, neginf=0.0)

        n[tissue <= 0] = 0.0
        m[tissue <= 0] = 0.0
        h[tissue <= 0] = 0.0
        s[tissue <= 0] = 0.0

        n = torch.clamp(n, min=0.0, max=1.0)
        m = torch.clamp(m, min=0.0, max=1.0)
        h = torch.clamp(h, min=0.0, max=1.0)
        s = torch.clamp(s, min=0.0)

        return torch.stack([n, m, h, s])

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
