"""
Directed tumor growth without chemotaxis (PDE system) with treatment support.

State fields:
    n : proliferating tumor cells
    m : quiescent tumor cells
    s : nutrient / oxygen

Auxiliary field:
    phi : velocity potential solved from Poisson equation each RHS call.

Governing equations:
    dn/dt = B*n*(1-n) - P(s)*n + div(Dn*(1-n)*grad(n)) - <grad(n), grad(phi)> - CT(t)*n
    dm/dt = P(s)*n - m*B*n - div(Dn*m*grad(n)) - <grad(m), grad(phi)> - CT(t)*m
    ds/dt = -k_s*n*s/(s+s_star) + Ds*Δs
    Δphi  = B*n

Radiotherapy (instantaneous, applied in callback_step):
    n_after = n_before * exp(-alpha_RT * d - beta_RT * d²)
    m_after = m_before * exp(-alpha_RT * d - beta_RT * d²)

Quiescence transition:
    P(s) = B*K*0.5*(1 - tanh((s - s_crit)/s_smooth))
"""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta
from typing import ClassVar, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from tumortwin.models.pde_system import PDEStateLayout, PDESystemModel3D
from tumortwin.preprocessing import bound_condition_maker
from tumortwin.spatial import FiniteDifferenceOperator3D
from tumortwin.treatments import (
    compute_radiotherapy_cell_survival_fraction,
    compute_total_cell_death_chemo,
)
from tumortwin.types import (
    ChemotherapySpecification,
    HGGPatientData,
    RadiotherapySpecification,
    TNBCPatientData,
)


class HemoInvasion3D(PDESystemModel3D):
    """
    Three-field PDE model (n, m, s) with convective transport (phi) and
    optional radiotherapy / chemotherapy treatment effects.

    Treatment handling mirrors ReactionDiffusion3D:
      - Chemotherapy: continuous decay term in forward(), proportional to n and m.
      - Radiotherapy: instantaneous cell kill in callback_step() at delivery times,
        using the linear-quadratic survival fraction model.
    """

    layout: ClassVar[PDEStateLayout] = PDEStateLayout(num_components=3)

    #: Stack order is ``(n, m, s)``. Total tumor occupancy comparable to ADC cellularity sums these.
    total_cellularity_component_indices: ClassVar[tuple[int, ...]] = (0, 1)

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
        # --- Treatment parameters (mirror ReactionDiffusion3D) ---
        initial_time: Optional[datetime] = None,
        radiotherapy_specification: Optional[RadiotherapySpecification] = None,
        chemotherapy_specifications: Optional[List[ChemotherapySpecification]] = None,
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

        # ----- Learnable parameters -----
        self.B      = nn.Parameter(B.to(device),      requires_grad=require_grad)
        self.Dn     = nn.Parameter(Dn.to(device),     requires_grad=require_grad)
        self.Ds     = nn.Parameter(Ds.to(device),     requires_grad=require_grad)
        self.k_s    = nn.Parameter(k_s.to(device),    requires_grad=require_grad)
        self.s_star = nn.Parameter(s_star.to(device), requires_grad=require_grad)

        if K is None:
            K = torch.tensor(1.0, dtype=torch.float32, device=device)
        self.K = nn.Parameter(K.to(device), requires_grad=require_grad)

        if s_crit is None:
            s_crit = torch.tensor(0.5, dtype=torch.float32, device=device)
        if s_smooth is None:
            s_smooth = s_x if s_x is not None else torch.tensor(0.1, dtype=torch.float32)
        self.s_crit   = nn.Parameter(s_crit.to(device),   requires_grad=require_grad)
        self.s_smooth = nn.Parameter(s_smooth.to(device), requires_grad=require_grad)

        self.register_buffer(
            "s_outside", torch.tensor(float(s_outside), dtype=torch.float32, device=device)
        )
        self.register_buffer(
            "s_vessel", torch.tensor(float(s_vessel), dtype=torch.float32, device=device)
        )

        # ----- Initial conditions -----
        self.register_buffer("n_initial", initial_n.to(device).float())

        if initial_m is None:
            self.register_buffer("m_initial", torch.zeros_like(self.n_initial))
        else:
            if initial_m.shape != self.n_initial.shape:
                raise ValueError(
                    f"initial_m shape {tuple(initial_m.shape)} != "
                    f"initial_n shape {tuple(self.n_initial.shape)}"
                )
            self.register_buffer("m_initial", initial_m.to(device).float())

        if initial_s is None:
            self.register_buffer("s_initial", torch.ones_like(self.n_initial))
        else:
            if initial_s.shape != self.n_initial.shape:
                raise ValueError(
                    f"initial_s shape {tuple(initial_s.shape)} != "
                    f"initial_n shape {tuple(self.n_initial.shape)}"
                )
            self.register_buffer("s_initial", initial_s.to(device).float())

        # ----- Anatomical mask & FD stencils -----
        mask_np = np.asarray(mask_image.array, dtype=float)
        if not (mask_np > 0).any():
            raise ValueError("Anatomical mask is empty (all zeros).")
        self.bcs = torch.from_numpy(bound_condition_maker(mask_image).array).to(device)
        comp = torch.as_tensor(mask_np, dtype=torch.float32).clamp(0.0, 1.0)
        self.register_buffer("comp_mask", (comp > 0).float().to(device))
        self.spacing = mask_image.spacing

        _nin  = float((self.n_initial * self.comp_mask).sum())
        _nsum = float(self.n_initial.sum())
        if _nsum > 1e-6 and _nin < 1e-4 * _nsum:
            warnings.warn(
                "Initial tumor mass is almost entirely outside the anatomical mask. "
                "Check mask / cellularity alignment after crop.",
                UserWarning,
                stacklevel=2,
            )
        self._prepare_fd_stencils()

        # ----- Vessel mask -----
        if vessel_mask is not None:
            vessel_mask_bool = vessel_mask.to(device).bool()
            if vessel_mask_bool.shape != self.n_initial.shape:
                raise ValueError(
                    f"vessel_mask shape {tuple(vessel_mask_bool.shape)} != "
                    f"initial_n shape {tuple(self.n_initial.shape)}"
                )
            self.register_buffer("vessel_mask", vessel_mask_bool)
        else:
            self.vessel_mask = None

        # ----- Treatment -----
        self.t_initial = initial_time

        self.radiotherapy_specification = radiotherapy_specification
        if radiotherapy_specification is not None:
            if initial_time is None:
                raise ValueError(
                    "initial_time must be provided when radiotherapy_specification is set."
                )
            # Use total fractional days (matching timedelta_to_days used in the solver)
            # so that adaptive-solver segment boundaries align with these keys.
            self.radiotherapy_days: dict = {
                (day - initial_time).total_seconds() / 86400.0: dose
                for day, dose in radiotherapy_specification.protocol.items()
            }
        else:
            self.radiotherapy_days = {}

        self.chemotherapy_specifications = chemotherapy_specifications
        if chemotherapy_specifications and initial_time is None:
            raise ValueError(
                "initial_time must be provided when chemotherapy_specifications is set."
            )
        # ParameterList keeps chemo sensitivities in the PyTorch graph for gradient computation.
        self.ct_sens = nn.ParameterList(
            [spec.sensitivity for spec in (chemotherapy_specifications or [])]
        )

        self.progress_bar = None

    # ------------------------------------------------------------------
    # Spatial operators
    # ------------------------------------------------------------------

    def _prepare_fd_stencils(self) -> None:
        sp = self.spacing
        self.spatial_fd = FiniteDifferenceOperator3D(self.bcs, [sp.x, sp.y, sp.z])

    # ------------------------------------------------------------------
    # Initial state
    # ------------------------------------------------------------------

    def get_initial_state(self) -> torch.Tensor:
        m = self.comp_mask
        return torch.stack(
            [self.n_initial * m, self.m_initial * m, self.s_initial * m],
            dim=0,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _P_transition(self, s: torch.Tensor) -> torch.Tensor:
        """Smooth proliferating → quiescent transition rate P(s)."""
        smooth = torch.clamp(self.s_smooth.to(s.device), min=1e-6)
        z = (s - self.s_crit.to(s.device)) / smooth
        return 0.5 * self.B.to(s.device) * self.K.to(s.device) * (1.0 - torch.tanh(z))

    @staticmethod
    def _dot_grad(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.sum(a * b, dim=0)

    def _solve_phi(self, rhs: torch.Tensor) -> torch.Tensor:
        """
        Iterative Richardson solver for  Δφ = rhs  (masked Jacobi).

        Converging sign: φ ← φ - α*(rhs - Δφ)
        The discrete Laplacian is negative semi-definite (eigenvalues ≤ 0),
        so the iteration matrix (I + αL) has spectral radius < 1 for
        α < h_min² / 6, which is guaranteed by the default relaxation choice.
        """
        phi = torch.zeros_like(rhs)
        alpha = self.poisson_relaxation
        mask = self.comp_mask.to(device=rhs.device, dtype=rhs.dtype)
        rhs_m = torch.nan_to_num(rhs * mask, nan=0.0, posinf=1e6, neginf=-1e6)
        for _ in range(self.poisson_iterations):
            lap = self.spatial_fd.laplacian(phi)
            phi = phi - alpha * (rhs_m - lap)
            phi = torch.nan_to_num(phi * mask, nan=0.0, posinf=1e6, neginf=-1e6)
        return phi

    # ------------------------------------------------------------------
    # ODE right-hand side
    # ------------------------------------------------------------------

    @torch.enable_grad()
    def forward(self, t: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        self.validate_state_shape(u, allow_batch=False)
        n, m, s = u[0], u[1], u[2]
        n = torch.clamp(torch.nan_to_num(n, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        m = torch.clamp(torch.nan_to_num(m, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        s = torch.clamp(torch.nan_to_num(s, nan=0.0, posinf=1.0, neginf=0.0), min=0.0)

        # --- Spatial operators ---
        grad_n   = self.spatial_fd.gradient(n)
        grad_m   = self.spatial_fd.gradient(m)
        lap_s    = self.spatial_fd.laplacian(s)
        phi      = self._solve_phi(self.B.to(n.device) * n)
        grad_phi = self.spatial_fd.gradient(phi)

        # --- Reaction / quiescence ---
        p_s = self._P_transition(s)

        # --- Diffusion ---
        diffusion_n = self.spatial_fd.divergence(
            self.Dn.to(n.device) * (1.0 - n).unsqueeze(0) * grad_n
        )
        diffusion_m = self.spatial_fd.divergence(
            self.Dn.to(n.device) * m.unsqueeze(0) * grad_n
        )

        # --- Convection ---
        conv_n = self._dot_grad(grad_n, grad_phi)
        conv_m = self._dot_grad(grad_m, grad_phi)

        # --- Chemotherapy (continuous exponential decay) ---
        chemo_effect = None
        if self.chemotherapy_specifications and self.t_initial is not None:
            chemo_effect = compute_total_cell_death_chemo(
                self.t_initial
                + timedelta(days=float(t.detach().reshape(()).item())),
                self.chemotherapy_specifications,
            )

        # --- Assemble rates ---
        dn_dt = (
            self.B.to(n.device) * n * (1.0 - n)
            - p_s * n
            + diffusion_n
            - conv_n
        )
        dm_dt = p_s * n - m * self.B.to(n.device) * n - diffusion_m - conv_m

        if chemo_effect is not None:
            dn_dt = dn_dt - chemo_effect * n
            dm_dt = dm_dt - chemo_effect * m

        denom = s + torch.clamp(self.s_star.to(s.device), min=1e-6)
        ds_dt = self.Ds.to(s.device) * lap_s - self.k_s.to(s.device) * n * s / denom

        out = torch.nan_to_num(
            torch.stack([dn_dt, dm_dt, ds_dt]),
            nan=0.0, posinf=1e6, neginf=-1e6,
        )
        return out / self.time_scale_days

    def _find_rt_dose(self, t: torch.Tensor, tol: float = 0.01) -> Optional[float]:
        """
        Return the RT dose for time t (days) if an event is within `tol` days,
        else None.  Tolerance comparison avoids exact-float mismatches between
        the solver grid and the protocol dict keys.
        """
        if self.radiotherapy_specification is None:
            return None
        # Solver time may require grad; comparing to protocol keys uses plain Python float.
        t_day = float(t.detach().reshape(()))
        for key, dose in self.radiotherapy_days.items():
            if abs(key - t_day) <= tol:
                return dose
        return None

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def callback_step(self, t, u, dt):
        if self.progress_bar is not None:
            try:
                new_n = int(t.detach().item() + dt.detach().item())
                delta = new_n - int(self.progress_bar.n)
                if delta > 0:
                    self.progress_bar.update(delta)
            except (TypeError, AttributeError, ValueError):
                pass
        self.validate_state_shape(u, allow_batch=False)

        tissue = self.comp_mask.to(device=u.device, dtype=u.dtype)
        n, m, s = u[0], u[1], u[2]

        # Apply tissue mask
        n = torch.nan_to_num(n * tissue, nan=0.0, posinf=1.0, neginf=0.0)
        m = torch.nan_to_num(m * tissue, nan=0.0, posinf=1.0, neginf=0.0)
        s = torch.nan_to_num(s * tissue, nan=0.0, posinf=1.0, neginf=0.0)

        # Boundary conditions for nutrient (functional only — in-place breaks autograd
        # through torchdiffeq adjoint when RT / Adam calibrates learnable parameters).
        outside = tissue <= 0
        s_out = self.s_outside.to(device=s.device, dtype=s.dtype)
        s = torch.where(outside, s_out, s)
        if self.vessel_mask is not None:
            vm = self.vessel_mask.to(device=s.device)
            s_v = self.s_vessel.to(device=s.device, dtype=s.dtype)
            s = torch.where(vm, s_v, s)

        # --- Radiotherapy: instantaneous cell kill (linear-quadratic model) ---
        rt_dose = self._find_rt_dose(t)
        if rt_dose is not None:
            sf = compute_radiotherapy_cell_survival_fraction(
                self.radiotherapy_specification, rt_dose
            )
            n = n * sf
            m = m * sf

        # Clamp & enforce occupancy ≤ 1
        n = torch.clamp(n, 0.0, 1.0)
        m = torch.clamp(m, 0.0, 1.0)
        s = torch.clamp(s, min=0.0)
        occ = n + m
        over = occ > 1.0
        inv = torch.where(
            over, 1.0 / occ.clamp(min=1e-12), torch.ones_like(occ)
        )
        n = n * inv
        m = m * inv

        # torchdiffeq fixed-step/adaptive callers ignore this return value and require
        # in-place mutation of ``u`` (see ReactionDiffusion3D.callback_step).
        u[0].copy_(n)
        u[1].copy_(m)
        u[2].copy_(s)
        return u

    def callback_step_adjoint(self, t, u, dt):
        """
        Backward pass callback. Mirrors the forward callback_step:
          - Applies tissue mask to adjoint state.
          - Multiplies adjoint of n and m by RT survival fraction at RT events
            (chain rule for the instantaneous multiplicative kill applied in forward).
        """
        mask = self.comp_mask.to(device=u[2].device, dtype=torch.bool)
        adj_y = u[2]

        # Apply tissue mask to all adjoint components
        if adj_y.dim() == 4:
            for c in range(adj_y.shape[0]):
                adj_y[c].mul_(mask)
        elif adj_y.dim() == 5:
            for c in range(adj_y.shape[1]):
                adj_y[:, c].mul_(mask)

        # RT adjoint: multiply adjoint of n (c=0) and m (c=1) by survival fraction
        rt_dose = self._find_rt_dose(t)
        if rt_dose is not None:
            sf = compute_radiotherapy_cell_survival_fraction(
                self.radiotherapy_specification, rt_dose
            )
            if adj_y.dim() == 4:
                adj_y[0].mul_(sf)
                adj_y[1].mul_(sf)
            elif adj_y.dim() == 5:
                adj_y[:, 0].mul_(sf)
                adj_y[:, 1].mul_(sf)

        return u
