from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import torch
import tqdm.auto as tqdm
from torchdiffeq import odeint, odeint_adjoint

from tumortwin.models.base import TumorGrowthModel3D
from tumortwin.solvers.base import ForwardSolver
from tumortwin.utils import days_since_first, timedelta_to_days

# torchdiffeq: ``grid_constructor`` (fixed-step) vs ``step_t`` / ``jump_t`` (adaptive RK).
# ``TorchDiffEqSolver`` maps the same temporal grid (step_size + treatments) to the
# correct option for each family so callbacks and discrete treatment days are honored.
ADAPTIVE_TORCHDIFFEQ_METHODS = frozenset(
    {
        "dopri5",
        "dopri8",
        "tsit5",
        "bosh3",
        "fehlberg2",
        "adaptive_heun",
    }
)


@dataclass
class TorchDiffEqSolverOptions:
    """
    Configuration options for the TorchDiffEqSolver.

    Attributes:
        step_size (timedelta): For **fixed-step** methods (e.g. ``rk4``), the integrator step.
            For **adaptive** methods (e.g. ``dopri5``), the spacing of mandatory internal
            nodes: the same merged grid as for fixed-step (uniform spacing up to this size,
            plus radiotherapy/chemotherapy times) is passed as ``step_t``; the solver may use
            smaller internal steps according to ``rtol`` / ``atol``.
        method (str): The ODE solver method to use (e.g., ``rk4``, ``dopri5``); see
            ``torchdiffeq`` for supported names.
        device (torch.device): The device on which to perform computations (e.g., CPU or GPU).
        use_adjoint (bool): Whether to use the adjoint method for memory-efficient backpropagation.
        rtol (float): Relative tolerance for **adaptive** solvers (passed to ``odeint``).
        atol (float): Absolute tolerance for **adaptive** solvers (passed to ``odeint``).
        ode_options (Optional[dict]): Extra options forwarded to the ``torchdiffeq`` integrator
            (e.g. ``max_num_steps``, ``first_step``). Do not set ``grid_constructor`` here for
            adaptive methods or ``step_t`` here for fixed-step methods; the solver supplies those.
    """

    step_size: timedelta = timedelta(days=2.0)
    method: str = "rk4"
    device: torch.device = torch.device("cpu")
    use_adjoint: bool = True
    rtol: float = 1e-5
    atol: float = 1e-7
    ode_options: Optional[dict] = None


class TorchDiffEqSolver(ForwardSolver):
    """
    ODE-based forward solver using the TorchDiffEq library.

    This solver integrates tumor growth models over specified timepoints
    using advanced ODE solvers and handles both radiotherapy and chemotherapy schedules.

    Fixed-step methods (e.g. ``rk4``) receive a ``grid_constructor`` that merges ``step_size``
    with treatment times. Adaptive methods (see :data:`ADAPTIVE_TORCHDIFFEQ_METHODS`, e.g.
    ``dopri5``) receive the same mesh as ``step_t`` so discrete ``callback_step`` logic and
    tolerances (:attr:`TorchDiffEqSolverOptions.rtol`, ``atol``) work as intended.

    Attributes:
        model (TumorGrowthModel3D): The tumor growth model to solve.
        solver_options (TorchDiffEqSolverOptions): Configuration options for the solver.
    """

    def __init__(
        self, model: TumorGrowthModel3D, solver_options: TorchDiffEqSolverOptions
    ):
        """
        Initializes the TorchDiffEqSolver.

        Args:
            model (TumorGrowthModel3D): The tumor growth model to solve.
            solver_options (TorchDiffEqSolverOptions): Configuration options for the solver.
        """
        self.model = model
        self.solver_options = solver_options

    @staticmethod
    def _interior_grid_times(t: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """Times strictly between integration endpoints (works for increasing or decreasing ``t``)."""
        t_lo = torch.minimum(t[0], t[-1])
        t_hi = torch.maximum(t[0], t[-1])
        mask = (grid > t_lo) & (grid < t_hi)
        return grid[mask]

    def _odeint_options(self, u_initial: torch.Tensor, t: torch.Tensor) -> dict:
        method = self.solver_options.method
        opts = dict(self.solver_options.ode_options or {})
        if method in ADAPTIVE_TORCHDIFFEQ_METHODS:
            opts.pop("grid_constructor", None)
            grid = self.grid_constructor(self.model, u_initial, t)
            step_t = self._interior_grid_times(t, grid)
            if step_t.numel() > 0:
                opts["step_t"] = step_t.to(device=t.device, dtype=t.dtype)
            return opts
        opts.pop("step_t", None)
        opts.pop("jump_t", None)
        opts["grid_constructor"] = self.grid_constructor
        return opts

    def solve(
        self, timepoints: List[datetime], u_initial: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Integrates ``model.forward(t, u)`` with ``torchdiffeq``.

        The state ``u`` may be a single field ``(D, H, W)`` or a coupled stack ``(C, D, H, W)``; the
        same shape is preserved along the time dimension in the output.

        Args:
            timepoints: Wall-clock times at which the solution is defined (first entry is the IC time).
            u_initial: Initial condition, same shape as the model expects (e.g. ``get_initial_state()``).

        Returns:
            ``(t, u)``:
                ``t`` — 1-D tensor, days since ``timepoints[0]``;
                ``u`` — tensor of shape ``(len(t), *u_initial.shape)`` (one state per row in time).

            For analysis or plotting one scalar field (e.g. tumor only in a multi-species model), use
            ``tumortwin.models.extract_trajectory_component(u, component_idx=0)``.
        """
        self.solver_options.device = self.model.device

        self.model.progress_bar = tqdm.tqdm(
            total=days_since_first(timepoints[-1], timepoints[0]),
            desc=f"Forward Simulation: [{timepoints[0]} to {timepoints[-1]} with timestep {timedelta_to_days(self.solver_options.step_size):.2f} days]",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n:.1f}/{total:.1f} days elapsed",
        )

        t = torch.tensor(
            [days_since_first(t, timepoints[0]) for t in timepoints],
            device=self.solver_options.device,
        )

        u_initial = u_initial.to(self.solver_options.device)
        integrator = odeint_adjoint if self.solver_options.use_adjoint else odeint
        u = integrator(
            self.model,
            u_initial,
            t,
            rtol=self.solver_options.rtol,
            atol=self.solver_options.atol,
            method=self.solver_options.method,
            options=self._odeint_options(u_initial, t),
        )
        return t, u

    @staticmethod
    def _schedule_time_to_days(
        event_time,
        *,
        t_initial: datetime,
        model_name: str,
        schedule_name: str,
    ) -> float:
        """
        Convert one schedule timepoint to solver day units.

        Accepts:
            - numeric day values (int/float/tensors that cast to float)
            - ``datetime`` values (requires ``t_initial``)
        """
        if isinstance(event_time, datetime):
            if t_initial is None:
                raise ValueError(
                    f"{model_name}.{schedule_name} contains datetime values but "
                    "model.t_initial is None. Set model.t_initial to the simulation start."
                )
            return timedelta_to_days(event_time - t_initial)
        try:
            return float(event_time)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Unsupported timepoint type in {model_name}.{schedule_name}: "
                f"{type(event_time)!r}. Use datetime or numeric day values."
            ) from exc

    def _extract_treatment_days(self, start_time, end_time) -> Tuple[List[float], List[float]]:
        """
        Collect treatment event days in (start_time, end_time).

        Uses attribute-safe access so PDE-system models without treatment metadata
        still work with the same solver/grid constructor.
        """
        model = self.model
        model_name = model.__class__.__name__
        t_initial = getattr(model, "t_initial", None)

        radiotherapy_days: List[float] = []
        rt_spec = getattr(model, "radiotherapy_specification", None)
        if rt_spec is not None:
            rt_times = list(getattr(rt_spec, "times", []))
            for t_event in rt_times:
                day = self._schedule_time_to_days(
                    t_event,
                    t_initial=t_initial,
                    model_name=model_name,
                    schedule_name="radiotherapy_specification.times",
                )
                if start_time < day < end_time:
                    radiotherapy_days.append(day)

        chemotherapy_days: List[float] = []
        chemo_specs = getattr(model, "chemotherapy_specifications", None) or []
        for spec in chemo_specs:
            spec_times = list(getattr(spec, "times", []))
            for t_event in spec_times:
                day = self._schedule_time_to_days(
                    t_event,
                    t_initial=t_initial,
                    model_name=model_name,
                    schedule_name="chemotherapy_specifications[].times",
                )
                if start_time < day < end_time:
                    chemotherapy_days.append(day)

        return radiotherapy_days, chemotherapy_days

    def grid_constructor(self, func, y0, t) -> torch.Tensor:
        """
        Constructs a grid of timesteps considering treatment schedules.

        Args:
            func: The ODE function (unused in this method but required by the API).
            y0: Initial state of the system (unused in this method but required by the API).
            t: Original list of timepoints requested by the solver.

        Returns:
            torch.Tensor: Tensor containing refined timepoints for integration.
        """
        start_time = t[0]
        end_time = t[-1]

        if end_time < start_time:
            isReverse = True
            start_time, end_time = end_time, start_time
        else:
            isReverse = False

        niters = torch.ceil(
            (end_time - start_time) / timedelta_to_days(self.solver_options.step_size)
            + 1
        ).item()
        solver_times = (
            torch.arange(0, niters, dtype=t.dtype, device=t.device)
            * timedelta_to_days(self.solver_options.step_size)
            + start_time
        )
        solver_times[-1] = end_time

        radiotherapy_days, chemotherapy_days = self._extract_treatment_days(
            start_time, end_time
        )
        radiotherapy_times = torch.tensor(
            radiotherapy_days, dtype=t.dtype, device=self.solver_options.device
        )
        chemotherapy_times = torch.tensor(
            chemotherapy_days, dtype=t.dtype, device=self.solver_options.device
        )

        # Merge all times and refine with steps
        allTimes, _ = torch.sort(
            torch.unique(
                torch.cat((solver_times, radiotherapy_times, chemotherapy_times), dim=0)
            ),
            descending=isReverse,
        )
        return allTimes
