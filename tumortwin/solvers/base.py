from datetime import datetime
from typing import List, Tuple

import torch


class ForwardSolver:
    """
    Abstract base class for forward solvers in tumor growth modeling.

    Integrates the model state (single field or stacked PDE system) forward in time.

    Methods:
        solve(timepoints, u_initial):
            Returns ``(t, u)`` tensors per the concrete solver (see ``TorchDiffEqSolver``).

    Raises:
        NotImplementedError: If a subclass does not implement the `solve` method.
    """

    def solve(
        self, timepoints: List[datetime], u_initial: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate the model from ``u_initial`` through the given wall-clock ``timepoints``.

        Args:
            timepoints: Integration output times (datetime).
            u_initial: Initial state; for a single PDE field typically ``(D, H, W)``, for a stacked
                system ``(C, D, H, W)`` (see ``PDESystemModel3D`` / ``ImmuneResponse3D``).

        Returns:
            ``(t, u)`` both tensors: ``t`` is 1-D (days since first timepoint); ``u`` stacks the state
            along time with shape ``(len(t), *u_initial.shape)`` (``torchdiffeq`` convention).

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError
