import torch
import torch.nn as nn


class TumorGrowthModel3D(nn.Module):
    """
    Base class for 3D tumor growth / reaction–diffusion models integrated with torchdiffeq.

    State convention:
        - Single PDE: ``u`` is ``(D, H, W)`` (legacy) or ``(1, D, H, W)`` if stacked.
        - Systems: ``u`` is ``(C, D, H, W)`` with ``C`` coupled scalar fields; optional
          batch dimension ``(B, C, D, H, W)``.

    Subclasses implement ``forward(t, u) -> du/dt`` with the same shape as ``u``.
    """

    def __init__(self):
        """
        Initializes the base class.

        This constructor calls the parent PyTorch `nn.Module` initializer.
        """
        super().__init__()

    @property
    def num_state_components(self) -> int:
        """Number of scalar PDE unknowns in the stacked state (default: single field)."""
        return 1

    def forward(self, t: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass for the tumor growth model.

        Subclasses must implement this method to specify the model's behavior
        during the forward pass.

        Args:
            t (torch.Tensor): A tensor representing time points, typically of shape `(batch_size, 1)`.
            u (torch.Tensor): A tensor representing the input state, such as tumor properties
                or environmental variables, typically of shape `(batch_size, ...)`.

        Returns:
            du_dt (torch.Tensor): A tensor representing the computed output state, typically of the same shape as `u`.

        Raises:
            NotImplementedError: If this method is not implemented in a subclass.
        """
        raise NotImplementedError
