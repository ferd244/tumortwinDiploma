from typing import Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

from tumortwin.types import RadiotherapySpecification, TreatmentTime


def compute_radiotherapy_cell_death_fractions(
    radiotherapy_specification: RadiotherapySpecification,
    alpha: float = 1.0,
    alpha_beta_ratio: float = 10.0,
) -> Dict[TreatmentTime, float]:
    """
    Compute cell death fractions for a given radiotherapy protocol.

    This uses the linear-quadratic model to calculate the fraction of cells killed by each
    dose in the protocol.

    Args:
        radiotherapy_specification (RadiotherapySpecification): Radiotherapy parameters,
            including protocol (dose and times), alpha, and alpha/beta ratio.
        alpha (float, optional): Intrinsic radiosensitivity of cells. Defaults to 1.0.
        alpha_beta_ratio (float, optional): The alpha-beta ratio. Defaults to 10.0.

    Returns:
        Dict[TreatmentTime, float]: A dictionary mapping treatment times to cell survival fractions.
    """
    rt_a = radiotherapy_specification.alpha
    if isinstance(rt_a, torch.Tensor):
        rt_a = float(rt_a.detach().cpu().item())
    beta = alpha / alpha_beta_ratio
    return {
        day: np.exp(-rt_a * (alpha * dose + beta * dose**2))
        for day, dose in radiotherapy_specification.protocol.items()
    }


def compute_radiotherapy_cell_survival_fraction(
    rt: RadiotherapySpecification, dose: float
) -> Union[float, torch.Tensor]:
    """
    Compute the cell survival fraction for a single radiotherapy dose.

    This function uses the linear-quadratic model to compute the survival fraction of cells
    after a given dose of radiation.

    Args:
        rt (RadiotherapySpecification): Radiotherapy parameters, including alpha and
            alpha/beta ratio.
        dose (float): The radiation dose administered.

    Returns:
        Scalar float, or a 0-dim tensor when ``rt.alpha`` is a tensor (e.g. learned).
    """
    beta = rt.alpha / rt.alpha_beta_ratio
    quad = rt.alpha * dose + beta * dose**2
    if isinstance(quad, torch.Tensor):
        return torch.exp(-quad)
    return float(np.exp(-quad))


def plot_radiotherapy(radiotherapy_specification: RadiotherapySpecification) -> None:
    """
    Plot the cell survival fractions for a radiotherapy protocol.

    This function computes and visualizes the cell survival fractions over time
    for the given radiotherapy protocol.

    Args:
        radiotherapy_specification (RadiotherapySpecification): Radiotherapy parameters,
            including protocol (dose and times), alpha, and alpha/beta ratio.
    """
    cell_death_fractions = compute_radiotherapy_cell_death_fractions(
        radiotherapy_specification
    )
    plt.stem(
        list(cell_death_fractions.keys()),
        list(cell_death_fractions.values()),
        label="Radiotherapy survival fractions",
        bottom=1,
    )
    plt.legend()
    plt.xlabel("Treatment Time")
    plt.ylabel("Survival Fraction")
    plt.title("Radiotherapy Cell Survival Fractions")
    plt.show()
