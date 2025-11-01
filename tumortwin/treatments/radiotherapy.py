from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

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
    beta = alpha / alpha_beta_ratio
    return {
        day: np.exp(-radiotherapy_specification.alpha * (alpha * dose + beta * dose**2))
        for day, dose in radiotherapy_specification.protocol.items()
    }

def compute_radiotherapy_cell_proliferation(
        rt: RadiotherapySpecification, radiotherapy_days: dict[float, float], time: float
) -> float:
    """
      Compute the reduction of proliferation due to cumulative dose of radiotherapy

      This function uses the linear-quadratic model to compute the survival fraction, which we assume relates to what
      proportion of the remaining cells are able to actively proliferate

      Args:
          rt (RadiotherapySpecification): Radiotherapy parameters, including alpha_lt {long term effects} and
              alpha/beta ratio.
          radiotherapy_days (dict): lists time and dose of radiotherapy delivered
           time (float): current simulation time

      Returns:
          float: The fraction of the tumor capable of proliferating
      """

    past_doses = [dose for day, dose in radiotherapy_days.items() if day <= time]
    if not past_doses:
        return 1.0
    beta = rt.alpha_proliferation / rt.alpha_beta_ratio
    survival_fractions = [
        np.exp(-(rt.alpha_proliferation * dose + beta * dose ** 2))
        for dose in past_doses
    ]

    # The total survival is the *product* of all individual survival fractions
    total_survival = np.prod(survival_fractions)
    return total_survival

def compute_radiotherapy_cell_death(
        rt: RadiotherapySpecification, radiotherapy_days: dict[float, float], time: float
) -> float:
    """
      Compute the increase of tumor death rate due to cumulative dose of radiotherapy

      This function uses the linear-quadratic model to compute the survival fraction, which we assume relates to the death rate

      Args:
          rt (RadiotherapySpecification): Radiotherapy parameters, including alpha_lt {long term effects} and
              alpha/beta ratio.
          radiotherapy_days (dict): lists time and dose of radiotherapy delivered
           time (float): current simulation time

      Returns:
          float: The fraction of the tumor that is dying
      """

    past_doses = [dose for day, dose in radiotherapy_days.items() if day <= time]
    if not past_doses:
        return 0.0
    beta = rt.alpha_death / rt.alpha_beta_ratio
    survival_fractions = [
        np.exp(-(rt.alpha_death * dose + beta * dose ** 2))
        for dose in past_doses
    ]

    total_kill_fraction = 1.0 - np.prod(survival_fractions)
    return total_kill_fraction

def compute_radiotherapy_cell_survival_fraction(rt: RadiotherapySpecification, dose: float
) -> float:
    """
    Compute the cell survival fraction for a single radiotherapy dose.

    This function uses the linear-quadratic model to compute the survival fraction of cells
    after a given dose of radiation.

    Args:
        rt (RadiotherapySpecification): Radiotherapy parameters, including alpha and
            alpha/beta ratio.
        dose (float): The radiation dose administered.

    Returns:
        float: The fraction of cells surviving the dose.
    """
    beta = rt.alpha / rt.alpha_beta_ratio
    return np.exp(-(rt.alpha * dose + beta * dose**2))


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
