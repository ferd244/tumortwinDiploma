from datetime import datetime
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes

from tumortwin.postprocessing.total_cell_count import compute_total_cell_count
from tumortwin.types.base import BasePatientData
from tumortwin.utils import days_since_first, find_best_slice


def overlay_cellularity_on_t1(
    cellularity: np.ndarray, t1: np.ndarray, threshold: float
):
    t1_min, t1_max = float(np.min(t1)), float(np.max(t1))
    if t1_max > t1_min:
        normalized_t1 = (t1 - t1_min) / (t1_max - t1_min)
    else:
        normalized_t1 = np.zeros_like(t1, dtype=np.float64)
    t1_rgb = np.stack(
        [normalized_t1] * 3, axis=-1
    )  # Convert grayscale to 3-channel RGB
    cellularity_colored = plt.cm.viridis(cellularity)[
        :, :, :3
    ]  # Apply colormap and remove alpha
    mask = cellularity >= threshold
    blended_image = t1_rgb.copy()
    blended_image[mask] = cellularity_colored[
        mask
    ]  # Replace only in high-cellularity areas
    blended_image = (blended_image * 255).astype(np.uint8)
    return blended_image


def plot_cellularity_map(
    solution: torch.Tensor,
    patient_data: BasePatientData,
    ax: Optional[Axes] = None,
    time: Optional[float] = None,
    threshold: float = 0.01,
    sum_component_indices: Optional[Sequence[int]] = None,
):
    """
    Plot a single 3D cellularity field (``(D, H, W)``). For coupled PDE trajectories
    ``(T, C, D, H, W)``, pass ``extract_trajectory_component(trajectory, 0)[i]`` or
    ``trajectory[i, 0]`` for the proliferating compartment at time ``i``.

    If a 4D field ``(C, D, H, W)`` is passed with ``sum_component_indices is None``, only
    component ``C=0`` is used. For ``HemoInvasion3D`` totals aligned with ADC-derived
    cellularity pass ``sum_component_indices=(0, 1)`` to plot ``n + m``.
    """
    arr = solution.detach().cpu().numpy()
    if arr.ndim == 4:
        if sum_component_indices is not None:
            ix = tuple(int(i) for i in sum_component_indices)
            arr = arr[ix, ...].sum(axis=0)
        else:
            arr = arr[0]
    elif arr.ndim != 3:
        raise ValueError(
            "plot_cellularity_map expects a 3D field (D, H, W) or 4D (C, D, H, W); "
            f"got shape {solution.shape}"
        )

    slice_id = find_best_slice(arr)
    slice_id = int(np.clip(slice_id, 0, arr.shape[2] - 1))

    cellularity_image = arr[:, :, slice_id]
    t1_vol = patient_data.T1_post_image.array
    t1_slice = int(np.clip(slice_id, 0, t1_vol.shape[2] - 1))
    t1_image = t1_vol[:, :, t1_slice]  # T1 image (grayscale)
    blended_image = overlay_cellularity_on_t1(
        cellularity=cellularity_image, t1=t1_image, threshold=threshold
    )

    # Create figure and subplots
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    # RGB uint8 from overlay; do not pass vmin/vmax (they are for scalar data, not 0–255 RGB).
    ax.imshow(np.asarray(blended_image))

    # Titles with Times New Roman font
    if time is not None:
        ax.set_title(
            f"t = {time}",
        )

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def plot_predicted_TCC(
    predicted_cellularity_maps: List[torch.Tensor],
    timepoints: List[datetime],
    ax: Optional[Axes] = None,
    color: str = "k",
    alpha: float = 1.0,
    carrying_capacity: float = 5062500,
):
    """
    Each element of ``predicted_cellularity_maps`` must be a single spatial field ``(D, H, W)``.
    If you have a full trajectory ``(T, C, D, H, W)``, either map with
    ``extract_trajectory_component(trajectory, 0)`` or, for ``HemoInvasion3D`` totals matching
    measured cellularity (``n + m``), use ``extract_trajectory_sum_components(trajectory, (0, 1))``
    or :func:`~tumortwin.pde_workflow.trajectory_to_map_list` with
    ``sum_component_indices=(0, 1)``.
    """
    predicted_cell_counts = [
        compute_total_cell_count(N, carrying_capacity)
        for N in predicted_cellularity_maps
    ]
    # plt.subplots(figsize=(7, 3.5))

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.plot(
        [days_since_first(t, timepoints[0]) for t in timepoints],
        [p.detach() for p in predicted_cell_counts],
        color=color,
        alpha=alpha,
    )

    ax.set_title("Total tumor cell count")
    ax.set_xlabel("Days since first image")
    ax.set_ylabel("Total tumor cell count")
    return ax
