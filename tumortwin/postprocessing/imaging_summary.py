import matplotlib.pyplot as plt
import numpy as np

from tumortwin.types.base import BasePatientData
from tumortwin.types.imaging import NibabelNifti
from tumortwin.utils import find_best_slice

plt.rcParams["figure.dpi"] = 300  # Adjust as needed (e.g., 600 for print quality)
plt.rcParams["savefig.dpi"] = 300  # For saving figures


def plot_imaging_summary(patient_data: BasePatientData):
    """
    Generates a summary figure for a given BasePatientData object.

    Each column represents a visit, and each row represents an imaging modality.

    Args:
        patient_data (BasePatientData): The patient data object to visualize.
    """
    num_cols = len(patient_data.visits)
    num_rows = 2

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 1.5))
    # fig, axes = plt.subplots(num_rows, num_cols, figsize=(7,2))

    # Ensure axes is always a 2D array for consistency
    if num_rows == 1:
        axes = np.array([axes])
    if num_cols == 1:
        axes = axes[:, np.newaxis]

    if num_cols == 0:
        print("No visits found in patient data.")
        return
    anatomic_mask = None
    anatomic_image = None
    adc = None
    roi = None

    for attr in dir(patient_data):
        if "mask" in attr and isinstance(
            getattr(patient_data, attr, None), (NibabelNifti, type(None))
        ):
            anatomic_mask = getattr(patient_data, attr, None)

        if "T1" in attr and isinstance(
            getattr(patient_data, attr, None), (NibabelNifti, type(None))
        ):
            anatomic_image = getattr(patient_data, attr, None)

    for col_idx, visit in enumerate(patient_data.visits):
        adc = None
        roi = None
        for attr in dir(visit):
            if "adc" in attr and isinstance(
                getattr(patient_data, attr, None), (NibabelNifti, type(None))
            ):
                adc = getattr(visit, attr, None)

            if "roi_enhance" in attr and isinstance(
                getattr(visit, attr, None), (NibabelNifti, type(None))
            ):
                roi = getattr(visit, attr, None)

        if col_idx == 0:
            ref = roi.array if roi is not None else adc.array if adc is not None else None
            if ref is None:
                print("No ADC/ROI on first visit for slice selection; skipping imaging summary.")
                plt.close(fig)
                return
            best_slice = find_best_slice(ref)

        if adc is not None:
            if anatomic_image is not None:
                axes[0, col_idx].imshow(
                    anatomic_image.array[:, :, best_slice], cmap="gray", aspect="auto"
                )

            axes[1, col_idx].imshow(
                adc.array[:, :, best_slice], cmap="jet", aspect="auto"
            )

            if anatomic_mask is not None:
                axes[0, col_idx].contour(
                    anatomic_mask.array[:, :, best_slice], colors="blue"
                )
                axes[1, col_idx].contour(
                    anatomic_mask.array[:, :, best_slice], colors="blue"
                )

            if roi is not None:
                axes[0, col_idx].contour(roi.array[:, :, best_slice], colors="red")
                axes[1, col_idx].contour(roi.array[:, :, best_slice], colors="red")

        axes[0, col_idx].set_title(f"Visit {col_idx + 1}")
        axes[0, 0].set_ylabel("T1")
        axes[1, 0].set_ylabel("ADC")

        for row_idx in range(num_rows):
            axes[row_idx, col_idx].axis("equal")
            axes[row_idx, col_idx].tick_params(
                left=False, labelleft=False, labelbottom=False, bottom=False
            )
            for spine in axes[row_idx, col_idx].spines.values():
                spine.set_visible(False)

    plt.tight_layout()
    plt.show()
