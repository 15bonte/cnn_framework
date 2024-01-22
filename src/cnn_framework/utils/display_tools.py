import sys
from matplotlib import pyplot as plt
import numpy as np
import psutil
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from skimage.exposure import equalize_adapthist
import torch
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from pint import UnitRegistry

from .preprocessing import (
    normalize_array,
    zero_one_scaler,
)
from .tools import get_image_type_max


def display_confusion_matrix(
    results,
    class_names,
    save_path=None,
    show=False,
    extension="pdf",
    normalize=None,
):
    (y_true, y_pred) = results

    # Switch to numpy
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()

    # Change labels to int if necessary
    if isinstance(y_true[0], str):
        y_true = np.array([class_names.index(label) for label in y_true])
        y_pred = np.array([class_names.index(label) for label in y_pred])

    m = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        normalize=normalize,
    )

    # Rotate only if names are long
    max_name_length = max([len(name) for name in class_names])
    if max_name_length > 15:
        xticks_rotation = 90
    elif max_name_length > 5:
        xticks_rotation = 45
    else:
        xticks_rotation = 0

    _, ax = plt.subplots(1, 1, figsize=(5, 5))
    ConfusionMatrixDisplay(m, display_labels=class_names).plot(
        ax=ax, cmap="Blues", colorbar=False, xticks_rotation=xticks_rotation
    )

    # Save plot
    if save_path is not None:
        plt.savefig(
            f"{save_path}/confusion_matrix.{extension}",
            transparent=True,
            bbox_inches="tight",
        )

    if show:
        plt.show()


def make_image_matplotlib_displayable(image, mean_std=None):
    """
    Make image displayable with matplotlib.
    """
    if len(image.shape) == 2:
        return image

    # Switch smallest dimension to the beginning to normalize
    channels_axis = np.argmin(image.shape)
    image = np.moveaxis(image, channels_axis, 0)

    # If mean and std are provided, normalize back to initial values
    if mean_std is not None:
        normalized_image = normalize_array(
            image, mean_std=mean_std, reverse=True
        )
        # Round to 6 decimals to avoid floating point errors
        # 6 is chosen arbitrary as enough to see uint16 precision (65535 ~ 1e6)
        # but not too much too avoid floating errors
        normalized_image = np.round(normalized_image, 6)
    else:
        normalized_image = image

    # Normalize between 0 and 1
    normalized_image = normalized_image / get_image_type_max(normalized_image)

    # Enhance contrast
    normalized_image = zero_one_scaler(normalized_image)
    normalized_image = equalize_adapthist(normalized_image)

    # Switch it back to the end for matpltolib
    normalized_image = np.moveaxis(normalized_image, 0, -1)

    # If more than 3 channels, only consider first 3
    if normalized_image.shape[-1] > 3:
        return normalized_image[:, :, :3]

    # Else, add dummy channels as copy of last
    for _ in range(3 - normalized_image.shape[-1]):
        copy_channel = np.copy(normalized_image[:, :, -1])
        normalized_image = np.concatenate(
            [normalized_image, copy_channel[..., None]], axis=-1
        )

    return normalized_image


def make_image_tiff_displayable(image, mean_std):
    """
    Make image displayable before saving in tiff.
    """

    # If only two channels, add dummy channel at beginning
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=0)

    # Switch smallest dimension to the beginning to normalize for tiff
    channels_axis = np.argmin(image.shape)
    image = np.moveaxis(image, channels_axis, 0)

    # If mean and std are provided, normalize back to initial values
    if mean_std is not None and len(mean_std["mean"]) > 0:
        normalized_image = normalize_array(
            image, mean_std=mean_std, reverse=True
        )
    # Else, just let it like that
    else:
        normalized_image = image

    return normalized_image


def display_progress(
    message,
    current,
    total,
    precision=1,
    additional_message="",
    cpu_memory=False,
):
    percentage = round(current / total * 100, precision)
    padded_percentage = str(percentage).ljust(precision + 3, "0")
    display_message = f"\r{message}: {padded_percentage}%"
    # Display additional message
    if additional_message:
        display_message += " | " + additional_message
    # Display CPU memory usage
    if cpu_memory:
        cpu_available = round(
            psutil.virtual_memory().available
            * 100
            / psutil.virtual_memory().total
        )
        cpu_message = f"CPU available: {cpu_available}%"
        display_message += " | " + cpu_message
    sys.stdout.write(display_message)
    sys.stdout.flush()


def generate_size_bar(image_width, ax, scale, unit, bar_width_ratio=1 / 3):
    # Initial bar width
    bar_width = int(image_width * bar_width_ratio)
    # Corresponding value to write
    value = int(bar_width * scale)
    # Round to nearest multiple of 10
    rounded_value = 10 if value < 10 else 10 * round(value / 10)
    # Adapt bar width
    rounded_bar_width = int(bar_width * rounded_value / value)

    # Define current quantity
    ureg = UnitRegistry()
    quantity = rounded_value * ureg(unit)
    # Adapt unit to more human-readable unit
    quantity = quantity.to_compact()

    # Define anchored bar
    fontprops = fm.FontProperties(size=30, weight="bold")
    bar = AnchoredSizeBar(
        ax.transData,
        rounded_bar_width,
        f"{int(quantity.magnitude)} {quantity.units}",
        4,  # right bottom
        sep=20,
        frameon=False,
        color="white",
        borderpad=1,
        fontproperties=fontprops,
        size_vertical=10,
    )

    # Add bar to axis
    ax.add_artist(bar)
