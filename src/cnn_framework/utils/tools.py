import random
import numpy as np
from albumentations.augmentations.utils import MAX_VALUES_BY_DTYPE
import fnmatch
import torch


CONSTANT_SEEDED_RD = random.Random(10)
BUFFER_RANDOM = None
BUFFER_RANGE = 0


def random_sample(range_sample, nb_sample):
    global BUFFER_RANGE
    global BUFFER_RANDOM

    if (
        BUFFER_RANDOM is None
        or nb_sample != len(BUFFER_RANDOM)
        or range_sample != BUFFER_RANGE
    ):
        BUFFER_RANGE = range_sample
        BUFFER_RANDOM = CONSTANT_SEEDED_RD.sample(range_sample, nb_sample)
    return BUFFER_RANDOM


# function to generate one hot encoded vector from a label
def to_one_hot(label, nb_classes):
    """
    label in [0, nb_classes-1]
    """
    one_hot_encoded = np.zeros(nb_classes)
    one_hot_encoded[label] = 1
    return one_hot_encoded


def read_categories_probability_from_name(filename):
    categories_and_probabilities = filename.split(".")[0].split("_c")[1:]
    category = int(categories_and_probabilities[0])
    probabilities = []
    for i in range(len(categories_and_probabilities) - 1):
        current_category_and_probability = categories_and_probabilities[i + 1]
        (
            current_category,
            current_probability,
        ) = current_category_and_probability.split("_")[:2]
        # Probabilities have to be given in order
        assert int(current_category) == i
        probabilities.append(int(current_probability) / 100)
    return category, probabilities


def handle_image_type(image):
    """
    Adapt image type to be used by torchvision.

    float32 -> make sure values are between 0 and 1.0
    float64 -> convert to float32

    uint8 -> does nothing
    uint16 -> convert to float32 as torchvision does not support uint16
    """

    if image.dtype == np.float64:
        image = image.astype("float32")
    if image.dtype == np.float32:
        if image.max() > 1.0:
            print("WARNING: float image should have values between 0 and 1.0")
    elif image.dtype == np.uint16:
        coef = MAX_VALUES_BY_DTYPE[image.dtype]
        image = image.astype(np.float32) * (1 / coef)
    elif image.dtype == np.uint8:  # to run VAE
        coef = MAX_VALUES_BY_DTYPE[image.dtype]
        image = image.astype(np.float32) * (1 / coef)

    return image


def extract_patterns(files, patterns):
    """Return files that match any of the given patterns."""
    result = []
    for name in files:
        for pattern in patterns:
            if fnmatch.fnmatch(name, pattern):
                result.append(name)
                break
    return result


def get_image_type_max(image):
    image_max_value = image.max()
    # Float case
    if image_max_value <= 1.0:
        return 1.0
    # Integer case
    possible_types = [np.uint8, np.uint16]
    for image_type in possible_types:
        if image_max_value > np.iinfo(image_type).max:
            continue
        return float(np.iinfo(image_type).max)
    raise ValueError(
        "Image max value is too high to be encoded in uint8 or uint16"
    )


def torch_from_numpy(np_array):
    """
    Converts a numpy array to a torch tensor.
    """
    if np_array is None:
        return None
    # uint16 not supported by torch.from_numpy
    if np_array.dtype == np.uint16:
        np_array = np_array.astype(np.int32)
    return torch.from_numpy(np_array)


def get_padding_coordinates(image) -> list[int]:
    """
    Automatically detect the padding of a torch image and return the coordinates.
    """
    same_rows = torch.all(image == image[0, :], axis=1)
    same_rows_index = torch.where(same_rows == False)[0]
    y_min, y_max = same_rows_index[0], same_rows_index[-1] + 1
    if y_min == 1:  # if only one column is different, no padding
        y_min = 0

    same_cols = torch.all(torch.transpose(image, 0, 1) == image[:, 0], axis=1)
    same_cols_index = torch.where(same_cols == False)[0]
    x_min, x_max = same_cols_index[0], same_cols_index[-1] + 1
    if x_min == 1:  # if only one row is different, no padding
        x_min = 0

    return y_min, y_max, x_min, x_max
