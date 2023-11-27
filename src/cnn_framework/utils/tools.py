import random
import numpy as np
from albumentations.augmentations.utils import MAX_VALUES_BY_DTYPE
from albumentations.core.transforms_interface import ImageOnlyTransform
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


class UnNormalize(ImageOnlyTransform):
    """
    Unnormalize a tensor image with mean and standard deviation.
    Tken from https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3
    """

    def __init__(
        self,
        mean,
        std,
        always_apply=False,
        p=1.0,
    ):
        super(UnNormalize, self).__init__(always_apply, p)
        self.mean = mean
        self.std = std

    def apply(self, img, **params):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(img, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return img

    def get_transform_init_args_names(self):
        return ("mean", "std")
