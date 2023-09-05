from abc import abstractmethod
import json
import numpy as np
from bigfish import stack
from skimage import io

from ..enum import NormalizeMethods, ProjectMethods
from ..preprocessing import (
    zero_one_scaler,
    normalize_array,
)
from ..tools import handle_image_type


class AbstractReader:
    """
    Base class for readers

    Parameters
    ----------
    file_path : str
        Path to the file to read
    normalize : NormalizeMethods
        Method to normalize the image
    project : ProjectMethods
        Method to project the image
        Can be a either a method, or a tuple (method, parameter), or a list of tuples
        Parameters are:
            - Maximum: int, dimension to project
            - Mean: int, dimension to project
            - Focus: int, dimension to project
            - Channel: ([int], int) list of channels to project, dimension to project
    mean_std_path : str
        Path to the mean and std file
    respect_initial_type : bool
        If True, the image will be kept in the same type as the original file.
        If False, the image will be converted to be adapted to torch.
    """

    def __init__(
        self,
        file_path,
        normalize=NormalizeMethods.none,
        project=ProjectMethods.none,
        mean_std_path=None,
        respect_initial_type=False,
    ):
        raw_image = io.imread(file_path)

        # Type management
        if not respect_initial_type:
            raw_image = handle_image_type(raw_image)
        self.image = raw_image

        self.preprocessing_done = False

        self.normalize = normalize
        if isinstance(project, list):
            self.project = project
        else:
            self.project = [project]

        self.file_path = file_path

        self.mean_std_path = mean_std_path

    def get_dimensions(self):
        return self.image.shape

    def normalize_image(self):
        if self.normalize == NormalizeMethods.none:
            pass
        elif self.normalize == NormalizeMethods.ZeroOneScaler:
            self.image = zero_one_scaler(self.image)
        elif self.normalize == NormalizeMethods.Standardize:
            self.image = normalize_array(self.image, None)
        elif self.normalize == NormalizeMethods.StandardizeImageNet:
            type_factor = np.iinfo(self.image.dtype).max
            mean_std = {
                0: {"mean": 0.485 * type_factor, "std": 0.229 * type_factor},
                1: {"mean": 0.456 * type_factor, "std": 0.224 * type_factor},
                2: {"mean": 0.406 * type_factor, "std": 0.225 * type_factor},
            }
            self.image = normalize_array(self.image, mean_std) / type_factor
        elif self.normalize == NormalizeMethods.CustomStandardize:
            with open(self.mean_std_path, "r") as points_file:
                mean_std = json.load(points_file)
            self.image = normalize_array(self.image, mean_std)
        else:
            raise ValueError("Unknown normalization method")

    def project_image(self, method):
        # Discriminate between projection method and projection with parameter
        if isinstance(method, tuple):
            projection_method, parameter = method
        else:
            projection_method, parameter = method, None
        # Apply projection
        if projection_method == ProjectMethods.none:
            pass
        elif projection_method == ProjectMethods.Maximum:
            axis = parameter if parameter is not None else 0
            self.image = self.image.max(axis=axis)
        elif projection_method == ProjectMethods.Mean:
            axis = parameter if parameter is not None else 0
            self.image = self.image.mean(axis=axis).astype(self.image.dtype)
        elif projection_method == ProjectMethods.Focus:
            proportion = parameter if parameter is not None else 1
            self.image = stack.focus_projection(self.image, proportion=proportion)
        elif projection_method == ProjectMethods.Channel:
            if isinstance(parameter[0], list):  # new channels management
                channels, axis = parameter
            else:
                channels, axis = parameter, 0
            self.image = np.take(self.image, channels, axis=axis).squeeze()
        else:
            raise ValueError("Unknown projection method")

    def preprocess_image(self):
        # NB: used to be a copy here, do not know why...
        # Project
        for method in self.project:
            self.project_image(method)
        # Normalize
        self.normalize_image()
        self.preprocessing_done = True

    def get_processed_image(self):
        if not self.preprocessing_done:
            self.preprocess_image()
        return self.image

    @abstractmethod
    def display_info(
        self,
        unit=None,
        scale=None,
        save_path="",
        dimensions=None,
        show=True,
        verbose=True,
    ):
        pass
