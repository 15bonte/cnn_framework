from abc import abstractmethod
import json
from typing import List
import numpy as np
from bigfish import stack
from skimage import io

from .utils.projection import Projection

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
        file_path: str,
        normalize=NormalizeMethods.none,
        project: List[Projection] = [Projection(method=ProjectMethods.none)],
        mean_std_path=None,
        respect_initial_type=False,
    ):
        self.file_path = file_path
        self._image = None
        self.respect_initial_type = respect_initial_type

        self.preprocessing_done = False

        self.normalize = normalize
        self.project = project

        self.file_path = file_path
        self.mean_std_path = mean_std_path

    @property
    def image(self) -> np.ndarray:
        if self._image is not None:
            return self._image

        # Read image from file
        raw_image = io.imread(self.file_path)
        # Type management
        if not self.respect_initial_type:
            raw_image = handle_image_type(raw_image)

        self._image = raw_image
        return self._image

    def get_processed_image(self):
        if not self.preprocessing_done:
            self.preprocess_image()
        return self._image

    def preprocess_image(self):
        # NB: used to be a copy here, do not know why...
        # Project
        for projection in self.project:
            self.project_image(projection)
        # Normalize
        self.normalize_image()
        self.preprocessing_done = True

    def get_dimensions(self):
        return self._image.shape

    def normalize_image(self):
        if self.normalize == NormalizeMethods.none:
            pass
        elif self.normalize == NormalizeMethods.ZeroOneScaler:
            self._image = zero_one_scaler(self._image)
        elif self.normalize == NormalizeMethods.Standardize:
            self._image = normalize_array(self._image, None)
        elif self.normalize == NormalizeMethods.StandardizeImageNet:
            type_factor = np.iinfo(self._image.dtype).max
            mean_std = {
                0: {"mean": 0.485 * type_factor, "std": 0.229 * type_factor},
                1: {"mean": 0.456 * type_factor, "std": 0.224 * type_factor},
                2: {"mean": 0.406 * type_factor, "std": 0.225 * type_factor},
            }
            self._image = normalize_array(self._image, mean_std) / type_factor
        elif self.normalize == NormalizeMethods.CustomStandardize:
            with open(self.mean_std_path, "r") as points_file:
                mean_std = json.load(points_file)
            self._image = normalize_array(self._image, mean_std)
        else:
            raise ValueError("Unknown normalization method")

    def project_image(self, projection: Projection):
        # Apply projection
        if projection.method == ProjectMethods.none:
            pass
        elif projection.method == ProjectMethods.Maximum:
            self._image = self._image.max(axis=projection.axis)
        elif projection.method == ProjectMethods.Mean:
            self._image = self._image.mean(axis=projection.axis).astype(self._image.dtype)
        elif projection.method == ProjectMethods.Focus:
            self._image = stack.focus_projection(self._image, proportion=projection.proportion)
        elif projection.method == ProjectMethods.Channel:
            self._image = np.take(self._image, projection.channels, axis=projection.axis).squeeze()
        else:
            raise ValueError("Unknown projection method")

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
