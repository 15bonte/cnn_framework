from abc import abstractmethod
from typing import List
import numpy as np
from bigfish import stack
from skimage import io

from .utils.normalization import Normalization
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
    normalize : Normalization
        Method to normalize the image
    project : List[Projection]
        Method to project the image
    respect_initial_type : bool
        If True, the image will be kept in the same type as the original file.
        If False, the image will be converted to be adapted to torch.
    """

    def __init__(
        self,
        file_path: str,
        normalize: Normalization = Normalization(method=NormalizeMethods.none),
        project: List[Projection] = [Projection(method=ProjectMethods.none)],
        respect_initial_type=False,
    ):
        # Read image from file
        self.image = self._read_image(file_path)
        # Type management
        if not respect_initial_type:
            self.image = handle_image_type(self.image)

        self.respect_initial_type = respect_initial_type

        self.preprocessing_done = False

        self.normalize = normalize
        self.project = project

    def _read_image(self, file_path: str) -> np.ndarray:
        return io.imread(file_path)

    def get_processed_image(self):
        if not self.preprocessing_done:
            # Project & normalize
            self.project_image()
            self.normalize_image()
            self.preprocessing_done = True
        return self.image

    def get_dimensions(self):
        return self.image.shape

    def normalize_image(self):
        if self.normalize.method == NormalizeMethods.none:
            pass
        elif self.normalize.method == NormalizeMethods.ZeroOneScaler:
            self.image = zero_one_scaler(self.image)
        elif self.normalize.method == NormalizeMethods.Standardize:
            self.image = normalize_array(self.image, channel_axis=self.normalize.axis)
        elif self.normalize.method == NormalizeMethods.StandardizeImageNet:
            type_factor = np.iinfo(self.image.dtype).max
            mean_std = {
                0: {"mean": 0.485 * type_factor, "std": 0.229 * type_factor},
                1: {"mean": 0.456 * type_factor, "std": 0.224 * type_factor},
                2: {"mean": 0.406 * type_factor, "std": 0.225 * type_factor},
            }
            self.image = normalize_array(self.image, mean_std) / type_factor
        else:
            raise ValueError("Unknown normalization method")

    def project_image(self):
        for projection in self.project:
            # Apply projection
            if projection.method == ProjectMethods.none:
                pass
            elif projection.method == ProjectMethods.Maximum:
                self.image = self.image.max(axis=projection.axis)
                self.image = np.expand_dims(self.image, projection.axis)
            elif projection.method == ProjectMethods.Mean:
                self.image = self.image.mean(axis=projection.axis).astype(
                    self.image.dtype
                )
                self.image = np.expand_dims(self.image, projection.axis)
            elif projection.method == ProjectMethods.Focus:
                self.image = stack.focus_projection(
                    self.image, proportion=projection.proportion
                )
            elif projection.method == ProjectMethods.Channel:
                self.image = np.take(
                    self.image, projection.channels, axis=projection.axis
                )
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
