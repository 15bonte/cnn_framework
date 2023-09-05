from abc import abstractmethod
from typing import Optional

import numpy as np
from torch.utils.data import Dataset
from albumentations import Compose

from ..readers.tiff_reader import TiffReader
from .dataset_output import DatasetOutput
from ..enum import NormalizeMethods, ProjectMethods
from ..tools import read_categories_probability_from_name, to_one_hot


class DataSource:
    def __init__(self, functions=[], projections=None, normalizations=None):
        self.functions = functions

        if projections is not None:
            assert len(functions) == len(projections)
            self.projections = projections
        else:
            self.projections = [ProjectMethods.none] * len(functions)

        if normalizations is not None:
            assert len(functions) == len(normalizations)
            self.normalizations = normalizations
        else:
            self.normalizations = [NormalizeMethods.none] * len(functions)

    def is_empty(self):
        return len(self.functions) == 0

    def get_image(self, filename, respect_initial_type=False, axis_to_merge=0):
        if self.is_empty():
            return None
        images = []
        for function, projection, normalization in zip(
            self.functions, self.projections, self.normalizations
        ):
            image_path = function(filename)
            image_reader = TiffReader(
                image_path,
                project=projection,
                normalize=normalization,
                respect_initial_type=respect_initial_type,
            )
            raw_image = image_reader.get_processed_image()

            # Add channel dimension if needed
            while raw_image.ndim < axis_to_merge + 1:
                raw_image = np.expand_dims(raw_image, axis=-1)

            # For global consistency, even projected image should have one channel
            if raw_image.ndim == 2:
                raw_image = np.expand_dims(raw_image, axis_to_merge)

            images.append(raw_image)

        raw_img = np.concatenate(images, axis=axis_to_merge)

        # For global consistency, axis_to_merge (=channels) should be last
        raw_img = np.moveaxis(raw_img, axis_to_merge, -1)  # H, W, C

        return raw_img


class AbstractDataSet(Dataset):
    def __init__(self, is_train, names, data_manager, params):
        # Is train (not val or test)
        self.is_train = is_train
        # Files names
        self.names = names
        # Model parameters
        self.params = params
        # Data manager
        self.data_manager = data_manager
        # Transforms
        self.transforms: Optional[Compose] = None
        # Initialize means and standard deviations as None, will be set afterwards
        self.mean_std = None
        # Create empty data sources
        self.input_data_source = DataSource()
        self.output_data_source = DataSource()
        self.additional_data_source = DataSource()

    def set_transforms(self):
        # No transforms
        pass

    def check_order(self):
        # Check if order of transforms makes sense
        transform_names = [transform.__class__.__name__ for transform in self.transforms]
        # Transform that have to be BEFORE and AFTER Normalize
        before_normalize = ["ColorJitter", "GaussNoise"]  # because they clip to 0-MAX_TYPE_VALUE
        after_normalize = ["GaussianBlur", "Rotate", "PadIfNeeded"]
        # Ignore if Normalize is not in transforms
        if "Normalize" not in transform_names:
            return
        # Check if transforms are in the right order
        normalize_seen = False
        for transform in transform_names:
            if transform == "Normalize":
                normalize_seen = True
            if not normalize_seen:
                assert transform not in after_normalize
            else:
                assert transform not in before_normalize

    def initialize_transforms(self):
        self.set_transforms()
        self.check_order()

    @abstractmethod
    def generate_raw_images(self, filename) -> DatasetOutput:
        raise NotImplementedError

    def read_output(self, filename, one_hot=False):
        """
        Used for classification models.
        Read category and/or probabilities from file name
        """
        # Read category from name
        categories_and_probabilities = filename.split(".")[0].split("_c")[1:]
        category = int(categories_and_probabilities[0])
        # Are there probabilities for classes?
        if len(categories_and_probabilities) > 1 and not one_hot:
            _, probabilities = read_categories_probability_from_name(filename)
        else:
            # Case where class is set to -1
            probabilities = to_one_hot(category, self.params.nb_classes) if category > -1 else None

        if probabilities is None:
            return np.zeros(self.params.nb_classes)

        # Length of probabilities has to be equal to number of classes
        assert self.params.nb_classes == len(probabilities)

        # Category has to be the index of the highest probability
        assert category == np.argmax(probabilities)

        return np.asarray(probabilities)

    @staticmethod
    def pass_image(image_x, image_y, image_z):
        return False

    # shape of inputs in the dataset
    def __len__(self):
        return len(self.names)

    def apply_transforms(self, data_set_output: DatasetOutput) -> None:
        # Transforms are not yet defined
        if self.transforms is None:
            return data_set_output

        # No target or additional images
        if not data_set_output.target_is_image() and not data_set_output.additional_is_image():
            transformed = self.transforms(image=data_set_output.input)
            data_set_output.input = np.moveaxis(transformed["image"], 2, 0)
        # Target and no additional images
        elif data_set_output.target_is_image() and not data_set_output.additional_is_image():
            fake_input = np.concatenate((data_set_output.input, data_set_output.target), axis=-1)
            transformed = self.transforms(image=fake_input)
            # Split image and target
            transformed_input = transformed["image"][:, :, : data_set_output.input.shape[-1]]
            transformed_target = transformed["image"][:, :, data_set_output.input.shape[-1] :]
            data_set_output.input = np.moveaxis(transformed_input, 2, 0)
            data_set_output.target = np.moveaxis(transformed_target, 2, 0)
        # No target and additional images
        elif not data_set_output.target_is_image() and data_set_output.additional_is_image():
            transformed = self.transforms(
                image=data_set_output.input, mask=data_set_output.additional
            )
            data_set_output.input = np.moveaxis(transformed["image"], 2, 0)
            data_set_output.additional = np.moveaxis(transformed["mask"], 2, 0)
        # Target and additional images
        elif data_set_output.target_is_image() and data_set_output.additional_is_image():
            fake_input = np.concatenate((data_set_output.input, data_set_output.target), axis=-1)
            transformed = self.transforms(image=fake_input, mask=data_set_output.additional)
            # Split image and target
            transformed_input = transformed["image"][:, :, : data_set_output.input.shape[-1]]
            transformed_target = transformed["image"][:, :, data_set_output.input.shape[-1] :]
            data_set_output.input = np.moveaxis(transformed_input, 2, 0)
            data_set_output.target = np.moveaxis(transformed_target, 2, 0)
            data_set_output.additional = np.moveaxis(transformed["mask"], 2, 0)

    def __getitem__(self, idx):
        # Read file and generate images
        data_set_output = self.generate_raw_images(self.names[idx])

        # Apply transforms
        self.apply_transforms(data_set_output)
        data_set_output.index = idx

        # Set to torch tensors
        data_set_output.to_torch()

        return data_set_output
