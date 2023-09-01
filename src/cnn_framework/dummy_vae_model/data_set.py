import albumentations as A
import numpy as np

from pythae.data.datasets import DatasetOutput as DatasetOutputVAE

from ..utils.data_sets.DatasetOutput import DatasetOutput
from ..utils.enum import NormalizeMethods, ProjectMethods
from ..utils.data_sets.AbstractDataSet import AbstractDataSet, DataSource


class DummyVAEDataSet(AbstractDataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Data sources
        self.input_data_source = DataSource(
            [self.data_manager.get_microscopy_image_path],
            [(ProjectMethods.Channel, ([0], 2))],
            [NormalizeMethods.none],
        )

        self.output_data_source = DataSource(
            [self.data_manager.get_microscopy_image_path],
            [(ProjectMethods.Channel, ([0, 1, 2], 2))],
            [NormalizeMethods.none],
        )

        # Mask
        self.additional_data_source = DataSource(
            [self.data_manager.get_microscopy_image_path],
            [(ProjectMethods.Channel, ([0], 2))],
            [NormalizeMethods.none],
        )

    def set_transforms(self):
        height, width = self.params.input_dimensions.to_tuple(False)
        if self.is_train:
            self.transforms = A.Compose(
                [
                    A.Normalize(
                        self.mean_std["mean"], std=self.mean_std["std"], max_pixel_value=1
                    ),
                    A.PadIfNeeded(min_height=height, min_width=width, border_mode=0, value=0, p=1),
                    A.CenterCrop(height=height, width=width, p=1),
                    A.Rotate(border_mode=0),
                    A.HorizontalFlip(),
                    A.VerticalFlip(),
                ]
            )
        else:
            self.transforms = A.Compose(
                [
                    A.Normalize(
                        self.mean_std["mean"], std=self.mean_std["std"], max_pixel_value=1
                    ),
                    A.PadIfNeeded(min_height=height, min_width=width, border_mode=0, value=0, p=1),
                    A.CenterCrop(height=height, width=width, p=1),
                ]
            )

    def generate_raw_images(self, filename):
        return DatasetOutput(
            input=self.input_data_source.get_image(filename, axis_to_merge=2),
            target_image=self.output_data_source.get_image(filename, axis_to_merge=2),
            additional=self.additional_data_source.get_image(filename, axis_to_merge=2),
        )

    def __getitem__(self, idx):
        # Read file and generate images
        filename = self.names[idx]
        raw_inputs = self.generate_raw_images(filename)

        # Category (0 for squares, 1 for circles)
        one_hot_probabilities = self.read_output(filename, one_hot=True)
        if np.max(one_hot_probabilities) == 0:
            category = -1
        else:
            category = np.argmax(one_hot_probabilities)

        # Apply transforms
        self.apply_transforms(raw_inputs)

        return DatasetOutputVAE(
            data=raw_inputs.input,
            target=raw_inputs.target,
            id=idx,
            mask=raw_inputs.additional,
            category=category,
        )