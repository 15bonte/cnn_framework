import albumentations as A
import numpy as np

from ..utils.data_sets.abstract_data_set import AbstractDataSet
from ..utils.readers.images_reader import ImagesReader
from ..utils.data_sets.dataset_output import DatasetOutput
from ..utils.enum import ProjectMethods
from ..utils.readers.utils.projection import Projection


class SimCLRDataSet(AbstractDataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Data sources
        self.input_data_source = ImagesReader(
            [self.data_manager.get_microscopy_image_path],
            [
                [
                    Projection(
                        method=ProjectMethods.Channel,
                        channels=self.params.c_indexes,
                        axis=-1,
                    )
                ]
            ],
        )

    def set_transforms(self):
        height, width = self.params.input_dimensions.to_tuple(False)
        # NB: ColorJitter is is not suited for our dummy data set, but should be used for real data
        self.transforms = A.Compose(
            [
                A.RandomCrop(height=height, width=width, p=1),
                A.HorizontalFlip(),
                A.VerticalFlip(),
                # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
                A.Normalize(
                    self.mean_std["mean"], std=self.mean_std["std"], max_pixel_value=1
                ),
                A.GaussianBlur(),
            ]
        )

    def generate_raw_images(self, filename):
        return DatasetOutput(
            input=self.input_data_source.get_image(filename, axis_to_merge=-1),
        )


class SimCLRDataSetWithoutTransforms(SimCLRDataSet):
    def set_transforms(self):
        height, width = self.params.input_dimensions.to_tuple(False)
        self.transforms = A.Compose(
            [
                A.Normalize(
                    self.mean_std["mean"], std=self.mean_std["std"], max_pixel_value=1
                ),
                A.CenterCrop(height=height, width=width, p=1),
            ]
        )
