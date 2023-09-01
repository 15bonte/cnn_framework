import albumentations as A
import numpy as np

from ..utils.data_sets.AbstractDataSet import AbstractDataSet, DataSource
from ..utils.data_sets.DatasetOutput import DatasetOutput
from ..utils.enum import NormalizeMethods, ProjectMethods


class DummyRegressionCnnDataSet(AbstractDataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Data sources
        nb_channels = self.params.nb_modalities
        self.input_data_source = DataSource(
            [self.data_manager.get_microscopy_image_path],
            [(ProjectMethods.Channel, (list(range(nb_channels)), 2))],
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
                    # A.Rotate(border_mode=0, p=1, value=1),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.GaussianBlur(),
                ]
            )
        else:
            self.transforms = A.Compose(
                [
                    A.Normalize(
                        mean=self.mean_std["mean"], std=self.mean_std["std"], max_pixel_value=1
                    ),
                    A.PadIfNeeded(min_height=height, min_width=width, border_mode=0, value=0, p=1),
                    A.CenterCrop(height=height, width=width, p=1),
                ]
            )

    def generate_raw_images(self, filename):
        # Output
        input_image = self.input_data_source.get_image(filename, axis_to_merge=2)
        non_zero_area = np.count_nonzero(input_image) / 3

        return DatasetOutput(
            input=input_image,
            target_array=np.array([non_zero_area]),
            additional=self.additional_data_source.get_image(filename, axis_to_merge=2),
        )