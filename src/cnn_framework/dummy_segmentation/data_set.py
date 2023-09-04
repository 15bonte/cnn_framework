import albumentations as A

from ..utils.data_sets.DatasetOutput import DatasetOutput
from ..utils.enum import ProjectMethods
from ..utils.data_sets.AbstractDataSet import AbstractDataSet, DataSource


class DummyDataSet(AbstractDataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Data sources
        self.input_data_source = DataSource(
            [self.data_manager.get_microscopy_image_path],
            [(ProjectMethods.Channel, ([0, 1, 2], 2))],
        )

        # First channel is always 255
        self.output_data_source = DataSource(
            [self.data_manager.get_microscopy_image_path],
            [(ProjectMethods.Channel, ([0], 2))],
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
        )
