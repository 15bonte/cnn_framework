from ...utils.enum import ProjectMethods
from ...utils.readers.utils.projection import Projection
from ..readers.images_reader import ImagesReader
from .dataset_output import DatasetOutput
from .abstract_data_set import AbstractDataSet


class BasicDataSet(AbstractDataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Data sources
        self.input_data_source = ImagesReader(
            [self.data_manager.get_microscopy_image_path],
            [
                [
                    Projection(
                        method=ProjectMethods.Channel,
                        channels=self.params.z_indexes,
                        axis=2,
                    ),
                    Projection(
                        method=ProjectMethods.Channel,
                        channels=self.params.c_indexes,
                        axis=1,
                    ),
                ]
            ],
        )

    def generate_images(self, filename):
        return DatasetOutput(
            input=self.input_data_source.get_image(
                filename, h5_file=self.h5_file, names=self.h5_names
            ),
        )
