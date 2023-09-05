import os

from .abstract_data_manager import AbstractDataManager


class DefaultDataManager(AbstractDataManager):
    def get_distinct_files(self):
        return os.listdir(self.data_set_dir)

    def get_microscopy_image_path(self, file):
        return os.path.join(self.data_set_dir, file)
