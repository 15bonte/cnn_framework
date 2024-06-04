import os
import h5py

from .abstract_data_manager import AbstractDataManager


class DefaultDataManager(AbstractDataManager):
    def get_distinct_files(self):
        if os.path.isfile(self.data_set_dir):  # h5 file case
            with h5py.File(self.data_set_dir, "r") as h5_file:
                # Get the list of keys
                keys = list(h5_file.keys())
            return keys
        return os.listdir(self.data_set_dir)

    def get_microscopy_image_path(self, file):
        if os.path.isfile(self.data_set_dir):  # h5 file case
            return file
        return os.path.join(self.data_set_dir, file)
