import json
import os
import h5py

from .abstract_data_manager import AbstractDataManager


class DefaultDataManager(AbstractDataManager):
    def get_distinct_files(self):
        if os.path.isfile(self.data_set_dir):  # h5 file case
            # Get json path
            names_json_path = os.path.join(
                os.path.dirname(self.data_set_dir), "names.json"
            )
            assert os.path.isfile(
                names_json_path
            ), f"Required {names_json_path} does not exist"
            # Load json
            with open(names_json_path, "r") as f:
                h5_names = json.load(f)
            return list(h5_names.keys())
        return os.listdir(self.data_set_dir)

    def get_microscopy_image_path(self, file):
        if os.path.isfile(self.data_set_dir):  # h5 file case
            return file
        return os.path.join(self.data_set_dir, file)
