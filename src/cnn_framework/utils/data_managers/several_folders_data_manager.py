import os

from .abstract_data_manager import AbstractDataManager
from .default_data_manager import DefaultDataManager


class SeveralFoldersDataManager(AbstractDataManager):
    """This class is used to manage data from several folders.
    It is useful when the data is split into several folders."""

    def __init__(self, data_set_dir: str, data_manager=DefaultDataManager):
        super().__init__()
        # Iterate over all folders in the data_set_dir and add them to the data manager
        sub_elements = []
        for root, dirs, _ in os.walk(data_set_dir, topdown=True):
            if dirs:
                sub_elements = [os.path.join(root, d) for d in dirs]
            break
        if not sub_elements:
            sub_elements = [data_set_dir]  # case of empty folder
        self.data_managers_container = []
        for sub_folder in sub_elements:
            self.data_managers_container.append(
                {"manager": data_manager(sub_folder)}
            )

    def get_distinct_files(self):
        distinct_files = []
        for dmc in self.data_managers_container:
            dmc_files = dmc["manager"].get_distinct_files()
            # Get files with extension only
            dmc_files = [f for f in dmc_files if "." in f]
            distinct_files += dmc_files
            dmc["files"] = dmc_files
        return distinct_files

    def get_microscopy_image_path(self, file):
        for dmc in self.data_managers_container:
            if file in dmc["files"]:
                return dmc["manager"].get_microscopy_image_path(file)
        raise FileNotFoundError(
            f"File {file} not found in any of the sub directories"
        )
