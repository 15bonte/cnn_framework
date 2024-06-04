import numpy as np

from .abstract_reader import AbstractReader


class H5Reader(AbstractReader):
    """
    Class to read image from h5 file.
    """

    def __init__(self, h5_file, *args, **kwargs):
        self.h5_file = h5_file
        super().__init__(*args, **kwargs)

    def _read_image(self, file_path: str) -> np.ndarray:
        image = self.h5_file[file_path][()]
        return image
