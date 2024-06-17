import numpy as np

from .abstract_reader import AbstractReader


class H5Reader(AbstractReader):
    """
    Class to read image from h5 file.
    """

    def __init__(self, h5_file, names, *args, **kwargs):
        self.h5_file = h5_file
        self.names = names
        super().__init__(*args, **kwargs)

    def _read_image(self, file_path: str) -> np.ndarray:
        position = self.names[file_path]
        image = self.h5_file["images"][position][()]
        return image
