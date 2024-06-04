import numpy as np
import h5py

from .abstract_reader import AbstractReader


class H5Reader(AbstractReader):
    """
    Class to read image from h5 file.
    """

    def _read_image(self, file_path: str) -> np.ndarray:
        h5_file, dataset = file_path.split(".h5")
        with h5py.File(h5_file + ".h5", "r") as f:
            image = f[dataset][()]
        return image
