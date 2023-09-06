from typing import Optional, List

import numpy as np

from ..readers.tiff_reader import TiffReader
from ..enum import NormalizeMethods, ProjectMethods

from .utils.projection import Projection


class ImagesReader:
    def __init__(
        self,
        functions=None,
        projections: Optional[List[List[Projection]]] = None,
        normalizations=None,
    ):
        self.functions = [] if functions is None else functions

        if projections is not None:
            assert len(self.functions) == len(projections)
            self.projections = projections
        else:
            self.projections = [[Projection(method=ProjectMethods.none)]] * len(self.functions)

        if normalizations is not None:
            assert len(self.functions) == len(normalizations)
            self.normalizations = normalizations
        else:
            self.normalizations = [NormalizeMethods.none] * len(self.functions)

    def is_empty(self):
        return len(self.functions) == 0

    def get_image(self, filename, respect_initial_type=False, axis_to_merge=0):
        if self.is_empty():
            return None
        images = []
        for function, projection, normalization in zip(
            self.functions, self.projections, self.normalizations
        ):
            image_path = function(filename)
            image_reader = TiffReader(
                image_path,
                project=projection,
                normalize=normalization,
                respect_initial_type=respect_initial_type,
            )
            raw_image = image_reader.get_processed_image()

            # Add channel dimension if needed
            while raw_image.ndim < axis_to_merge + 1:
                raw_image = np.expand_dims(raw_image, axis=-1)

            # For global consistency, even projected image should have one channel
            if raw_image.ndim == 2:
                raw_image = np.expand_dims(raw_image, axis_to_merge)

            images.append(raw_image)

        raw_img = np.concatenate(images, axis=axis_to_merge)

        # For global consistency, axis_to_merge (=channels) should be last
        raw_img = np.moveaxis(raw_img, axis_to_merge, -1)  # H, W, C

        return raw_img
