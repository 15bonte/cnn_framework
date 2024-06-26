from typing import Optional, List

import numpy as np

from ..readers.tiff_reader import TiffReader
from ..readers.h5_reader import H5Reader
from ..enum import NormalizeMethods, ProjectMethods

from .utils.projection import Projection
from .utils.normalization import Normalization


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
            self.projections = [
                [Projection(method=ProjectMethods.none)]
            ] * len(self.functions)

        if normalizations is not None:
            assert len(self.functions) == len(normalizations)
            self.normalizations = normalizations
        else:
            self.normalizations = [
                Normalization(method=NormalizeMethods.none)
            ] * len(self.functions)

    def is_empty(self):
        return len(self.functions) == 0

    def get_image(
        self,
        filename,
        respect_initial_type=False,
        axis_to_merge=1,
        for_training=True,
        h5_file=None,
        names=None,
    ):
        """
        Expected dimensions order is TCZYX. Hence, axis to merge is usually 1 for channels.
        """
        if self.is_empty():
            return None

        try:
            images = []
            for function, projection, normalization in zip(
                self.functions, self.projections, self.normalizations
            ):
                image_path = function(filename)

                if h5_file is not None:
                    image_reader = H5Reader(
                        h5_file,
                        names,
                        image_path,
                        project=projection,
                        normalize=normalization,
                        respect_initial_type=respect_initial_type,
                    )
                else:
                    image_reader = TiffReader(
                        image_path,
                        project=projection,
                        normalize=normalization,
                        respect_initial_type=respect_initial_type,
                    )

                raw_image = image_reader.get_processed_image()

                if raw_image.ndim < 5:
                    print("Old behavior. Should be investigated.")

                    # Add channel dimension if needed
                    while raw_image.ndim < axis_to_merge + 1:
                        raw_image = np.expand_dims(raw_image, axis=-1)

                    # For global consistency, even projected image should have one channel
                    if raw_image.ndim == 2:
                        raw_image = np.expand_dims(raw_image, axis_to_merge)

                images.append(raw_image)

            concatenated_image = np.concatenate(images, axis=axis_to_merge)

            # For global consistency, axis_to_merge (=channels) should be last
            # raw_img = np.moveaxis(raw_img, axis_to_merge, -1)  # H, W, C

            if for_training:  # need to match YXC
                # From here, expect STCZYX or TCZYX format
                # STCZ or TCZ formats will be merged in one single dimension named C
                new_dim = np.prod(concatenated_image.shape[:-2])
                concatenated_image = concatenated_image.reshape(
                    new_dim, *concatenated_image.shape[-2:]
                )  # CYX
                # To be sure that C is first dimension so far
                channel_dim = np.argmin(concatenated_image.shape)
                concatenated_image = np.moveaxis(
                    concatenated_image, channel_dim, -1
                )  # YXC

            return concatenated_image

        except IndexError:  # handle indexes out of bound
            return None
