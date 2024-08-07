from typing import Optional
import math
import matplotlib.pyplot as plt
import numpy as np
from aicsimageio import AICSImage

from .abstract_reader import AbstractReader
from ..display_tools import generate_size_bar


class TiffReader(AbstractReader):
    """
    Class to read Tiff file.
    Handles 2D or 3D images, coded on 8 or 16bit.
    """

    def _read_image(self, file_path: str) -> np.ndarray:
        aics_img = AICSImage(file_path)
        image = self.reorganize_channels(
            aics_img.data, "TCZYX", aics_img.dims.order
        )
        return image

    @staticmethod
    def reorganize_channels(
        image: np.ndarray,
        target_order: Optional[str],
        original_order: Optional[str],
    ):
        """
        Make sure image dimensions order matches dim_order.
        """
        # Add missing dimensions if necessary
        for dim in target_order:
            if dim not in original_order:
                original_order = dim + original_order
                image = np.expand_dims(image, axis=0)

        indexes = [original_order.index(dim) for dim in target_order]
        return np.moveaxis(image, indexes, list(range(len(target_order))))

    def display_info(
        self,
        unit=None,
        scale=None,
        save_path="",
        dimensions=None,
        show=True,
    ):
        image_to_plot = self.get_processed_image().squeeze()
        assert image_to_plot.ndim in [2, 3]

        if image_to_plot.ndim == 2:
            image_to_plot = np.expand_dims(image_to_plot, 0)

        if dimensions is not None:
            image_to_plot = image_to_plot[
                :,
                dimensions["min_y"] : dimensions["max_y"],
                dimensions["min_x"] : dimensions["max_x"],
            ]

        z_slice_to_plot = image_to_plot.shape[0]
        nb_col_row = math.ceil(math.sqrt(z_slice_to_plot))
        _, ax = plt.subplots(nb_col_row, nb_col_row)
        ax = np.array(ax)

        for i, local_ax in enumerate(ax.reshape(-1)):
            local_ax.set_yticklabels([])
            local_ax.set_xticklabels([])
            local_ax.set_xticks([])
            local_ax.set_yticks([])
            local_ax.spines["top"].set_visible(False)
            local_ax.spines["right"].set_visible(False)
            local_ax.spines["bottom"].set_visible(False)
            local_ax.spines["left"].set_visible(False)

            if i < z_slice_to_plot:
                local_ax.imshow(image_to_plot[i, :, :], cmap="gray")

                # Add scale bar
                if scale is not None and unit is not None:
                    generate_size_bar(
                        image_to_plot.shape[2], local_ax, scale, unit
                    )

        plt.tight_layout()

        # Display on whole screen
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()

        if save_path:
            plt.savefig(
                save_path,
                bbox_inches="tight",
                pad_inches=0.0,
                transparent=True,
            )

        if show:
            plt.show()
