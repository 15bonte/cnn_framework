from matplotlib import pyplot as plt

from .abstract_reader import AbstractReader
from ..display_tools import generate_size_bar


class PngReader(AbstractReader):
    """
    Class to read PNG file.
    """

    def display_info(self, unit=None, scale=None, save_path="", dimensions=None, _=True, __=True):
        image_to_plot = self.get_processed_image()

        if dimensions is not None:
            image_to_plot = image_to_plot[
                dimensions["min_x"] : dimensions["max_x"],
                dimensions["min_y"] : dimensions["max_y"],
            ]

        if unit is None and save_path:
            plt.imsave(save_path, image_to_plot, cmap="gray")
            return

        _, ax = plt.subplots()

        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

        ax.imshow(image_to_plot)

        # Add scale bar
        if scale is not None and unit is not None:
            generate_size_bar(image_to_plot.shape[0], ax, scale, unit)

        plt.tight_layout()

        # Display on whole screen
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0.0, transparent=True)

        plt.show()


if __name__ == "__main__":
    PNG_FILE_TO_DISPLAY = r"C:\Users\thoma\data\Other\img05.png"
    UNIT, SCALE = None, None  # "nm", 500  # ("µm", 0.1) => 1px = 0.1µm
    DIMENSIONS = {"min_x": 0, "max_x": 500, "min_y": 0, "max_y": 500}
    SAVE_PATH = ""

    if PNG_FILE_TO_DISPLAY is not None:
        reader = PngReader(PNG_FILE_TO_DISPLAY)
        reader.display_info(UNIT, SCALE, SAVE_PATH, DIMENSIONS)
