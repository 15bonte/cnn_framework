import slideio
import numpy as np
import matplotlib.pyplot as plt

from ..enum import NormalizeMethods, ProjectMethods
from ..preprocessing import normalize_array


class CIZReader:
    """
    Class to read CIZ file.
    """

    def __init__(
        self, path, channels=None, normalize=NormalizeMethods.none, project=ProjectMethods.none
    ):
        print("May be deprecated because unused for a while.")
        self.slide = slideio.open_slide(path, "CZI")
        self.scene = self.slide.get_scene(0)

        # Get all channels or only specified ones if asked
        if channels is None:
            self.channels_names = {
                channel: f"{self.scene.get_channel_name(channel)}_{channel}"
                for channel in range(self.scene.num_channels)
            }
        else:
            self.channels_names = {}
            for channel in channels:
                index = int(channel.split("_")[-1])
                self.channels_names[index] = channel

        self.z_stacks = {}
        self.image_preprocessed = False
        self.normalize = normalize
        self.project = project

    def group_project_stacks(self, channel, name):
        stacks = []
        for z_slice in range(self.scene.num_z_slices):
            block = self.scene.read_block(
                channel_indices=[channel], slices=(z_slice, z_slice + 1)
            )  # HW
            stacks.append(block)
        array_stacks = np.asarray(stacks)  # DHW
        if name in self.project:  # Project only if asked
            array_stacks = array_stacks.max(axis=0)  # HW
            array_stacks = np.expand_dims(array_stacks, axis=0)  # DHW, D=1
        return array_stacks

    def preprocess_image(self):
        # Iterate over CIZ channels
        for channel in range(self.scene.num_channels):
            # Ignore if not asked by user
            if channel not in self.channels_names:
                continue
            # Group z-stacks together in DHW format & normalize
            name = self.channels_names[channel]
            z_stack_array = self.group_project_stacks(channel, name)  # DHW
            # Normalize only if asked
            if name in self.normalize:
                z_stack_array = normalize_array(z_stack_array, None)
            # Swap axes to match HWD format
            if len(z_stack_array.shape) > 2:
                z_stack_array = np.moveaxis(z_stack_array, 0, 2)  # HWD
            # Force float64 type
            if z_stack_array.dtype == np.uint16:
                z_stack_array = z_stack_array * 1.0
            # Store final result
            self.z_stacks[name] = np.expand_dims(z_stack_array, axis=0)  # CHWD (gray-scale)
        self.image_preprocessed = True

    def get_processed_image(self, channel):
        if not self.image_preprocessed:
            self.preprocess_image()
        if channel not in self.z_stacks:
            raise ValueError(f"Channel {channel} does not exist in image.")
        return self.z_stacks[channel]

    def display_info(self, _=None, __=None, ___="", ____=None):
        # Print some data information
        print(f"Scenes: {self.slide.num_scenes} \nScene name: {self.scene.name}")
        print(f"Rectangle: {self.scene.rect} \nChannels: {self.scene.num_channels}")
        print(f"Resolution: {self.scene.resolution} \nZ-slices: {self.scene.num_z_slices}")
        print(f"Channels names: {self.channels_names}")

        z_slices_to_plot = [0, self.scene.num_z_slices // 2, self.scene.num_z_slices - 1]
        rows, columns = len(z_slices_to_plot), self.scene.num_channels
        fig = plt.figure()

        for idx, z_slice in enumerate(z_slices_to_plot):
            for scene_channel in range(self.scene.num_channels):
                block = self.scene.read_block(
                    channel_indices=[scene_channel], slices=(z_slice, z_slice + 1)
                )
                fig.add_subplot(rows, columns, idx * columns + scene_channel + 1)
                plt.title(f"{self.channels_names[scene_channel]} - Z{z_slice}")
                plt.imshow(block, cmap="gray")

        plt.show()


if __name__ == "__main__":
    FILE_PATH = r"C:\Users\thoma\data\Data Allen\st6gal1.tar\st6gal1\3500001232_100X_20170825_2-Scene-04-P15-E05.czi"
    ONLY_CHANNELS = ["H3342_5", "EGFP_3"]

    reader = CIZReader(
        FILE_PATH,
        project=["CMDRP_0", "CMDRP_1", "EGFP_2", "EGFP_3", "H3342_4", "H3342_5", "Bright_6"],
        channels=ONLY_CHANNELS,
    )
    reader.display_max_intensity_projection()
