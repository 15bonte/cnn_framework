import os
import warnings
import numpy as np
from skimage import io


def generate_square(image_size, nb_channels, color):
    # Create black image with square in the middle
    image = np.zeros((image_size, image_size, nb_channels), dtype=np.uint8)

    # Square
    square_size = np.random.randint(2 * image_size / 5, 2 * image_size / 3)

    # Top left random corner
    random_x = np.random.randint(0, image_size - square_size)
    random_y = np.random.randint(0, image_size - square_size)

    if color is None:
        # First channel is always 255 as it is used for segmentation
        color = [255, np.random.randint(0, 255), np.random.randint(0, 255)]

    image[
        random_y : random_y + square_size,
        random_x : random_x + square_size,
        :,
    ] = [255, np.random.randint(0, 255), np.random.randint(0, 255)]

    return image


def generate_circle(image_size, nb_channels, color):
    # Create black image with circle in the middle
    image = np.zeros((image_size, image_size, nb_channels), dtype=np.uint8)

    # Circle

    # Radius
    radius = np.random.randint(image_size / 5, image_size / 3)

    # Center
    center_x = np.random.randint(radius, image_size - radius)
    center_y = np.random.randint(radius, image_size - radius)

    if color is None:
        color = [255, np.random.randint(0, 255), np.random.randint(0, 255)]

    for channel, channel_color in enumerate(color):
        image[:, :, channel] = channel_color
        for x in range(image_size):
            for y in range(image_size):
                if (x - center_x) ** 2 + (y - center_y) ** 2 > radius**2:
                    image[y, x, channel] = 0

    return image


def generate_data_set(
    save_dir,
    image_size=32,
    nb_channels=3,
    nb_elements_per_class=1000,
    extension="tiff",
    same_color=False,  # if True, square are all yellow and circle are all purple, if False, random colors
):
    # Create save_dir if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for idx in range(nb_elements_per_class):
        square_image = generate_square(
            image_size, nb_channels, color=[255, 255, 0] if same_color else None
        )
        # Save image without warnings
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", category=UserWarning)
            io.imsave(f"{save_dir}/random_{idx}_c0.{extension}", square_image)

        circle_image = generate_circle(
            image_size, nb_channels, color=[255, 0, 255] if same_color else None
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", category=UserWarning)
            io.imsave(f"{save_dir}/random_{idx}_c1.{extension}", circle_image)
