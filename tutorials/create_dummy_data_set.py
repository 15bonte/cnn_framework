import os
import numpy as np
from skimage import io


def generate_square(image_size, nb_channels, color):
    # Create black image with square in the middle
    image = np.zeros((image_size, image_size, nb_channels), dtype=np.uint8)

    # Square
    # Top left random corner
    random_x = np.random.randint(0, image_size // 2)
    random_y = np.random.randint(0, image_size // 2)

    if color is None:
        for channel in range(nb_channels):
            image[
                random_y : random_y + image_size // 2,
                random_x : random_x + image_size // 2,
                channel,
            ] = np.random.randint(0, 255)
    else:
        image[
            random_y : random_y + image_size // 2,
            random_x : random_x + image_size // 2,
            :,
        ] = color
    return image


def generate_circle(image_size, nb_channels, color):
    # Create black image with circle in the middle
    image = np.zeros((image_size, image_size, nb_channels), dtype=np.uint8)

    # Circle
    # Center
    center_x = np.random.randint(image_size // 4, 3 * image_size // 4)
    center_y = np.random.randint(image_size // 4, 3 * image_size // 4)
    # Radius
    radius = image_size // 4

    if color is None:
        for channel in range(nb_channels):
            image[:, :, channel] = np.random.randint(0, 255)
            for x in range(image_size):
                for y in range(image_size):
                    if (x - center_x) ** 2 + (y - center_y) ** 2 > radius**2:
                        image[y, x, channel] = 0
    else:
        for channel, channel_color in enumerate(color):
            image[:, :, channel] = channel_color
            for x in range(image_size):
                for y in range(image_size):
                    if (x - center_x) ** 2 + (y - center_y) ** 2 > radius**2:
                        image[y, x, channel] = 0

    return image


def main(image_size, nb_channels, nb_elements_per_class, save_dir, extension, same_color):
    # Create save_dir if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for idx in range(nb_elements_per_class):
        square_image = generate_square(
            image_size, nb_channels, color=[255, 255, 0] if same_color else None
        )
        io.imsave(f"{save_dir}/random_{idx}_c0.{extension}", square_image)

        circle_image = generate_circle(
            image_size, nb_channels, color=[255, 0, 255] if same_color else None
        )
        io.imsave(f"{save_dir}/random_{idx}_c1.{extension}", circle_image)


if __name__ == "__main__":
    IMAGE_SIZE = 128
    NB_CHANNELS = 3
    NB_ELEMENTS_PER_CLASS = 100
    SAVE_DIR = ""
    EXTENSION = "tiff"
    SAME_COLOR = (
        True  # if True, square are all yellow and circle are all purple, if False, random colors
    )

    main(IMAGE_SIZE, NB_CHANNELS, NB_ELEMENTS_PER_CLASS, SAVE_DIR, EXTENSION, SAME_COLOR)
