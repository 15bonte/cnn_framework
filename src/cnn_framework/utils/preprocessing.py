import numpy as np
from sklearn.preprocessing import MinMaxScaler


def normalize_array(input_array, mean_std=None, reverse=False, channel_axis=None):
    """
    Works for both 2D and 3D arrays.
    If reverse, then the normalization is reversed.

    mean_std = dict(channel, dict(mean, std))
    """

    # Get number of channels
    nb_channels = np.min(input_array.shape)

    # Find channel axis if not already given
    if channel_axis is None:
        channel_axis = np.argmin(input_array.shape)

    # Put channel axis in front
    input_array = np.moveaxis(input_array, channel_axis, 0)

    if mean_std is not None:  # custom mean and std
        assert nb_channels == len(mean_std["mean"])
        mean, std = mean_std["mean"], mean_std["std"]
        for _ in range(2):  # 2 for H and W
            mean = np.expand_dims(mean, -1)
            std = np.expand_dims(std, -1)
    else:  # compute mean and std
        axes = tuple(range(1, len(input_array.shape)))
        mean = np.mean(input_array, axis=axes, keepdims=True)
        std = np.std(input_array, axis=axes, keepdims=True)

    # If reverse, apply reverse normalization
    if reverse:
        mean = -mean / std
        std = 1 / std

    normalized_array = (input_array - mean) / std

    # Cast to float32 as float64 is not supported by albumentations
    normalized_array = normalized_array.astype(np.float32)

    # Put channel axis back to original position
    normalized_array = np.moveaxis(normalized_array, 0, channel_axis)

    return normalized_array


def zero_one_scaler(input_array, feature_range=(0, 1)):
    """
    Normalize array between 0 and 1.
    Works for both 2D and 3D arrays.
    """
    # 2D
    if len(input_array.shape) <= 2:
        return MinMaxScaler().fit_transform(input_array)

    # Find channel axis
    channel_axis = np.argmin(input_array.shape)
    nb_channels = input_array.shape[channel_axis]
    # Put channel axis in back
    new_input_array = np.moveaxis(input_array, channel_axis, -1)
    # Reshape to 2D
    as_columns = new_input_array.reshape(-1, nb_channels)
    # Apply min max scaler
    transformed_columns = MinMaxScaler(feature_range=feature_range).fit_transform(
        as_columns
    )
    # Reshape to original shape
    transformed = transformed_columns.reshape(new_input_array.shape)
    # Put channel axis back to original position
    transformed = np.moveaxis(transformed, -1, channel_axis)

    # Make sure values are between 0 and 1
    transformed = np.clip(transformed, 0, 1)

    return transformed
