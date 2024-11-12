import json
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data._utils.collate import default_collate
from typing import Optional

from ..data_managers.default_data_manager import DefaultDataManager
from ..model_params.base_model_params import DataSplit
from ..data_sets.dataset_output import DatasetOutput
from ..tools import handle_image_type
from ..display_tools import display_progress


def check_dimensions_order(params, dataset_output):
    assert dataset_output.input.shape[-1] == len(params.c_indexes) * len(
        params.z_indexes
    )
    if dataset_output.target_is_image():
        assert dataset_output.target.shape[-1] == params.out_channels


def post_process_mean_std(mean_std, mode, nb_std=3):
    """
    Used to normalize between 0 and 1 instead of -1 and 1.

    - Gaussian mode: suppose data is distributed according to a gaussian distribution

    nb_std = 0 -> do nothing
    nb_std = 1 -> keep 68% of the data between 0 and 1
    nb_std = 2 -> keep 95% of the data between 0 and 1
    nb_std = 3 -> keep 99% of the data between 0 and 1

    - Max mode:

    Just make sure that the max value is 1 and the min value is 0

    """
    if mode == "standard":
        return mean_std

    if mode == "gaussian":
        if nb_std == 0:
            return mean_std

        new_mean_std = {}
        new_mean_std["mean"] = [
            m - nb_std * std
            for m, std in zip(mean_std["mean"], mean_std["std"])
        ]
        new_mean_std["std"] = [std * 2 * nb_std for std in mean_std["std"]]

    elif mode == "max":
        new_mean_std = {}
        new_mean_std["mean"] = [0 for _ in mean_std["mean"]]
        new_mean_std["std"] = list(mean_std["max"])

    elif mode == "isl":  # mimic ISL paper (Christiansen 2018)
        # Set mean to 0.25 and std to 0.125
        target_mean = 0.25
        target_std = 0.125

        new_mean_std = {}
        new_mean_std["mean"] = [
            m - target_mean * std / target_std
            for m, std in zip(mean_std["mean"], mean_std["std"])
        ]
        new_mean_std["std"] = [std / target_std for std in mean_std["std"]]

    else:
        return ValueError("Unknown mean/std mode.")

    return new_mean_std


def get_mean_and_std(
    data_loaders: list[DataLoader],
    max_percentile=90,
    mean_std_path: Optional[str] = None,
    max_nb_imgs: Optional[int] = 10000,
) -> dict[str, list[float]]:
    """
    Args:
        dataloader: DataLoader with a mil dataset
        max_nb_images: in case of huge data sets, consider using only first images to compute mean and std
    Returns:
        Mean and std of images in dataloader.
    """

    # If provided, read mean and std from file directly
    if mean_std_path is not None:
        with open(mean_std_path, "r") as mean_std_file:
            mean_std = json.load(mean_std_file)
        return mean_std

    # If max_nb_imgs is provided, compute mean and std on first images
    if max_nb_imgs is not None:
        print(f"Mean and std are computed on only {max_nb_imgs} images max.")
    else:
        max_nb_imgs = np.inf

    # Initialize mean and std
    params = data_loaders[0].dataset.params
    in_channels = len(params.c_indexes) * len(params.z_indexes)
    channels = in_channels + params.out_channels
    channels_sum, channels_squared_sum, channels_max = (
        np.zeros(channels),
        np.zeros(channels),
        [],
    )
    nb_imgs, nb_pixels = 0, 0

    total_files = sum(
        [len(data_loader.dataset.names) for data_loader in data_loaders]
    )
    for data_loader in data_loaders:
        for filename in data_loader.dataset.names:
            # Break if max_nb_images is reached
            if nb_imgs >= max_nb_imgs:
                break
            dataset_output = data_loader.dataset.generate_raw_images(
                filename
            )  # H, W, C
            check_dimensions_order(params, dataset_output)
            img, target = dataset_output.input, dataset_output.target
            # If target is image, concatenate it to img to compute its mean and std
            if dataset_output.target_is_image():
                img = np.concatenate([img, target], axis=-1)
            # Handle type
            img = handle_image_type(img)
            # Cast img to float32 to avoid overflow
            img = img.astype("float32")
            # Compute sum, squared sum, max
            channels_sum += np.sum(img, axis=(0, 1))
            channels_squared_sum += np.sum(np.square(img), axis=(0, 1))
            channels_max.append(
                np.percentile(img, max_percentile, axis=(0, 1))
            )
            # Update number of images and pixels
            nb_imgs += 1
            nb_pixels += img.shape[0] * img.shape[1]
            display_progress(
                "Mean/std computation in progress",
                nb_imgs,
                min(total_files, max_nb_imgs),
                additional_message=f"Image {nb_imgs}/{min(total_files, max_nb_imgs)}",
            )

    mean = channels_sum / nb_pixels
    std = np.sqrt((channels_squared_sum / nb_pixels - np.square(mean)))
    channels_percent_max = np.percentile(
        np.array(channels_max), max_percentile, axis=0
    ).astype(
        "float64"
    )  # to enable json serialization

    return {
        "mean": list(mean),
        "std": list(std),
        "max": list(channels_percent_max),
    }


def collate_dataset_output(batch):
    """Collate function that treats the `DatasetOutput` class correctly."""
    if isinstance(batch[0], DatasetOutput):
        # `default_collate` returns a dict for older versions of PyTorch.
        return DatasetOutput(
            **default_collate([element.to_dict() for element in batch])
        )
    else:
        return default_collate(batch)


class DataLoaderGenerator:
    """
    Class used to generate data loaders from params and data folder.
    Note DATA_SET_CLASS is related to the MODEL,
    while the DATA_MANAGER is related to the DATA.
    """

    def __init__(
        self,
        params,
        data_set_class,
        data_manager_class=DefaultDataManager,
        collate_fn=collate_dataset_output,
    ):
        self.data_set_class = data_set_class
        self.data_manager = data_manager_class(params.data_dir)
        self.params = params
        self.collate_fn = collate_fn

    def generate_train_weights(
        self, data_set_train, data_set_val, data_set_test
    ):
        # Print data sets size
        for set_name, data_set in zip(
            ["train", "val", "test"],
            [data_set_train, data_set_val, data_set_test],
        ):
            print(f"{set_name} has {len(data_set.names)} images.")

        print("###################")

        # By default, no oversampling is applied
        return [1 for _ in data_set_train.names]

    def generate_data_loader(
        self,
        single_image_test_batch=False,
        train_as_test=False,
        shuffle_test=False,
    ):
        """
        single_image_test_batch should be set to True to force batch_size=1 for test set.
        Useful to enable extraction of multi sub images for testing, for detection models.

        If train_as_test is True, train is not shuffled and no augmentation is applied.
        """
        files = self.data_manager.get_distinct_files()
        files.sort()
        print("File names correctly loaded.")

        data_split = DataSplit(self.params, files)
        (
            names_train,
            names_val,
            names_test,
        ) = data_split.generate_train_val_test_list(
            files, self.params.data_dir
        )

        self.params.names_train = names_train
        self.params.names_val = names_val
        self.params.names_test = names_test

        # Data sets
        dataset_train = self.data_set_class(
            (not train_as_test), names_train, self.data_manager, self.params
        )
        dataset_val = self.data_set_class(
            False, names_val, self.data_manager, self.params
        )
        dataset_test = self.data_set_class(
            False, names_test, self.data_manager, self.params
        )

        # Generate weights for train set (i.e. oversampling of some images)
        weights_train = self.generate_train_weights(
            dataset_train, dataset_val, dataset_test
        )

        # Train
        sampler_train = (
            WeightedRandomSampler(
                torch.DoubleTensor(weights_train), len(names_train)
            )
            if len(names_train) and not train_as_test
            else None
        )
        train_dl = DataLoader(
            dataset_train,
            batch_size=self.params.batch_size,
            sampler=sampler_train,
            collate_fn=self.collate_fn,
            num_workers=self.params.num_workers,
            persistent_workers=True if self.params.num_workers else False,
            pin_memory=True,
        )

        # Val
        val_dl = DataLoader(
            dataset_val,
            batch_size=self.params.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.params.num_workers,
            persistent_workers=True if self.params.num_workers else False,
            pin_memory=True,
        )

        # Test
        sampler_test = (
            WeightedRandomSampler(
                torch.DoubleTensor([1 for _ in dataset_test.names]),
                len(names_test),
            )
            if shuffle_test and len(names_test)
            else None
        )

        test_dl = DataLoader(
            dataset_test,
            batch_size=(
                1 if single_image_test_batch else self.params.batch_size
            ),
            collate_fn=self.collate_fn,
            pin_memory=True,
            sampler=sampler_test,
        )

        return train_dl, val_dl, test_dl
