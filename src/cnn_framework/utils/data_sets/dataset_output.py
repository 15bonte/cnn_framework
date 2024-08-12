from __future__ import annotations
from typing import Optional
import numpy as np
import torch

from ..tools import torch_from_numpy


class DatasetOutput:
    """
    Class to handle the output of a dataset.
    """

    def __init__(
        self,
        index: Optional[int] = None,
        input: Optional[np.array] = None,
        target_image: Optional[np.array] = None,
        target_array: Optional[np.array] = None,
        additional: Optional[np.array] = None,
        prediction: Optional[np.array] = None,
        encoded_file_name: Optional[int] = -1,
    ):
        self.index = index
        self.input = input

        # Make sure only one target is specified
        assert (target_image is None) or (target_array is None)

        self.target_image = target_image
        self.target_array = target_array

        self.additional = additional
        self.prediction = prediction

        self.encoded_file_name = encoded_file_name

    @property
    def target(self):
        return (
            self.target_image if self.target_is_image() else self.target_array
        )

    @target.setter
    def target(self, new_target: np.array):
        if self.target_image is not None:
            self.target_image = new_target
        else:
            self.target_array = new_target

    def __getitem__(self, idx):
        item_dict = {
            key: value[idx, ...].squeeze()
            for key, value in self.__dict__.items()
            if value is not None and key != "target"
        }
        return DatasetOutput(**item_dict)

    def to_torch(self) -> None:
        self.index = torch.tensor(self.index)
        self.input = torch_from_numpy(self.input)
        self.target = torch_from_numpy(self.target)
        self.additional = torch_from_numpy(self.additional)
        self.prediction = torch_from_numpy(self.prediction)
        self.encoded_file_name = torch.tensor(self.encoded_file_name)

    def to_dict(self) -> dict:
        return {
            key: value
            for key, value in self.__dict__.items()
            if value is not None and key != "target"
        }

    def to_device(self, device) -> None:
        self.index = self.index.to(device)
        self.input = self.input.to(device)
        self.target = self.target.to(device)
        self.additional = (
            self.additional.to(device) if self.additional_is_image() else None
        )
        self.prediction = (
            self.prediction.to(device) if self.prediction is not None else None
        )
        self.encoded_file_name = self.encoded_file_name.to(device)

    def get_numpy_dataset_output(self):
        numpy_dict = {
            key: value.detach().cpu().numpy()
            for key, value in self.__dict__.items()
            if value is not None and key != "target"
        }
        return DatasetOutput(**numpy_dict)

    def target_is_image(self):
        return self.target_image is not None

    def additional_is_image(self):
        return self.additional is not None

    @staticmethod
    def merge(list_dataset: list[DatasetOutput]) -> DatasetOutput:
        """
        Merge a list of DatasetOutput into a single DatasetOutput.
        """
        merged = {}
        for key in list_dataset[0].__dict__.keys():
            if key in ["index", "target", "encoded_file_name"]:
                continue
            if getattr(list_dataset[0], key) is None:
                merged[key] = None
            else:
                merged[key] = np.concatenate(
                    [getattr(data, key) for data in list_dataset], axis=-1
                )
        return DatasetOutput(**merged)

    def split(self, nb_splits: int) -> list[DatasetOutput]:
        """
        Split the DatasetOutput into a list of DatasetOutput.
        """
        split_size = self.input.shape[-1] // nb_splits
        split_data = []
        for i in range(nb_splits):
            split_dict = {
                key: value[..., i * split_size : (i + 1) * split_size]
                for key, value in self.__dict__.items()
                if value is not None
                and key not in ["index", "target", "encoded_file_name"]
            }
            split_data.append(DatasetOutput(**split_dict))
        return split_data
