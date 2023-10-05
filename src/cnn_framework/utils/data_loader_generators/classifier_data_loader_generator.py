import random
from collections import Counter
import numpy as np

from .data_loader_generator import DataLoaderGenerator


constant_seeded_rd = random.Random(10)


class ClassifierDataLoaderGenerator(DataLoaderGenerator):
    """
    Class used to generate data loaders from params and data folder.
    Specific for CNN classifier.
    """

    def generate_train_weights(self, data_set_train, data_set_val, data_set_test):
        """
        For train, remove all images with label > number of classes, and balance dataset.
        For val, test, keep as it is.
        """
        for set_name, data_set in zip(
            ["train", "val", "test"], [data_set_train, data_set_val, data_set_test]
        ):
            if len(data_set.names) == 0:
                if set_name == "train":
                    weights_train = []
                continue
            # Create list of class ids
            class_probabilities = np.array([data_set.read_output(name) for name in data_set.names])
            class_ids = np.argmax(class_probabilities, axis=1)
            # Count occurrences/sum of probabilities of each class
            count = np.sum(class_probabilities, axis=0)
            # Create weights for each class
            if set_name == "train":  # oversampling for train set
                weights_train = [
                    1.0 / count[class_id] if count[class_id] > 0 else 0 for class_id in class_ids
                ]

            # Print information
            count = Counter(class_ids)
            class_counts = ", ".join(
                [f"{count[class_id]} images for class {class_id}" for class_id in sorted(count)]
            )
            oversampling_message = " (oversampling applied)" if set_name == "train" else ""
            print(
                f"{set_name} has {class_counts}, total {len(data_set.names)} images{oversampling_message}"
            )

        print("###################")

        return weights_train
