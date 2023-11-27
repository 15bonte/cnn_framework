import warnings
from matplotlib import pyplot as plt
from skimage import io
import numpy as np
from torch.utils.data import DataLoader

from ..data_sets.dataset_output import DatasetOutput
from .model_manager import ModelManager
from ..display_tools import (
    make_image_matplotlib_displayable,
    make_image_tiff_displayable,
)
from ..metrics.abstract_metric import AbstractMetric


class RegressionModelManager(ModelManager):
    """
    Model manager for CNN regression models.
    """

    def save_results(
        self, name: str, dl_element: DatasetOutput, mean_std
    ) -> None:
        input_np = dl_element.input
        target_np = dl_element.target
        prediction_np = dl_element.prediction
        add_np = dl_element.additional

        # Save inputs, targets & predictions as tiff images
        for data_image, data_type, data_mean_std in zip(
            [input_np, add_np],
            ["input", "additional"],
            [mean_std, None],  # No normalization for additional data
        ):
            if data_image is None:
                continue
            image_to_save = make_image_tiff_displayable(
                data_image, data_mean_std
            )

            # Save inputs with prediction and ground truth in name
            if (
                len(target_np.shape) == 0
            ):  # case where regression predicts only one value
                target_np = np.array([target_np])
                prediction_np = np.array([prediction_np])
            ground_truth = "_".join(
                [str(local_target) for local_target in target_np]
            )
            prediction = "_".join(
                [str(local_prediction) for local_prediction in prediction_np]
            )
            with warnings.catch_warnings():
                warnings.filterwarnings(action="ignore", category=UserWarning)
                io.imsave(
                    f"{self.params.output_dir}/{name}_{data_type}_g{ground_truth}_p{prediction}.tiff",
                    image_to_save,
                )

    def write_images_to_tensorboard(
        self,
        current_batch: int,
        dl_element: DatasetOutput,
        name: str,
    ) -> None:
        # Get numpy arrays
        numpy_dl_element = dl_element.get_numpy_dataset_output()

        # Get images name
        current_dl_file_names = [
            file_name.split(".")[0]
            for file_name in self.dl[name].dataset.names
        ]
        image_names = [
            current_dl_file_names[image_index]
            for image_index in numpy_dl_element.index
        ]

        # Log the results images
        for i, (input_np, target_np, prediction_np, image_name) in enumerate(
            zip(
                numpy_dl_element.input,
                numpy_dl_element.target,
                numpy_dl_element.prediction,
                image_names,
            )
        ):
            # Do not save too many images
            if i == self.params.nb_tensorboard_images_max:
                break
            # Get class with max probability
            ground_truth = ",".join(
                [str(int(local_target)) for local_target in target_np]
            )
            prediction = ",".join(
                [
                    str(int(local_prediction))
                    for local_prediction in prediction_np
                ]
            )

            # Log input image
            plt.title(f"Ground truth {ground_truth} - Predicted {prediction}")
            input_mat = make_image_matplotlib_displayable(
                input_np, self.dl[name].dataset.mean_std
            )
            plt.imshow(input_mat)
            self.writer.add_figure(
                f"{name}/{image_name}",
                plt.gcf(),
                current_batch,
            )

    def model_prediction(
        self,
        dl_element: DatasetOutput,
        dl_metric: AbstractMetric,
        data_loader: DataLoader,
    ) -> None:
        """
        Function to generate outputs from inputs for given model.
        No softmax is applied here.
        """
        dl_element.to_device(self.device)

        predictions = self.model(dl_element.input.float())
        dl_element.prediction = predictions

        # Update metric
        dl_metric.update(
            predictions,
            dl_element.target,
            adds=dl_element.additional,
            mean_std=data_loader.dataset.mean_std,
        )
