import os
import warnings
import re
from matplotlib import pyplot as plt
from skimage import io
import numpy as np
import torch
from torch.utils.data import DataLoader


from ..metrics.abstract_metric import AbstractMetric
from ..data_sets.dataset_output import DatasetOutput
from .model_manager import ModelManager
from ..display_tools import (
    display_confusion_matrix,
    make_image_matplotlib_displayable,
    make_image_tiff_displayable,
)


def get_nested_attribute(obj, attr_string):
    # Split the attribute string by '.' to handle nested attributes
    attributes = attr_string.split(".")

    for attr in attributes:
        # Check if the attribute contains list indexing, e.g., "c[-1]"
        match = re.match(r"(\w+)\[(\-?\d+)\]", attr)
        if match:
            attr_name, index = match.groups()
            obj = getattr(obj, attr_name)  # Get the attribute (a list)
            obj = obj[int(index)]  # Access the specific index in the list
        else:
            obj = getattr(obj, attr)  # Normal attribute access

    return obj


class CnnModelManager(ModelManager):
    """
    Model manager for CNN classification models.
    """

    def save_results(
        self,
        name: str,
        dl_element: DatasetOutput,
        mean_std: dict[str, list[float]],
        save_only_wrong: bool = False,
    ):
        input_np = dl_element.input
        target_np = dl_element.target
        prediction_np = dl_element.prediction
        add_np = dl_element.additional

        prediction_class = np.argmax(prediction_np, axis=0)
        ground_truth_class = np.argmax(target_np, axis=0)

        # Choose folder to save
        folder_to_save = (
            os.path.join(self.params.output_dir, "right")
            if prediction_class == ground_truth_class
            else os.path.join(self.params.output_dir, "wrong")
        )
        os.makedirs(folder_to_save, exist_ok=True)

        # Save inputs, targets & predictions as tiff images
        for data_image, data_type, data_mean_std in zip(
            [input_np, add_np],
            ["input", "additional"],
            [mean_std, None],  # No normalization for additional data
        ):
            if data_image is None:  # case when additional data is None
                continue
            if save_only_wrong and ground_truth_class == prediction_class:
                continue
            image_to_save = make_image_tiff_displayable(
                data_image, data_mean_std
            )

            # Save inputs with prediction and ground truth in name
            with warnings.catch_warnings():
                warnings.filterwarnings(action="ignore", category=UserWarning)
                io.imsave(
                    f"{folder_to_save}/{name}_{data_type}_g{ground_truth_class}_p{prediction_class}.tiff",
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
            prediction = np.argmax(prediction_np, axis=0)
            ground_truth = np.argmax(target_np, axis=0)

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
        Careful, softmax is applied here and not in the model.
        """
        dl_element.to_device(self.device)

        if hasattr(
            self.model, "target_layer"
        ):  # apply GradCam only if target_layer is defined

            # Import here since tests crash when inside cnn_framework, for some reason
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.image import show_cam_on_image

            target_layers = [
                get_nested_attribute(self.model, self.model.target_layer)
            ]

            # Construct the CAM object once, and then re-use it on many images.
            with GradCAM(model=self.model, target_layers=target_layers) as cam:
                # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
                # If targets is None, the highest scoring category (for every member in the batch) will be used.
                grayscale_cam = cam(
                    input_tensor=dl_element.input.float(), targets=None
                )
                predictions = torch.softmax(cam.outputs, dim=-1)
                # In this example grayscale_cam has only one image in the batch:
                grad_cam_images = []
                for input_img, grad_cam_img in zip(
                    dl_element.input, grayscale_cam
                ):
                    input_img = np.max(input_img.cpu().numpy(), axis=0)
                    input_img_rgb = np.stack([input_img] * 3, axis=-1)
                    processed = show_cam_on_image(
                        input_img_rgb, grad_cam_img, use_rgb=True
                    )
                    processed = torch.from_numpy(np.moveaxis(processed, -1, 0))
                    grad_cam_images.append(processed)

            dl_element.additional = torch.stack(grad_cam_images)

        else:
            predictions = torch.softmax(
                self.model(dl_element.input.float()), dim=-1
            )

        dl_element.prediction = predictions

        # Update metric
        dl_metric.update(
            predictions,
            dl_element.target,
            adds=dl_element.additional,
            mean_std=data_loader.dataset.mean_std,
        )

    def plot_confusion_matrix(self, results) -> None:
        display_confusion_matrix(
            results, self.params.class_names, self.params.output_dir
        )
