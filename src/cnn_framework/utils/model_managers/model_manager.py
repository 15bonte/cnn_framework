import math
from pathlib import Path
import warnings
import datetime
import json
import os
import time
from typing import Optional
from matplotlib import pyplot as plt
import numpy as np
from skimage import io
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)
import tensorflow as tf

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer

from .utils.training_information import TrainingInformation
from ..enum import PredictMode
from ..losses.loss_manager import LossManager
from ..model_params.base_model_params import BaseModelParams
from ..data_sets.dataset_output import DatasetOutput
from ..metrics.abstract_metric import AbstractMetric
from ..tools import extract_patterns, random_sample
from ..data_loader_generators.data_loader_generator import get_mean_and_std
from ..display_tools import (
    display_progress,
    make_image_tiff_displayable,
)


def adapt_mean_std(mean_std):
    """
    Used to read old fashioned mean_std files
    """

    # If "mean" is already a key, nothing to change
    if "mean" in mean_std:
        return mean_std

    # Else, adapt accordingly
    mean = [mean_std[str(idx)]["mean"] for idx in range(len(mean_std))]
    std = [mean_std[str(idx)]["std"] for idx in range(len(mean_std))]

    return {"mean": mean, "std": std}


class ModelManager:
    """
    Class with all useful functions to train, test, ... a CNN-based model
    """

    def __init__(
        self,
        model: nn.Module,
        params: BaseModelParams,
        metric_class: type[AbstractMetric],
    ):
        # Device to train model
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = model.float()
        self.model.to(self.device)

        self.params = params

        self.metric_class = metric_class

        self.parameters_path = os.path.join(
            self.params.models_folder, "parameters.csv"
        )

        # Display current git hash to follow up

        # Used in prediction
        self.image_index = 0
        # Used in training
        self.train_loss_manager = None
        self.val_loss_manager = None
        self.lr_scheduler = None

        # Useful information
        self.dl = {}  # data loaders dictionary
        self.training_information = TrainingInformation(
            int(self.params.num_epochs)
        )

        # Tensorboard writer
        os.makedirs(self.params.tensorboard_folder_path, exist_ok=True)
        self.writer = SummaryWriter(self.params.tensorboard_folder_path)
        self.model_save_path = (
            f"{self.params.models_folder}/{self.params.model_save_name}"
        )
        self.model_save_path_early_stopping = f"{self.params.models_folder}/early_stopping_{self.params.model_save_name}"

    def write_images_to_tensorboard(
        self, current_batch: int, dl_element: DatasetOutput, name: str
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
        for i, (prediction_np, target_np, image_name) in enumerate(
            zip(
                numpy_dl_element.prediction,
                numpy_dl_element.target,
                image_names,
            )
        ):
            # Do not save too many images
            if i == self.params.nb_tensorboard_images_max:
                break
            for channel in range(target_np.shape[0]):
                # ... log the ground truth image
                plt.imshow(target_np[channel], cmap="gray")
                self.writer.add_figure(
                    f"{name}/{image_name}/{channel}/groundtruth",
                    plt.gcf(),
                    current_batch,
                )

                plt.imshow(prediction_np[channel], cmap="gray")
                # ... log the model output image
                self.writer.add_figure(
                    f"{name}/{image_name}/{channel}/predicted",
                    plt.gcf(),
                    current_batch,
                )

    def compute_loss(
        self,
        dl_element: DatasetOutput,
        dl_metric: AbstractMetric,
        data_loader: DataLoader,
        loss_manager: Optional[LossManager] = None,
    ):
        inputs = dl_element.input.to(self.device)  # B, C, H, W
        targets = dl_element.target.to(self.device)

        # Compute the model output
        predictions = self.model(inputs.float())
        dl_element.prediction = predictions

        # Update metric
        dl_metric.update(
            predictions,
            targets,
            adds=dl_element.additional,
            mean_std=data_loader.dataset.mean_std,
        )

        # No need to compute loss if used in test
        if loss_manager is None:
            return None

        # Compute loss
        loss = loss_manager(predictions, targets.float())
        return loss

    def log_train_progress(self, train_metric: AbstractMetric) -> None:
        current_batch = self.training_information.get_current_batch()
        # Graphs
        plot_step = int(self.params.plot_step)
        if (
            current_batch % plot_step == plot_step - 1
        ):  # every plot_step mini-batches...
            # ... log the running loss
            running_losses = self.train_loss_manager.get_running_losses()
            for name, loss in running_losses:
                self.writer.add_scalar(
                    f"train/{name}",
                    loss.item() / plot_step,
                    current_batch,
                )

            # ... log running metric
            score, _ = train_metric.get_score()
            self.writer.add_scalar(
                f"train/{train_metric.get_name()}", score, current_batch
            )
            train_metric.reset()

    def log_val_progress(self, val_metric: AbstractMetric) -> None:
        current_batch = self.training_information.get_current_batch()

        # ...log the running loss
        running_losses = self.val_loss_manager.get_running_losses()
        for name, loss in running_losses:
            val_dl_length = len(self.dl["val"])
            self.writer.add_scalar(
                f"val/{name}", loss.item() / val_dl_length, current_batch
            )

        # ...log the running metric
        score, _ = val_metric.get_score()
        self.writer.add_scalar(
            f"val/{val_metric.get_name()}", score, current_batch
        )
        return score

    def log_images(self, dl_element: DatasetOutput, name: str) -> None:
        # Get needed training information
        current_batch = self.training_information.get_current_batch()
        num_batches_train = self.training_information.num_batches_train
        batch_index = self.training_information.batch_index
        epoch = self.training_information.epoch

        epochs_to_plot = np.linspace(
            1,
            self.training_information.num_epochs,
            self.params.nb_plot_images,
            dtype=int,
        )

        # Condition to apply only for training
        batch_condition = (
            True if name == "val" else batch_index == num_batches_train - 1
        )
        if epoch in epochs_to_plot and batch_condition:
            # Plot last training batch of epoch
            self.write_images_to_tensorboard(current_batch, dl_element, name)

    def fit_core(self, optimizer: Optimizer) -> None:
        # Batch information
        self.training_information.num_batches_train = len(self.dl["train"])
        best_val_loss, best_val_score = np.Infinity, -np.Infinity
        model_epoch = -1

        # Create train metric
        train_metric = self.metric_class(self.device, self.params.nb_classes)

        # Define scaler
        scaler = GradScaler(enabled=self.params.fp16_precision)

        for epoch in range(self.training_information.num_epochs):
            self.training_information.epoch = epoch + 1
            # Enumerate mini batches
            self.model.train()  # set model to train mode
            for batch_index, dl_element in enumerate(
                self.dl["train"]
            ):  # indexes, inputs, targets, adds
                self.training_information.batch_index = batch_index + 1

                # Perform training loop
                with autocast(enabled=self.params.fp16_precision):
                    loss = self.compute_loss(
                        dl_element,
                        train_metric,
                        self.dl["train"],
                        self.train_loss_manager,
                    )

                # Clear the gradients
                optimizer.zero_grad()
                # Credit assignment
                scaler.scale(loss).backward()
                # Update model weights
                scaler.step(optimizer)
                scaler.update()

                # Log progress
                self.log_train_progress(train_metric)
                self.log_images(dl_element, "train")
                display_progress(
                    "Training in progress",
                    self.training_information.get_current_batch(),
                    self.training_information.get_total_batches(),
                    additional_message=f"Local step {self.training_information.batch_index} | Epoch {self.training_information.epoch}",
                )

            # Start by lr_scheduler as warmup starts by 0 otherwise
            self.lr_scheduler.step()

            evaluate = len(self.dl["val"]) > 0
            val_loss, val_score = 0, 0
            # If val_dl is not empty then perform evaluation and compute loss
            if evaluate:
                val_metric = self.metric_class(
                    self.device, self.params.nb_classes
                )
                self.model.eval()  # set model to evaluation mode
                with torch.no_grad():
                    for dl_element in self.dl["val"]:
                        loss = self.compute_loss(
                            dl_element,
                            val_metric,
                            self.dl["val"],
                            self.val_loss_manager,
                        )
                        val_loss += loss

                    # Log progress
                    val_score = self.log_val_progress(val_metric)
                    self.log_images(dl_element, "val")

            # Save only if better than current best loss, or if no evaluation is possible
            if (
                (not evaluate)
                or (
                    val_score >= best_val_score
                )  # best model is model with highest val score
                or (
                    val_score == best_val_score and val_loss < best_val_loss
                )  # or lowest val loss if val score is constant
            ):
                torch.save(
                    self.model.state_dict(),
                    self.model_save_path_early_stopping,
                )
                best_val_loss = val_loss
                best_val_score = val_score
                model_epoch = self.training_information.epoch

        # Save model at last epoch
        torch.save(self.model.state_dict(), self.model_save_path)

        # Best model epoch
        self.training_information.best_model_epoch = model_epoch
        print(f"\nBest model saved at epoch {model_epoch}.")

    def compute_and_save_mean_std(
        self,
        train_dl: DataLoader,
        val_dl: DataLoader,
        mean_std_path: Optional[str] = None,
    ) -> dict[str, list[float]]:
        """
        Compute mean and std.
        """
        data_set_mean_std = get_mean_and_std(
            [train_dl, val_dl], mean_std_path=mean_std_path
        )

        # If necessary, use only first channels
        current_nb_channels = len(self.params.c_indexes) * len(
            self.params.z_indexes
        )
        data_set_mean_std["mean"] = data_set_mean_std["mean"][
            :current_nb_channels
        ]
        data_set_mean_std["std"] = data_set_mean_std["std"][
            :current_nb_channels
        ]

        # Save in model folder
        mean_std_file = os.path.join(
            self.params.models_folder, "mean_std.json"
        )
        with open(mean_std_file, "w") as write_file:
            json.dump(data_set_mean_std, write_file, indent=4)

        # Update train and val accordingly
        train_dl.dataset.mean_std = data_set_mean_std
        val_dl.dataset.mean_std = data_set_mean_std

        return data_set_mean_std

    def fit(
        self,
        train_dl: DataLoader,
        val_dl: DataLoader,
        optimizer: Optimizer,
        loss_function,
        lr_scheduler=None,
        mean_std_path: Optional[str] = None,
    ) -> None:
        # Create folder to save model
        os.makedirs(self.params.models_folder, exist_ok=True)

        # Create csv file with all parameters
        with open(self.parameters_path, "a") as f:
            for key in self.params.__dict__.keys():
                f.write("%s;%s\n" % (key, self.params.__dict__[key]))
        f.close()

        # Compute mean and std of dataset
        self.compute_and_save_mean_std(train_dl, val_dl, mean_std_path)

        if len(train_dl) == 0:
            raise ValueError("No data to train.")

        # Initialise transforms before prediction
        train_dl.dataset.initialize_transforms()
        val_dl.dataset.initialize_transforms()

        # Define data loaders names
        self.dl["train"] = train_dl
        self.dl["val"] = val_dl

        # Define lr_scheduler
        if lr_scheduler is None:
            self.lr_scheduler = StepLR(
                optimizer, step_size=1, gamma=1
            )  # Constant lr
        else:
            self.lr_scheduler = lr_scheduler

        # Monitor training time
        start = time.time()

        # Loss managers initializer
        if isinstance(loss_function, list):
            self.train_loss_manager = LossManager(*loss_function)
            self.val_loss_manager = LossManager(*loss_function)
        else:
            self.train_loss_manager = LossManager(loss_function)
            self.val_loss_manager = LossManager(loss_function)

        # Core fit function with training loop
        self.fit_core(optimizer)

        self.writer.close()
        end = time.time()

        # Training time
        training_time = end - start
        self.training_information.training_time = training_time
        print(
            f"\nTraining successfully finished in {datetime.timedelta(seconds = training_time)}."
        )

        # Update model with saved one
        self.model.load_state_dict(torch.load(self.model_save_path))

    def model_prediction(
        self,
        dl_element: DatasetOutput,
        dl_metric: AbstractMetric,
        data_loader: DataLoader,
    ) -> None:
        """
        Function to generate outputs from inputs for given model.
        By default, does the same thing as compute_loss.
        """
        self.compute_loss(dl_element, dl_metric, data_loader)

    def save_results(
        self, name: str, dl_element: DatasetOutput, mean_std
    ) -> None:
        # Possible target normalization
        target_mean_std = None
        # Save inputs, targets & predictions as tiff images
        for data_image, data_type in zip(
            [
                dl_element.input,
                dl_element.target,
                dl_element.prediction,
                dl_element.additional,
            ],
            ["input", "groundtruth", "predicted", "additional"],
        ):
            # C, H, W for data_image
            if data_image is None:  # case when additional data is None
                continue
            if (
                data_type == "input" and mean_std is not None
            ):  # input has been normalized
                # Case where both input and target have been normalized
                if data_image.shape[0] != len(mean_std["mean"]):
                    nb_input_channels = len(self.params.c_indexes) * len(
                        self.params.z_indexes
                    )
                    input_mean_std = {
                        "mean": mean_std["mean"][:nb_input_channels],
                        "std": mean_std["std"][:nb_input_channels],
                    }
                    target_mean_std = {
                        "mean": mean_std["mean"][nb_input_channels:],
                        "std": mean_std["std"][nb_input_channels:],
                    }
                else:
                    input_mean_std = mean_std
                image_to_save = make_image_tiff_displayable(
                    data_image, input_mean_std
                )
            elif (
                data_type in ["groundtruth", "predicted"]
                and target_mean_std is not None
            ):
                image_to_save = make_image_tiff_displayable(
                    data_image, target_mean_std
                )
            else:
                image_to_save = make_image_tiff_displayable(data_image, None)
            if len(image_to_save) == 0:  # protect against empty image
                continue
            with warnings.catch_warnings():
                warnings.filterwarnings(action="ignore", category=UserWarning)
                io.imsave(
                    f"{self.params.output_dir}/{name}_{data_type}.tiff",
                    image_to_save,
                )

    def batch_predict(
        self,
        test_dl: DataLoader,
        images_to_save: list[int],
        num_batches_test: int,
        test_metric: AbstractMetric,
        predict_mode: PredictMode,
    ):
        all_predictions_np = []
        for batch_idx, dl_element in enumerate(test_dl):
            # Reshape in case of multiple images stacked together
            # Transform shape to (S, C, H, W) to mimic (B, C, H, W)
            if (
                len(dl_element.input.shape) == 5
            ):  # (B, S, C, H, W) = (1, S, C, H, W)
                dl_element.input = dl_element.input.view(
                    *dl_element.input.shape[:0],
                    -1,
                    *dl_element.input.shape[2:],
                )  # (S, C, H, W)
                dl_element.target = dl_element.target.view(
                    *dl_element.target.shape[:0],
                    -1,
                    *dl_element.target.shape[2:],
                )
                dl_element.additional = dl_element.additional.view(
                    *dl_element.additional.shape[:0],
                    -1,
                    *dl_element.additional.shape[2:],
                )

            # Run prediction
            self.model_prediction(dl_element, test_metric, test_dl)

            # Get numpy elements
            dl_element_numpy = dl_element.get_numpy_dataset_output()
            all_predictions_np = all_predictions_np + [
                *dl_element_numpy.prediction
            ]

            if (
                predict_mode == PredictMode.Standard
                and dl_element.target is not None
            ):
                # Save few images
                for idx in range(dl_element.target.shape[0]):
                    if self.image_index in images_to_save:
                        image_id = (batch_idx * test_dl.batch_size) + idx
                        image_name = test_dl.dataset.names[image_id].split(
                            "."
                        )[0]

                        self.save_results(
                            f"{image_name}_{self.image_index}",
                            dl_element_numpy[idx],
                            test_dl.dataset.mean_std,
                        )
                    self.image_index += 1

            display_progress(
                "Model evaluation in progress",
                batch_idx + 1,
                num_batches_test,
                additional_message=f"Batch #{batch_idx}",
            )

        return all_predictions_np

    def plot_confusion_matrix(self, _) -> None:
        # Only used for classification models
        return

    @staticmethod
    def read_mean_std(test_dl: DataLoader, params) -> None:
        # Get mean and standard deviation from saved file
        # Either from model that have just been trained
        if os.path.exists(params.models_folder):
            mean_std_path = f"{params.models_folder}/mean_std.json"
        # Or from model that have been loaded from model_load_path
        else:
            # mean_std may be saved in parent folders
            # Iterate over parent folders to find mean_std.json
            mean_std_path = None
            potential_paths = [
                params.model_load_path,
            ] + list(Path(params.model_load_path).parents)
            for parent_folder in potential_paths:
                mean_std_path = f"{parent_folder}/mean_std.json"
                if os.path.isfile(mean_std_path):
                    break

        if os.path.isfile(mean_std_path):
            with open(mean_std_path, "r") as mean_std_file:
                raw_mean_std = json.load(mean_std_file)
                mean_std = adapt_mean_std(raw_mean_std)
                test_dl.dataset.mean_std = mean_std

    def predict(
        self,
        test_dl: DataLoader,
        predict_mode=PredictMode.Standard,
        nb_images_to_save=10,
        compute_own_mean_std=False,
    ) -> Optional[list]:
        """
        Parameters
        ----------
        test_dl : DataLoader
            DataLoader for test set
        predict_mode : PredictMode, optional
            By default, do not return predictions
        nb_images_to_save : int, optional
            Number of images to save, by default 10
            nb_images_to_save == -1 => save all images

        """

        # Create folder to save predictions
        os.makedirs(self.params.output_dir, exist_ok=True)

        # Update test_dl with saved mean and std
        if compute_own_mean_std:
            mean_std = get_mean_and_std([test_dl])
            test_dl.dataset.mean_std = mean_std
        else:
            self.read_mean_std(test_dl, self.params)

        # Initialise transforms before prediction
        test_dl.dataset.initialize_transforms()

        self.model.eval()  # Set eval mode for model

        # Create list with images indexes to save predictions, to avoid saving all
        num_batches_test = len(test_dl)
        total_images = num_batches_test * test_dl.batch_size
        if nb_images_to_save == -1:
            nb_images_to_save = total_images
        else:
            nb_images_to_save = min(total_images, nb_images_to_save)
        images_to_save = random_sample(range(total_images), nb_images_to_save)

        # Reset metric
        test_metric = self.metric_class(self.device, self.params.nb_classes)

        with torch.no_grad():
            # Use trained model to predict on test set
            predictions = self.batch_predict(
                test_dl,
                images_to_save,
                num_batches_test,
                test_metric,
                predict_mode,
            )

        if predict_mode != PredictMode.Standard:
            return predictions

        # Display box plot
        score, additional_results = test_metric.get_score()

        self.plot_confusion_matrix(additional_results)

        # Compute accuracy
        accuracy_message = (
            f"Average {test_metric.get_name()}: {round(score, 2)}"
        )
        print("\n" + accuracy_message)
        self.training_information.score = score

    def write_useful_information(self) -> None:
        # Update parameters file with all useful information
        os.makedirs(self.params.models_folder, exist_ok=True)
        with open(self.parameters_path, "a") as f:
            for attribute, value in vars(self.training_information).items():
                f.write("%s;%s\n" % (attribute, value))
        f.close()

        if self.params.global_results_path == "":
            return

        # If global results file does not exist, create it
        if not os.path.exists(self.params.global_results_path):
            with open(self.params.global_results_path, "w") as f:
                f.write(
                    "model;git hash;data;model id;train number;val number;test number;epochs;learning rate;batch size;training_time;score;additional score\n"
                )
            f.close()

        # Store useful results in global results file
        with open(self.params.global_results_path, "a") as f:
            f.write(f"{self.params.name};")
            f.write(f"{self.training_information.git_hash};")
            f.write(f"{self.params.data_dir};")
            f.write(f"{self.params.format_now};")
            f.write(f"{self.params.train_number};")
            f.write(f"{self.params.val_number};")
            f.write(f"{self.params.test_number};")
            f.write(f"{self.params.num_epochs};")
            f.write(f"{self.params.learning_rate};")
            f.write(f"{self.params.batch_size};")
            f.write(f"{self.training_information.training_time};")
            f.write(f"{self.training_information.score};")
            f.write(f"{self.training_information.additional_score};\n")
        f.close()

    def display_training_curves(self):
        """
        Show tensorboard curves in a matplotlib figure.
        """
        event_acc = EventAccumulator(self.params.tensorboard_folder_path)
        event_acc.Reload()
        scalar_tags = event_acc.Tags()["scalars"]

        fig = plt.figure(figsize=(12, 12))

        nb_columns = 2
        nb_lines = math.ceil(len(scalar_tags) / nb_columns)
        for idx, scalar_tag in enumerate(scalar_tags):
            _, step_nums, vals = zip(*event_acc.Scalars(scalar_tag))
            ax = fig.add_subplot(nb_lines, nb_columns, idx + 1)
            ax.plot(step_nums, vals)
            ax.set_title(scalar_tag)
            ax.set_xlabel("Steps")

        plt.show()

    def display_training_images(self, patterns: Optional[list[str]] = None):
        """
        Show tensorboard images in a matplotlib figure.
        Plot first 8 images, last printed to tensorboard.
        """
        if patterns is None:
            patterns = ["val*"]

        event_acc = EventAccumulator(self.params.tensorboard_folder_path)
        event_acc.Reload()
        image_tags = event_acc.Tags()["images"]
        filtered_images_tags = extract_patterns(image_tags, patterns)

        fig = plt.figure(figsize=(4, 6))

        for idx, image_tag in enumerate(filtered_images_tags):
            if idx == 6:
                break
            is_ground_truth = idx % 2 == 0
            _, _, encoded_string, _, _ = zip(*event_acc.Images(image_tag))
            # -1 for last image (most recent)
            image = tf.image.decode_image(
                encoded_string[-1], channels=1
            ).numpy()
            ax = fig.add_subplot(3, 2, idx + 1)
            ax.imshow(image, cmap="gray")
            ax.axis("off")
            ax.set_title(
                "Ground truth mask" if is_ground_truth else "Predicted mask"
            )

        plt.show()
