from typing import Optional
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..losses.loss_manager import LossManager
from ..data_sets.dataset_output import DatasetOutput
from ..display_tools import make_image_matplotlib_displayable
from ..metrics.abstract_metric import AbstractMetric
from .model_manager import ModelManager


class ContrastiveModelManager(ModelManager):
    """
    Contrastive learning model manager.
    """

    def compute_loss(
        self,
        dl_element: DatasetOutput,
        dl_metric: AbstractMetric,
        data_loader: DataLoader,
        loss_manager: Optional[LossManager] = None,
    ):
        # Read data loader element
        inputs = dl_element.input
        batch_size = inputs.shape[0]

        # Construct all_inputs: [input[0], input[1], input[2], ..., input_bis[0], input_bis[1], input_bis[2], ...]
        all_inputs = torch.zeros(
            (inputs.shape[0] * self.params.n_views, *inputs.shape[1:])
        )
        for input_index, image_index in enumerate(dl_element.index):
            dl_element_bis = data_loader.dataset[image_index.item()]
            assert dl_element_bis.index == image_index
            # Fill input
            all_inputs[input_index] = inputs[input_index]
            all_inputs[input_index + batch_size] = dl_element_bis.input

        all_inputs = all_inputs.to(self.device).float()
        features = self.model(all_inputs)

        # Define labels
        labels = torch.cat(
            [torch.arange(batch_size) for i in range(self.params.n_views)],
            dim=0,
        )
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # Discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(
            similarity_matrix.shape[0], -1
        )

        # Select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # Select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(
            similarity_matrix.shape[0], -1
        )

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        # Update metric
        dl_metric.update(logits, labels, mean_std=data_loader.dataset.mean_std)

        # Use other inputs as fake predictions
        dl_element.prediction = all_inputs[batch_size:]

        # Compute loss if possible
        if loss_manager is None:
            return None
        loss = loss_manager(logits, labels)

        return loss

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

        # Current data set mean & std
        mean_std = self.dl[name].dataset.mean_std

        # Log the results images
        for i, (input_np, input_np_bis, image_name) in enumerate(
            zip(
                numpy_dl_element.input,
                numpy_dl_element.prediction,
                image_names,
            )
        ):
            # Do not save too many images
            if i == self.params.nb_tensorboard_images_max:
                break
            plt.imshow(
                make_image_matplotlib_displayable(input_np, mean_std=mean_std)
            )
            self.writer.add_figure(
                f"{name}/{image_name}/first_transformed",
                plt.gcf(),
                current_batch,
            )

            plt.imshow(
                make_image_matplotlib_displayable(
                    input_np_bis, mean_std=mean_std
                )
            )
            self.writer.add_figure(
                f"{name}/{image_name}/second_transformed",
                plt.gcf(),
                current_batch,
            )

    def save_results(self, _, __, ___):
        pass
