from torchmetrics.image import StructuralSimilarityIndexMeasure

from ..augmentations.unnormalize import UnNormalize
from ..tools import get_padding_coordinates
from .abstract_metric import AbstractMetric


class SSIM(AbstractMetric):
    """
    Structural Similarity Index Metric.
    """

    def __init__(self, *args):
        super().__init__(*args)
        self.metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(
            self.device
        )

    def update(self, predictions, targets, adds=None, mean_std=None):
        # if mean_std is not None:
        #     # Expect reconstruction images to be first channels
        #     # Unormalize images to use SSIM
        #     nb_input_channels = predictions.shape[1]
        #     un_normalize = UnNormalize(
        #         mean=mean_std["mean"][:nb_input_channels],
        #         std=mean_std["std"][:nb_input_channels],
        #     )
        #     predictions_copy, targets_copy = (
        #         predictions.clone(),
        #         targets.clone(),
        #     )
        #     for idx in range(predictions_copy.size(0)):
        #         predictions_copy[idx] = un_normalize(
        #             image=predictions_copy[idx]
        #         )["image"]
        #         targets_copy[idx] = un_normalize(image=targets_copy[idx])[
        #             "image"
        #         ]
        # else:
        #     predictions_copy, targets_copy = predictions, targets
        # # Remove padding
        # y_min, y_max, x_min, x_max = get_padding_coordinates(
        #     targets_copy[0, 0]
        # )
        # self.metric.update(
        #     predictions[..., y_min:y_max, x_min:x_max],
        #     targets[..., y_min:y_max, x_min:x_max],
        # )
        self.metric.update(predictions, targets)

    def get_score(self):
        return self.metric.compute().item(), None

    def reset(self):
        self.metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(
            self.device
        )
