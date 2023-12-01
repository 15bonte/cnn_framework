from torchmetrics import MeanSquaredError

from .abstract_metric import AbstractMetric


class MeanSquaredErrorMetric(AbstractMetric):
    """
    Mean Squared Error.
    """

    def __init__(self, *args):
        super().__init__(*args)
        self.metric = MeanSquaredError(squared=True).to(self.device)

    def update(self, predictions, targets, adds=None, mean_std=None):
        # Update metric
        self.metric.update(
            predictions,
            targets,
        )

    def get_score(self):
        return -self.metric.compute().item(), None

    def reset(self):
        self.metric = MeanSquaredError(squared=True).to(self.device)
