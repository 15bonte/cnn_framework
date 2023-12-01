import torch
from torchmetrics.classification import Accuracy

from .abstract_metric import AbstractMetric


class ClassificationAccuracy(AbstractMetric):
    """
    Classification accuracy.
    """

    def __init__(self, *args):
        super().__init__(*args)
        self.true, self.pred = torch.empty(0).to(self.device), torch.empty(
            0
        ).to(self.device)
        self.metric = Accuracy(
            task="multiclass", num_classes=self.num_classes, average="macro"
        ).to(self.device)

    def update(self, predictions, targets, adds=None, mean_std=None):
        # From vector to classification
        predictions_argmax = torch.argmax(predictions, dim=1)
        targets_argmax = torch.argmax(targets, dim=1)
        # Update metric
        self.metric.update(
            predictions_argmax,
            targets_argmax,
        )
        # Update current values
        self.true = torch.cat((self.true, targets_argmax))
        self.pred = torch.cat((self.pred, predictions_argmax))

    def get_score(self):
        return self.metric.compute().item(), (self.true, self.pred)

    def reset(self):
        self.metric = Accuracy(
            task="multiclass", num_classes=self.num_classes, average="macro"
        ).to(self.device)
        self.true, self.pred = torch.empty(0).to(self.device), torch.empty(
            0
        ).to(self.device)
