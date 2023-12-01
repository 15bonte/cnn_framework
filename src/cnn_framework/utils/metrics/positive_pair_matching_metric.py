import torch
import numpy as np

from .abstract_metric import AbstractMetric


class PositivePairMatchingMetric(AbstractMetric):
    """
    Computes the accuracy over the k top predictions for the specified values of k.
    """

    def __init__(self, *args):
        super().__init__(*args)
        self.local_matchings = []

    def update(self, predictions, targets, adds=None, mean_std=None):
        topk = (1,)

        with torch.no_grad():
            maxk = topk[0]
            batch_size = targets.size(0)

            _, pred = predictions.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(targets.view(1, -1).expand_as(pred))

            correct_k = correct[:maxk].reshape(-1).float().sum(0, keepdim=True)
            matching = correct_k.div_(batch_size)[0].item()
            self.local_matchings.append(matching)

    def get_score(self):
        return np.mean(self.local_matchings), None

    def reset(self):
        self.local_matchings = []
