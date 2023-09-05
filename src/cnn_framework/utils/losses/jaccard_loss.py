import torch.nn as nn
from torchmetrics import JaccardIndex


class JaccardLoss(nn.Module):
    def __init__(self, num_classes=2):
        super(JaccardLoss, self).__init__()
        self.jaccard_score = JaccardIndex(num_classes=num_classes, task="binary")

    def forward(self, inputs, targets):
        return 1 - self.jaccard_score(inputs, targets)
