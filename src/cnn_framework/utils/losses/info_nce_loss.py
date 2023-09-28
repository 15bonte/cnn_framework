import torch.nn as nn
import torch


class InfoNceLoss(nn.Module):
    """
    Loss function for contrastive learning.
    Taken from https://github.com/sthalles/SimCLR/blob/1848fc934ad844ae630e6c452300433fe99acfd9/simclr.py#L76.
    """

    def __init__(self, device, temperature):
        super().__init__()
        self.device = device
        self.temperature = temperature
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

    def forward(self, logits, labels):
        logits = logits / self.temperature
        loss = self.criterion(logits, labels)
        return loss
