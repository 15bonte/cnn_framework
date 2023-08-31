from typing import Optional
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

# Taken from https://github.com/AdeelH/pytorch-multi-class-focal-loss/blob/master/focal_loss.py


class FocalLossAdeelH(nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.
    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.
    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(
        self,
        alpha: Optional[Tensor] = None,
        gamma: float = 0.0,
        reduction: str = "mean",
        ignore_index: int = -100,
    ):
        """Constructor.
        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ("mean", "sum", "none"):
            raise ValueError('Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(weight=alpha, reduction="none", ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ["alpha", "gamma", "ignore_index", "reduction"]
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_vals = [f"'{v}'" if isinstance(v, str) else v for v in arg_vals]
        arg_strs = [f"{k}={v}" for k, v in zip(arg_keys, arg_vals)]
        arg_str = ", ".join(arg_strs)
        return f"{type(self).__name__}({arg_str})"

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.0)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt) ** self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class FocalLoss(nn.Module):
    r"""Implementation of the `focal loss <https://arxiv.org/pdf/1708.02002.pdf>`
    Implements FocalLoss with either probabilities or logits.
    ----------
    alpha: float
        Focal Loss ``alpha`` parameter
        If negative, balance is ignored.
    gamma: float
        Focal Loss ``gamma`` parameter
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.use_cuda = torch.cuda.is_available()

        self.softmax = nn.Softmax(dim=-1)
        self.log_soft_max = nn.LogSoftmax(dim=-1)

    def forward(self, scores: Tensor, target: Tensor) -> Tensor:
        r"""
        Parameters
        ----------
        input: Tensor
            input tensor with scores
        target: Tensor
            target tensor with the actual classes or probabilities
            B or (B, C)
        --------
        """

        # Get useful dimensions
        num_batch = scores.size(0)
        num_class = scores.size(1)

        # Convert target to one-hot if necessary
        if len(target.shape) == 1:
            target = target.view(-1, 1).squeeze()
            # Adapt to special case of one batch
            if num_batch == 1:
                target = torch.unsqueeze(target, 0)
            target = torch.eye(num_class)[target]  # B, C

        # Adapt to special case of one batch
        if num_batch == 1:
            scores = torch.unsqueeze(scores, 0)

        # Move to GPU if possible
        if self.use_cuda:
            target = target.cuda()
            scores = scores.cuda()

        target = target.contiguous()

        # Compute weights is alpha is positive
        if self.alpha > 0:
            balanced_weights = self.alpha * target + (1 - self.alpha) * (1 - target)
        else:
            balanced_weights = 1

        probabilities = self.softmax(scores)
        weights = (balanced_weights * (1 - probabilities).pow(self.gamma)).detach()
        logs = self.log_soft_max(scores)

        cross_entropy_loss = -weights * logs * target  # B, C

        return torch.sum(cross_entropy_loss) / num_batch
