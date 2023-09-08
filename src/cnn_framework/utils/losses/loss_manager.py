from typing import List
import torch.nn as nn


class LossManager:
    def __init__(self, *args) -> None:
        self.losses_function: List[nn.Module] = args
        self.running_losses = [0] * len(self.losses_function)

    def __call__(self, *args, **kwargs):
        # First compute main loss
        main_loss = self.losses_function[0](*args, **kwargs)
        self.running_losses[0] += main_loss
        # Then compute other losses, used only for logging
        for i, loss_function in enumerate(self.losses_function[1:]):
            self.running_losses[i + 1] += loss_function(*args, **kwargs)
        return main_loss

    def get_running_losses(self):
        """
        Return a list of tuples (loss_name, loss_value)
        """
        named_running_losses = []
        for i, loss_function in enumerate(self.losses_function):
            named_running_losses.append((loss_function.__class__.__name__, self.running_losses[i]))
            self.running_losses[i] = 0
        return named_running_losses
