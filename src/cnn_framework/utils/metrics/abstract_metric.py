from abc import abstractmethod


class AbstractMetric:
    """
    Abstract class for metrics.
    """

    def __init__(self, device="cpu", num_classes=None):
        self.device = device
        self.num_classes = num_classes

    def get_name(self):
        return self.__class__.__name__

    @abstractmethod
    def update(self, predictions, targets, adds=None, mean_std=None):
        pass

    @abstractmethod
    def get_score(self):
        return 0, None

    @abstractmethod
    def reset(self):
        pass
