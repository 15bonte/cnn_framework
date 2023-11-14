from cnn_framework.utils.enum import NormalizeMethods


class Normalization:
    def __init__(self, method: NormalizeMethods, axis=None) -> None:
        self.method = method
        self.axis = axis
