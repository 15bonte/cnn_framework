from cnn_framework.utils.enum import NormalizeMethods


class Normalization:
    def __init__(self, method: NormalizeMethods) -> None:
        self.method = method
