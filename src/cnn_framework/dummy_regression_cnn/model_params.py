from ..utils.model_params.base_model_params import BaseModelParams


class DummyModelParams(BaseModelParams):
    """
    Dummy model params.
    """

    def __init__(self):
        super().__init__("dummy_regression_cnn")

        self.train_ratio = 0.8
        self.val_ratio = 0.1
        self.test_ratio = 0.1

        self.num_epochs = 25
        self.learning_rate = 1e-3

        self.nb_classes = 1
        self.c_indexes = [0, 1, 2]
