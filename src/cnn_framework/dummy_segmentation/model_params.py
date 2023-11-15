from ..utils.model_params.base_model_params import BaseModelParams


class DummyModelParams(BaseModelParams):
    """
    Segmentation model params.
    """

    def __init__(self):
        super().__init__("dummy_segmentation")
        self.learning_rate = 1e-4

        self.train_ratio = 0.8
        self.val_ratio = 0.1
        self.test_ratio = 0.1

        self.out_channels = 1
        self.c_indexes = [0, 1, 2]
