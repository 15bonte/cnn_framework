from ..utils.ModelParams import ModelParams
from ..utils.dimensions import Dimensions


class DummyModelParams(ModelParams):
    """
    Segmentation model params.
    """

    def __init__(self):
        super().__init__("dummy_segmentation")

        self.input_dimensions = Dimensions(height=128, width=128)
        self.learning_rate = 1e-4

        self.train_ratio = 0.8
        self.val_ratio = 0.2
        self.test_ratio = 0.2

        self.out_channels = 1
        self.nb_modalities = 3