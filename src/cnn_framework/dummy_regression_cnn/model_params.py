from ..utils.ModelParams import ModelParams
from ..utils.dimensions import Dimensions


class DummyModelParams(ModelParams):
    """
    Dummy model params.
    """

    def __init__(self):
        super().__init__("dummy_regression_cnn")

        self.input_dimensions = Dimensions(height=128, width=128)

        self.num_epochs = 10
        self.learning_rate = 1e-2

        self.nb_classes = 1
        self.nb_modalities = 3
