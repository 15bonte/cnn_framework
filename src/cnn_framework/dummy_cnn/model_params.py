from ..utils.ModelParams import ModelParams
from ..utils.dimensions import Dimensions


class DummyModelParams(ModelParams):
    """
    Dummy model params.
    """

    def __init__(self):
        super().__init__("dummy_cnn")

        self.input_dimensions = Dimensions(height=128, width=128)

        self.num_epochs = 10
        self.learning_rate = 1e-4

        self.nb_classes = 2
        self.class_names = ["Square", "Circle"]
        self.nb_modalities = 3
