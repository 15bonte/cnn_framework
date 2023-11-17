from cnn_framework.dummy_segmentation.model_params import DummyModelParams
from cnn_framework.dummy_segmentation.train import training
from cnn_framework.utils.parsers.cnn_parser import CnnParser


if __name__ == "__main__":
    parser = CnnParser()
    args = parser.arguments_parser.parse_args()

    parameters = DummyModelParams()
    parameters.update(args)

    training(parameters)
