from cnn_framework.dummy_regression_cnn.model_params import DummyModelParams
from cnn_framework.utils.parsers.cnn_parser import CnnParser
from cnn_framework.dummy_regression_cnn.train import training


if __name__ == "__main__":
    parser = CnnParser()
    args = parser.arguments_parser.parse_args()

    parameters = DummyModelParams()
    parameters.update(args)

    training(parameters)
