from cnn_framework.dummy_cnn.model_params import DummyModelParams
from cnn_framework.dummy_cnn.test import testing
from cnn_framework.utils.parsers.cnn_parser import CnnParser


if __name__ == "__main__":
    parser = CnnParser()
    args = parser.arguments_parser.parse_args()

    parameters = DummyModelParams()
    parameters.update(args)

    testing(parameters)
