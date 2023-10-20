import torch

from cnn_framework.dummy_regression_cnn.data_set import DummyRegressionCnnDataSet
from cnn_framework.dummy_regression_cnn.model_params import DummyModelParams
from cnn_framework.dummy_regression_cnn.model import DummyCnn

from cnn_framework.utils.data_loader_generators.data_loader_generator import DataLoaderGenerator
from cnn_framework.utils.model_managers.regression_model_manager import RegressionModelManager
from cnn_framework.utils.data_managers.default_data_manager import DefaultDataManager
from cnn_framework.utils.metrics.mean_squared_error_metric import MeanSquaredErrorMetric
from cnn_framework.utils.parsers.cnn_parser import CnnParser


def main(params):
    loader_generator = DataLoaderGenerator(params, DummyRegressionCnnDataSet, DefaultDataManager)
    _, _, test_dl = loader_generator.generate_data_loader()

    # Model definition
    model = DummyCnn(
        nb_classes=params.nb_classes,
        nb_input_channels=len(params.c_indexes) * len(params.z_indexes),
    )
    model.load_state_dict(torch.load(params.model_load_path))

    manager = RegressionModelManager(model, params, MeanSquaredErrorMetric)

    manager.predict(test_dl)

    manager.write_useful_information()


if __name__ == "__main__":
    parser = CnnParser()
    args = parser.arguments_parser.parse_args()

    parameters = DummyModelParams()
    parameters.update(args)

    main(parameters)
