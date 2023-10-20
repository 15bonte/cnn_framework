import torch

from cnn_framework.utils.data_loader_generators.data_loader_generator import DataLoaderGenerator
from cnn_framework.utils.data_managers.default_data_manager import DefaultDataManager
from cnn_framework.utils.metrics.positive_pair_matching_metric import PositivePairMatchingMetric
from cnn_framework.utils.parsers.cnn_parser import CnnParser
from cnn_framework.utils.model_managers.contrastive_model_manager import ContrastiveModelManager

from cnn_framework.dummy_sim_clr.data_set import SimCLRDataSet
from cnn_framework.dummy_sim_clr.model import ResNetSimCLR
from cnn_framework.dummy_sim_clr.model_params import SimCLRModelParams


def main(params):
    loader_generator = DataLoaderGenerator(params, SimCLRDataSet, DefaultDataManager)
    _, _, test_dl = loader_generator.generate_data_loader()

    model = ResNetSimCLR(
        nb_input_channels=len(params.c_indexes) * len(params.z_indexes),
    )
    manager = ContrastiveModelManager(model, params, PositivePairMatchingMetric)

    model.load_state_dict(torch.load(params.model_load_path))

    manager.predict(test_dl)

    manager.write_useful_information()


if __name__ == "__main__":
    parser = CnnParser()
    args = parser.arguments_parser.parse_args()

    parameters = SimCLRModelParams()
    parameters.update(args)

    main(parameters)
