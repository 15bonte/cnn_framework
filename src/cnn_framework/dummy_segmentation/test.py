import torch

from ..utils.DataManagers import DefaultDataManager
from ..utils.data_loader_generators.DataLoaderGenerator import DataLoaderGenerator
from ..utils.metrics import PCC
from ..utils.model_managers.ModelManager import ModelManager
from ..utils.parsers.CnnParser import CnnParser

from .data_set import DummyDataSet
from .model_params import DummyModelParams
from .model import UNet


def main(params):
    # Data loading
    loader_generator = DataLoaderGenerator(params, DummyDataSet, DefaultDataManager)
    _, _, test_dl = loader_generator.generate_data_loader()

    # Model definition
    # Load pretrained model
    model = UNet(
        nb_classes=params.out_channels,
        nb_input_channels=params.nb_modalities * params.nb_stacks_per_modality,
    )
    model.load_state_dict(torch.load(params.model_load_path))

    manager = ModelManager(model, params, PCC)

    manager.predict(test_dl)

    manager.write_useful_information()


if __name__ == "__main__":
    parser = CnnParser()
    args = parser.arguments_parser.parse_args()

    parameters = DummyModelParams()
    parameters.update(args)

    main(parameters)
