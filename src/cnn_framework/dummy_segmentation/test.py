import torch

from ..utils.data_managers.default_data_manager import DefaultDataManager
from ..utils.data_loader_generators.data_loader_generator import DataLoaderGenerator
from ..utils.metrics.pcc import PCC
from ..utils.model_managers.model_manager import ModelManager

from .data_set import DummyDataSet
from .model import UNet


def testing(params):
    """
    Testing function for dummy regression.
    """

    # Data loading
    loader_generator = DataLoaderGenerator(params, DummyDataSet, DefaultDataManager)
    _, _, test_dl = loader_generator.generate_data_loader()

    # Model definition
    # Load pretrained model
    model = UNet(
        nb_classes=params.out_channels,
        nb_input_channels=len(params.c_indexes) * len(params.z_indexes),
    )
    model.load_state_dict(torch.load(params.model_load_path))

    manager = ModelManager(model, params, PCC)

    manager.predict(test_dl)

    manager.write_useful_information()
