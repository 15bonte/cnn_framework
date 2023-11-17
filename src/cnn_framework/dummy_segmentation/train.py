from torch import optim
from torch import nn

from ..utils.data_loader_generators.data_loader_generator import DataLoaderGenerator
from ..utils.model_managers.model_manager import ModelManager
from ..utils.data_managers.default_data_manager import DefaultDataManager
from ..utils.metrics.pcc import PCC

from .data_set import DummyDataSet
from .model import UNet


def training(params):
    """
    Training function for dummy segmentation.
    """

    loader_generator = DataLoaderGenerator(params, DummyDataSet, DefaultDataManager)
    train_dl, val_dl, test_dl = loader_generator.generate_data_loader()

    # Load pretrained model
    model = UNet(
        nb_classes=params.out_channels,
        nb_input_channels=len(params.c_indexes) * len(params.z_indexes),
    )
    manager = ModelManager(model, params, PCC)

    optimizer = optim.Adam(
        model.parameters(),
        lr=float(params.learning_rate),
        betas=(params.beta1, params.beta2),
    )  # define the optimization
    loss_function = nn.L1Loss()
    manager.fit(train_dl, val_dl, optimizer, loss_function)

    manager.predict(test_dl)

    manager.write_useful_information()

    return manager.training_information.score
