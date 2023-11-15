import torch
import torch.nn as nn
from torch import optim

from ..utils.data_loader_generators.data_loader_generator import (
    DataLoaderGenerator,
)
from ..utils.model_managers.regression_model_manager import (
    RegressionModelManager,
)
from ..utils.data_managers.default_data_manager import DefaultDataManager
from ..utils.metrics.mean_error_metric import MeanErrorMetric

from .model import DummyCnn
from .data_set import DummyRegressionCnnDataSet


def training(params):
    """
    Training function for dummy regression.
    """

    loader_generator = DataLoaderGenerator(
        params, DummyRegressionCnnDataSet, DefaultDataManager
    )
    train_dl, val_dl, test_dl = loader_generator.generate_data_loader()

    # Load pretrained model
    model = DummyCnn(
        nb_classes=params.nb_classes,
        nb_input_channels=len(params.c_indexes) * len(params.z_indexes),
    )
    manager = RegressionModelManager(model, params, MeanErrorMetric)

    optimizer = optim.Adam(
        model.parameters(),
        lr=float(params.learning_rate),
        betas=(params.beta1, params.beta2),
    )  # define the optimization
    loss_function = nn.L1Loss()  # define the loss function

    manager.fit(train_dl, val_dl, optimizer, loss_function)

    for model_path, name in zip(
        [manager.model_save_path_early_stopping, manager.model_save_path],
        ["early stopping", "final"],
    ):
        print(f"\nPredicting with {name} model.")
        # Update model with saved one
        manager.model.load_state_dict(torch.load(model_path))
        manager.predict(test_dl)

    manager.write_useful_information()

    return manager.training_information.score
