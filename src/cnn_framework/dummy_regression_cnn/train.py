from cProfile import Profile
from pstats import Stats

import torch
import torch.nn as nn
from torch import optim

from .data_set import DummyRegressionCnnDataSet
from .model_params import DummyModelParams
from .model import DummyCnn

from ..utils.data_loader_generators.data_loader_generator import DataLoaderGenerator
from ..utils.model_managers.RegressionModelManager import RegressionModelManager
from ..utils.DataManagers import DefaultDataManager
from ..utils.metrics import MeanSquaredErrorMetric
from ..utils.parsers.CnnParser import CnnParser

MONITOR_FUNCTIONS = False


def main(params):
    loader_generator = DataLoaderGenerator(params, DummyRegressionCnnDataSet, DefaultDataManager)
    train_dl, val_dl, test_dl = loader_generator.generate_data_loader()

    # Load pretrained model
    model = DummyCnn(
        nb_classes=params.nb_classes,
        nb_input_channels=params.nb_modalities * params.nb_stacks_per_modality,
    )
    manager = RegressionModelManager(model, params, MeanSquaredErrorMetric)

    optimizer = optim.Adam(
        model.parameters(),
        lr=float(params.learning_rate),
        betas=(params.beta1, params.beta2),
    )  # define the optimization
    loss_function = nn.L1Loss()  # define the loss function

    if MONITOR_FUNCTIONS:
        profiler = Profile()
        training_test_function = lambda: manager.fit(train_dl, val_dl, optimizer, loss_function)
        profiler.runcall(training_test_function)

        stats = Stats(profiler)
        stats.strip_dirs()
        stats.sort_stats("cumulative")
        stats.print_stats(20)
    else:
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


if __name__ == "__main__":
    parser = CnnParser()
    args = parser.arguments_parser.parse_args()

    parameters = DummyModelParams()
    parameters.update(args)

    main(parameters)
