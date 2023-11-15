import torch

from .data_set import DummyCnnDataSet

from .model import DummyCnn

from ..utils.data_loader_generators.classifier_data_loader_generator import (
    ClassifierDataLoaderGenerator,
)
from ..utils.model_managers.cnn_model_manager import CnnModelManager
from ..utils.data_managers.default_data_manager import DefaultDataManager
from ..utils.metrics.classification_accuracy import ClassificationAccuracy


def testing(params):
    """
    Testing function for dummy classification.
    """

    loader_generator = ClassifierDataLoaderGenerator(
        params, DummyCnnDataSet, DefaultDataManager
    )
    _, _, test_dl = loader_generator.generate_data_loader()

    # Model definition
    model = DummyCnn(
        nb_classes=params.nb_classes,
        nb_input_channels=len(params.c_indexes) * len(params.z_indexes),
    )
    model.load_state_dict(torch.load(params.model_load_path))

    manager = CnnModelManager(model, params, ClassificationAccuracy)

    manager.predict(test_dl)

    manager.write_useful_information()
