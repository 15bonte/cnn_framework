import torch

from cnn_framework.utils.data_managers.default_data_manager import DefaultDataManager
from cnn_framework.utils.data_loader_generators.data_loader_generator import DataLoaderGenerator
from cnn_framework.utils.metrics.pcc import PCC
from cnn_framework.utils.model_managers.model_manager import ModelManager
from cnn_framework.utils.parsers.cnn_parser import CnnParser

from cnn_framework.dummy_segmentation.data_set import DummyDataSet
from cnn_framework.dummy_segmentation.model_params import DummyModelParams
from cnn_framework.dummy_segmentation.model import UNet


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