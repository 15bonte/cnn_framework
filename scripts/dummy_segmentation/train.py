from torch import optim
from torch import nn

from cnn_framework.dummy_segmentation.data_set import DummyDataSet
from cnn_framework.dummy_segmentation.model_params import DummyModelParams
from cnn_framework.dummy_segmentation.model import UNet

from cnn_framework.utils.parsers.cnn_parser import CnnParser
from cnn_framework.utils.data_loader_generators.data_loader_generator import DataLoaderGenerator
from cnn_framework.utils.model_managers.model_manager import ModelManager
from cnn_framework.utils.data_managers.default_data_manager import DefaultDataManager
from cnn_framework.utils.metrics.pcc import PCC


def main(params):
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


if __name__ == "__main__":
    parser = CnnParser()
    args = parser.arguments_parser.parse_args()

    parameters = DummyModelParams()
    parameters.update(args)

    main(parameters)
