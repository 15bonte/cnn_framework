from torch import optim
from torch import nn

from .data_set import DummyDataSet
from .model_params import DummyModelParams
from .model import UNet

from ..utils.parsers.CnnParser import CnnParser
from ..utils.data_loader_generators.data_loader_generator import DataLoaderGenerator
from ..utils.model_managers.ModelManager import ModelManager
from ..utils.DataManagers import DefaultDataManager
from ..utils.metrics import PCC


def main(params):
    loader_generator = DataLoaderGenerator(params, DummyDataSet, DefaultDataManager)
    train_dl, val_dl, test_dl = loader_generator.generate_data_loader()

    # Load pretrained model
    model = UNet(
        nb_classes=params.out_channels,
        nb_input_channels=params.nb_modalities * params.nb_stacks_per_modality,
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
