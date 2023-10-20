from cnn_framework.utils.data_loader_generators.data_loader_generator import DataLoaderGenerator
from cnn_framework.utils.data_managers.default_data_manager import DefaultDataManager
from cnn_framework.utils.lr_schedulers.linear_warmup_cosine_annealing_lr import (
    LinearWarmupCosineAnnealingLR,
)
from cnn_framework.utils.metrics.positive_pair_matching_metric import PositivePairMatchingMetric
from cnn_framework.utils.optimizers.lars import create_optimizer_lars
from cnn_framework.utils.parsers.cnn_parser import CnnParser
from cnn_framework.utils.model_managers.contrastive_model_manager import ContrastiveModelManager
from cnn_framework.utils.losses.info_nce_loss import InfoNceLoss

from cnn_framework.dummy_sim_clr.data_set import SimCLRDataSet
from cnn_framework.dummy_sim_clr.model import ResNetSimCLR
from cnn_framework.dummy_sim_clr.model_params import SimCLRModelParams


def main(params):
    loader_generator = DataLoaderGenerator(params, SimCLRDataSet, DefaultDataManager)
    train_dl, val_dl, test_dl = loader_generator.generate_data_loader()

    model = ResNetSimCLR(
        nb_input_channels=len(params.c_indexes) * len(params.z_indexes),
    )
    manager = ContrastiveModelManager(model, params, PositivePairMatchingMetric)

    optimizer = create_optimizer_lars(
        model,
        lr=params.learning_rate,
        momentum=0.9,
        weight_decay=params.weight_decay,
        bn_bias_separately=True,
        epsilon=1e-5,
    )
    loss_function = InfoNceLoss(manager.device, params.temperature)  # define the loss function
    lr_scheduler = LinearWarmupCosineAnnealingLR(
        optimizer, warmup_epochs=params.nb_warmup_epochs, max_epochs=params.num_epochs
    )
    manager.fit(train_dl, val_dl, optimizer, loss_function, lr_scheduler=lr_scheduler)

    manager.predict(test_dl)

    manager.write_useful_information()


if __name__ == "__main__":
    parser = CnnParser()
    args = parser.arguments_parser.parse_args()

    parameters = SimCLRModelParams()
    parameters.update(args)

    main(parameters)
