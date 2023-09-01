import os

from pythae.pipelines import TrainingPipeline
from pythae.trainers import BaseTrainerConfig
from pythae.trainers.training_callbacks import WandbCallback
from pythae.models import AutoModel, FucciVAEConfig, FucciVAE

from .data_set import DummyVAEDataSet
from .model_params import DummyVAEModelParams

from ..utils.DataManagers import DefaultDataManager
from ..utils.data_loader_generators.DataLoaderGenerator import DataLoaderGenerator
from ..utils.metrics import MeanSquaredErrorMetric
from ..utils.model_managers.ModelManager import ModelManager
from ..utils.model_managers.VAEModelManager import VAEModelManager
from ..utils.parsers.VAEParser import VAEParser

from .decoder import CustomDecoder
from .encoder import CustomEncoder


def main(params):
    loader_generator = DataLoaderGenerator(params, DummyVAEDataSet, DefaultDataManager)
    train_dl, val_dl, test_dl = loader_generator.generate_data_loader()

    # Create folder to save model
    os.makedirs(params.models_folder, exist_ok=True)

    my_training_config = BaseTrainerConfig(
        output_dir=params.models_folder,
        num_epochs=params.num_epochs,
        learning_rate=params.learning_rate,
        per_device_train_batch_size=params.batch_size,
        per_device_eval_batch_size=params.batch_size,
        train_dataloader_num_workers=params.num_workers,
        eval_dataloader_num_workers=params.num_workers,
        steps_saving=None,
        optimizer_cls="AdamW",
        optimizer_params={
            "weight_decay": params.weight_decay,
            "betas": (params.beta1, params.beta2),
        },
        scheduler_cls="ReduceLROnPlateau",
        scheduler_params={"patience": 5, "factor": 0.5},
    )

    # Set up the model configuration
    my_vae_config = FucciVAEConfig(
        gamma=params.gamma,
        delta=params.delta,
        nb_classes=params.nb_classes,
        reconstruction_loss=params.reconstruction_loss,
        kld_loss=params.kld_loss,
        input_dim=(
            params.nb_modalities * params.nb_stacks_per_modality,
            params.input_dimensions.height,
            params.input_dimensions.width,
        ),
        latent_dim=params.latent_dim,
        beta=params.beta,
        uses_default_decoder=False,
        uses_default_encoder=False,
    )

    # Build the model
    if params.model_pretrained_path:
        vae_model = AutoModel.load_from_folder(params.model_pretrained_path)
        # Update modifiable parameters
        vae_model.model_config = my_vae_config
        vae_model.beta = my_vae_config.beta
        vae_model.gamma = my_vae_config.gamma
        vae_model.delta = my_vae_config.delta
    else:
        encoder = CustomEncoder(params, my_vae_config)
        print(f"Number of parameters in encoder: {sum(p.numel() for p in encoder.parameters())}")
        decoder = CustomDecoder(params, my_vae_config)
        print(f"Number of parameters in decoder: {sum(p.numel() for p in decoder.parameters())}")
        vae_model = FucciVAE(encoder=encoder, decoder=decoder, model_config=my_vae_config)

    # Build the Pipeline
    pipeline = TrainingPipeline(training_config=my_training_config, model=vae_model)

    # Compute mean_std for future normalization
    model_manager = ModelManager(vae_model, params, None)
    model_manager.compute_and_save_mean_std(train_dl, val_dl)

    train_dl.dataset.initialize_transforms()
    val_dl.dataset.initialize_transforms()

    # Create you callback
    callbacks = []  # the TrainingPipeline expects a list of callbacks
    wandb_cb = WandbCallback()  # Build the callback
    # SetUp the callback
    wandb_cb.setup(
        training_config=my_training_config,  # training config
        model_config=my_vae_config,  # model config
        project_name="vae-dummy",  # specify your wandb project
        entity_name="cbio-bis",  # specify your wandb entity
        run_name=params.format_now,  # name of the run
    )
    callbacks.append(wandb_cb)  # Add it to the callbacks list

    # Launch the Pipeline
    pipeline(
        train_data=train_dl.dataset,  # must be torch.Tensor, np.array or torch datasets
        eval_data=val_dl.dataset,  # must be torch.Tensor, np.array or torch datasets
        callbacks=callbacks,
    )

    # Test and save images
    manager = VAEModelManager(vae_model, params, MeanSquaredErrorMetric)
    manager.predict(test_dl)
    manager.write_useful_information()

    # Evaluate logistic regression from latent space to classify between G1, S, G2
    evaluate_logistic_regression(loader_generator, manager, params, display_matrix=False)


if __name__ == "__main__":
    parser = VAEParser()
    args = parser.arguments_parser.parse_args()

    parameters = DummyVAEModelParams()
    parameters.update(args)

    main(parameters)