import os

from pythae.pipelines import TrainingPipeline
from pythae.trainers import BaseTrainerConfig
from pythae.trainers.training_callbacks import WandbCallback
from pythae.models import AutoModel, BetaVAE, BetaVAEConfig

from cnn_framework.dummy_vae_model.data_set import DummyVAEDataSet
from cnn_framework.dummy_vae_model.model_params import DummyVAEModelParams

from cnn_framework.utils.data_managers.default_data_manager import DefaultDataManager
from cnn_framework.utils.data_loader_generators.data_loader_generator import DataLoaderGenerator
from cnn_framework.utils.metrics.mean_squared_error_metric import MeanSquaredErrorMetric
from cnn_framework.utils.model_managers.model_manager import ModelManager
from cnn_framework.utils.model_managers.vae_model_manager import VAEModelManager
from cnn_framework.utils.parsers.vae_parser import VAEParser

from cnn_framework.dummy_vae_model.decoder import CustomDecoder
from cnn_framework.dummy_vae_model.encoder import CustomEncoder


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
    my_vae_config = BetaVAEConfig(
        reconstruction_loss=params.reconstruction_loss,
        input_dim=(
            len(params.c_indexes) * len(params.z_indexes),
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
    else:
        encoder = CustomEncoder(params, my_vae_config)
        print(f"Number of parameters in encoder: {sum(p.numel() for p in encoder.parameters())}")
        decoder = CustomDecoder(params, my_vae_config)
        print(f"Number of parameters in decoder: {sum(p.numel() for p in decoder.parameters())}")
        vae_model = BetaVAE(encoder=encoder, decoder=decoder, model_config=my_vae_config)

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
        project_name=params.wandb_project,  # specify your wandb project
        entity_name=params.wandb_entity,  # specify your wandb entity
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


if __name__ == "__main__":
    parser = VAEParser()
    args = parser.arguments_parser.parse_args()

    parameters = DummyVAEModelParams()
    parameters.update(args)

    main(parameters)
