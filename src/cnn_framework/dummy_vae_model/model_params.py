from ..utils.VAEModelParams import VAEModelParams

from ..utils.dimensions import Dimensions


class DummyVAEModelParams(VAEModelParams):
    """
    VAE model params.
    """

    def __init__(self):
        super().__init__("dummy_vae")

        self.input_dimensions = Dimensions(height=128, width=128)

        self.num_epochs = 30

        self.nb_modalities = 3  # RGB
        self.nb_stacks_per_modality = 1

        self.nb_classes = 2  # square, circle
        self.depth = 5

        self.train_ratio = 0.8
        self.val_ratio = 0.1
        self.test_ratio = 0.1

        self.learning_rate = 1e-4
        self.weight_decay = 0.05
        self.beta1 = 0.91
        self.beta2 = 0.995
        self.reconstruction_loss = "mse"  # "mse"
        self.kld_loss = "standard"  # "standard"
        self.encoder_name = "timm-efficientnet-b0"

        self.latent_dim = 16
        self.beta = 1  # weight of KLD loss

        self.model_pretrained_path = ""
