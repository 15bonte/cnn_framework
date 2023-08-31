from ..utils.VAEModelParams import VAEModelParams

from ..utils.dimensions import Dimensions

class DummyVAEModelParams(VAEModelParams):
    """
    VAE model params.
    """

    def __init__(self):
        super().__init__("dummy_vae")

        self.input_dimensions = Dimensions(height=128, width=128)

        self.num_epochs = 10

        self.nb_modalities = 1  # RGB or grayscale
        self.nb_stacks_per_modality = 1

        self.nb_classes = 2  # square, circle
        self.out_channels = 3  # R, G, B
        self.depth = 5

        self.learning_rate = 1e-4
        self.weight_decay = 0.05
        self.beta1 = 0.91
        self.beta2 = 0.995
        self.reconstruction_loss = "mse"  # "l1", "bce" or "mse"
        self.kld_loss = "standard"  # "standard" or "cl-vae"
        self.encoder_name = "timm-efficientnet-b0"

        self.latent_dim = 16
        self.beta = 1  # weight of KLD loss
        self.gamma = 0  # weight of classification loss
        self.delta = 0  # weight of segmentation loss

        self.model_pretrained_path = ""
