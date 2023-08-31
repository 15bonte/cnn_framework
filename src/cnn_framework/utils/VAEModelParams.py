from .ModelParams import ModelParams
from .dimensions import Dimensions


class VAEModelParams(ModelParams):
    """
    VAE model params.
    """

    def __init__(self, name="vae"):
        super().__init__(name)

        self.input_dimensions = Dimensions(height=160, width=160)

        self.nb_modalities = 2
        self.nb_stacks_per_modality = 1

        self.nb_classes = 3  # G1, S, G2
        self.out_channels = 2  # FUCCI red and green
        self.depth = 5

        self.learning_rate = 1e-3
        self.weight_decay = 0.05
        self.beta1 = 0.91
        self.beta2 = 0.995
        self.reconstruction_loss = "mse"  # "l1", "bce" or "mse"
        self.kld_loss = "standard"  # "standard" or "cl-vae"
        self.encoder_name = "timm-efficientnet-b0"

        self.latent_dim = 256
        self.beta = 1  # weight of KLD loss
        self.gamma = 0  # weight of classification loss
        self.delta = 0  # weight of segmentation loss

        self.model_pretrained_path = ""

    def update(self, args=None):
        # Finish by parent class as it prints the parameters

        if args is not None:
            if args.latent_dim:
                self.latent_dim = args.latent_dim
            if args.beta:
                self.beta = float(args.beta)
            if args.gamma:
                self.gamma = float(args.gamma)
            if args.delta:
                self.delta = float(args.delta)
            if args.reconstruction_loss:
                self.reconstruction_loss = args.reconstruction_loss
            if args.model_pretrained_path:
                self.model_pretrained_path = args.model_pretrained_path
            if args.depth:
                self.depth = int(args.depth)
            if args.kld_loss:
                self.kld_loss = args.kld_loss
            if args.encoder_name:
                self.encoder_name = args.encoder_name

        super().update(args)

    def get_useful_training_parameters(self):
        parent_parameters = super().get_useful_training_parameters()
        parameters = (
            parent_parameters
            + f" | latent dim {self.latent_dim}"
            + f" | beta {self.beta}"
            + f" | gamma {self.gamma}"
            + f" | delta {self.delta}"
            + f" | depth {self.depth}"
            + f" | kld loss {self.kld_loss}"
            + f" | encoder name {self.encoder_name}"
        )
        return parameters
