from .cnn_parser import CnnParser


class VAEParser(CnnParser):
    """
    VAE parsing class.
    """

    def __init__(self):
        super().__init__()

        self.arguments_parser.add_argument("--latent_dim", help="Latent space dimension")
        self.arguments_parser.add_argument("--beta", help="KD loss weight")
        self.arguments_parser.add_argument("--gamma", help="classification loss weight")
        self.arguments_parser.add_argument("--delta", help="segmentation loss weight")
        self.arguments_parser.add_argument("--reconstruction_loss", help="mse, bce or l1")
        self.arguments_parser.add_argument(
            "--model_pretrained_path", help="Path to pretrained model"
        )
        self.arguments_parser.add_argument("--depth", help="Depth of encoder")
        self.arguments_parser.add_argument("--kld_loss", help="standard or cl-vae")
        self.arguments_parser.add_argument("--encoder_name", help="Encoder name")
