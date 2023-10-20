from torch import nn
import torch

from pythae.models.nn import BaseEncoder
from pythae.models.base.base_utils import ModelOutput

from ..utils.model_managers.utils.custom_get_encoder import get_encoder


class CustomEncoder(BaseEncoder):
    def __init__(self, params, args):
        BaseEncoder.__init__(self)

        in_channels = len(params.c_indexes) * len(params.z_indexes)
        self.conv_layers = get_encoder(
            params.encoder_name,
            in_channels=in_channels,
            weights="imagenet",
            depth=params.depth,
            drop_rate=params.dropout,
        )

        # Infer size of images after convolutions
        # Create random input to infer size of output
        random_input = torch.randn(
            1, in_channels, params.input_dimensions.height, params.input_dimensions.width
        )
        random_output = self.conv_layers(random_input)
        output_size = random_output[-1].flatten().size(dim=0)

        self.embedding = nn.Linear(output_size, args.latent_dim)
        self.log_var = nn.Linear(output_size, args.latent_dim)

    def forward(self, x: torch.Tensor):
        h1 = self.conv_layers(x)[-1].reshape(x.shape[0], -1)
        output = ModelOutput(embedding=self.embedding(h1), log_covariance=self.log_var(h1))
        return output
