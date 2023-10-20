import numpy as np

from torch import nn
import torch
import torch.nn.functional as F

from segmentation_models_pytorch.base.heads import SegmentationHead
from segmentation_models_pytorch.base import modules as md

from pythae.models.nn import BaseDecoder
from pythae.models.base.base_utils import ModelOutput


class DecoderBlock(nn.Module):
    """
    Inspired by segmentation_models_pytorch
    """

    def __init__(self, in_channels, out_channels, use_batchnorm):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class CustomDecoder(BaseDecoder):
    def __init__(self, params, args):
        BaseDecoder.__init__(self)

        self.head_channels = 512
        out_channels = np.array([256, 128, 64, 32, 16])
        in_channels = [self.head_channels] + list(out_channels[:-1])

        # Infer size of images after convolutions
        self.first_square_size = params.input_dimensions.height
        for _ in range(len(out_channels)):
            self.first_square_size = self.first_square_size // 2

        self.fc = nn.Linear(
            args.latent_dim,
            self.head_channels * self.first_square_size * self.first_square_size,
        )
        blocks = [
            DecoderBlock(in_ch, out_ch, use_batchnorm=True)
            for in_ch, out_ch in zip(in_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

        # Reconstruction
        self.segmentation_head = SegmentationHead(
            in_channels=out_channels[-1],
            out_channels=len(params.c_indexes) * len(params.z_indexes),
            activation=None,
            kernel_size=3,
        )

    def forward(self, z: torch.Tensor):
        x = self.fc(z).reshape(
            z.shape[0], self.head_channels, self.first_square_size, self.first_square_size
        )

        for decoder_block in self.blocks:
            x = decoder_block(x)

        reconstruction = self.segmentation_head(x)
        output = ModelOutput(reconstruction=reconstruction)

        return output
