from torchvision.models import (
    resnet18,
    ResNet18_Weights,
    resnet34,
    ResNet34_Weights,
    resnet50,
    ResNet50_Weights,
)
import torch.nn as nn


class ResnetClassifier(nn.Module):
    """
    Standard ResNet classifier.
    """

    def __init__(
        self,
        nb_classes: int,
        nb_input_channels: int,
        encoder_name: str,
    ):
        super().__init__()

        if encoder_name == "resnet18":
            encoder = resnet18
            weights = ResNet18_Weights.DEFAULT
        elif encoder_name == "resnet34":
            encoder = resnet34
            weights = ResNet34_Weights.DEFAULT
        elif encoder_name == "resnet50":
            encoder = resnet50
            weights = ResNet50_Weights.DEFAULT
        else:
            raise ValueError(f"Unknown encoder name: {encoder_name}")

        if nb_input_channels == 3:
            self.cnn = encoder(weights=weights)
        else:
            self.cnn = encoder(weights=weights)
            # Modify first layer to accept input channels number
            self.cnn.conv1 = nn.Conv2d(
                nb_input_channels,
                self.cnn.conv1.out_channels,
                kernel_size=self.cnn.conv1.kernel_size,
                stride=self.cnn.conv1.stride,
                padding=self.cnn.conv1.padding,
                bias=self.cnn.conv1.bias is not None,
            )
        num_features = self.cnn.fc.in_features

        self.cnn.fc = nn.Linear(num_features, nb_classes)

    def forward(self, x):
        """
        Forward pass.
        """
        return self.cnn(x)
