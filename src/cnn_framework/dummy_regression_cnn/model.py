from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn


class DummyCnn(nn.Module):
    def __init__(self, nb_classes, nb_input_channels):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if nb_input_channels == 3 else None
        self.cnn = resnet18(weights=weights)

        # Modify first layer to accept input channels number
        self.cnn.conv1 = nn.Conv2d(
            nb_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        num_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Linear(num_features, nb_classes)

    def forward(self, x):
        return self.cnn(x)
