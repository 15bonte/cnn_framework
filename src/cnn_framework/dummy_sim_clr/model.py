import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNetSimCLR(nn.Module):
    """
    ResNet backbone with MLP head for SimCLR
    Here, f outputs a dim_mlp-dim vector and g outputs a out_dim-dim vector in the latent space.

    f is the ResNet without its final fc layer.
    g is this final fc layer, a MLP with 2 hidden layers.
    """

    def __init__(self, nb_input_channels, out_dim=128, **kwargs):
        super().__init__()

        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Modify first layer to accept input channels number
        self.backbone.conv1 = nn.Conv2d(
            nb_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # Modify fc to match out_dim
        dim_mlp = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, out_dim, bias=True)

        # Add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def forward(self, x):
        return self.backbone(x)
