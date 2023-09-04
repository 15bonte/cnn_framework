import segmentation_models_pytorch as smp


class UNet(smp.Unet):
    def __init__(self, nb_classes, nb_input_channels):
        super().__init__(
            encoder_name="resnet18",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=nb_input_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=nb_classes,  # model output channels (number of classes in your dataset),
        )
