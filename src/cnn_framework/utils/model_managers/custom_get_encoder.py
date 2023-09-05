import torch.utils.model_zoo as model_zoo

from segmentation_models_pytorch.encoders.resnet import resnet_encoders
from segmentation_models_pytorch.encoders.dpn import dpn_encoders
from segmentation_models_pytorch.encoders.vgg import vgg_encoders
from segmentation_models_pytorch.encoders.senet import senet_encoders
from segmentation_models_pytorch.encoders.densenet import densenet_encoders
from segmentation_models_pytorch.encoders.inceptionresnetv2 import inceptionresnetv2_encoders
from segmentation_models_pytorch.encoders.inceptionv4 import inceptionv4_encoders
from segmentation_models_pytorch.encoders.efficientnet import efficient_net_encoders
from segmentation_models_pytorch.encoders.mobilenet import mobilenet_encoders
from segmentation_models_pytorch.encoders.xception import xception_encoders
from segmentation_models_pytorch.encoders.timm_efficientnet import timm_efficientnet_encoders
from segmentation_models_pytorch.encoders.timm_resnest import timm_resnest_encoders
from segmentation_models_pytorch.encoders.timm_res2net import timm_res2net_encoders
from segmentation_models_pytorch.encoders.timm_regnet import timm_regnet_encoders
from segmentation_models_pytorch.encoders.timm_sknet import timm_sknet_encoders
from segmentation_models_pytorch.encoders.timm_mobilenetv3 import timm_mobilenetv3_encoders
from segmentation_models_pytorch.encoders.timm_gernet import timm_gernet_encoders

from segmentation_models_pytorch.encoders.timm_universal import TimmUniversalEncoder

# Modified from segmentation_models_pytorch to enable drop_out parameter change 
# Look for "Only change from original code" 

encoders = {}
encoders.update(resnet_encoders)
encoders.update(dpn_encoders)
encoders.update(vgg_encoders)
encoders.update(senet_encoders)
encoders.update(densenet_encoders)
encoders.update(inceptionresnetv2_encoders)
encoders.update(inceptionv4_encoders)
encoders.update(efficient_net_encoders)
encoders.update(mobilenet_encoders)
encoders.update(xception_encoders)
encoders.update(timm_efficientnet_encoders)
encoders.update(timm_resnest_encoders)
encoders.update(timm_res2net_encoders)
encoders.update(timm_regnet_encoders)
encoders.update(timm_sknet_encoders)
encoders.update(timm_mobilenetv3_encoders)
encoders.update(timm_gernet_encoders)


def get_encoder(
    name, in_channels=3, depth=5, drop_rate=0.2, weights=None, output_stride=32, **kwargs
):

    if name.startswith("tu-"):
        name = name[3:]
        encoder = TimmUniversalEncoder(
            name=name,
            in_channels=in_channels,
            depth=depth,
            output_stride=output_stride,
            pretrained=weights is not None,
            **kwargs,
        )
        return encoder

    try:
        Encoder = encoders[name]["encoder"]
    except KeyError:
        raise KeyError(
            "Wrong encoder name `{}`, supported encoders: {}".format(name, list(encoders.keys()))
        )

    params = encoders[name]["params"]
    params.update(depth=depth)

    # Only change from original code - update drop_rate only if it is in params
    if "drop_rate" in params:
        params.update(drop_rate=drop_rate)

    encoder = Encoder(**params)

    if weights is not None:
        try:
            settings = encoders[name]["pretrained_settings"][weights]
        except KeyError:
            raise KeyError(
                "Wrong pretrained weights `{}` for encoder `{}`. Available options are: {}".format(
                    weights, name, list(encoders[name]["pretrained_settings"].keys()),
                )
            )
        encoder.load_state_dict(model_zoo.load_url(settings["url"]))

    encoder.set_in_channels(in_channels, pretrained=weights is not None)
    if output_stride != 32:
        encoder.make_dilated(output_stride)

    return encoder
