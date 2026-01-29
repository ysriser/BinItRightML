from typing import Tuple

import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B3_Weights,
    MobileNet_V3_Large_Weights,
    ConvNeXt_Tiny_Weights,
    efficientnet_b0,
    efficientnet_b3,
    mobilenet_v3_large,
    convnext_tiny,
)


def get_backbone(name: str, pretrained: bool = True, dropout: float = 0.2) -> Tuple[nn.Module, int]:
    if name == "convnext_tiny":
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = convnext_tiny(weights=weights)
        in_features = model.classifier[-1].in_features
        return model, in_features

    if name == "mobilenet_v3_large":
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        model = mobilenet_v3_large(weights=weights)
        in_features = model.classifier[-1].in_features
        return model, in_features

    if name == "efficientnet_b3":
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        model = efficientnet_b3(weights=weights, dropout=dropout)
        in_features = model.classifier[1].in_features
        return model, in_features

    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights, dropout=dropout)
    in_features = model.classifier[1].in_features
    return model, in_features
