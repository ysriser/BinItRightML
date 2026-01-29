from typing import Tuple

import torch.nn as nn

from .backbone import get_backbone


def build_classifier(
    num_classes: int,
    backbone: str = "efficientnet_b0",
    pretrained: bool = True,
    dropout: float = 0.2,
) -> Tuple[nn.Module, str]:
    model, in_features = get_backbone(backbone, pretrained=pretrained, dropout=dropout)
    if backbone == "mobilenet_v3_large":
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif backbone == "convnext_tiny":
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        model.classifier[1] = nn.Linear(in_features, num_classes)
    return model, backbone
