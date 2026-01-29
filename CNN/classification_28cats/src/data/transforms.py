from typing import Tuple

from torchvision import transforms
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B3_Weights,
    MobileNet_V3_Large_Weights,
    ConvNeXt_Tiny_Weights,
)


def build_transforms(backbone: str, image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    if backbone == "mobilenet_v3_large":
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
    elif backbone == "efficientnet_b3":
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1
    elif backbone == "convnext_tiny":
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    else:
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1

    preset = weights.transforms()
    mean, std = preset.mean, preset.std

    train_tfm = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    eval_tfm = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_tfm, eval_tfm
