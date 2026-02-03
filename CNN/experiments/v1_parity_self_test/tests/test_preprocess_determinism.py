from __future__ import annotations

from pathlib import Path
from PIL import Image

from CNN.experiments.v1_parity_self_test import parity_utils


def test_preprocess_determinism() -> None:
    sample_path = Path(
        "CNN/experiments/v1_parity_self_test/samples/sample_red.png"
    )
    img = Image.open(sample_path).convert("RGB")
    cfg = parity_utils.InferConfig(
        img_size=224,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        resize_scale=1.15,
        interpolation="BICUBIC",
    )

    tensor1, _ = parity_utils.preprocess_with_audit(img, cfg)
    tensor2, _ = parity_utils.preprocess_with_audit(img, cfg)

    assert tensor1.shape == tensor2.shape
    assert parity_utils.hash_array(tensor1) == parity_utils.hash_array(tensor2)
