from __future__ import annotations

import numpy as np
from PIL import Image

from CNN.shared import multicrop


def test_multicrop_triggers_extra_crops() -> None:
    img = Image.new("RGB", (256, 256), color=(128, 128, 128))

    def infer_fn(_: np.ndarray, crop_name: str | None) -> np.ndarray:
        if crop_name == "center":
            return np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        if crop_name == "zoom_center":
            return np.array([[5.0, 0.0, 0.0]], dtype=np.float32)
        return np.array([[0.0, 5.0, 0.0]], dtype=np.float32)

    cfg = {
        "enabled": True,
        "trigger": {"conf": 0.9, "margin": 0.1},
        "extra_crops": ["zoom_center", "full_resize"],
        "combine": "best_score",
    }

    probs, meta = multicrop.run_multicrop(
        img=img,
        infer_fn=infer_fn,
        img_size=224,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        cfg=cfg,
    )

    assert meta["multicrop_used"] is True
    assert meta["selected_crop"] in {"zoom_center", "full_resize"}
    assert int(np.argmax(probs)) in {0, 1}
