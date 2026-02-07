from __future__ import annotations

import numpy as np

from CNN.shared import decision


def test_strict_class_low_confidence() -> None:
    labels = [
        "paper",
        "plastic",
        "metal",
        "glass",
        "other_uncertain",
        "e-waste",
        "textile",
    ]
    probs = np.array([0.05, 0.8, 0.05, 0.04, 0.03, 0.02, 0.01], dtype=np.float32)
    probs = probs / probs.sum()

    thresholds = {
        "conf": 0.75,
        "margin": 0.1,
        "reject_to_other": True,
        "strict_per_class": {"plastic": {"conf": 0.85}},
    }

    result = decision.decide_from_probs(probs, labels, thresholds)
    assert result["escalate"] is True
    assert "strict_class_low_conf" in result["reasons"]
    assert result["final_label"] == "other_uncertain"
