from __future__ import annotations

import numpy as np
import pytest

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


def test_compute_metrics_requires_1d_array() -> None:
    probs = np.array([[0.5, 0.5], [0.2, 0.8]], dtype=np.float32)
    with pytest.raises(ValueError, match="1D"):
        decision.compute_metrics(probs)


def test_decide_from_probs_triggers_all_reason_types_without_forced_reject() -> None:
    labels = ["plastic", "paper", "glass", "other_uncertain"]
    probs = np.array([0.34, 0.33, 0.33, 0.00], dtype=np.float32)
    probs = probs / probs.sum()
    thresholds = {
        "topk": 10,
        "conf": 0.9,
        "margin": 0.2,
        "entropy": 0.1,
        "reject_to_other": False,
        "strict_per_class": {"plastic": {"conf": 0.95, "margin": 0.3, "entropy": 0.2}},
    }

    result = decision.decide_from_probs(probs, labels, thresholds)
    reasons = set(result["reasons"])

    assert result["escalate"] is True
    assert result["final_label"] == "plastic"
    assert "low_confidence" in reasons
    assert "small_margin" in reasons
    assert "high_entropy" in reasons
    assert "strict_class_low_conf" in reasons
    assert "strict_class_low_margin" in reasons
    assert "strict_class_high_entropy" in reasons
    assert len(result["top3"]) == len(labels)


def test_decide_uses_softmax_and_handles_non_dict_strict_cfg() -> None:
    labels = ["other_uncertain", "paper", "metal"]
    logits = np.array([[2.0, 1.0, 0.0]], dtype=np.float32)
    result = decision.decide(
        logits,
        labels,
        {
            "strict_per_class": ["unexpected-type"],
            "reject_to_other": True,
        },
    )
    assert result["top1_label"] == "other_uncertain"
    assert "label_other_uncertain" in result["reasons"]
    assert result["final_label"] == "other_uncertain"
