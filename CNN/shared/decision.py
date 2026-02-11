"""Decision utilities for rejection/escalation."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits).squeeze()
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    return exp / np.sum(exp)


def compute_metrics(probs: np.ndarray) -> Dict[str, float]:
    probs = np.asarray(probs).squeeze()
    if probs.ndim != 1:
        raise ValueError("probs must be a 1D array")
    top_idxs = np.argsort(probs)[::-1]
    top1 = int(top_idxs[0]) if len(top_idxs) > 0 else -1
    top2 = int(top_idxs[1]) if len(top_idxs) > 1 else top1
    max_prob = float(probs[top1]) if top1 >= 0 else 0.0
    second_prob = float(probs[top2]) if top2 >= 0 else 0.0
    margin = max_prob - second_prob
    entropy = float(-np.sum(probs * np.log(probs + 1e-9)))
    return {
        "max_prob": max_prob,
        "margin": margin,
        "entropy": entropy,
        "top1_idx": top1,
        "top2_idx": top2,
    }


def _topk(probs: np.ndarray, labels: Sequence[str], k: int) -> List[Dict[str, float]]:
    k = max(1, min(k, len(labels)))
    top_idxs = np.argsort(probs)[::-1][:k]
    return [{"label": labels[i], "p": float(probs[i])} for i in top_idxs]


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _append_base_reasons(
    reasons: List[str],
    top1_label: str,
    top1_prob: float,
    margin: float,
    entropy: float,
    conf_th: float | None,
    margin_th: float | None,
    entropy_th: float | None,
) -> None:
    if top1_label == "other_uncertain":
        reasons.append("label_other_uncertain")
    if conf_th is not None and top1_prob < conf_th:
        reasons.append("low_confidence")
    if margin_th is not None and margin < margin_th:
        reasons.append("small_margin")
    if entropy_th is not None and entropy > entropy_th:
        reasons.append("high_entropy")


def _append_strict_reasons(
    reasons: List[str],
    strict: Any,
    top1_label: str,
    top1_prob: float,
    margin: float,
    entropy: float,
) -> None:
    if not isinstance(strict, dict) or top1_label not in strict:
        return
    strict_cfg = strict[top1_label] or {}
    strict_conf = _as_float(strict_cfg.get("conf"))
    strict_margin = _as_float(strict_cfg.get("margin"))
    strict_entropy = _as_float(strict_cfg.get("entropy"))
    if strict_conf is not None and top1_prob < strict_conf:
        reasons.append("strict_class_low_conf")
    if strict_margin is not None and margin < strict_margin:
        reasons.append("strict_class_low_margin")
    if strict_entropy is not None and entropy > strict_entropy:
        reasons.append("strict_class_high_entropy")


def _resolve_final_label(
    top1_label: str,
    labels: Sequence[str],
    escalate: bool,
    reject_to_other: bool,
) -> str:
    if escalate and reject_to_other and "other_uncertain" in labels:
        return "other_uncertain"
    return top1_label


def decide_from_probs(
    probs: np.ndarray,
    labels: Sequence[str],
    thresholds: Dict[str, Any],
) -> Dict[str, Any]:
    metrics = compute_metrics(probs)
    top1_idx = int(metrics["top1_idx"])
    top1_label = labels[top1_idx]
    top1_prob = float(metrics["max_prob"])
    margin = float(metrics["margin"])
    entropy = float(metrics["entropy"])

    topk = int(thresholds.get("topk", 3))
    top3 = _topk(probs, labels, topk)

    reasons: List[str] = []
    conf_th = _as_float(thresholds.get("conf"))
    margin_th = _as_float(thresholds.get("margin"))
    entropy_th = _as_float(thresholds.get("entropy"))
    reject_to_other = bool(thresholds.get("reject_to_other", True))

    _append_base_reasons(
        reasons,
        top1_label,
        top1_prob,
        margin,
        entropy,
        conf_th,
        margin_th,
        entropy_th,
    )
    _append_strict_reasons(
        reasons,
        thresholds.get("strict_per_class", {}) or {},
        top1_label,
        top1_prob,
        margin,
        entropy,
    )

    escalate = len(reasons) > 0
    final_label = _resolve_final_label(
        top1_label=top1_label,
        labels=labels,
        escalate=escalate,
        reject_to_other=reject_to_other,
    )

    return {
        "final_label": final_label,
        "top1_label": top1_label,
        "top1_prob": top1_prob,
        "top3": top3,
        "metrics": {"max_prob": top1_prob, "margin": margin, "entropy": entropy},
        "escalate": escalate,
        "reasons": reasons,
    }


def decide(
    logits: np.ndarray,
    labels: Sequence[str],
    thresholds: Dict[str, Any],
) -> Dict[str, Any]:
    probs = softmax(logits)
    return decide_from_probs(probs, labels, thresholds)
