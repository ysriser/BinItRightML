"""Scan service contract helpers (v0.1).

This module documents the external response shape we align on with Android/Spring Boot.
It is intentionally stdlib-only so it can be imported anywhere without heavy deps.

Doc: `CNN/docs/SCAN_SERVICE_SPEC_v0_1.md`
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional, TypedDict


# Tier-2 ExpertDecision uses a small stream enum. We map it to a UI routing hint
# internally (not returned in the API response).
ExpertBinType = Literal["recyclable", "non_recyclable", "textile", "e_waste"]
ContaminationFlag = Literal["clean", "contaminated", "unknown"]


class TopK(TypedDict):
    label: str
    p: float


class Tier1Result(TypedDict):
    category: str
    confidence: float
    top3: List[TopK]
    escalate: bool


class Decision(TypedDict, total=False):
    used_tier2: bool
    reason_codes: List[str]
    thresholds: Dict[str, float]


class FinalResult(TypedDict):
    """Android UI contract (v0.1): only these 5 fields are used."""

    category: str
    recyclable: bool
    confidence: float
    instruction: str
    instructions: List[str]


class ScanData(TypedDict, total=False):
    tier1: Optional[Tier1Result]
    decision: Decision
    final: FinalResult
    meta: Dict[str, Any]


class ScanResponse(TypedDict, total=False):
    status: Literal["success", "error"]
    request_id: str
    data: Optional[ScanData]
    code: str
    message: str


class ExpertDecision(TypedDict, total=False):
    """Template-friendly Tier-2 decision (mock for v0.1).

    This is intentionally structured (no free-text paragraphs) so we can later
    store it in a DB and render final UI instructions from templates.
    """

    refined_name: str
    bin_type: ExpertBinType
    contamination_flag: ContaminationFlag
    needs_confirmation: bool


def ensure_special_prefix(category: str, tier1_label: str) -> str:
    """Ensure Android-routing prefixes for special streams.

    Android's v0.1 UI routing should be able to infer special flows from the
    returned `final.category` string (in addition to `recyclable`).

    Rules:
    - e-waste must start with "E-waste - "
    - textile must start with "Textile - "

    For Tier-1-only predictions we may not know the refined item name, so we
    fall back to a generic suffix ("Item").
    """

    cat = str(category or "").strip()
    label = str(tier1_label or "").strip().lower()

    if label == "e-waste":
        if cat.lower().startswith("e-waste - "):
            return cat
        if cat.lower().startswith("e-waste"):
            cat = cat[len("e-waste") :].lstrip(" -")
        return f"E-waste - {cat or 'Item'}"

    if label == "textile":
        if cat.lower().startswith("textile - "):
            return cat
        if cat.lower().startswith("textile"):
            cat = cat[len("textile") :].lstrip(" -")
        return f"Textile - {cat or 'Item'}"

    return cat


REASON_CODE_MAP: Dict[str, str] = {
    # From CNN/shared/decision.py reasons -> external reason_codes
    "label_other_uncertain": "PRED_OTHER_UNCERTAIN",
    "low_confidence": "LOW_CONFIDENCE",
    "small_margin": "LOW_MARGIN",
    "high_entropy": "HIGH_ENTROPY",
    "strict_class_low_conf": "STRICT_CLASS_LOW_CONF",
    "strict_class_low_margin": "STRICT_CLASS_LOW_MARGIN",
    "strict_class_high_entropy": "STRICT_CLASS_HIGH_ENTROPY",
}


def map_reason_codes(reasons: List[str]) -> List[str]:
    codes: List[str] = []
    for reason in reasons:
        codes.append(REASON_CODE_MAP.get(reason, reason.upper()))
    return codes


TIER1_RULES: Dict[str, Dict[str, Any]] = {
    "paper": {
        "category": "Paper",
        "recyclable": True,
        "instructions": ["Empty and dry before recycling.", "Remove food residue."],
    },
    "plastic": {
        "category": "Plastic",
        "recyclable": True,
        "instructions": ["Rinse and empty containers.", "Remove caps if possible."],
    },
    "metal": {
        "category": "Metal",
        "recyclable": True,
        "instructions": ["Rinse cans and remove residue."],
    },
    "glass": {
        "category": "Glass",
        "recyclable": True,
        "instructions": ["Rinse glass containers.", "Avoid breaking glass."],
    },
    "e-waste": {
        "category": "E-waste - Item",
        "recyclable": False,
        "instructions": ["Do not bin. Bring to an e-waste collection point."],
    },
    "textile": {
        "category": "Textile - Item",
        # v0.1 meaning: recyclable means "blue bin flow". Textile is a separate stream.
        "recyclable": False,
        "instructions": ["Donate if usable; otherwise bag it for drop-off."],
    },
    "other_uncertain": {
        "category": "Uncertain",
        "recyclable": False,
        "instructions": [
            "Dispose as general waste to avoid contaminating recycling.",
            "If it contains electronics/battery, bring to an e-waste collection point.",
            "If it is mostly fabric, bag it for textile drop-off.",
            "If you need confirmation, open the Questionnaire flow in the app.",
        ],
    },
}


def final_from_tier1(tier1: Tier1Result) -> FinalResult:
    label = tier1["category"]
    rule = TIER1_RULES.get(label, TIER1_RULES["other_uncertain"])
    instructions = list(rule.get("instructions", []))
    primary = instructions[0] if instructions else "Follow local guidelines."
    category = ensure_special_prefix(str(rule.get("category", label)), label)
    return {
        # v3.3 minimal contract: Android UI depends only on these 5 fields.
        "category": category,
        "recyclable": bool(rule.get("recyclable", False)),
        "confidence": float(tier1["confidence"]),
        "instruction": primary,
        "instructions": instructions,
    }


def recyclable_from_bin_type(bin_type: str) -> bool:
    # v0.1 meaning: recyclable means "normal recycling bin (blue bin) flow".
    return bin_type == "blue"


EXPERT_BIN_TO_UI_BIN: Dict[str, str] = {
    "recyclable": "blue",
    "non_recyclable": "general",
    "textile": "textile",
    "e_waste": "ewaste",
}


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "unknown"


# Template key precedence (most specific -> least specific):
#   label.bin_type.contamination
#   label.bin_type.any
#   label.any.contamination
#   label.any.any
#   default
INSTRUCTION_TEMPLATES: Dict[str, List[str]] = {
    "plastic.blue.clean": [
        "Empty contents.",
        "Rinse the container.",
        "Remove caps if possible.",
    ],
    "plastic.blue.unknown": [
        "If the item is clean and empty, recycle in the blue bin.",
        "If it has food residue, dispose as general waste.",
        "Remove non-plastic parts if possible.",
    ],
    "paper.blue.clean": [
        "Empty and dry the paper.",
        "Remove food residue.",
        "Flatten if possible to save space.",
    ],
    "paper.blue.unknown": [
        "If clean and dry, recycle in the blue bin.",
        "If oily/food-stained, dispose as general waste.",
        "Remove clean parts if separable.",
    ],
    "paper.general.contaminated": [
        "If oily/food-stained, dispose as general waste.",
        "Remove clean parts if separable.",
        "Do not place wet/dirty paper in the blue bin.",
    ],
    "metal.blue.clean": [
        "Empty contents.",
        "Rinse to remove residue.",
        "Crush cans if safe to do so.",
    ],
    "metal.blue.unknown": [
        "If clean, recycle in the blue bin.",
        "If heavily contaminated, dispose as general waste.",
        "Rinse if possible before recycling.",
    ],
    "glass.blue.clean": [
        "Empty contents.",
        "Rinse the glass container.",
        "Handle carefully to avoid breaking.",
    ],
    "glass.blue.unknown": [
        "If clean, recycle in the blue bin.",
        "If contaminated with food/liquid, rinse before recycling.",
        "Handle carefully to avoid breaking.",
    ],
    "textile.textile.any": [
        "Keep textiles clean and dry.",
        "Bag the item for drop-off.",
        "Donate if usable; otherwise recycle at a textile point.",
    ],
    "e-waste.ewaste.any": [
        "Do not place in the blue bin.",
        "Bring to an e-waste collection point.",
        "Remove batteries if safe, and keep parts separated.",
    ],
    "other_uncertain.unknown.any": [
        "Dispose as general waste to avoid contaminating recycling.",
        "If it contains electronics/battery, bring to an e-waste collection point.",
        "If it is mostly fabric, bag it for textile drop-off.",
    ],
    "default": [
        "If uncertain, dispose as general waste to avoid contaminating recycling.",
        "If it contains electronics/battery, bring to an e-waste collection point.",
        "Keep items clean and dry before using the blue recycling bin.",
        "If you need confirmation, open the Questionnaire flow in the app.",
    ],
}


def select_template(
    tier1_label: str,
    bin_type: str,
    contamination_flag: str,
) -> List[str]:
    keys = [
        f"{tier1_label}.{bin_type}.{contamination_flag}",
        f"{tier1_label}.{bin_type}.any",
        f"{tier1_label}.any.{contamination_flag}",
        f"{tier1_label}.any.any",
        "default",
    ]
    for key in keys:
        value = INSTRUCTION_TEMPLATES.get(key)
        if value:
            return list(value)
    return list(INSTRUCTION_TEMPLATES["default"])


def final_from_expert_decision(
    tier1: Tier1Result,
    expert: ExpertDecision,
) -> FinalResult:
    """Compose v0.1 FinalResult deterministically from Tier-1 + expert decision."""

    base = final_from_tier1(tier1)
    tier1_label = str(tier1.get("category", "other_uncertain"))

    refined = str(expert.get("refined_name", "")).strip() if expert else ""
    expert_bin = str(expert.get("bin_type", "non_recyclable"))
    bin_type = EXPERT_BIN_TO_UI_BIN.get(expert_bin, "unknown")
    contamination = str(expert.get("contamination_flag", "unknown"))
    needs_confirmation = bool(expert.get("needs_confirmation", False))

    category = refined if refined else str(base.get("category", tier1_label))
    category = ensure_special_prefix(category, tier1_label)
    instructions = select_template(tier1_label, bin_type, contamination)
    primary = instructions[0] if instructions else "Follow local guidelines."

    # Tier-2 confidence: avoid leaking Tier-1 "other_uncertain" high probabilities as a misleading
    # final confidence. This is a conservative heuristic until we have a calibrated Tier-2 model.
    tier1_conf = float(tier1.get("confidence", 0.0))
    if tier1_label == "other_uncertain" or needs_confirmation:
        confidence = min(tier1_conf, 0.55)
    else:
        if contamination == "clean":
            confidence = min(tier1_conf, 0.92)
        elif contamination == "contaminated":
            confidence = min(tier1_conf, 0.75)
        else:
            confidence = min(tier1_conf, 0.82)
    confidence = max(0.0, min(1.0, float(confidence)))

    return {
        # v3.3 minimal contract: Android UI depends only on these 5 fields.
        "category": category,
        "recyclable": recyclable_from_bin_type(bin_type),
        "confidence": confidence,
        "instruction": primary,
        "instructions": instructions,
    }
