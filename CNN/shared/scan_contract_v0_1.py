"""Scan service contract helpers (v0.1).

This module documents the external response shape we align on with Android/Spring Boot.
It is intentionally stdlib-only so it can be imported anywhere without heavy deps.

Doc: `CNN/docs/SCAN_SERVICE_SPEC_v0_1.md`
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict


BinType = Literal["blue", "general", "ewaste", "textile", "special", "unknown"]


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


class FollowupQuestion(TypedDict):
    id: str
    type: Literal["single_choice"]
    question: str
    options: List[str]


class Followup(TypedDict):
    needs_confirmation: bool
    questions: List[FollowupQuestion]


class FinalResult(TypedDict, total=False):
    category: str
    category_id: str
    recyclable: bool
    confidence: float
    instruction: str
    instructions: List[str]
    disposal_method: str
    bin_type: BinType
    rationale_tags: List[str]


class ScanData(TypedDict, total=False):
    tier1: Optional[Tier1Result]
    decision: Decision
    final: FinalResult
    followup: Optional[Followup]
    meta: Dict[str, Any]


class ScanResponse(TypedDict, total=False):
    status: Literal["success", "error"]
    request_id: str
    data: Optional[ScanData]
    code: str
    message: str


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
        "category_id": "tier1.paper",
        "recyclable": True,
        "bin_type": "blue",
        "disposal_method": "Blue Recycling Bin",
        "instructions": ["Empty and dry before recycling.", "Remove food residue."],
    },
    "plastic": {
        "category": "Plastic",
        "category_id": "tier1.plastic",
        "recyclable": True,
        "bin_type": "blue",
        "disposal_method": "Blue Recycling Bin",
        "instructions": ["Rinse and empty containers.", "Remove caps if possible."],
    },
    "metal": {
        "category": "Metal",
        "category_id": "tier1.metal",
        "recyclable": True,
        "bin_type": "blue",
        "disposal_method": "Blue Recycling Bin",
        "instructions": ["Rinse cans and remove residue."],
    },
    "glass": {
        "category": "Glass",
        "category_id": "tier1.glass",
        "recyclable": True,
        "bin_type": "blue",
        "disposal_method": "Blue Recycling Bin",
        "instructions": ["Rinse glass containers.", "Avoid breaking glass."],
    },
    "e-waste": {
        "category": "E-waste",
        "category_id": "tier1.ewaste",
        "recyclable": True,
        "bin_type": "ewaste",
        "disposal_method": "E-waste Collection Point",
        "instructions": ["Do not bin. Bring to an e-waste collection point."],
    },
    "textile": {
        "category": "Textile",
        "category_id": "tier1.textile",
        "recyclable": True,
        "bin_type": "textile",
        "disposal_method": "Textile Drop-off",
        "instructions": ["Donate if usable; otherwise bag it for drop-off."],
    },
    "other_uncertain": {
        "category": "Uncertain",
        "category_id": "tier1.other_uncertain",
        "recyclable": False,
        "bin_type": "unknown",
        "disposal_method": "Unknown",
        "instructions": ["Use expert scan or check local guidelines."],
    },
}


def final_from_tier1(tier1: Tier1Result) -> FinalResult:
    label = tier1["category"]
    rule = TIER1_RULES.get(label, TIER1_RULES["other_uncertain"])
    instructions = list(rule.get("instructions", []))
    primary = instructions[0] if instructions else "Follow local guidelines."
    return {
        "category": str(rule.get("category", label)),
        "category_id": str(rule.get("category_id", f"tier1.{label}")),
        "recyclable": bool(rule.get("recyclable", False)),
        "confidence": float(tier1["confidence"]),
        "instruction": primary,
        "instructions": instructions,
        "disposal_method": str(rule.get("disposal_method", "")),
        "bin_type": rule.get("bin_type", "unknown"),
    }

