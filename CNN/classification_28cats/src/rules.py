from typing import Dict, List


RULES: Dict[str, Dict[str, object]] = {
    # Example defaults. Extend based on local recycling rules.
    "plastic": {
        "recyclable_default": True,
        "instructions": ["Rinse and dry", "Remove caps if possible"],
        "risk_level": "medium",
    },
    "glass": {
        "recyclable_default": True,
        "instructions": ["Rinse and dry", "Do not include broken glass"],
        "risk_level": "high",
    },
    "paper": {
        "recyclable_default": True,
        "instructions": ["Keep clean and dry"],
        "risk_level": "low",
    },
    "metal": {
        "recyclable_default": True,
        "instructions": ["Rinse and dry"],
        "risk_level": "low",
    },
    "other": {
        "recyclable_default": False,
        "instructions": ["Dispose as general waste"],
        "risk_level": "high",
    },
}


FOLLOWUP_QUESTIONS: Dict[str, List[Dict[str, object]]] = {
    "glass": [
        {
            "id": "q_glass_broken",
            "type": "single_choice",
            "question": "Is the glass broken?",
            "options": ["Yes", "No"],
        }
    ],
    "plastic": [
        {
            "id": "q_plastic_food",
            "type": "single_choice",
            "question": "Is the plastic heavily soiled with food?",
            "options": ["Yes", "No"],
        }
    ],
}


def apply_rules(category: str, confidence: float, threshold: float, high_risk: List[str]) -> Dict[str, object]:
    rule = RULES.get(category, RULES.get("other", {}))
    recyclable_default = bool(rule.get("recyclable_default", False))
    instructions = list(rule.get("instructions", []))
    risk_level = rule.get("risk_level", "medium")

    needs_confirmation = confidence < threshold or category in high_risk or risk_level == "high"
    followup_questions = FOLLOWUP_QUESTIONS.get(category, []) if needs_confirmation else []

    return {
        "recyclable": recyclable_default,
        "instructions": instructions,
        "needs_confirmation": needs_confirmation,
        "followup_questions": followup_questions,
    }
