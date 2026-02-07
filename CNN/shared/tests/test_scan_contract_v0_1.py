from CNN.shared import scan_contract_v0_1 as contract


def test_ensure_special_prefix_for_ewaste_and_textile():
    assert contract.ensure_special_prefix("Phone Charger", "e-waste") == "E-waste - Phone Charger"
    assert contract.ensure_special_prefix("Shirt", "textile") == "Textile - Shirt"
    assert contract.ensure_special_prefix("Glass Jar", "glass") == "Glass Jar"


def test_final_from_tier1_returns_minimum_five_fields():
    tier1 = {
        "category": "plastic",
        "confidence": 0.91,
        "top3": [
            {"label": "plastic", "p": 0.91},
            {"label": "paper", "p": 0.05},
            {"label": "glass", "p": 0.04},
        ],
        "escalate": False,
    }

    final = contract.final_from_tier1(tier1)

    assert set(final.keys()) == {
        "category",
        "recyclable",
        "confidence",
        "instruction",
        "instructions",
    }
    assert final["category"] == "Plastic"
    assert final["recyclable"] is True
    assert isinstance(final["instructions"], list)
    assert len(final["instructions"]) >= 1


def test_final_from_expert_decision_uses_refined_name_and_confidence_guardrail():
    tier1 = {
        "category": "other_uncertain",
        "confidence": 0.95,
        "top3": [
            {"label": "other_uncertain", "p": 0.95},
            {"label": "metal", "p": 0.03},
            {"label": "paper", "p": 0.02},
        ],
        "escalate": True,
    }
    expert = {
        "refined_name": "Ceramic mug",
        "bin_type": "non_recyclable",
        "contamination_flag": "unknown",
        "needs_confirmation": True,
    }

    final = contract.final_from_expert_decision(tier1, expert)

    assert final["category"] == "Ceramic mug"
    assert final["recyclable"] is False
    assert final["confidence"] <= 0.55
    assert len(final["instructions"]) >= 2


def test_map_reason_codes_maps_known_and_unknown_values():
    reasons = ["low_confidence", "strict_class_low_conf", "my_custom_reason"]
    mapped = contract.map_reason_codes(reasons)

    assert "LOW_CONFIDENCE" in mapped
    assert "STRICT_CLASS_LOW_CONF" in mapped
    assert "MY_CUSTOM_REASON" in mapped