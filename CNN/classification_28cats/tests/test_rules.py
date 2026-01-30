from src.rules import apply_rules


def test_apply_rules_low_confidence_requires_confirmation():
    result = apply_rules("glass", confidence=0.2, threshold=0.6, high_risk=[])
    assert result["needs_confirmation"] is True
    assert result["followup_questions"] != []


def test_apply_rules_high_confidence_no_risk():
    result = apply_rules("paper", confidence=0.9, threshold=0.6, high_risk=[])
    assert result["needs_confirmation"] is False
    assert result["followup_questions"] == []
    assert result["recyclable"] is True


def test_apply_rules_high_risk_list_forces_confirmation():
    result = apply_rules("paper", confidence=0.95, threshold=0.6, high_risk=["paper"])
    assert result["needs_confirmation"] is True
