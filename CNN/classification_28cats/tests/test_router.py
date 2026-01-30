from src.router import Router


def test_router_returns_base_result():
    router = Router(expert_categories=["plastic"])
    base = {"category": "plastic", "confidence": 0.9}
    result = router.route("plastic", base)
    assert result == base
