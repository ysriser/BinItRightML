from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def _fake_locust_module():
    def task(weight=1):
        def decorator(fn):
            fn._task_weight = weight
            return fn

        return decorator

    return SimpleNamespace(HttpUser=object, task=task, between=lambda a, b: (a, b))


def test_forecast_locustfile_imports_without_real_locust(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "locust", _fake_locust_module())

    path = Path(__file__).resolve().parents[3] / "forecast" / "locustfile.py"
    spec = importlib.util.spec_from_file_location("forecast_locustfile_test", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert hasattr(mod, "BinItRightUser")
    assert hasattr(mod.BinItRightUser, "get_forecast")
    assert hasattr(mod.BinItRightUser, "load_openapi")

    calls = []
    fake_self = SimpleNamespace(client=SimpleNamespace(get=lambda path: calls.append(path)))
    mod.BinItRightUser.get_forecast(fake_self)
    mod.BinItRightUser.load_openapi(fake_self)

    assert calls == ["/forecast", "/openapi.json"]


def test_root_locustfile_reexports_user(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "locust", _fake_locust_module())

    forecast_path = Path(__file__).resolve().parents[3] / "forecast" / "locustfile.py"
    forecast_spec = importlib.util.spec_from_file_location("forecast.locustfile", forecast_path)
    assert forecast_spec and forecast_spec.loader
    forecast_mod = importlib.util.module_from_spec(forecast_spec)
    forecast_spec.loader.exec_module(forecast_mod)
    monkeypatch.setitem(sys.modules, "forecast.locustfile", forecast_mod)

    root_path = Path(__file__).resolve().parents[3] / "locustfile.py"
    root_spec = importlib.util.spec_from_file_location("root_locustfile_test", root_path)
    assert root_spec and root_spec.loader
    root_mod = importlib.util.module_from_spec(root_spec)
    root_spec.loader.exec_module(root_mod)

    assert root_mod.BinItRightUser is forecast_mod.BinItRightUser
    assert "BinItRightUser" in root_mod.__all__
