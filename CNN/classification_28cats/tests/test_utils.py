import re

import pytest
import torch

from src.utils import load_json, load_yaml, resolve_device, save_json, save_yaml, timestamp


def test_timestamp_format():
    value = timestamp()
    assert re.match(r"^\d{8}_\d{6}$", value)


def test_save_load_json_roundtrip(tmp_path):
    payload = {"a": 1, "b": {"c": 2}}
    path = tmp_path / "data.json"
    save_json(payload, path)
    loaded = load_json(path)
    assert loaded == payload


def test_save_load_yaml_roundtrip(tmp_path):
    payload = {"a": 1, "b": {"c": 2}}
    path = tmp_path / "data.yaml"
    save_yaml(payload, path)
    loaded = load_yaml(path)
    assert loaded == payload


def test_load_yaml_rejects_non_mapping(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_yaml(path)


def test_resolve_device_cpu():
    device = resolve_device("cpu", require_cuda=False)
    assert device.type == "cpu"


def test_resolve_device_require_cuda():
    if torch.cuda.is_available():
        device = resolve_device("cuda", require_cuda=True)
        assert device.type == "cuda"
    else:
        with pytest.raises(RuntimeError):
            resolve_device("cuda", require_cuda=True)
