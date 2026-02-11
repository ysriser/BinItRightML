from __future__ import annotations

import importlib
import importlib.machinery
import json
import sys
import types
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _stub_onnxruntime() -> types.ModuleType:
    fake = types.ModuleType("onnxruntime")
    fake.__spec__ = importlib.machinery.ModuleSpec("onnxruntime", loader=None)

    class _Session:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def get_inputs(self):
            return [types.SimpleNamespace(name="input")]

        def get_outputs(self):
            return [types.SimpleNamespace(name="logits")]

        def run(self, _out_names, _feeds):
            return [np.array([[2.0, 0.2]], dtype=np.float32)]

    fake.InferenceSession = _Session
    fake.get_available_providers = lambda: ["CPUExecutionProvider"]
    return fake


def _reload_eval_v3_calibrate(monkeypatch: pytest.MonkeyPatch):
    module_name = "CNN.experiments.v3_generalization_upgrade.scripts.eval_v3_calibrate"
    if module_name in sys.modules:
        del sys.modules[module_name]
    monkeypatch.setitem(sys.modules, "onnxruntime", _stub_onnxruntime())
    module = importlib.import_module(module_name)
    return importlib.reload(module)


def test_eval_v3_calibrate_metrics_and_sweep(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _reload_eval_v3_calibrate(monkeypatch)

    logits = np.array([[3.0, 0.2], [0.5, 1.5], [2.0, 1.0]], dtype=np.float32)
    labels = np.array([0, 1, 0], dtype=np.int64)
    probs = mod.softmax(logits)

    nll = mod.negative_log_likelihood(probs, labels)
    best_t, best_nll = mod.fit_temperature(logits, labels, t_min=0.5, t_max=2.0, t_step=0.5)
    ece = mod.expected_calibration_error(probs, labels, bins=5)
    metrics = mod.metric_summary(probs, labels)

    assert nll >= 0
    assert 0.5 <= best_t <= 2.0
    assert best_nll >= 0
    assert 0.0 <= ece <= 1.0
    assert set(metrics.keys()) == {"top1", "top3", "macro_f1", "nll"}

    class_names = ["paper", "other_uncertain"]
    final_preds, escalated = mod.apply_reject_rules(
        probs=probs,
        class_names=class_names,
        conf=0.7,
        margin=0.1,
        strict_per_class={"paper": 0.75},
    )
    best, rows = mod.sweep_thresholds(
        probs=probs,
        labels=labels,
        class_names=class_names,
        sweep_cfg={
            "conf_min": 0.6,
            "conf_max": 0.8,
            "conf_step": 0.1,
            "margin_min": 0.05,
            "margin_max": 0.1,
            "margin_step": 0.05,
            "target_selective_acc": 0.5,
        },
    )
    assert final_preds.shape == labels.shape
    assert escalated.shape == labels.shape
    assert best and rows


def test_eval_v3_calibrate_list_split_and_main_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _reload_eval_v3_calibrate(monkeypatch)

    data_dir = tmp_path / "data"
    (data_dir / "paper").mkdir(parents=True)
    (data_dir / "other_uncertain").mkdir(parents=True)
    Image.new("RGB", (10, 10), color=(120, 10, 10)).save(data_dir / "paper" / "p1.jpg")
    Image.new("RGB", (10, 10), color=(10, 120, 10)).save(
        data_dir / "other_uncertain" / "o1.jpg"
    )

    label_to_idx = {"paper": 0, "other_uncertain": 1}
    listed = mod.list_items(data_dir, label_to_idx)
    assert len(listed) == 2

    calib_items, eval_items = mod.stratified_split(listed, ratio=0.5, seed=123)
    assert calib_items
    assert eval_items

    cfg_path = tmp_path / "eval_v3.yaml"
    cfg_path.write_text(
        """
paths:
  robust_model: model.onnx
  robust_infer: infer_config.json
  label_map: label_map.json
  hardset_dir: data
  output_dir: out
calibration:
  split_ratio: 0.5
  seed: 42
  temperature_search:
    min: 0.5
    max: 1.5
    step: 0.5
reject_sweep:
  conf_min: 0.6
  conf_max: 0.8
  conf_step: 0.1
  margin_min: 0.05
  margin_max: 0.1
  margin_step: 0.05
  target_selective_acc: 0.5
ece:
  bins: 5
""".strip(),
        encoding="utf-8",
    )
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"onnx")
    infer_path = tmp_path / "infer_config.json"
    infer_path.write_text(
        json.dumps({"img_size": 16, "mean": [0.5, 0.5, 0.5], "std": [0.2, 0.2, 0.2]}),
        encoding="utf-8",
    )
    label_map_path = tmp_path / "label_map.json"
    label_map_path.write_text(
        json.dumps({"labels": ["paper", "other_uncertain"]}),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    monkeypatch.setattr(
        mod,
        "parse_args",
        lambda: Namespace(
            config=cfg_path,
            model=model_path,
            infer=infer_path,
            label_map=label_map_path,
            data_dir=data_dir,
            output_dir=output_dir,
            prefer_cuda=False,
        ),
    )
    monkeypatch.setattr(
        mod,
        "list_items",
        lambda _data, _map: [(Path("a.jpg"), 0), (Path("b.jpg"), 1)],
    )

    def _fake_run_logits(_session, _transform, items):
        if len(items) == 1:
            return (
                np.array([[2.0, 0.1]], dtype=np.float32),
                np.array([0], dtype=np.int64),
            )
        return (
            np.array([[1.5, 0.5]], dtype=np.float32),
            np.array([0], dtype=np.int64),
        )

    monkeypatch.setattr(mod, "run_logits", _fake_run_logits)
    monkeypatch.setattr(mod, "save_confusion_image", lambda *_a, **_k: None)
    monkeypatch.setattr(mod, "save_reliability_plot", lambda *_a, **_k: None)

    mod.main()

    reports = list(output_dir.rglob("eval_v3_calibration.json"))
    sweeps = list(output_dir.rglob("threshold_sweep.csv"))
    assert reports and sweeps


def test_eval_compare_v3_main_builds_subprocess_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import CNN.experiments.v3_generalization_upgrade.scripts.eval_compare_v3 as mod

    cfg = {
        "paths": {
            "baseline_model": "CNN/models/base.onnx",
            "baseline_infer": "CNN/models/base_infer.json",
            "robust_model": "CNN/models/v3.onnx",
            "robust_infer": "CNN/models/infer_config.json",
            "label_map": "CNN/models/label_map.json",
            "hardset_dir": "CNN/data/hardset",
            "output_dir": "CNN/experiments/v3_generalization_upgrade/outputs",
        }
    }
    cfg_path = tmp_path / "eval_v3.yaml"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    calls: list[list[str]] = []

    monkeypatch.setattr(mod, "parse_args", lambda: Namespace(config=cfg_path, prefer_cuda=True))
    monkeypatch.setattr(mod, "load_yaml", lambda _path: cfg)
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda cmd, check, cwd: calls.append(cmd),
    )

    mod.main()
    assert calls
    assert "--prefer-cuda" in calls[0]


def test_train_v3_apply_calibration_and_main_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import CNN.experiments.v3_generalization_upgrade.scripts.train_v3 as mod

    infer_path = tmp_path / "infer_config.json"
    infer_path.write_text("{}", encoding="utf-8")
    report_path = tmp_path / "eval_v3_calibration.json"
    report_path.write_text(
        json.dumps(
            {
                "temperature_scaling": {"temperature": 1.25},
                "recommended_reject_thresholds": {
                    "conf_threshold": 0.72,
                    "margin_threshold": 0.11,
                    "strict_per_class": {"paper": 0.8},
                },
            }
        ),
        encoding="utf-8",
    )
    updated = mod.apply_calibration_to_infer(infer_path, report_path)
    assert updated["temperature"] == 1.25
    assert updated["conf_threshold"] == 0.72

    train_cfg_path = tmp_path / "train_v3.yaml"
    eval_cfg_path = tmp_path / "eval_v3.yaml"
    artifact_root = tmp_path / "artifacts"
    models_dir = tmp_path / "models"
    output_root = tmp_path / "outputs"
    run_dir = artifact_root / "robust_v2_001"
    run_dir.mkdir(parents=True)
    calib_run_dir = output_root / "v3_eval_001"
    calib_run_dir.mkdir(parents=True)

    (run_dir / "model.onnx").write_bytes(b"onnx")
    (run_dir / "label_map.json").write_text('{"labels":["paper","other_uncertain"]}', encoding="utf-8")
    (run_dir / "infer_config.json").write_text("{}", encoding="utf-8")
    (calib_run_dir / "eval_v3_calibration.json").write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
    models_dir.mkdir(parents=True)
    (models_dir / "infer_config.json").write_text("{}", encoding="utf-8")

    train_cfg_path.write_text(
        json.dumps(
            {
                "paths": {
                    "artifact_dir": str(artifact_root),
                    "models_dir": str(models_dir),
                }
            }
        ),
        encoding="utf-8",
    )
    eval_cfg_path.write_text(
        json.dumps(
            {
                "paths": {
                    "hardset_dir": str(tmp_path / "hardset"),
                    "output_dir": str(output_root),
                }
            }
        ),
        encoding="utf-8",
    )

    calls: list[list[str]] = []
    monkeypatch.setattr(
        mod,
        "parse_args",
        lambda: (
            Namespace(
                config=train_cfg_path,
                eval_config=eval_cfg_path,
                skip_train=False,
                skip_calibration=False,
            ),
            ["--epochs", "1"],
        ),
    )
    monkeypatch.setattr(
        mod,
        "latest_run_dir",
        lambda root, prefix: run_dir if prefix.startswith("robust_v2_") else calib_run_dir,
    )
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda cmd, check, cwd: calls.append(cmd),
    )

    mod.main()
    assert len(calls) == 2
    assert (run_dir / "v3_post_calibration_summary.json").exists()
