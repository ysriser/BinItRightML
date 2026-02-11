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
import torch
from PIL import Image


class _TinyModel(torch.nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.pool = torch.nn.AdaptiveAvgPool2d((8, 8))
        self.flatten = torch.nn.Flatten()
        self.classifier = torch.nn.Linear(3 * 8 * 8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.flatten(x)
        return self.classifier(x)


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
            return [np.array([[2.0, 0.5]], dtype=np.float32)]

    fake.InferenceSession = _Session
    fake.get_available_providers = lambda: ["CPUExecutionProvider"]
    return fake


def _stub_timm() -> types.ModuleType:
    fake = types.ModuleType("timm")
    fake.__spec__ = importlib.machinery.ModuleSpec("timm", loader=None)
    fake.create_model = (
        lambda _backbone, pretrained, num_classes: _TinyModel(num_classes=num_classes)
    )
    fake.data = types.SimpleNamespace(
        resolve_data_config=lambda *_a, **_k: {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }
    )
    return fake


def _reload_with_stubs(monkeypatch: pytest.MonkeyPatch, module_name: str):
    if module_name in sys.modules:
        del sys.modules[module_name]
    monkeypatch.setitem(sys.modules, "onnxruntime", _stub_onnxruntime())
    monkeypatch.setitem(sys.modules, "timm", _stub_timm())
    module = importlib.import_module(module_name)
    return importlib.reload(module)


def test_eval_compare_v2_helpers_and_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _reload_with_stubs(
        monkeypatch, "CNN.experiments.v2_robust_finetune.scripts.eval_compare_v2"
    )

    paper = tmp_path / "paper"
    paper.mkdir()
    img_path = paper / "a.jpg"
    Image.new("RGB", (16, 16), color=(30, 40, 50)).save(img_path)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("filepath,label\na.jpg,paper\nmissing.jpg,unknown\n", encoding="utf-8")

    label_to_idx = {"paper": 0, "other_uncertain": 1}
    items = mod.list_images_from_manifest(manifest, label_to_idx)
    assert items and items[0][0].name == "a.jpg"

    class _DecisionUtils:
        @staticmethod
        def decide_from_probs(probs, class_names, thresholds):
            _ = thresholds
            pred = int(np.argmax(probs))
            score = float(np.max(probs))
            escalate = score < 0.6
            final_label = "other_uncertain" if escalate else class_names[pred]
            return {"final_label": final_label, "escalate": escalate}

    monkeypatch.setattr(mod, "load_decision_utils", lambda: _DecisionUtils)
    probs_list = [
        np.array([0.90, 0.10], dtype=np.float32),
        np.array([0.55, 0.45], dtype=np.float32),
    ]
    labels = [0, 0]
    class_names = ["paper", "other_uncertain"]

    selective = mod.selective_metrics(
        probs_list=probs_list,
        labels=labels,
        class_names=class_names,
        thresholds={"conf": 0.5, "margin": 0.1},
    )
    best = mod.sweep_thresholds(
        probs_list=probs_list,
        labels=labels,
        class_names=class_names,
        base_thresholds={"conf": 0.5, "margin": 0.1},
    )

    assert 0.0 <= selective["coverage"] <= 1.0
    assert "selective_acc" in selective
    assert set(best.keys()) == {"conf", "margin", "coverage", "selective_acc"}


def test_eval_compare_v2_main_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _reload_with_stubs(
        monkeypatch, "CNN.experiments.v2_robust_finetune.scripts.eval_compare_v2"
    )

    label_map = tmp_path / "label_map.json"
    label_map.write_text(json.dumps({"labels": ["paper", "other_uncertain"]}), encoding="utf-8")
    infer_cfg = tmp_path / "infer.json"
    infer_cfg.write_text(
        json.dumps({"img_size": 32, "mean": [0.4, 0.4, 0.4], "std": [0.2, 0.2, 0.2]}),
        encoding="utf-8",
    )
    reject_cfg = tmp_path / "reject.yaml"
    reject_cfg.write_text("thresholds: {conf: 0.6, margin: 0.1}\noutput: {topk: 3}\n", encoding="utf-8")
    output_dir = tmp_path / "outputs"
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    monkeypatch.setattr(
        mod,
        "parse_args",
        lambda: Namespace(
            baseline_model=tmp_path / "baseline.onnx",
            baseline_infer=infer_cfg,
            robust_model=tmp_path / "robust.onnx",
            robust_infer=infer_cfg,
            label_map=label_map,
            data_dir=data_dir,
            reject_config=reject_cfg,
            output_dir=output_dir,
            prefer_cuda=False,
        ),
    )
    monkeypatch.setattr(
        mod,
        "build_groups",
        lambda _data_dir, _label_to_idx: [mod.EvalGroup("overall", [])],
    )
    monkeypatch.setattr(
        mod,
        "evaluate_group",
        lambda *_a, **_k: (
            {"top1": 1.0, "top3": 1.0, "macro_f1": 1.0},
            [0],
            [0],
            [np.array([0.9, 0.1], dtype=np.float32)],
        ),
    )
    monkeypatch.setattr(mod, "save_confusion", lambda *_a, **_k: None)

    class _DecisionUtils:
        @staticmethod
        def decide_from_probs(_probs, class_names, _thresholds):
            return {"final_label": class_names[0], "escalate": False}

    monkeypatch.setattr(mod, "load_decision_utils", lambda: _DecisionUtils)

    mod.main()

    json_reports = list(output_dir.rglob("eval_compare_v2.json"))
    csv_reports = list(output_dir.rglob("eval_compare_v2.csv"))
    assert json_reports and csv_reports


def test_export_v2_list_images_and_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _reload_with_stubs(
        monkeypatch, "CNN.experiments.v2_robust_finetune.scripts.export_v2"
    )

    (tmp_path / "cls").mkdir()
    a = tmp_path / "cls" / "a.jpg"
    b = tmp_path / "cls" / "b.png"
    Image.new("RGB", (8, 8), color=(1, 2, 3)).save(a)
    Image.new("RGB", (8, 8), color=(4, 5, 6)).save(b)
    assert [p.name for p in mod.list_images(tmp_path, limit=1)] == ["a.jpg"]

    csv_path = tmp_path / "items.csv"
    csv_path.write_text("filepath\ncls/a.jpg\nmissing.jpg\n", encoding="utf-8")
    loaded = mod.list_images_from_csv(csv_path, limit=5)
    assert loaded == [tmp_path / "cls" / "a.jpg"]


def test_export_v2_main_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _reload_with_stubs(
        monkeypatch, "CNN.experiments.v2_robust_finetune.scripts.export_v2"
    )

    artifact_dir = tmp_path / "artifacts" / "robust_v2_run"
    artifact_dir.mkdir(parents=True)
    labels = ["paper", "other_uncertain"]
    torch.save(_TinyModel(len(labels)).state_dict(), artifact_dir / "best.pt")

    cfg = {
        "labels": labels,
        "model": {"backbone": "tiny", "img_size": 16},
        "onnx": {"opset": 17, "sanity_images": 2},
        "infer": {"topk": 3, "conf_threshold": 0.7, "margin_threshold": 0.1},
        "paths": {"output_dir": str(tmp_path / "out"), "g3_data_dir": str(tmp_path / "missing")},
    }
    cfg_path = tmp_path / "train_v2.yaml"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    monkeypatch.setattr(
        mod,
        "parse_args",
        lambda: Namespace(
            config=cfg_path,
            checkpoint=None,
            artifact_dir=artifact_dir,
            sanity_images=None,
            prefer_cuda=False,
        ),
    )
    monkeypatch.setattr(mod, "list_images", lambda *_a, **_k: [])
    monkeypatch.setattr(mod, "list_images_from_csv", lambda *_a, **_k: [])

    def _fake_export(_model, _dummy, out_path, **_kwargs):
        Path(out_path).write_bytes(b"onnx")

    monkeypatch.setattr(torch.onnx, "export", _fake_export)
    mod.main()

    assert (artifact_dir / "model.onnx").exists()
    assert (artifact_dir / "infer_config.json").exists()
    assert (artifact_dir / "label_map.json").exists()


def test_train_v2_helpers_cover_core_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _reload_with_stubs(
        monkeypatch, "CNN.experiments.v2_robust_finetune.scripts.train_v2"
    )

    data_dir = tmp_path / "data"
    (data_dir / "paper").mkdir(parents=True)
    img = data_dir / "paper" / "x.jpg"
    Image.new("RGB", (12, 12), color=(9, 8, 7)).save(img)

    label_to_idx = {"paper": 0, "other_uncertain": 1}
    folder_items = mod.list_images_from_folder(data_dir, label_to_idx)
    assert folder_items and folder_items[0][1] == 0

    manifest = data_dir / "manifest.csv"
    manifest.write_text("filepath,label\npaper/x.jpg,paper\n", encoding="utf-8")
    manifest_items = mod.list_images_from_manifest(manifest, label_to_idx)
    assert manifest_items and manifest_items[0][0].name == "x.jpg"
    assert mod.build_domain_items(data_dir, label_to_idx)

    split = mod.split_items(folder_items * 2, val_ratio=0.5, seed=7)
    assert len(split.train) + len(split.val) == 2

    weights = mod.compute_class_weights([(img, 0), (img, 0), (img, 1)], num_classes=2)
    assert tuple(weights.shape) == (2,)

    labels = torch.tensor([0, 1], dtype=torch.long)
    hot = mod.one_hot(labels, num_classes=2)
    assert tuple(hot.shape) == (2, 2)

    logits = torch.tensor([[3.0, 1.0], [0.2, 2.1]], dtype=torch.float32)
    loss = mod.soft_target_ce(logits, hot)
    focal = mod.FocalCrossEntropy(gamma=1.0)(logits, labels)
    assert float(loss) > 0
    assert float(focal) > 0

    x1, y1, x2, y2 = mod.rand_bbox(10, 8, lam=0.5)
    assert 0 <= x1 <= x2 <= 10
    assert 0 <= y1 <= y2 <= 8

    images = torch.ones((2, 3, 4, 4), dtype=torch.float32)
    mixed, mixed_targets, used = mod.apply_mixup_cutmix(
        images=images.clone(),
        labels=labels,
        num_classes=2,
        mixup_cfg={"mixup_alpha": 1.0, "mixup_p": 1.0, "cutmix_alpha": 0.0, "cutmix_p": 0.0},
    )
    assert used is True
    assert tuple(mixed.shape) == (2, 3, 4, 4)
    assert tuple(mixed_targets.shape) == (2, 2)

    disabled = mod.apply_mixup_cutmix(
        images=images.clone(),
        labels=labels,
        num_classes=2,
        mixup_cfg={"mixup_alpha": 0.0, "mixup_p": 0.0, "cutmix_alpha": 0.0, "cutmix_p": 0.0},
    )
    assert disabled[2] is False

    summary = mod.summarize_phases(
        [
            {"phase": 1, "epoch": 1, "val_f1": 0.3, "val_top1": 0.4, "val_top3": 0.5},
            {"phase": 1, "epoch": 2, "val_f1": 0.6, "val_top1": 0.7, "val_top3": 0.8},
            {"phase": 2, "epoch": 1, "val_f1": 0.5, "val_top1": 0.6, "val_top3": 0.7},
        ]
    )
    assert summary["phase_1"]["epoch"] == 2
    assert summary["phase_2"]["epoch"] == 1

    model = _TinyModel(num_classes=2)
    head = mod.resolve_head_module(model)
    assert isinstance(head, torch.nn.Module)

    backbone_params, head_params = mod.split_backbone_head_params(model)
    assert head_params
    assert len(backbone_params) + len(head_params) == len(list(model.parameters()))
    mod.set_requires_grad(head_params, False)
    assert all(not p.requires_grad for p in head_params)

    class _Result:
        returncode = 0
        stdout = "abc123\n"
        stderr = ""

    monkeypatch.setattr(mod.subprocess, "run", lambda *_a, **_k: _Result())
    assert mod.get_git_commit(tmp_path) == "abc123"

    artifact_dir = tmp_path / "artifact"
    models_dir = tmp_path / "models"
    artifact_dir.mkdir()
    (artifact_dir / "best.pt").write_bytes(b"pt")
    (artifact_dir / "model.onnx").write_bytes(b"onnx")
    (artifact_dir / "infer_config.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "label_map.json").write_text('{"labels":["paper","other_uncertain"]}', encoding="utf-8")
    mod.update_models_dir(artifact_dir, models_dir)
    assert (models_dir / "tier1_best.pt").exists()
    assert (models_dir / "tier1.onnx").exists()

    def _fake_export(_model, _dummy, out_path, **_kwargs):
        Path(out_path).write_bytes(b"onnx")

    monkeypatch.setattr(torch.onnx, "export", _fake_export)
    onnx_path = mod.export_onnx_model(
        model=model,
        artifact_dir=artifact_dir,
        labels=["paper", "other_uncertain"],
        img_size=16,
        mean=[0.1, 0.2, 0.3],
        std=[0.4, 0.5, 0.6],
        opset=17,
        backbone="tiny",
        infer_cfg={"topk": 3, "conf_threshold": 0.7, "margin_threshold": 0.1},
    )
    assert onnx_path.exists()

    phase_a = mod.save_phase_a_latest(model, artifact_dir, models_dir)
    assert phase_a.exists()
    assert (models_dir / "tier1_phase_a_latest.pt").exists()

    assert mod.setup_wandb({"wandb": {"enabled": False}}, artifact_dir, "run") is None
    mod.log_wandb(None, {"x": 1.0})


def test_train_v2_datasets_transforms_and_eval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _reload_with_stubs(
        monkeypatch, "CNN.experiments.v2_robust_finetune.scripts.train_v2"
    )

    img1 = tmp_path / "i1.jpg"
    img2 = tmp_path / "i2.jpg"
    Image.new("RGB", (20, 20), color=(40, 50, 60)).save(img1)
    Image.new("RGB", (20, 20), color=(60, 50, 40)).save(img2)

    csv_path = tmp_path / "data.csv"
    csv_path.write_text(
        f"filepath,final_label\n{img1.as_posix()},paper\n{img2.as_posix()},paper\n",
        encoding="utf-8",
    )
    ds = mod.CsvDataset(
        csv_path=csv_path,
        label_to_idx={"paper": 0},
        transform=None,
        verify_images=False,
    )
    assert len(ds) == 2
    sample_img, sample_label = ds[0]
    assert sample_label == 0
    assert sample_img.size == (20, 20)

    list_ds = mod.ListDataset(items=[(img1, 0)], transform=None)
    out_img, out_label = list_ds[0]
    assert out_label == 0
    assert out_img.size == (20, 20)

    scaled = mod.ScaleDownPad(
        img_size=16,
        scale_range=(0.5, 0.8),
        prob=1.0,
        random_position=False,
        pad_value=(0, 0, 0),
    )(Image.new("RGB", (16, 16), color=(1, 2, 3)))
    assert scaled.size == (16, 16)

    tfm = mod.build_train_transform(
        cfg={
            "augment": {
                "rrc_p": 1.0,
                "rrc_scale_min": 0.8,
                "rrc_scale_max": 1.0,
                "rrc_ratio": [0.9, 1.1],
                "hflip_p": 0.0,
                "scale_down_pad": {"enabled": True, "prob": 0.0},
                "color_jitter_p": 0.0,
                "random_erasing_p": 0.0,
            }
        },
        img_size=16,
        mean=[0.5, 0.5, 0.5],
        std=[0.2, 0.2, 0.2],
        phase="a",
    )
    tensor = tfm(Image.new("RGB", (18, 18), color=(9, 9, 9)))
    assert tuple(tensor.shape) == (3, 16, 16)

    class _Model(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            _ = x
            return torch.tensor([[3.0, 0.2], [0.1, 2.5]], dtype=torch.float32)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.zeros((2, 3, 16, 16), dtype=torch.float32),
            torch.tensor([0, 1], dtype=torch.long),
        ),
        batch_size=2,
    )
    metrics, labels, preds = mod.evaluate(
        model=_Model(),
        loader=loader,
        device=torch.device("cpu"),
        class_names=["paper", "other_uncertain"],
    )
    assert metrics["top1"] == 1.0
    assert labels == [0, 1]
    assert preds == [0, 1]


def test_train_v2_save_history_and_wandb_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _reload_with_stubs(
        monkeypatch, "CNN.experiments.v2_robust_finetune.scripts.train_v2"
    )

    rows = [
        {
            "phase": 1,
            "epoch": 1,
            "global_epoch": 1,
            "train_loss": 0.5,
            "val_top1": 0.8,
            "val_f1": 0.7,
        }
    ]
    mod.save_history(rows, tmp_path)
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "metrics.json").exists()

    monkeypatch.setattr(mod, "wandb", None)
    with pytest.raises(ImportError):
        mod.setup_wandb({"wandb": {"enabled": True}}, tmp_path, "run")
