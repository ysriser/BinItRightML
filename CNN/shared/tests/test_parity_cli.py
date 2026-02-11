from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from CNN.experiments.v1_parity_self_test import parity_cli as cli


def test_resolve_path_relative_and_absolute(tmp_path: Path) -> None:
    rel = cli.resolve_path("a/b.txt", tmp_path)
    abs_path = cli.resolve_path(str(tmp_path / "z.txt"), tmp_path)

    assert rel == tmp_path / "a" / "b.txt"
    assert abs_path == tmp_path / "z.txt"


def test_parse_args_reads_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["parity_cli.py", "--mode", "golden"])
    args = cli.parse_args()
    assert args.mode == "golden"


def test_run_golden_happy_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {}

    def save_json(path: Path, data: dict) -> None:
        calls["json_path"] = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

    def write_csv(path: Path, records) -> None:
        calls["csv_path"] = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("csv", encoding="utf-8")

    fake_utils = SimpleNamespace(
        load_infer_config=lambda path, resize_scale: "cfg",
        load_labels=lambda _: ["paper"],
        build_session=lambda model_path, prefer_cuda: "sess",
        list_items=lambda images_dir, manifest_path: [
            {"path": tmp_path / "x.jpg", "label": "paper", "image_id": "x.jpg"}
        ],
        record_for_image=lambda **kwargs: {"image_id": "x.jpg", "top3": []},
        build_output=lambda **kwargs: {"meta": {"total": 1}, "images": [{}]},
        save_json=save_json,
        write_csv=write_csv,
    )

    monkeypatch.setattr(cli, "load_utils", lambda: (fake_utils, tmp_path))

    args = Namespace(
        model="model.onnx",
        infer_config="infer.json",
        label_map="label.json",
        output_dir="out",
        images=None,
        manifest=None,
        resize_scale=1.15,
        prefer_cuda=False,
        topk=3,
    )

    code = cli.run_golden(args)

    assert code == 0
    assert calls["json_path"].name == "parity_golden.json"
    assert calls["csv_path"].name == "parity_golden.csv"


def test_run_golden_raises_when_no_items(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_utils = SimpleNamespace(
        load_infer_config=lambda path, resize_scale: "cfg",
        load_labels=lambda _: ["paper"],
        build_session=lambda model_path, prefer_cuda: "sess",
        list_items=lambda images_dir, manifest_path: [],
    )
    monkeypatch.setattr(cli, "load_utils", lambda: (fake_utils, tmp_path))

    args = Namespace(
        model="model.onnx",
        infer_config="infer.json",
        label_map="label.json",
        output_dir="out",
        images=None,
        manifest=None,
        resize_scale=1.15,
        prefer_cuda=False,
        topk=3,
    )

    with pytest.raises(ValueError, match="No images"):
        cli.run_golden(args)


def test_run_compare_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    saved = {}

    fake_report = {
        "mismatch_rate": 0.0,
        "max_prob_diff": 0.0,
        "missing": [],
        "worst": [],
    }
    fake_utils = SimpleNamespace(
        load_json=lambda path: {"images": []},
        compare_runs=lambda **kwargs: (fake_report, 0),
        save_json=lambda path, data: saved.setdefault("path", path),
    )
    monkeypatch.setattr(cli, "load_utils", lambda: (fake_utils, tmp_path))

    args = Namespace(
        golden="golden.json",
        current="current.json",
        output_dir="out",
        max_prob_diff=0.02,
        max_top1_mismatch=0.0,
        worst_n=5,
    )

    code = cli.run_compare(args)
    assert code == 0
    assert saved["path"].name == "parity_compare.json"


def test_run_compare_fails_when_compare_runs_reports_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_report = {
        "mismatch_rate": 0.1,
        "max_prob_diff": 0.3,
        "missing": ["a.jpg"],
        "worst": [
            {
                "image_id": "a",
                "max_prob_diff": 0.3,
                "top1_mismatch": True,
                "top3_overlap": 1,
                "reasons": ["input_tensor_hash_mismatch"],
            }
        ],
    }
    fake_utils = SimpleNamespace(
        load_json=lambda path: {"images": []},
        compare_runs=lambda **kwargs: (fake_report, 1),
        save_json=lambda path, data: None,
    )
    monkeypatch.setattr(cli, "load_utils", lambda: (fake_utils, tmp_path))

    args = Namespace(
        golden="golden.json",
        current="current.json",
        output_dir="out",
        max_prob_diff=0.02,
        max_top1_mismatch=0.0,
        worst_n=5,
    )

    assert cli.run_compare(args) == 1


def test_run_android_compare_happy_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tensor_path = tmp_path / "input.bin"
    probs_path = tmp_path / "probs.json"
    np.zeros((1, 3, 2, 2), dtype=np.float32).tofile(tensor_path)
    probs_path.write_text("[0.9, 0.1]", encoding="utf-8")

    fake_utils = SimpleNamespace(
        load_infer_config=lambda infer_path, resize_scale: SimpleNamespace(img_size=2),
        build_session=lambda model_path, prefer_cuda: "sess",
        run_onnx=lambda session, tensor: np.array([[9.0, 1.0]], dtype=np.float32),
        softmax=lambda logits: np.array([0.9, 0.1], dtype=np.float32),
        parse_android_probs=lambda path: np.array([0.9, 0.1], dtype=np.float32),
    )
    monkeypatch.setattr(cli, "load_utils", lambda: (fake_utils, tmp_path))

    args = Namespace(
        model="model.onnx",
        infer_config="infer.json",
        android_tensor=str(tensor_path),
        android_probs=str(probs_path),
        max_prob_diff=0.02,
        prefer_cuda=False,
        resize_scale=1.15,
    )

    assert cli.run_android_compare(args) == 0


def test_run_android_compare_returns_failure_on_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tensor_path = tmp_path / "input.bin"
    probs_path = tmp_path / "probs.json"
    np.zeros((1, 3, 2, 2), dtype=np.float32).tofile(tensor_path)
    probs_path.write_text("[0.1, 0.9]", encoding="utf-8")

    fake_utils = SimpleNamespace(
        load_infer_config=lambda infer_path, resize_scale: SimpleNamespace(img_size=2),
        build_session=lambda model_path, prefer_cuda: "sess",
        run_onnx=lambda session, tensor: np.array([[9.0, 1.0]], dtype=np.float32),
        softmax=lambda logits: np.array([0.9, 0.1], dtype=np.float32),
        parse_android_probs=lambda path: np.array([0.1, 0.9], dtype=np.float32),
    )
    monkeypatch.setattr(cli, "load_utils", lambda: (fake_utils, tmp_path))

    args = Namespace(
        model="model.onnx",
        infer_config="infer.json",
        android_tensor=str(tensor_path),
        android_probs=str(probs_path),
        max_prob_diff=0.02,
        prefer_cuda=False,
        resize_scale=1.15,
    )

    assert cli.run_android_compare(args) == 1


def test_main_dispatches_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "parse_args", lambda: Namespace(mode="golden"))
    monkeypatch.setattr(cli, "run_golden", lambda args: 0)
    with pytest.raises(SystemExit) as exc_info:
        cli.main()
    assert exc_info.value.code == 0
