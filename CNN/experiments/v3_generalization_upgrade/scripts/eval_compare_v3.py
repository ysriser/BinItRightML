"""Run baseline-vs-v3 comparison by delegating to eval_compare_v2."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML mapping")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline vs robust v3")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("CNN/experiments/v3_generalization_upgrade/configs/eval_v3.yaml"),
    )
    parser.add_argument("--prefer-cuda", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ml_root = Path(__file__).resolve().parents[4]

    config_path = args.config
    if not config_path.is_absolute():
        config_path = ml_root / config_path
    cfg = load_yaml(config_path)
    paths = cfg.get("paths", {})

    eval_v2 = ml_root / "CNN/experiments/v2_robust_finetune/scripts/eval_compare_v2.py"
    if not eval_v2.exists():
        raise FileNotFoundError(f"Missing eval_compare_v2 script: {eval_v2}")

    baseline_model = ml_root / paths.get(
        "baseline_model", "CNN/versions/v0_classification_6cats_new/models/tier1.onnx"
    )
    baseline_infer = ml_root / paths.get(
        "baseline_infer", "CNN/versions/v0_classification_6cats_new/models/infer_config.json"
    )
    robust_model = ml_root / paths.get("robust_model", "CNN/models/tier1.onnx")
    robust_infer = ml_root / paths.get("robust_infer", "CNN/models/infer_config.json")
    label_map = ml_root / paths.get("label_map", "CNN/models/label_map.json")
    data_dir = ml_root / paths.get("hardset_dir", "CNN/data/hardset")
    output_dir = ml_root / paths.get("output_dir", "CNN/experiments/v3_generalization_upgrade/outputs")

    cmd = [
        sys.executable,
        str(eval_v2),
        "--baseline-model",
        str(baseline_model),
        "--baseline-infer",
        str(baseline_infer),
        "--robust-model",
        str(robust_model),
        "--robust-infer",
        str(robust_infer),
        "--label-map",
        str(label_map),
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(output_dir),
    ]
    if args.prefer_cuda:
        cmd.append("--prefer-cuda")

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ml_root)


if __name__ == "__main__":
    main()