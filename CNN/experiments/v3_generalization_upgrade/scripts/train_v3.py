"""Train robust v3 model and auto-apply calibration thresholds."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML mapping")
    return data


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Train robust v3 model")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("CNN/experiments/v3_generalization_upgrade/configs/train_v3.yaml"),
    )
    parser.add_argument(
        "--eval-config",
        type=Path,
        default=Path("CNN/experiments/v3_generalization_upgrade/configs/eval_v3.yaml"),
    )
    parser.add_argument("--skip-train", action="store_true", default=False)
    parser.add_argument("--skip-calibration", action="store_true", default=False)
    return parser.parse_known_args()


def latest_run_dir(root: Path, prefix: str) -> Path:
    candidates = [d for d in root.glob(f"{prefix}*") if d.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under: {root}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def apply_calibration_to_infer(
    infer_path: Path,
    calibration_report_path: Path,
) -> Dict[str, Any]:
    infer_cfg = load_json(infer_path)
    report = load_json(calibration_report_path)

    temp = report.get("temperature_scaling", {}).get("temperature")
    rec = report.get("recommended_reject_thresholds", {})

    if temp is not None:
        infer_cfg["temperature"] = float(temp)

    conf = rec.get("conf_threshold")
    margin = rec.get("margin_threshold")
    if conf is not None:
        infer_cfg["conf_threshold"] = float(conf)
    if margin is not None:
        infer_cfg["margin_threshold"] = float(margin)

    strict = rec.get("strict_per_class")
    if isinstance(strict, dict) and strict:
        infer_cfg["strict_per_class"] = strict

    save_json(infer_path, infer_cfg)
    return infer_cfg


def main() -> None:
    args, passthrough = parse_args()

    ml_root = Path(__file__).resolve().parents[4]
    train_v2 = ml_root / "CNN/experiments/v2_robust_finetune/scripts/train_v2.py"
    eval_cal_script = ml_root / "CNN/experiments/v3_generalization_upgrade/scripts/eval_v3_calibrate.py"

    if not train_v2.exists():
        raise FileNotFoundError(f"Missing train_v2 script: {train_v2}")
    if not eval_cal_script.exists():
        raise FileNotFoundError(f"Missing calibration script: {eval_cal_script}")

    train_cfg_path = args.config
    if not train_cfg_path.is_absolute():
        train_cfg_path = ml_root / train_cfg_path
    eval_cfg_path = args.eval_config
    if not eval_cfg_path.is_absolute():
        eval_cfg_path = ml_root / eval_cfg_path

    train_cfg = load_yaml(train_cfg_path)
    eval_cfg = load_yaml(eval_cfg_path)

    paths = train_cfg.get("paths", {})
    artifact_root = ml_root / paths.get(
        "artifact_dir", "CNN/experiments/v3_generalization_upgrade/artifacts"
    )
    models_dir = ml_root / paths.get("models_dir", "CNN/models")

    if not args.skip_train:
        train_cmd = [
            sys.executable,
            str(train_v2),
            "--config",
            str(train_cfg_path),
            *passthrough,
        ]
        print("Running train:", " ".join(train_cmd))
        subprocess.run(train_cmd, check=True, cwd=ml_root)

    run_dir = latest_run_dir(artifact_root, "robust_v2_")
    print(f"Using run artifact: {run_dir}")

    if args.skip_calibration:
        print("Calibration skipped by flag.")
        return

    eval_paths = eval_cfg.get("paths", {})
    hardset_dir = ml_root / eval_paths.get("hardset_dir", "CNN/data/hardset")
    label_map = run_dir / "label_map.json"
    infer_cfg_path = run_dir / "infer_config.json"
    model_path = run_dir / "model.onnx"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing ONNX model in artifact: {model_path}")

    calib_cmd = [
        sys.executable,
        str(eval_cal_script),
        "--config",
        str(eval_cfg_path),
        "--model",
        str(model_path),
        "--infer",
        str(infer_cfg_path),
        "--label-map",
        str(label_map),
        "--data-dir",
        str(hardset_dir),
    ]
    print("Running calibration:", " ".join(calib_cmd))
    subprocess.run(calib_cmd, check=True, cwd=ml_root)

    eval_output_root = ml_root / eval_paths.get(
        "output_dir", "CNN/experiments/v3_generalization_upgrade/outputs"
    )
    calib_run_dir = latest_run_dir(eval_output_root, "v3_eval_")
    calibration_report = calib_run_dir / "eval_v3_calibration.json"
    if not calibration_report.exists():
        raise FileNotFoundError(f"Missing calibration report: {calibration_report}")

    updated_artifact_infer = apply_calibration_to_infer(infer_cfg_path, calibration_report)

    models_infer = models_dir / "infer_config.json"
    if models_infer.exists():
        updated_models_infer = apply_calibration_to_infer(models_infer, calibration_report)
    else:
        updated_models_infer = None

    summary = {
        "train_config": str(train_cfg_path),
        "eval_config": str(eval_cfg_path),
        "artifact_run": str(run_dir),
        "calibration_run": str(calib_run_dir),
        "artifact_infer_after_calibration": updated_artifact_infer,
        "models_infer_after_calibration": updated_models_infer,
    }
    save_json(run_dir / "v3_post_calibration_summary.json", summary)

    print(f"Saved post-calibration summary -> {run_dir / 'v3_post_calibration_summary.json'}")


if __name__ == "__main__":
    main()