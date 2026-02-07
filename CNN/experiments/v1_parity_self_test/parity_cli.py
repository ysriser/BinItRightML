"""Inference parity self-test CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm


def load_utils():
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from CNN.experiments.v1_parity_self_test import parity_utils as utils

    return utils, repo_root


def resolve_path(path_str: str, repo_root: Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else repo_root / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference parity self-test")
    parser.add_argument(
        "--mode",
        choices=["golden", "compare", "android_compare"],
        required=True,
    )
    parser.add_argument("--model", type=str, default="CNN/models/tier1.onnx")
    parser.add_argument(
        "--infer-config",
        type=str,
        default="CNN/models/infer_config.json",
    )
    parser.add_argument(
        "--label-map",
        type=str,
        default="CNN/models/label_map.json",
    )
    parser.add_argument("--images", type=str, default=None)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="CNN/experiments/v1_parity_self_test/outputs",
    )
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--resize-scale", type=float, default=1.15)
    parser.add_argument("--golden", type=str, default=None)
    parser.add_argument("--current", type=str, default=None)
    parser.add_argument("--max-prob-diff", type=float, default=0.02)
    parser.add_argument("--max-top1-mismatch", type=float, default=0.0)
    parser.add_argument("--worst-n", type=int, default=10)
    parser.add_argument("--android-tensor", type=str, default=None)
    parser.add_argument("--android-probs", type=str, default=None)
    parser.add_argument("--prefer-cuda", action="store_true", default=False)
    return parser.parse_args()


def run_golden(args: argparse.Namespace) -> int:
    utils, repo_root = load_utils()

    model_path = resolve_path(args.model, repo_root)
    infer_path = resolve_path(args.infer_config, repo_root)
    label_map_path = resolve_path(args.label_map, repo_root)
    output_dir = resolve_path(args.output_dir, repo_root)

    images_dir = resolve_path(args.images, repo_root) if args.images else None
    manifest_path = resolve_path(args.manifest, repo_root) if args.manifest else None

    cfg = utils.load_infer_config(infer_path, resize_scale=args.resize_scale)
    labels = utils.load_labels(label_map_path)
    session = utils.build_session(model_path, prefer_cuda=args.prefer_cuda)

    items = utils.list_items(images_dir, manifest_path)
    if not items:
        raise ValueError("No images found for parity test")

    records: List[Dict[str, Any]] = []
    for item in tqdm(items, desc="Parity", ncols=100):
        record = utils.record_for_image(
            session=session,
            cfg=cfg,
            labels=labels,
            path=item["path"],
            label=item.get("label"),
            image_id=item["image_id"],
            topk_n=args.topk,
        )
        records.append(record)

    output = utils.build_output(
        records=records,
        model_path=model_path,
        infer_path=infer_path,
        label_map_path=label_map_path,
        cfg=cfg,
    )

    json_path = output_dir / "parity_golden.json"
    csv_path = output_dir / "parity_golden.csv"
    utils.save_json(json_path, output)
    utils.write_csv(csv_path, records)

    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")
    return 0


def run_compare(args: argparse.Namespace) -> int:
    utils, repo_root = load_utils()

    if not args.golden or not args.current:
        raise ValueError("Compare mode requires --golden and --current")

    golden_path = resolve_path(args.golden, repo_root)
    current_path = resolve_path(args.current, repo_root)
    output_dir = resolve_path(args.output_dir, repo_root)

    golden = utils.load_json(golden_path)
    current = utils.load_json(current_path)

    report, failures = utils.compare_runs(
        golden=golden,
        current=current,
        max_prob_diff=args.max_prob_diff,
        max_top1_mismatch=args.max_top1_mismatch,
        worst_n=args.worst_n,
    )

    report_path = output_dir / "parity_compare.json"
    utils.save_json(report_path, report)

    print(f"Saved: {report_path}")
    print(
        "Summary -> "
        f"mismatch_rate={report['mismatch_rate']:.3f} "
        f"max_prob_diff={report['max_prob_diff']:.4f}"
    )

    if report["missing"]:
        print(f"Missing images: {len(report['missing'])}")

    if report["worst"]:
        print("Worst diffs:")
        for diff in report["worst"]:
            reasons = ",".join(diff["reasons"]) if diff["reasons"] else "none"
            print(
                "- "
                f"{diff['image_id']} "
                f"max_diff={diff['max_prob_diff']:.4f} "
                f"top1_mismatch={diff['top1_mismatch']} "
                f"top3_overlap={diff['top3_overlap']} "
                f"reasons={reasons}"
            )

    if failures:
        print("Parity check FAILED")
        return 1
    print("Parity check PASSED")
    return 0


def run_android_compare(args: argparse.Namespace) -> int:
    utils, repo_root = load_utils()

    if not args.android_tensor or not args.android_probs:
        raise ValueError(
            "android_compare requires --android-tensor and --android-probs"
        )

    model_path = resolve_path(args.model, repo_root)
    infer_path = resolve_path(args.infer_config, repo_root)
    tensor_path = resolve_path(args.android_tensor, repo_root)
    probs_path = resolve_path(args.android_probs, repo_root)

    cfg = utils.load_infer_config(infer_path, resize_scale=args.resize_scale)
    session = utils.build_session(model_path, prefer_cuda=args.prefer_cuda)

    tensor = np.fromfile(tensor_path, dtype=np.float32)
    expected = 1 * 3 * cfg.img_size * cfg.img_size
    if tensor.size != expected:
        raise ValueError(
            f"Tensor size mismatch: got {tensor.size}, expected {expected}"
        )
    tensor = tensor.reshape((1, 3, cfg.img_size, cfg.img_size))

    logits = utils.run_onnx(session, tensor)
    probs_py = utils.softmax(logits)
    probs_android = utils.parse_android_probs(probs_path)

    if probs_android.size != probs_py.size:
        raise ValueError("Android probs size mismatch")

    max_diff = float(np.max(np.abs(probs_py - probs_android)))
    top1_py = int(np.argmax(probs_py))
    top1_android = int(np.argmax(probs_android))
    top1_mismatch = top1_py != top1_android

    print(f"Android compare max_diff={max_diff:.6f} top1_mismatch={top1_mismatch}")
    if max_diff > args.max_prob_diff or top1_mismatch:
        return 1
    return 0


def main() -> None:
    args = parse_args()
    if args.mode == "golden":
        sys.exit(run_golden(args))
    if args.mode == "compare":
        sys.exit(run_compare(args))
    if args.mode == "android_compare":
        sys.exit(run_android_compare(args))


if __name__ == "__main__":
    main()
