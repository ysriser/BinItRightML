import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML config must be a mapping (key: value).")
    return data


def save_yaml(payload: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def save_json(payload: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(state: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    torch.save(state, path)


def load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=True)


def resolve_device(device: str, require_cuda: bool) -> torch.device:
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("require_cuda is true but CUDA is not available.")
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def snapshot_config(cfg: Dict[str, Any], out_dir: Path) -> None:
    save_yaml(cfg, out_dir / "config_snapshot.yaml")

