from pathlib import Path
from typing import Dict, List, Tuple

from ..utils import load_yaml


def load_label_map(path: Path) -> Tuple[Dict[str, str], List[str]]:
    data = load_yaml(path)
    fine_to_super = data.get("fine_to_super", {})
    super_classes = data.get("super_classes", [])
    if not fine_to_super or not super_classes:
        raise ValueError(
            "label_map is empty. Generate it by running data_tools/build_taco_cls_dataset.py"
        )
    if not isinstance(fine_to_super, dict) or not isinstance(super_classes, list):
        raise ValueError("label_map file has invalid structure.")
    return fine_to_super, super_classes


def build_label_map_from_coco(coco_json: dict) -> Tuple[Dict[str, str], List[str]]:
    fine_to_super: Dict[str, str] = {}
    super_set = set()
    for cat in coco_json.get("categories", []):
        name = cat.get("name")
        super_name = cat.get("supercategory")
        if not name or not super_name:
            continue
        fine_to_super[name] = super_name
        super_set.add(super_name)
    super_classes = sorted(super_set)
    return fine_to_super, super_classes
