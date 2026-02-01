from pathlib import Path

import pytest

from CNN.classification_6cats_new.scripts.build_manifest_and_splits import (
    allocate_by_weights,
    resolve_class_root,
    stable_int,
)


def _write_image(path: Path) -> None:
    # Create a minimal fake file to simulate an image path.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake")


def test_allocate_by_weights_basic():
    # Step 1: allocate 10 items equally.
    result = allocate_by_weights(10, [1.0, 1.0])
    # Step 2: verify totals and balance.
    assert sum(result) == 10
    assert result == [5, 5]


def test_allocate_by_weights_zero_total():
    # Step 1: allocate zero items.
    result = allocate_by_weights(0, [1.0, 2.0])
    # Step 2: result should be all zeros.
    assert result == [0, 0]


def test_allocate_by_weights_invalid_weights():
    # Step 1: weights sum to 0 should raise.
    with pytest.raises(ValueError):
        allocate_by_weights(5, [0.0, 0.0])


def test_stable_int_is_deterministic():
    # Step 1: same input -> same output.
    assert stable_int("a") == stable_int("a")
    # Step 2: different input -> different output.
    assert stable_int("a") != stable_int("b")


def test_resolve_class_root_explicit(tmp_path: Path):
    # Step 1: create explicit class_root.
    root = tmp_path / "dataset"
    class_root = root / "classes"
    _write_image(class_root / "paper" / "img.jpg")
    # Step 2: resolve using config with class_root.
    resolved = resolve_class_root({"path": str(root), "class_root": "classes"})
    assert resolved == class_root


def test_resolve_class_root_auto_detect_base(tmp_path: Path):
    # Step 1: create class folders directly under base.
    root = tmp_path / "dataset"
    _write_image(root / "plastic" / "img.jpg")
    # Step 2: resolve should return base.
    resolved = resolve_class_root({"path": str(root)})
    assert resolved == root


def test_resolve_class_root_auto_detect_nested(tmp_path: Path):
    # Step 1: create nested class folders.
    root = tmp_path / "dataset"
    nested = root / "nested"
    _write_image(nested / "glass" / "img.jpg")
    # Step 2: resolve should return nested folder.
    resolved = resolve_class_root({"path": str(root)})
    assert resolved == nested
