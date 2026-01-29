from pathlib import Path

import pytest
import torch


def test_exported_model_inference():
    artifact_dir = Path("ml/artifacts/classification_28cats/latest")
    model_path = artifact_dir / "model.ts"
    labels_path = artifact_dir / "model.labels.json"

    if not model_path.exists() or not labels_path.exists():
        pytest.skip("Exported model not found. Run export.py first.")

    model = torch.jit.load(model_path, map_location="cpu")
    model.eval()

    # Create a dummy sample input
    x = torch.rand(1, 3, 224, 224)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape[0] == 1
