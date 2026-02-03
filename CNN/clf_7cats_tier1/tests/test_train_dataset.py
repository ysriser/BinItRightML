from pathlib import Path

from PIL import Image
from torchvision import transforms

from CNN.clf_7cats_tier1.train import CsvDataset


def _write_image(path: Path) -> None:
    # Create a small valid RGB image on disk.
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (32, 32), color=(10, 20, 30))
    img.save(path)


def test_csv_dataset_reads_single_item(tmp_path: Path):
    # Step 1: create a tiny image file.
    img_path = tmp_path / "paper.jpg"
    _write_image(img_path)

    # Step 2: write a CSV pointing to the image.
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(
        "filepath,final_label\n"
        f"{img_path.as_posix()},paper\n",
        encoding="utf-8",
    )

    # Step 3: build dataset and read one item.
    ds = CsvDataset(
        csv_path=csv_path,
        label_to_idx={"paper": 0},
        transform=transforms.ToTensor(),
        verify_images=False,
    )

    # Step 4: verify output.
    assert len(ds) == 1
    image, label = ds[0]
    assert label == 0
    assert image.shape[0] == 3
