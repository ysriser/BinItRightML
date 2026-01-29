from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from .splits import create_dataloaders


class DatasetAdapter(ABC):
    @abstractmethod
    def get_dataloaders(self, device: torch.device) -> Tuple[Dict[str, DataLoader], torch.Tensor, List[str]]:
        """Return dataloaders, class weights, and class names."""


class ImageFolderAdapter(DatasetAdapter):
    def __init__(
        self,
        data_dir: Path,
        backbone: str,
        image_size: int,
        batch_size: int,
        num_workers: int,
        persistent_workers: bool,
        prefetch_factor: int,
        use_weighted_sampler: bool,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.backbone = backbone
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.use_weighted_sampler = use_weighted_sampler

    def get_dataloaders(self, device: torch.device):
        return create_dataloaders(
            data_dir=self.data_dir,
            backbone=self.backbone,
            image_size=self.image_size,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            use_weighted_sampler=self.use_weighted_sampler,
            device=device,
        )


class CsvAdapter(DatasetAdapter):
    """Placeholder for Phase 3: CSV-labeled datasets."""

    def get_dataloaders(self, device: torch.device):
        raise NotImplementedError("CSV adapter not implemented yet.")
