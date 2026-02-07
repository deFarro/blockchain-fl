# Dataset interfaces and implementations

from pathlib import Path
from typing import Optional
from shared.datasets.dataset_interface import DatasetInterface
from shared.datasets.mnist_dataset import MNISTDataset
from shared.datasets.caltech101_dataset import Caltech101Dataset
from shared.config import settings

__all__ = [
    "DatasetInterface",
    "MNISTDataset",
    "Caltech101Dataset",
    "get_dataset",
]


def get_dataset(
    dataset_name: Optional[str] = None,
    data_dir: Optional[str] = None,
    seed: Optional[int] = None,
) -> DatasetInterface:
    """
    Factory function to get dataset instance based on name.

    Args:
        dataset_name: Name of dataset (e.g., "mnist", "caltech101"). If None, uses settings.
        data_dir: Directory for dataset. If None, uses settings.data_dir / dataset_name.
        seed: Random seed. If None, uses settings.dataset_seed or default 42.

    Returns:
        DatasetInterface instance

    Raises:
        ValueError: If dataset_name is not supported
    """
    if dataset_name is None:
        dataset_name = getattr(settings, "dataset_name", "mnist")

    if data_dir is None:
        base = getattr(settings, "data_dir", None)
        if base is not None:
            data_dir = str(Path(base) / dataset_name)
        else:
            data_dir = f"data/{dataset_name}"

    if seed is None:
        seed = getattr(settings, "dataset_seed", 42)

    dataset_name_lower = dataset_name.lower()

    if dataset_name_lower == "mnist":
        return MNISTDataset(data_dir=data_dir, seed=seed)
    if dataset_name_lower == "caltech101":
        return Caltech101Dataset(data_dir=data_dir, seed=seed)
    raise ValueError(
        f"Unsupported dataset: {dataset_name}. "
        f"Supported datasets: mnist, caltech101"
    )

