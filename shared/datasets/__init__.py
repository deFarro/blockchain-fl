# Dataset interfaces and implementations

from typing import Optional
from shared.datasets.dataset_interface import DatasetInterface
from shared.datasets.mnist_dataset import MNISTDataset
from shared.config import settings

__all__ = [
    "DatasetInterface",
    "MNISTDataset",
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
        dataset_name: Name of dataset (e.g., "mnist"). If None, uses settings.
        data_dir: Directory for dataset. If None, uses settings.data_dir.
        seed: Random seed. If None, uses settings.dataset_seed or default 42.
    
    Returns:
        DatasetInterface instance
    
    Raises:
        ValueError: If dataset_name is not supported
    """
    # Get dataset name from settings if not provided
    if dataset_name is None:
        dataset_name = getattr(settings, 'dataset_name', 'mnist')
    
    # Get data directory from settings if not provided
    if data_dir is None:
        data_dir = str(settings.data_dir)
    
    # Get seed from settings if not provided
    if seed is None:
        seed = getattr(settings, 'dataset_seed', 42)
    
    # Create dataset instance based on name
    dataset_name_lower = dataset_name.lower()
    
    if dataset_name_lower == "mnist":
        return MNISTDataset(data_dir=data_dir, seed=seed)
    else:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            f"Supported datasets: mnist"
        )

