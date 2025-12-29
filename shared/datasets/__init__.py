# Dataset interfaces and implementations

from shared.datasets.dataset_interface import DatasetInterface
from shared.datasets.mnist_dataset import MNISTDataset

__all__ = [
    "DatasetInterface",
    "MNISTDataset"
]

