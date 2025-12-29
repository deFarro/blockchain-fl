"""Dataset interface for federated learning."""

from abc import ABC, abstractmethod
from typing import Optional
import torch
from torch.utils.data import Dataset, Subset


class DatasetInterface(ABC):
    """Abstract base class for datasets used in federated learning."""

    @abstractmethod
    def load_training_data(
        self,
        client_id: Optional[int] = None,
        num_clients: Optional[int] = None,
        split_type: str = "iid",
        seed: int = 42,
    ) -> Dataset:
        """
        Load training dataset for a specific client.

        If client_id and num_clients are provided, returns only that client's slice.
        If not provided, returns full training dataset.

        Args:
            client_id: Client identifier (0, 1, 2, ...). If None, returns full dataset.
            num_clients: Total number of clients. Required if client_id is provided.
            split_type: Type of split ("iid" or "non_iid"). Default: "iid"
            seed: Random seed for reproducibility. Default: 42

        Returns:
            PyTorch Dataset containing client's training samples (or full dataset if client_id is None)
        """
        pass

    @abstractmethod
    def load_test_data(self) -> Dataset:
        """
        Load the full test dataset.

        Returns:
            PyTorch Dataset containing all test samples
        """
        pass

    @abstractmethod
    def get_num_classes(self) -> int:
        """
        Get the number of classes in the dataset.

        Returns:
            Number of classes
        """
        pass

    @abstractmethod
    def get_class_names(self) -> list:
        """
        Get the names of classes in the dataset.

        Returns:
            List of class names
        """
        pass
