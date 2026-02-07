"""USPS dataset implementation for federated learning."""

import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
from typing import Optional
import numpy as np
from shared.datasets.dataset_interface import DatasetInterface


class USPSDataset(DatasetInterface):
    """USPS dataset implementation (handwritten digits, 16x16 grayscale)."""

    def __init__(self, data_dir: str = "data/usps", seed: int = 42):
        """
        Initialize USPS dataset.

        Args:
            data_dir: Directory to store/load USPS data
            seed: Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.seed = seed
        # USPS images are 16x16, pixel values [0, 255] in torchvision; normalize similarly to MNIST
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        self._full_training_data = None
        self._test_data = None

    def load_training_data(
        self,
        client_id: Optional[int] = None,
        num_clients: Optional[int] = None,
        split_type: str = "iid",
        seed: Optional[int] = None,
    ) -> Dataset:
        """
        Load training dataset for a specific client.

        If client_id and num_clients are provided, returns only that client's slice.
        If not provided, returns full training dataset.
        """
        if self._full_training_data is None:
            self._full_training_data = datasets.USPS(
                root=self.data_dir,
                train=True,
                download=True,
                transform=self.transform,
            )

        if client_id is None or num_clients is None:
            if self._full_training_data is None:
                raise RuntimeError("Training data not loaded")
            return self._full_training_data

        if seed is None:
            seed = self.seed

        if split_type == "iid":
            return self._get_iid_slice(client_id, num_clients, seed)
        elif split_type == "non_iid":
            return self._get_non_iid_slice(client_id, num_clients, seed)
        else:
            raise ValueError(f"Unknown split type: {split_type}")

    def _get_iid_slice(self, client_id: int, num_clients: int, seed: int) -> Subset:
        """Get IID slice for a specific client."""
        if self._full_training_data is None:
            raise RuntimeError(
                "Training data not loaded. Call load_training_data() first."
            )
        dataset = self._full_training_data
        total_samples = len(dataset)
        samples_per_client = total_samples // num_clients

        np.random.seed(seed)
        torch.manual_seed(seed)
        indices = np.random.permutation(total_samples)

        start_idx = client_id * samples_per_client
        end_idx = (
            start_idx + samples_per_client
            if client_id < num_clients - 1
            else total_samples
        )
        client_indices = indices[start_idx:end_idx]
        return Subset(dataset, client_indices)

    def _get_non_iid_slice(
        self, client_id: int, num_clients: int, seed: int
    ) -> Subset:
        """Get non-IID slice (class-based) for a specific client."""
        if self._full_training_data is None:
            raise RuntimeError(
                "Training data not loaded. Call load_training_data() first."
            )
        dataset = self._full_training_data
        num_classes = self.get_num_classes()

        np.random.seed(seed)
        torch.manual_seed(seed)

        class_indices: dict[int, list[int]] = {i: [] for i in range(num_classes)}
        for idx, (_, label) in enumerate(dataset):
            class_indices[int(label)].append(idx)

        classes_per_client = num_classes // num_clients
        start_class = client_id * classes_per_client
        end_class = (
            start_class + classes_per_client
            if client_id < num_clients - 1
            else num_classes
        )
        client_classes = list(range(start_class, end_class))

        client_indices = []
        for class_id in client_classes:
            client_indices.extend(class_indices[class_id])
        np.random.shuffle(client_indices)
        return Subset(dataset, client_indices)

    def load_test_data(self) -> Dataset:
        """Load full USPS test dataset."""
        if self._test_data is None:
            self._test_data = datasets.USPS(
                root=self.data_dir,
                train=False,
                download=True,
                transform=self.transform,
            )
        test_data = self._test_data
        if test_data is None:
            raise RuntimeError("Failed to load test data")
        return test_data

    def get_num_classes(self) -> int:
        """Get number of classes in USPS (10 digits)."""
        return 10

    def get_class_names(self) -> list:
        """Get class names (digits 0-9)."""
        return [str(i) for i in range(10)]

    def get_in_channels(self) -> int:
        """Get number of input channels (1 for grayscale USPS)."""
        return 1
