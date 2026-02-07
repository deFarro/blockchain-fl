"""Caltech101 dataset implementation for federated learning."""

import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
from typing import Optional
import numpy as np
from shared.datasets.dataset_interface import DatasetInterface

# Default image size for Caltech101 (RGB, variable original size)
DEFAULT_IMAGE_SIZE = 128


class Caltech101Dataset(DatasetInterface):
    """Caltech101 dataset implementation (101 object categories)."""

    def __init__(
        self,
        data_dir: str = "data/caltech101",
        seed: int = 42,
        image_size: int = DEFAULT_IMAGE_SIZE,
        train_ratio: float = 0.8,
    ):
        """
        Initialize Caltech101 dataset.

        Args:
            data_dir: Directory to store/load Caltech101 data (root for torchvision).
            seed: Random seed for reproducibility (used for train/test split).
            image_size: Target size for resize (images are resized to image_size x image_size).
            train_ratio: Fraction of data used for training (rest for test). Default 0.8.
        """
        self.data_dir = data_dir
        self.seed = seed
        self.train_ratio = train_ratio
        self._image_size = image_size
        # Some Caltech101 images are grayscale; convert to RGB so Normalize (3 channels) always works
        self.transform = transforms.Compose(
            [
                transforms.Lambda(
                    lambda img: img.convert("RGB") if img.mode != "RGB" else img
                ),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),  # ImageNet-style
            ]
        )
        self._full_dataset = None
        self._train_indices = None
        self._test_indices = None

    def _load_full_dataset(self) -> datasets.Caltech101:
        """Load full Caltech101 dataset (no built-in train/test split)."""
        if self._full_dataset is None:
            self._full_dataset = datasets.Caltech101(
                root=self.data_dir,
                target_type="category",
                transform=self.transform,
                download=True,
            )
            self._build_train_test_split()
        return self._full_dataset

    def _build_train_test_split(self) -> None:
        """Build reproducible train/test indices (no official split in Caltech101)."""
        if self._full_dataset is None or self._train_indices is not None:
            return
        n = len(self._full_dataset)
        np.random.seed(self.seed)
        indices = np.random.permutation(n)
        n_train = int(n * self.train_ratio)
        self._train_indices = indices[:n_train].tolist()
        self._test_indices = indices[n_train:].tolist()

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
        If not provided, returns full training partition.
        """
        self._load_full_dataset()
        train_dataset = Subset(self._full_dataset, self._train_indices)

        if client_id is None or num_clients is None:
            return train_dataset

        use_seed = seed if seed is not None else self.seed

        if split_type == "iid":
            return self._get_iid_slice(train_dataset, client_id, num_clients, use_seed)
        elif split_type == "non_iid":
            return self._get_non_iid_slice(
                train_dataset, client_id, num_clients, use_seed
            )
        else:
            raise ValueError(f"Unknown split type: {split_type}")

    def _get_iid_slice(
        self,
        train_dataset: Subset,
        client_id: int,
        num_clients: int,
        seed: int,
    ) -> Subset:
        """Get IID slice for a specific client."""
        total_samples = len(train_dataset)
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
        # Subset of Subset: we need indices into the original train indices
        base_indices = train_dataset.indices  # type: ignore[attr-defined]
        mapped = [base_indices[i] for i in client_indices]
        return Subset(self._full_dataset, mapped)

    def _get_non_iid_slice(
        self,
        train_dataset: Subset,
        client_id: int,
        num_clients: int,
        seed: int,
    ) -> Subset:
        """Get non-IID slice (class-based) for a specific client."""
        num_classes = self.get_num_classes()
        np.random.seed(seed)
        torch.manual_seed(seed)
        base_indices = train_dataset.indices  # type: ignore[attr-defined]
        class_indices: dict[int, list[int]] = {i: [] for i in range(num_classes)}
        for pos, idx in enumerate(base_indices):
            _, label = self._full_dataset[idx]
            class_indices[int(label)].append(pos)
        classes_per_client = max(1, num_classes // num_clients)
        start_class = client_id * classes_per_client
        end_class = (
            start_class + classes_per_client
            if client_id < num_clients - 1
            else num_classes
        )
        client_classes = list(range(start_class, end_class))
        client_positions = []
        for c in client_classes:
            client_positions.extend(class_indices[c])
        np.random.shuffle(client_positions)
        mapped = [base_indices[i] for i in client_positions]
        return Subset(self._full_dataset, mapped)

    def load_test_data(self) -> Dataset:
        """Load test partition of Caltech101."""
        self._load_full_dataset()
        return Subset(self._full_dataset, self._test_indices)

    def get_num_classes(self) -> int:
        """Get number of classes (101 object categories)."""
        return 101

    def get_class_names(self) -> list:
        """Get class names (categories from dataset if available)."""
        self._load_full_dataset()
        if hasattr(self._full_dataset, "categories"):
            return list(self._full_dataset.categories)
        return [f"class_{i}" for i in range(self.get_num_classes())]

    def get_in_channels(self) -> int:
        """Get number of input channels (3 for RGB Caltech101)."""
        return 3
