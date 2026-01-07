"""Training logic for client service."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Dict, Any, Tuple, cast
from client_service.training.model import SimpleCNN
from client_service.config import config
from shared.datasets import get_dataset
from shared.logger import setup_logger

logger = setup_logger(__name__)


class Trainer:
    """Handles model training for a client."""

    def __init__(
        self,
        model: Optional[SimpleCNN] = None,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        epochs: int = 10,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model (creates new one if None)
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            epochs: Number of training epochs per iteration
            device: PyTorch device (auto-detects if None)
        """
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Initialize model
        if model is None:
            self.model = SimpleCNN(num_classes=10).to(self.device)
        else:
            self.model = model.to(self.device)

        # Optimizer and loss
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        self.criterion = nn.NLLLoss()

        logger.info(
            f"Trainer initialized: device={self.device}, lr={learning_rate}, "
            f"batch_size={batch_size}, epochs={epochs}"
        )

    def load_dataset(
        self, client_id: Optional[int] = None
    ) -> Tuple[DataLoader, Dataset]:
        """
        Load training dataset for this client.

        Args:
            client_id: Client ID (uses config if None)

        Returns:
            Tuple of (DataLoader, Dataset)
        """
        if client_id is None:
            client_id = config.get_client_id()

        # Load dataset using factory function
        dataset_loader = get_dataset(
            dataset_name=getattr(config, 'dataset_name', None),
            data_dir=str(config.data_dir),
            seed=config.dataset_seed,
        )
        train_dataset = dataset_loader.load_training_data(
            client_id=client_id,
            num_clients=config.num_clients,
            split_type=config.split_type,
            seed=config.dataset_seed,
        )

        # Create DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 for compatibility
        )

        # PyTorch Dataset implements __len__, but mypy doesn't know this
        # Cast to help mypy understand that Dataset has __len__
        dataset_len = len(cast(Any, train_dataset))
        logger.info(f"Loaded dataset for client {client_id}: {dataset_len} samples")

        return train_loader, train_dataset

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: DataLoader for training data

        Returns:
            Dictionary with training metrics (loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

        # Calculate metrics
        avg_loss = running_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        return {"loss": avg_loss, "accuracy": accuracy, "samples": total}

    def train(
        self,
        train_loader: Optional[DataLoader] = None,
        client_id: Optional[int] = None,
        previous_weights: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, Any]]:
        """
        Train the model.

        Args:
            train_loader: DataLoader for training (loads dataset if None)
            client_id: Client ID (uses config if None)
            previous_weights: Previous model weights for diff computation

        Returns:
            Tuple of (weight_diff, training_metrics, initial_weights)
            - weight_diff: Dictionary of weight differences (current - initial)
            - training_metrics: Dictionary with training metrics (loss, accuracy, samples, epochs)
            - initial_weights: Dictionary of weights actually used at the start of training
        """
        # Load dataset if not provided
        if train_loader is None:
            train_loader, _ = self.load_dataset(client_id=client_id)

        # Store initial weights if computing diff
        if previous_weights is not None:
            self.model.set_weights(previous_weights)

        initial_weights = self.model.get_weights()

        # Train for specified epochs
        logger.info(f"Starting training for {self.epochs} epoch(s)...")
        all_metrics = []

        for epoch in range(self.epochs):
            metrics = self.train_epoch(train_loader)
            all_metrics.append(metrics)
            logger.info(
                f"Epoch {epoch + 1}/{self.epochs}: "
                f"loss={metrics['loss']:.4f}, accuracy={metrics['accuracy']:.2f}%"
            )

        # Compute final metrics (average across epochs)
        final_metrics = {
            "loss": sum(m["loss"] for m in all_metrics) / len(all_metrics),
            "accuracy": sum(m["accuracy"] for m in all_metrics) / len(all_metrics),
            "samples": all_metrics[0]["samples"],
            "epochs": self.epochs,
        }

        # Compute weight diff
        final_weights = self.model.get_weights()
        weight_diff = self.model.compute_weight_diff(initial_weights)

        logger.info(
            f"Training completed: final_loss={final_metrics['loss']:.4f}, "
            f"final_accuracy={final_metrics['accuracy']:.2f}%"
        )

        return weight_diff, final_metrics, initial_weights

    def get_model(self) -> SimpleCNN:
        """
        Get the current model.

        Returns:
            The model instance
        """
        return self.model

    def set_model_weights(self, weights: Dict[str, Any]):
        """
        Set model weights.

        Args:
            weights: Dictionary of weights
        """
        self.model.set_weights(weights)
        logger.debug("Model weights updated")
