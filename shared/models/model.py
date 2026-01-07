"""PyTorch model definition for MNIST classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import json


class SimpleCNN(nn.Module):
    """
    Simple Convolutional Neural Network for MNIST classification.

    Architecture:
    - Conv2d(1, 32, kernel_size=3) -> ReLU -> MaxPool2d(2)
    - Conv2d(32, 64, kernel_size=3) -> ReLU -> MaxPool2d(2)
    - Flatten
    - Linear(9216, 128) -> ReLU -> Dropout(0.5)
    - Linear(128, 10) -> LogSoftmax
    """

    def __init__(self, num_classes: int = 10):
        """
        Initialize the CNN model.

        Args:
            num_classes: Number of output classes (default: 10 for MNIST)
        """
        super(SimpleCNN, self).__init__()
        self.num_classes = num_classes

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        # After two conv+pool layers: 28x28 -> 14x14 -> 7x7
        # 7 * 7 * 64 = 3136, but we'll use adaptive pooling to be safe
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # First conv block
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 -> 14x14

        # Second conv block
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 -> 7x7

        # Flatten
        x = x.view(-1, 64 * 7 * 7)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

    def get_weights(self) -> Dict[str, Any]:
        """
        Get model weights as a dictionary.

        Returns:
            Dictionary mapping parameter names to numpy arrays
        """
        weights = {}
        for name, param in self.named_parameters():
            weights[name] = param.data.cpu().clone().detach()
        return weights

    def set_weights(self, weights: Dict[str, Any]):
        """
        Set model weights from a dictionary.

        Args:
            weights: Dictionary mapping parameter names to tensors/arrays
        """
        for name, param in self.named_parameters():
            if name in weights:
                weight_data = weights[name]
                # Convert to tensor if needed
                if not isinstance(weight_data, torch.Tensor):
                    weight_data = torch.tensor(weight_data)
                param.data = weight_data.to(param.device)
            else:
                raise ValueError(f"Weight {name} not found in provided weights")

    def compute_weight_diff(
        self, previous_weights: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Compute the difference between current weights and previous weights.

        Args:
            previous_weights: Previous model weights (None for initial weights)

        Returns:
            Dictionary of weight differences (current - previous)
        """
        current_weights = self.get_weights()

        if previous_weights is None:
            # If no previous weights, return current weights as the "diff"
            return current_weights

        # Compute difference
        diff = {}
        for name in current_weights:
            if name in previous_weights:
                current_tensor = current_weights[name]
                prev_tensor = previous_weights[name]

                # Ensure both are tensors on CPU with same dtype
                if not isinstance(prev_tensor, torch.Tensor):
                    prev_tensor = torch.tensor(prev_tensor)
                else:
                    prev_tensor = prev_tensor.cpu().clone().detach()

                # Ensure same dtype
                if prev_tensor.dtype != current_tensor.dtype:
                    prev_tensor = prev_tensor.to(dtype=current_tensor.dtype)

                # Compute diff
                diff[name] = current_tensor - prev_tensor
            else:
                # New parameter, use current weight as diff
                diff[name] = current_weights[name].clone()

        return diff

    def apply_weight_diff(self, diff: Dict[str, Any]):
        """
        Apply weight differences to current model weights.

        Args:
            diff: Dictionary of weight differences to apply
        """
        current_weights = self.get_weights()

        # Apply diff to each weight
        updated_weights = {}
        for name in current_weights:
            if name in diff:
                # Ensure both tensors are on CPU and have same dtype
                current_tensor = current_weights[name]
                diff_tensor = diff[name]

                # Convert diff to tensor if needed and ensure it's on CPU
                if not isinstance(diff_tensor, torch.Tensor):
                    diff_tensor = torch.tensor(diff_tensor)
                else:
                    # Clone to avoid modifying the original
                    diff_tensor = diff_tensor.cpu().clone().detach()

                # Ensure same dtype
                if diff_tensor.dtype != current_tensor.dtype:
                    diff_tensor = diff_tensor.to(dtype=current_tensor.dtype)

                # Apply diff
                updated_weights[name] = current_tensor + diff_tensor
            else:
                # Parameter not in diff, keep current
                updated_weights[name] = current_weights[name]

        # Handle new parameters in diff that aren't in current weights
        for name in diff:
            if name not in current_weights:
                # New parameter - ensure it's a tensor
                if not isinstance(diff[name], torch.Tensor):
                    updated_weights[name] = torch.tensor(diff[name])
                else:
                    updated_weights[name] = diff[name].cpu().clone().detach()

        self.set_weights(updated_weights)

    def weights_to_bytes(self) -> bytes:
        """
        Serialize model weights to bytes (for transmission/storage).

        Returns:
            Serialized weights as bytes
        """
        weights = self.get_weights()
        # Convert tensors to lists for JSON serialization
        serializable_weights = {}
        for name, tensor in weights.items():
            serializable_weights[name] = tensor.numpy().tolist()

        json_str = json.dumps(serializable_weights)
        return json_str.encode("utf-8")

    @classmethod
    def weights_from_bytes(cls, data: bytes) -> Dict[str, Any]:
        """
        Deserialize model weights from bytes.

        Args:
            data: Serialized weights as bytes

        Returns:
            Dictionary of weights
        """
        json_str = data.decode("utf-8")
        serializable_weights = json.loads(json_str)

        # Convert lists back to tensors
        weights = {}
        for name, tensor_list in serializable_weights.items():
            weights[name] = torch.tensor(tensor_list)

        return weights


def create_model(num_classes: int = 10) -> SimpleCNN:
    """
    Create a new model instance.

    Args:
        num_classes: Number of output classes

    Returns:
        Initialized model
    """
    return SimpleCNN(num_classes=num_classes)

