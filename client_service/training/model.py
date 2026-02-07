"""Re-export shared model for client service (MNIST, Caltech101, etc.)."""

from shared.models.model import SimpleCNN, create_model

__all__ = ["SimpleCNN", "create_model"]
