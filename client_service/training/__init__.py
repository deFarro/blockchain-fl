"""Client training module."""

from client_service.training.model import SimpleCNN, create_model
from client_service.training.trainer import Trainer

__all__ = ["SimpleCNN", "create_model", "Trainer"]
