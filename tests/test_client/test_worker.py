"""Test client service worker queue integration."""

import sys
from pathlib import Path
import time
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Set localhost for local testing
if "RABBITMQ_HOST" not in os.environ:
    os.environ["RABBITMQ_HOST"] = "localhost"

import pytest
import pika
import torch
from torch.utils.data import TensorDataset
from unittest.mock import Mock, AsyncMock, patch
from shared.queue.publisher import QueuePublisher
from shared.models.task import Task, TaskType, TrainTaskPayload, TaskMetadata
from client_service.worker import ClientWorker
from client_service.config import config
from client_service.training.model import SimpleCNN
from client_service.training.trainer import Trainer

GET_DATASET_PATCH = "client_service.training.trainer.get_dataset"


def _make_mock_dataset(num_classes=10, in_channels=1, num_samples=64):
    """Mock dataset so tests don't download real data."""
    data = torch.rand(num_samples, in_channels, 28, 28)
    labels = torch.randint(0, num_classes, (num_samples,))
    fake = TensorDataset(data, labels)
    mock = Mock()
    mock.get_num_classes.return_value = num_classes
    mock.get_in_channels.return_value = in_channels
    mock.load_training_data.return_value = fake
    return mock


def test_worker_train_task():
    """Test worker processing a TRAIN task."""
    print("=" * 60)
    print("Testing Client Worker - TRAIN Task Processing")
    print("=" * 60)
    print()

    # Set client ID for testing
    os.environ["CLIENT_ID"] = "0"

    # Create a test task (client_id is optional - universal tasks don't require it)
    test_task = Task(
        task_id="test-train-001",
        task_type=TaskType.TRAIN,
        payload=TrainTaskPayload(
            weights_cid=None,  # Start from scratch
            iteration=1,
        ).model_dump(),
        metadata=TaskMetadata(source="test_script"),
    )

    print(f"Created test task: {test_task.task_id}")
    print(f"Task type: {test_task.task_type}")
    print(f"Iteration: {test_task.payload['iteration']}")
    print()

    print("Note: Testing worker directly (not publishing to queue)")
    print()

    # Mock dataset for entire test (Trainer init and train() both call get_dataset)
    mock_dataset = _make_mock_dataset()
    with patch(GET_DATASET_PATCH, return_value=mock_dataset):
        # Create worker with trainer
        print("Creating client worker...")
        trainer = Trainer(epochs=1)  # Use 1 epoch for faster testing
        worker = ClientWorker(trainer=trainer)

        # Mock the publisher to avoid RabbitMQ connection
        mock_publisher = Mock()
        mock_publisher.publish_dict = Mock(return_value=None)
        worker.publisher = mock_publisher

        # Mock IPFS client to avoid connection errors
        mock_cid = "QmTestWeightDiff123"
        with patch("shared.storage.ipfs_client.IPFSClient") as mock_ipfs_class:
            mock_client = AsyncMock()
            mock_client.add_bytes = AsyncMock(return_value=mock_cid)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_ipfs_class.return_value = mock_client

            # Process task directly (train() calls load_dataset -> get_dataset)
            print("Processing task...")
            success = worker._handle_train_task(test_task)

    # Verify that publish_dict was called (worker tried to publish the update)
    assert (
        mock_publisher.publish_dict.called
    ), "Worker should attempt to publish client update"

    assert success, "Task processing should succeed"
    print("✓ Task processed successfully")
    print()

    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_worker_train_task()
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Test FAILED: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
