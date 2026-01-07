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
from unittest.mock import Mock, patch
from shared.queue.publisher import QueuePublisher
from shared.models.task import Task, TaskType, TrainTaskPayload, TaskMetadata
from client_service.worker import ClientWorker
from client_service.config import config
from client_service.training.model import SimpleCNN
from client_service.training.trainer import Trainer


def test_worker_train_task():
    """Test worker processing a TRAIN task."""
    print("=" * 60)
    print("Testing Client Worker - TRAIN Task Processing")
    print("=" * 60)
    print()

    # Set client ID for testing
    os.environ["CLIENT_ID"] = "0"

    # Create a test task
    test_task = Task(
        task_id="test-train-001",
        task_type=TaskType.TRAIN,
        payload=TrainTaskPayload(
            weights_cid=None,  # Start from scratch
            iteration=1,
            client_id="client_0",
        ).model_dump(),
        metadata=TaskMetadata(source="test_script"),
    )

    print(f"Created test task: {test_task.task_id}")
    print(f"Task type: {test_task.task_type}")
    print(f"Client ID: {test_task.payload['client_id']}")
    print()

    # Note: We don't actually need to publish to RabbitMQ for this test
    # since we're calling _handle_train_task directly
    # But if we want to test publishing, we need RabbitMQ
    # For now, skip publishing and test worker directly
    print("Note: Testing worker directly (not publishing to queue)")
    print()

    # Create worker with trainer using 1 epoch for faster testing
    print("Creating client worker...")
    trainer = Trainer(epochs=1)  # Use 1 epoch for faster testing
    worker = ClientWorker(trainer=trainer)

    # Mock the publisher to avoid RabbitMQ connection
    # The worker will try to publish client updates, but we don't need RabbitMQ for this test
    mock_publisher = Mock()
    mock_publisher.publish_dict = Mock(return_value=None)
    worker.publisher = mock_publisher

    # Process task directly (for testing, not using consume loop)
    print("Processing task...")
    success = worker._handle_train_task(test_task)
    
    # Verify that publish_dict was called (worker tried to publish the update)
    assert mock_publisher.publish_dict.called, "Worker should attempt to publish client update"

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
