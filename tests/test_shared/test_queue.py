#!/usr/bin/env python3
"""Test script for queue publish/subscribe functionality."""

import sys
import os
import time
from pathlib import Path

# Add project root to path
project_root = Path(
    __file__
).parent.parent.parent  # Go up from test_shared/ -> tests/ -> project root
sys.path.insert(0, str(project_root))

# Set localhost for local testing (not in Docker)
# This overrides the default "rabbitmq" hostname used in Docker
if "RABBITMQ_HOST" not in os.environ:
    os.environ["RABBITMQ_HOST"] = "localhost"

from shared.queue.connection import QueueConnection
from shared.queue.publisher import QueuePublisher
from shared.queue.consumer import QueueConsumer
from shared.models.task import Task, TaskType, TrainTaskPayload, TaskMetadata
from shared.logger import setup_logger

logger = setup_logger(__name__)


def test_publish_subscribe():
    """Test basic publish/subscribe functionality."""
    test_queue = "test_queue"

    # Create a test task
    test_task = Task(
        task_id="test-task-001",
        task_type=TaskType.TRAIN,
        payload=TrainTaskPayload(
            weights_cid=None,
            iteration=1,
            client_id="client_0",
        ).model_dump(),
        metadata=TaskMetadata(source="test_script"),
    )

    print(f"Testing queue publish/subscribe on queue: {test_queue}")
    print(f"Publishing task: {test_task.task_id}")

    # Test publisher
    received_tasks = []

    def message_handler(task: Task):
        """Handle received task."""
        print(f"✓ Received task: {task.task_id} (type: {task.task_type})")
        received_tasks.append(task)

    # Start consumer in a separate thread (simplified - in real usage, use threading)
    print("\nStarting consumer...")
    print("(Note: This is a blocking test. Press Ctrl+C to stop)")

    try:
        # Test 1: Publish a message
        with QueuePublisher() as publisher:
            publisher.publish_task(test_task, test_queue)
            print(f"✓ Published task to queue: {test_queue}")

        # Test 2: Consume the message
        print("\nConsuming messages (will wait for 5 seconds)...")
        with QueueConsumer() as consumer:
            # Start consuming in a non-blocking way for testing
            import threading

            def consume():
                try:
                    consumer.consume_tasks(test_queue, message_handler)
                except KeyboardInterrupt:
                    pass

            consumer_thread = threading.Thread(target=consume, daemon=True)
            consumer_thread.start()

            # Wait a bit for message to be consumed
            time.sleep(2)

            # Publish another message while consumer is running
            with QueuePublisher() as publisher:
                publisher.publish_task(test_task, test_queue)
                print(f"✓ Published another task while consumer is running")

            # Wait for consumption
            time.sleep(3)

            consumer.stop()

        # Verify
        if received_tasks:
            print(f"\n✓ Success! Received {len(received_tasks)} task(s)")
            for task in received_tasks:
                print(f"  - Task ID: {task.task_id}, Type: {task.task_type}")
            return True
        else:
            print("\n✗ Failed! No tasks received")
            return False

    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Queue Infrastructure Test")
    print("=" * 60)
    print("\nMake sure RabbitMQ is running:")
    print("  docker-compose up -d rabbitmq")
    print("\nNote: This script uses 'localhost' for local testing.")
    print("      If running in Docker, set RABBITMQ_HOST=rabbitmq")
    print()

    success = test_publish_subscribe()

    print("\n" + "=" * 60)
    if success:
        print("✓ Test PASSED")
        sys.exit(0)
    else:
        print("✗ Test FAILED")
        sys.exit(1)
