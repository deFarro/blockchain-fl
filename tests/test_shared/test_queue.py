#!/usr/bin/env python3
"""Test script for queue publish/subscribe functionality."""

import sys
import os
import time
import threading
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

import pytest
import pika
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
    stop_flag = threading.Event()

    def message_handler(task: Task):
        """Handle received task."""
        print(f"✓ Received task: {task.task_id} (type: {task.task_type})")
        received_tasks.append(task)
        # Stop after receiving 2 messages
        if len(received_tasks) >= 2:
            stop_flag.set()

    try:
        # Test 1: Publish messages first
        # Check if RabbitMQ is available, skip test if not
        try:
            connection = QueueConnection()
            connection.connect()
            connection.close()
        except (pika.exceptions.AMQPConnectionError, ConnectionRefusedError, OSError) as e:
            pytest.skip(f"RabbitMQ is not available: {e}. Start RabbitMQ with: docker-compose up -d rabbitmq")
        
        with QueuePublisher() as publisher:
            publisher.publish_task(test_task, test_queue)
            print(f"✓ Published task 1 to queue: {test_queue}")

            # Publish second message
            test_task_2 = Task(
                task_id="test-task-002",
                task_type=TaskType.TRAIN,
                payload=TrainTaskPayload(
                    weights_cid=None,
                    iteration=2,
                    client_id="client_1",
                ).model_dump(),
                metadata=TaskMetadata(source="test_script"),
            )
            publisher.publish_task(test_task_2, test_queue)
            print(f"✓ Published task 2 to queue: {test_queue}")

        # Test 2: Consume the messages using threading with timeout
        print("\nConsuming messages (with timeout)...")

        def consume_with_timeout():
            """Consume messages with a timeout."""
            consumer = None
            try:
                consumer = QueueConsumer()
                consumer.consume_tasks(test_queue, message_handler)
            except (KeyboardInterrupt, SystemExit):
                # Expected when stopping
                pass
            except Exception as e:
                # Connection errors are expected when stopping/closing
                error_str = str(e).lower()
                if not any(
                    keyword in error_str
                    for keyword in [
                        "closed",
                        "bad file descriptor",
                        "channel",
                        "closing",
                        "connection lost",
                        "stream connection",
                    ]
                ):
                    # Only log unexpected errors
                    logger.warning(f"Consumer error (may be expected): {e}")
            finally:
                # Try to stop gracefully, but ignore all errors
                if consumer:
                    try:
                        if hasattr(consumer, "_consuming") and consumer._consuming:
                            consumer.stop()
                    except Exception:
                        pass  # Ignore all errors when stopping
                    try:
                        consumer.close()
                    except Exception:
                        pass  # Ignore all errors when closing

        # Start consumer in a daemon thread
        consumer_thread = threading.Thread(target=consume_with_timeout, daemon=True)
        consumer_thread.start()

        # Wait for messages (up to 5 seconds)
        time.sleep(5)

        # The thread will be cleaned up automatically as a daemon thread
        # No need to explicitly stop it - just verify we received messages

        # Verify
        assert (
            len(received_tasks) >= 1
        ), f"Expected at least 1 task, got {len(received_tasks)}"
        print(f"\n✓ Success! Received {len(received_tasks)} task(s)")
        for task in received_tasks:
            print(f"  - Task ID: {task.task_id}, Type: {task.task_type}")

    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback

        traceback.print_exc()
        raise


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
