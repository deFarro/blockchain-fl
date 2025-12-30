"""Generic message publisher for RabbitMQ."""

import json
from typing import Any, Dict, Optional
import pika
from pika.channel import Channel
from shared.queue.connection import QueueConnection
from shared.models.task import Task
from shared.logger import setup_logger

logger = setup_logger(__name__)


class QueuePublisher:
    """Publishes messages to RabbitMQ queues."""

    def __init__(self, connection: Optional[QueueConnection] = None):
        """
        Initialize publisher.

        Args:
            connection: QueueConnection instance (creates new one if None)
        """
        self.connection = connection or QueueConnection()
        self._own_connection = connection is None

    def _ensure_connected(self) -> Channel:
        """Ensure connection is active."""
        return self.connection.ensure_connected()

    def declare_queue(
        self,
        queue_name: str,
        durable: bool = True,
        exclusive: bool = False,
        auto_delete: bool = False,
    ) -> None:
        """
        Declare a queue.

        Args:
            queue_name: Name of the queue
            durable: If True, queue survives broker restart
            exclusive: If True, queue is only accessible by this connection
            auto_delete: If True, queue is deleted when connection closes
        """
        channel = self._ensure_connected()
        channel.queue_declare(
            queue=queue_name,
            durable=durable,
            exclusive=exclusive,
            auto_delete=auto_delete,
        )
        logger.debug(f"Declared queue: {queue_name}")

    def publish_task(
        self,
        task: Task,
        queue_name: str,
        routing_key: Optional[str] = None,
        exchange: str = "",
        durable: bool = True,
    ) -> None:
        """
        Publish a Task to a queue.

        Args:
            task: Task object to publish
            queue_name: Name of the queue
            routing_key: Routing key (defaults to queue_name)
            exchange: Exchange name (default: "" for default exchange)
            durable: If True, queue is durable

        Raises:
            pika.exceptions.AMQPConnectionError: If connection fails
        """
        channel = self._ensure_connected()

        # Declare queue if not already declared
        self.declare_queue(queue_name, durable=durable)

        # Convert task to dictionary
        message_body = task.to_dict()

        # Use routing_key if provided, otherwise use queue_name
        routing = routing_key or queue_name

        # Publish message
        channel.basic_publish(
            exchange=exchange,
            routing_key=routing,
            body=json.dumps(message_body),
            properties=pika.BasicProperties(
                delivery_mode=2,  # Make message persistent
                content_type="application/json",
            ),
        )

        logger.info(
            f"Published task {task.task_id} (type: {task.task_type}) to queue {queue_name}"
        )

    def publish_dict(
        self,
        message: Dict[str, Any],
        queue_name: str,
        routing_key: Optional[str] = None,
        exchange: str = "",
        durable: bool = True,
    ) -> None:
        """
        Publish a dictionary message to a queue.

        Args:
            message: Dictionary to publish
            queue_name: Name of the queue
            routing_key: Routing key (defaults to queue_name)
            exchange: Exchange name (default: "" for default exchange)
            durable: If True, queue is durable

        Raises:
            pika.exceptions.AMQPConnectionError: If connection fails
        """
        channel = self._ensure_connected()

        # Declare queue if not already declared
        self.declare_queue(queue_name, durable=durable)

        # Use routing_key if provided, otherwise use queue_name
        routing = routing_key or queue_name

        # Publish message
        channel.basic_publish(
            exchange=exchange,
            routing_key=routing,
            body=json.dumps(message),
            properties=pika.BasicProperties(
                delivery_mode=2,  # Make message persistent
                content_type="application/json",
            ),
        )

        logger.debug(f"Published message to queue {queue_name}")

    def close(self):
        """Close connection if we own it."""
        if self._own_connection:
            self.connection.close()

    def __enter__(self):
        """Context manager entry."""
        self.connection.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

