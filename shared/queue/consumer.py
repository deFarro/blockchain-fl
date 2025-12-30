"""Generic message consumer for RabbitMQ."""

import json
from typing import Callable, Optional, Dict, Any
from functools import wraps
import pika
from pika.channel import Channel
from shared.queue.connection import QueueConnection
from shared.models.task import Task
from shared.logger import setup_logger

logger = setup_logger(__name__)


def handle_connection_error(func: Callable) -> Callable:
    """Decorator to handle connection errors and retry."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except (pika.exceptions.AMQPConnectionError, pika.exceptions.StreamLostError) as e:
            logger.error(f"Connection error in {func.__name__}: {str(e)}")
            if self.auto_reconnect:
                logger.info("Attempting to reconnect...")
                self.connection.reconnect()
                return func(self, *args, **kwargs)
            raise

    return wrapper


class QueueConsumer:
    """Consumes messages from RabbitMQ queues."""

    def __init__(
        self,
        connection: Optional[QueueConnection] = None,
        auto_reconnect: bool = True,
    ):
        """
        Initialize consumer.

        Args:
            connection: QueueConnection instance (creates new one if None)
            auto_reconnect: If True, automatically reconnect on connection errors
        """
        self.connection = connection or QueueConnection()
        self._own_connection = connection is None
        self.auto_reconnect = auto_reconnect
        self._consuming = False

    def _ensure_connected(self) -> Channel:
        """Ensure connection is active."""
        return self.connection.ensure_connected()

    @handle_connection_error
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

    def _on_message(
        self,
        channel: Channel,
        method: pika.spec.Basic.Deliver,
        properties: pika.spec.BasicProperties,
        body: bytes,
        callback: Callable[[Task], None],
    ):
        """
        Internal message handler.

        Args:
            channel: RabbitMQ channel
            method: Delivery method
            properties: Message properties
            body: Message body
            callback: Callback function to process task
        """
        try:
            # Parse message
            message_dict = json.loads(body)
            task = Task.from_dict(message_dict)

            logger.debug(
                f"Received task {task.task_id} (type: {task.task_type}) from queue {method.routing_key}"
            )

            # Call user callback
            callback(task)

            # Acknowledge message
            channel.basic_ack(delivery_tag=method.delivery_tag)

            logger.debug(f"Acknowledged task {task.task_id}")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message: {str(e)}")
            # Reject message and don't requeue (malformed message)
            channel.basic_nack(
                delivery_tag=method.delivery_tag, requeue=False
            )

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            # Reject message and requeue (temporary error)
            channel.basic_nack(
                delivery_tag=method.delivery_tag, requeue=True
            )

    @handle_connection_error
    def consume_tasks(
        self,
        queue_name: str,
        callback: Callable[[Task], None],
        durable: bool = True,
        prefetch_count: int = 1,
    ) -> None:
        """
        Consume tasks from a queue.

        Args:
            queue_name: Name of the queue to consume from
            callback: Function to call for each task (Task) -> None
            durable: If True, queue is durable
            prefetch_count: Number of unacknowledged messages to prefetch

        Raises:
            pika.exceptions.AMQPConnectionError: If connection fails
        """
        channel = self._ensure_connected()

        # Declare queue
        self.declare_queue(queue_name, durable=durable)

        # Set QoS to limit unacknowledged messages
        channel.basic_qos(prefetch_count=prefetch_count)

        # Set up consumer
        channel.basic_consume(
            queue=queue_name,
            on_message_callback=lambda ch, method, props, body: self._on_message(
                ch, method, props, body, callback
            ),
        )

        self._consuming = True
        logger.info(f"Started consuming from queue: {queue_name}")

        try:
            # Start consuming (blocking)
            channel.start_consuming()
        except KeyboardInterrupt:
            logger.info("Stopping consumer...")
            channel.stop_consuming()
            self._consuming = False

    @handle_connection_error
    def consume_dict(
        self,
        queue_name: str,
        callback: Callable[[Dict[str, Any]], None],
        durable: bool = True,
        prefetch_count: int = 1,
    ) -> None:
        """
        Consume raw dictionary messages from a queue.

        Args:
            queue_name: Name of the queue to consume from
            callback: Function to call for each message (Dict) -> None
            durable: If True, queue is durable
            prefetch_count: Number of unacknowledged messages to prefetch

        Raises:
            pika.exceptions.AMQPConnectionError: If connection fails
        """
        channel = self._ensure_connected()

        # Declare queue
        self.declare_queue(queue_name, durable=durable)

        # Set QoS to limit unacknowledged messages
        channel.basic_qos(prefetch_count=prefetch_count)

        def on_message(
            ch: Channel,
            method: pika.spec.Basic.Deliver,
            properties: pika.spec.BasicProperties,
            body: bytes,
        ):
            try:
                message_dict = json.loads(body)
                logger.debug(
                    f"Received message from queue {method.routing_key}"
                )
                callback(message_dict)
                ch.basic_ack(delivery_tag=method.delivery_tag)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse message: {str(e)}")
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

        # Set up consumer
        channel.basic_consume(
            queue=queue_name,
            on_message_callback=on_message,
        )

        self._consuming = True
        logger.info(f"Started consuming from queue: {queue_name}")

        try:
            # Start consuming (blocking)
            channel.start_consuming()
        except KeyboardInterrupt:
            logger.info("Stopping consumer...")
            channel.stop_consuming()
            self._consuming = False

    def stop(self):
        """Stop consuming messages."""
        if self._consuming:
            channel = self._ensure_connected()
            channel.stop_consuming()
            self._consuming = False
            logger.info("Stopped consuming")

    def close(self):
        """Close connection if we own it."""
        self.stop()
        if self._own_connection:
            self.connection.close()

    def __enter__(self):
        """Context manager entry."""
        self.connection.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

