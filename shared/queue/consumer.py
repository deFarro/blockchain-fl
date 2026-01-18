"""Generic message consumer for RabbitMQ."""

import json
from typing import Callable, Optional, Dict, Any, Union
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
        except (
            pika.exceptions.AMQPConnectionError,
            pika.exceptions.StreamLostError,
        ) as e:
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
        self._active_channel: Optional[Channel] = None  # Store channel for stopping

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

            # Acknowledge message (only if channel is still open)
            if channel.is_open:
                try:
                    channel.basic_ack(delivery_tag=method.delivery_tag)
                    logger.debug(f"Acknowledged task {task.task_id}")
                except pika.exceptions.ChannelClosedByBroker as e:
                    logger.warning(
                        f"Channel closed by broker while acknowledging task {task.task_id}: {str(e)}"
                    )
                except pika.exceptions.ChannelWrongStateError as e:
                    logger.warning(
                        f"Channel in wrong state while acknowledging task {task.task_id}: {str(e)}"
                    )
            else:
                logger.warning(
                    f"Channel closed, cannot acknowledge task {task.task_id} (delivery_tag: {method.delivery_tag})"
                )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message: {str(e)}")
            # Reject message and don't requeue (malformed message)
            if channel.is_open:
                try:
                    channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
                except (
                    pika.exceptions.ChannelClosedByBroker,
                    pika.exceptions.ChannelWrongStateError,
                ) as nack_error:
                    logger.debug(
                        f"Channel closed/wrong state while nacking malformed message: {nack_error}"
                    )

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            # Reject message and requeue (temporary error)
            if channel.is_open:
                try:
                    channel.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
                except (
                    pika.exceptions.ChannelClosedByBroker,
                    pika.exceptions.ChannelWrongStateError,
                ) as nack_error:
                    logger.debug(
                        f"Channel closed/wrong state while nacking failed message: {nack_error}"
                    )

    @handle_connection_error
    def consume_tasks(
        self,
        queue_name: str,
        callback: Callable[[Task], None],
        durable: bool = True,
        prefetch_count: int = 1,
        use_fanout: bool = False,
        consumer_id: Optional[str] = None,
    ) -> None:
        """
        Consume tasks from a queue.

        Args:
            queue_name: Name of the queue to consume from
            callback: Function to call for each task (Task) -> None
            durable: If True, queue is durable
            prefetch_count: Number of unacknowledged messages to prefetch
            use_fanout: If True, consume from fanout exchange (each consumer gets own queue)
            consumer_id: Unique ID for this consumer (required if use_fanout=True)

        Raises:
            pika.exceptions.AMQPConnectionError: If connection fails
        """
        channel = self._ensure_connected()

        if use_fanout:
            # Use fanout exchange - each consumer gets its own queue
            if not consumer_id:
                raise ValueError("consumer_id is required when use_fanout=True")

            fanout_exchange = f"{queue_name}_fanout"
            # Declare fanout exchange
            channel.exchange_declare(
                exchange=fanout_exchange,
                exchange_type="fanout",
                durable=durable,
            )

            # Create unique queue for this consumer
            consumer_queue = f"{queue_name}_{consumer_id}"
            self.declare_queue(
                consumer_queue, durable=durable, exclusive=False, auto_delete=True
            )

            # Bind queue to fanout exchange
            channel.queue_bind(exchange=fanout_exchange, queue=consumer_queue)

            logger.info(
                f"Consuming from fanout exchange {fanout_exchange} via queue {consumer_queue}"
            )
            actual_queue = consumer_queue
        else:
            # Use direct queue
            self.declare_queue(queue_name, durable=durable)
            actual_queue = queue_name
            logger.info(f"Started consuming from queue: {queue_name}")

        # Set QoS to limit unacknowledged messages
        channel.basic_qos(prefetch_count=prefetch_count)

        # Set up consumer
        channel.basic_consume(
            queue=actual_queue,
            on_message_callback=lambda ch, method, props, body: self._on_message(
                ch, method, props, body, callback
            ),
        )

        self._consuming = True
        self._active_channel = channel  # Store channel reference for stopping

        try:
            # Start consuming (blocking)
            # This will block until stop_consuming() is called or connection is lost
            channel.start_consuming()
            # If we get here, consuming stopped normally (not due to exception)
            logger.debug("Consuming stopped normally")
        except KeyboardInterrupt:
            logger.info("Stopping consumer (KeyboardInterrupt)...")
            # Don't try to stop again if already stopped
            if channel.is_open:
                try:
                    channel.stop_consuming()
                except Exception:
                    pass  # Already stopped or connection lost
        except pika.exceptions.StreamLostError as e:
            # Connection was closed (e.g., by stop() from another thread or network issue)
            logger.debug(f"Stream connection lost while consuming: {str(e)}")
            # Don't re-raise - this is expected when stop() is called
        except pika.exceptions.AMQPConnectionError as e:
            # Connection error
            logger.debug(f"AMQP connection error while consuming: {str(e)}")
            raise
        except AttributeError as e:
            # This can happen when connection is being closed while start_consuming() is running
            # pika's internal state (_processing_fd_event_map) becomes None during cleanup
            # This is expected when stop() is called or connection is closed
            if "'NoneType' object has no attribute 'clear'" in str(
                e
            ) or "_processing_fd_event_map" in str(e):
                logger.debug(
                    f"Connection cleanup in progress while consuming: {str(e)}"
                )
                # Don't re-raise - this is expected when connection is being closed
            else:
                # Unexpected AttributeError, re-raise it
                logger.error(
                    f"Unexpected AttributeError in consume_tasks: {str(e)}",
                    exc_info=True,
                )
                raise
        except Exception as e:
            logger.error(f"Unexpected error in consume_tasks: {str(e)}", exc_info=True)
            raise
        finally:
            # Always reset state, even if we exited normally
            self._consuming = False
            self._active_channel = None

    @handle_connection_error
    def consume_dict(
        self,
        queue_name: str,
        callback: Callable[
            ..., None
        ],  # Accepts either (message_dict) or (message_dict, channel, delivery_tag)
        durable: bool = True,
        prefetch_count: int = 1,
        auto_ack: bool = True,
    ) -> None:
        """
        Consume raw dictionary messages from a queue.

        Args:
            queue_name: Name of the queue to consume from
            callback: Function to call for each message.
                     If auto_ack=False, callback signature should be:
                     callback(message_dict, channel, delivery_tag)
                     If auto_ack=True, callback signature is:
                     callback(message_dict)
            durable: If True, queue is durable
            prefetch_count: Number of unacknowledged messages to prefetch
            auto_ack: If True, automatically acknowledge messages after callback.
                     If False, callback must acknowledge manually.

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
                logger.debug(f"Received message from queue {method.routing_key}")
                if auto_ack:
                    callback(message_dict)
                    # Check if channel is still open before acking
                    if ch.is_open:
                        try:
                            ch.basic_ack(delivery_tag=method.delivery_tag)
                        except Exception as ack_error:
                            logger.debug(
                                f"Failed to ack message (channel may be closed): {ack_error}"
                            )
                else:
                    # Pass channel and delivery_tag to callback for manual ack
                    # Type checker can't verify this at compile time since callback signature
                    # depends on auto_ack parameter
                    callback(message_dict, ch, method.delivery_tag)  # type: ignore[misc]
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse message: {str(e)}")
                # Check if channel is still open before nacking
                if ch.is_open:
                    try:
                        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
                    except Exception as nack_error:
                        logger.debug(
                            f"Failed to nack message after JSON decode error (channel may be closed): {nack_error}"
                        )
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
                # Check if channel is still open before nacking
                if ch.is_open:
                    try:
                        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
                    except Exception as nack_error:
                        logger.debug(
                            f"Failed to nack message after processing error (channel may be closed): {nack_error}"
                        )

        # Set up consumer
        channel.basic_consume(
            queue=queue_name,
            on_message_callback=on_message,
        )

        self._consuming = True
        self._active_channel = channel  # Store channel reference for stopping
        logger.info(f"Started consuming from queue: {queue_name}")

        try:
            # Start consuming (blocking)
            # This will block until stop_consuming() is called or connection is lost
            channel.start_consuming()
            # If we get here, consuming stopped normally (not due to exception)
            logger.debug("Consuming stopped normally")
        except KeyboardInterrupt:
            logger.info("Stopping consumer (KeyboardInterrupt)...")
            # Don't try to stop again if already stopped
            if channel.is_open:
                try:
                    channel.stop_consuming()
                except Exception:
                    pass  # Already stopped or connection lost
        except pika.exceptions.StreamLostError as e:
            # Connection was closed (e.g., by stop() from another thread or network issue)
            logger.debug(f"Stream connection lost while consuming: {str(e)}")
            # Don't re-raise - this is expected when stop() is called
        except pika.exceptions.ChannelClosedByBroker as e:
            # Channel was closed by broker (e.g., PRECONDITION_FAILED - unknown delivery tag)
            # This can happen when stop() is called after messages have been acknowledged
            # It's safe to ignore - consuming has already stopped
            logger.debug(f"Channel closed by broker while consuming: {str(e)}")
            # Don't re-raise - this is expected when stop() is called
        except pika.exceptions.AMQPConnectionError as e:
            # Connection error
            logger.debug(f"AMQP connection error while consuming: {str(e)}")
            raise
        except AttributeError as e:
            # This can happen when connection is being closed while start_consuming() is running
            # pika's internal state (_processing_fd_event_map) becomes None during cleanup
            # This is expected when stop() is called or connection is closed
            if "'NoneType' object has no attribute 'clear'" in str(
                e
            ) or "_processing_fd_event_map" in str(e):
                logger.debug(
                    f"Connection cleanup in progress while consuming: {str(e)}"
                )
                # Don't re-raise - this is expected when connection is being closed
            else:
                # Unexpected AttributeError, re-raise it
                logger.error(
                    f"Unexpected AttributeError in consume_dict: {str(e)}",
                    exc_info=True,
                )
                raise
        except Exception as e:
            logger.error(f"Unexpected error in consume_dict: {str(e)}", exc_info=True)
            raise
        finally:
            # Always reset state, even if we exited normally
            self._consuming = False
            self._active_channel = None

    def stop(self):
        """Stop consuming messages."""
        if not self._consuming:
            # Already stopped, nothing to do
            return

        # Use the stored channel reference if available
        if self._active_channel is not None:
            try:
                # Check if channel is still open before stopping
                if self._active_channel.is_open:
                    self._active_channel.stop_consuming()
                    logger.debug("Stopped consuming on active channel")
            except pika.exceptions.StreamLostError:
                # Connection already lost, that's okay
                logger.debug("Connection already lost when stopping")
            except pika.exceptions.ChannelClosedByBroker as e:
                # Channel was closed by broker (e.g., PRECONDITION_FAILED - unknown delivery tag)
                # This can happen when messages have already been acknowledged and we try to stop
                # It's safe to ignore - the channel is already closed
                logger.debug(f"Channel closed by broker when stopping: {str(e)}")
            except Exception as e:
                logger.debug(f"Error stopping channel: {str(e)}")
        else:
            # No active channel, try to get one and stop it (fallback)
            try:
                if self.connection.is_connected():
                    channel = self.connection.channel
                    if channel and channel.is_open:
                        channel.stop_consuming()
                        logger.debug("Stopped consuming on fallback channel")
            except pika.exceptions.ChannelClosedByBroker as e:
                # Channel was closed by broker - safe to ignore
                logger.debug(
                    f"Channel closed by broker when stopping (fallback): {str(e)}"
                )
            except Exception as e:
                logger.debug(f"Error stopping consumer (fallback): {str(e)}")

        self._consuming = False
        self._active_channel = None
        logger.info("Stopped consuming")

    def close(self):
        """Close connection if we own it."""
        # Stop consuming first (idempotent)
        self.stop()
        # Only close connection if we own it
        if self._own_connection and self.connection:
            try:
                self.connection.close()
            except Exception as e:
                logger.debug(f"Error closing connection: {str(e)}")

    def __enter__(self):
        """Context manager entry."""
        self.connection.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
