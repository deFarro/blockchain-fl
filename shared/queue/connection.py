"""RabbitMQ connection manager with reconnection logic."""

import pika
from typing import Optional
from pika.connection import Connection
from pika.channel import Channel
from shared.config import settings
from shared.logger import setup_logger

logger = setup_logger(__name__)


class QueueConnection:
    """Manages RabbitMQ connection with automatic reconnection."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        vhost: Optional[str] = None,
    ):
        """
        Initialize queue connection.

        Args:
            host: RabbitMQ host (defaults to settings.rabbitmq_host)
            port: RabbitMQ port (defaults to settings.rabbitmq_port)
            username: RabbitMQ username (defaults to settings.rabbitmq_user)
            password: RabbitMQ password (defaults to settings.rabbitmq_password)
            vhost: RabbitMQ virtual host (defaults to settings.rabbitmq_vhost)
        """
        self.host = host or settings.rabbitmq_host
        self.port = port or settings.rabbitmq_port
        self.username = username or settings.rabbitmq_user
        self.password = password or settings.rabbitmq_password
        self.vhost = vhost or settings.rabbitmq_vhost

        self.connection: Optional[Connection] = None
        self.channel: Optional[Channel] = None

    def connect(self) -> Channel:
        """
        Establish connection to RabbitMQ and return channel.

        Returns:
            RabbitMQ channel

        Raises:
            pika.exceptions.AMQPConnectionError: If connection fails
        """
        try:
            credentials = pika.PlainCredentials(self.username, self.password)
            parameters = pika.ConnectionParameters(
                host=self.host,
                port=self.port,
                virtual_host=self.vhost,
                credentials=credentials,
                heartbeat=600,  # 10 minutes
                blocked_connection_timeout=300,  # 5 minutes
            )

            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()

            logger.info(
                f"Connected to RabbitMQ at {self.host}:{self.port}/{self.vhost}"
            )
            return self.channel

        except pika.exceptions.AMQPConnectionError as e:
            logger.error(f"Failed to connect to RabbitMQ: {str(e)}")
            raise

    def reconnect(self) -> Channel:
        """
        Reconnect to RabbitMQ.

        Returns:
            RabbitMQ channel

        Raises:
            pika.exceptions.AMQPConnectionError: If reconnection fails
        """
        self.close()
        return self.connect()

    def close(self):
        """Close connection and channel."""
        if self.channel and self.channel.is_open:
            try:
                self.channel.close()
            except Exception as e:
                logger.warning(f"Error closing channel: {str(e)}")

        if self.connection and self.connection.is_open:
            try:
                self.connection.close()
            except Exception as e:
                logger.warning(f"Error closing connection: {str(e)}")

        self.channel = None
        self.connection = None
        logger.debug("RabbitMQ connection closed")

    def is_connected(self) -> bool:
        """
        Check if connection is active.

        Returns:
            True if connected and channel is open
        """
        return (
            self.connection is not None
            and self.connection.is_open
            and self.channel is not None
            and self.channel.is_open
        )

    def ensure_connected(self) -> Channel:
        """
        Ensure connection is active, reconnect if needed.

        Returns:
            RabbitMQ channel
        """
        if not self.is_connected():
            logger.warning("Connection lost, reconnecting...")
            return self.reconnect()
        return self.channel

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

