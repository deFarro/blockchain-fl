"""Client service configuration."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
from shared.logger import setup_logger

logger = setup_logger(__name__)

load_dotenv()


class ClientConfig(BaseSettings):
    """Client service configuration."""

    # Client identification
    num_clients: int = 2  # Number of client instances
    split_type: str = "iid"  # Dataset split type: iid or non_iid
    dataset_seed: int = 42  # Seed for dataset splitting

    # RabbitMQ Configuration
    rabbitmq_host: str = (
        "rabbitmq"  # Use service name in Docker, localhost for local dev
    )
    rabbitmq_port: int = 5672
    rabbitmq_user: str = "admin"
    rabbitmq_password: str = "admin"
    rabbitmq_vhost: str = "/"

    # Application Configuration
    log_level: str = "INFO"
    environment: str = "development"

    # Dataset Configuration
    dataset_name: str = "mnist"  # Dataset to use: mnist, caltech101, usps, cifar10

    # Training Configuration
    epochs: int = 1  # Number of epochs per training iteration

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    @property
    def data_dir(self) -> str:
        """Dataset directory, derived from dataset_name (e.g. data/mnist, data/caltech101, data/usps, data/cifar10)."""
        return f"data/{self.dataset_name}"

    @property
    def rabbitmq_url(self) -> str:
        """Get RabbitMQ connection URL."""
        return f"amqp://{self.rabbitmq_user}:{self.rabbitmq_password}@{self.rabbitmq_host}:{self.rabbitmq_port}{self.rabbitmq_vhost}"


# Global config instance
config = ClientConfig()
