"""Client service configuration."""

import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
from shared.logger import setup_logger

logger = setup_logger(__name__)

load_dotenv()


class ClientConfig(BaseSettings):
    """Client service configuration."""

    # Client identification
    num_clients: int = int(os.getenv("NUM_CLIENTS", "2"))
    split_type: str = os.getenv("SPLIT_TYPE", "iid")  # iid or non_iid
    dataset_seed: int = int(os.getenv("DATASET_SEED", "42"))

    # RabbitMQ Configuration
    rabbitmq_host: str = os.getenv("RABBITMQ_HOST", "rabbitmq")
    rabbitmq_port: int = int(os.getenv("RABBITMQ_PORT", "5672"))
    rabbitmq_user: str = os.getenv("RABBITMQ_USER", "admin")
    rabbitmq_password: str = os.getenv("RABBITMQ_PASSWORD", "admin")
    rabbitmq_vhost: str = os.getenv("RABBITMQ_VHOST", "/")

    # Application Configuration
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    environment: str = os.getenv("ENVIRONMENT", "development")

    # Dataset Configuration
    data_dir: str = "data/mnist"

    # Training Configuration
    epochs: int = int(
        os.getenv("EPOCHS", "1")
    )  # Number of epochs per training iteration

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    @property
    def rabbitmq_url(self) -> str:
        """Get RabbitMQ connection URL."""
        return f"amqp://{self.rabbitmq_user}:{self.rabbitmq_password}@{self.rabbitmq_host}:{self.rabbitmq_port}{self.rabbitmq_vhost}"


# Global config instance
config = ClientConfig()
