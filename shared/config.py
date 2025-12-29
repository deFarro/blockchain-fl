"""Shared configuration management."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings."""

    # RabbitMQ Configuration
    rabbitmq_host: str = (
        "rabbitmq"  # Use service name in Docker, localhost for local dev
    )
    rabbitmq_port: int = 5672
    rabbitmq_user: str = "admin"
    rabbitmq_password: str = "admin"
    rabbitmq_vhost: str = "/"

    # IPFS Configuration
    ipfs_host: str = "ipfs"  # Use service name in Docker, localhost for local dev
    ipfs_port: int = 5001
    ipfs_protocol: str = "http"

    # Vault Configuration
    vault_addr: str = (
        "http://vault:8200"  # Use service name in Docker, localhost for local dev
    )
    vault_token: str = "root-token"

    # Hyperledger Fabric Configuration
    fabric_network_profile: str = "network.json"
    fabric_channel_name: str = "mychannel"
    fabric_chaincode_name: str = "model_provenance"

    # Application Configuration
    log_level: str = "INFO"
    environment: str = "development"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    @property
    def project_root(self) -> Path:
        """Get project root directory."""
        return Path(__file__).parent.parent

    @property
    def data_dir(self) -> Path:
        """Get data directory."""
        return self.project_root / "data"

    @property
    def rabbitmq_url(self) -> str:
        """Get RabbitMQ connection URL."""
        return f"amqp://{self.rabbitmq_user}:{self.rabbitmq_password}@{self.rabbitmq_host}:{self.rabbitmq_port}{self.rabbitmq_vhost}"

    @property
    def ipfs_url(self) -> str:
        """Get IPFS API URL."""
        return f"{self.ipfs_protocol}://{self.ipfs_host}:{self.ipfs_port}"


# Global settings instance
settings = Settings()
