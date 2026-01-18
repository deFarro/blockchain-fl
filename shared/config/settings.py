"""Shared configuration management."""

import base64
from pathlib import Path
from typing import Optional, List
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
from shared.utils.crypto import AES256GCM

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

    # Encryption Key Configuration
    # Key provided via ENCRYPTION_KEY environment variable
    # Must be a Base64-encoded 32-byte key
    # Generate with: python -c "import base64, os; print(base64.b64encode(os.urandom(32)).decode())"
    encryption_key: Optional[str] = None

    # Blockchain Service Configuration
    # Blockchain service is a separate Go microservice
    blockchain_service_url: str = (
        "http://blockchain-service:8080"  # Use service name in Docker, localhost for local dev
    )

    # Application Configuration
    log_level: str = "INFO"
    environment: str = "development"

    # API Configuration
    api_key: Optional[str] = None  # API key for authentication
    api_host: str = "0.0.0.0"  # API server host
    api_port: int = 8000  # API server port

    # Aggregation Configuration (for main service)
    min_clients_for_aggregation: int = 1
    aggregation_timeout: int = 60  # seconds to wait for client updates

    # Client Exclusion Configuration (for regression diagnosis)
    enable_client_exclusion: bool = (
        False  # Enable client exclusion after regression diagnosis
    )
    excluded_clients: List[str] = []  # List of client IDs to exclude from aggregation

    # Training Configuration
    target_accuracy: float = 95.0  # Target accuracy to achieve (percentage)
    max_iterations: int = 100  # Maximum training iterations
    max_rollbacks: int = 5  # Maximum rollbacks before stopping
    convergence_patience: int = 10  # Iterations without improvement before convergence
    accuracy_tolerance: float = 0.5  # Allowed accuracy drop (percentage)
    patience_threshold: int = 3  # Consecutive bad iterations before rollback
    severe_drop_threshold: float = 2.0  # Immediate rollback threshold (percentage)
    num_clients: int = 2  # Number of client instances

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    @property
    def project_root(self) -> Path:
        """Get project root directory."""
        return Path(__file__).parent.parent.parent

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

    def get_encryption_key(self) -> bytes:
        """
        Get encryption key from ENCRYPTION_KEY environment variable.

        Expects a Base64-encoded 32-byte key.

        Returns:
            bytes: 32-byte encryption key

        Raises:
            ValueError: If ENCRYPTION_KEY is not set or invalid
        """
        if not self.encryption_key:
            raise ValueError(
                "ENCRYPTION_KEY environment variable is required. "
                'Generate a key with: python -c "import base64, os; print(base64.b64encode(os.urandom(32)).decode())"'
            )

        # Decode Base64-encoded key
        try:
            key = base64.b64decode(self.encryption_key)
            if len(key) != AES256GCM.KEY_SIZE:
                raise ValueError(
                    f"Encryption key must be {AES256GCM.KEY_SIZE} bytes after Base64 decoding, "
                    f"got {len(key)} bytes"
                )
            return key
        except Exception as e:
            raise ValueError(
                f"Invalid ENCRYPTION_KEY: must be a valid Base64-encoded 32-byte key. Error: {str(e)}"
            ) from e


# Global settings instance
settings = Settings()
