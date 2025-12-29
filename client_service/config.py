"""Client service configuration."""
import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()


class ClientConfig(BaseSettings):
    """Client service configuration."""
    
    # Client identification
    # client_id is determined from docker-compose replica number (0, 1, 2, ...)
    # This is set automatically by docker-compose using HOSTNAME
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
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    def get_client_id(self) -> int:
        """
        Get client ID from docker-compose replica number.
        
        When using docker-compose with --scale, containers are named:
        - blockchain-fl-client-service-1 -> client_id = 0
        - blockchain-fl-client-service-2 -> client_id = 1
        - blockchain-fl-client-service-3 -> client_id = 2
        etc.
        
        HOSTNAME inside container matches the container name.
        """
        # Extract from HOSTNAME (docker-compose format: service-name-replica-number)
        hostname = os.getenv("HOSTNAME", "")
        if hostname:
            # Format: blockchain-fl-client-service-1, blockchain-fl-client-service-2, etc.
            # Last part is the replica number
            parts = hostname.split("-")
            if len(parts) > 0:
                try:
                    # Last part is replica number, convert to 0-based index
                    replica_num = int(parts[-1])
                    return replica_num - 1  # Convert to 0-based (replica 1 -> client_id 0)
                except ValueError:
                    pass
        
        # Fallback: try CLIENT_ID env var (for local development)
        client_id_env = os.getenv("CLIENT_ID")
        if client_id_env:
            if client_id_env.isdigit():
                return int(client_id_env)
            elif client_id_env.startswith("client_"):
                return int(client_id_env.split("_")[1])
        
        # Default to 0 if can't determine
        return 0
    
    @property
    def rabbitmq_url(self) -> str:
        """Get RabbitMQ connection URL."""
        return f"amqp://{self.rabbitmq_user}:{self.rabbitmq_password}@{self.rabbitmq_host}:{self.rabbitmq_port}{self.rabbitmq_vhost}"


# Global config instance
config = ClientConfig()

