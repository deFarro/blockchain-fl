# Setup Instructions

## Prerequisites

- Python 3.10 or higher
- Docker and Docker Compose
- Git

## Initial Setup

### 1. Set Up Environment Variables

Create a `.env` file from the template:

```bash
cp env_template.txt .env
```

Edit `.env` and update any values if needed (defaults work for development).

**Important:** Change default passwords in production!

### 2. Start Infrastructure Services

Start RabbitMQ, IPFS, and Vault using Docker Compose:

```bash
docker-compose up -d rabbitmq ipfs vault
```

Verify services are running:

```bash
docker-compose ps
```

Access services:

- RabbitMQ Management UI: http://localhost:15672 (credentials from .env)
- IPFS API: http://localhost:5001
- Vault UI: http://localhost:8200 (token from .env)

### 3. Build and Start Application Services

Build the Docker images for main service and client service:

```bash
docker-compose build
```

Start the services:

```bash
# Start all services
docker-compose up -d

# Or start individually
docker-compose up -d main-service
docker-compose up -d client-service
```

### 4. Verify Setup

Check that all services are running:

```bash
docker-compose ps
```

View logs:

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f main-service
docker-compose logs -f client-service
```

### 5. Scale Client Services (Optional)

To run multiple client instances:

```bash
# Scale to 2 clients
docker-compose up -d --scale client-service=2

# Or set CLIENT_ID environment variable for each instance
CLIENT_ID=client_0 docker-compose up -d client-service
CLIENT_ID=client_1 docker-compose up -d client-service
```

## Local Development (Without Docker)

If you want to develop locally without Docker containers:

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip

# Install main service dependencies
pip install -r main_service/requirements.txt

# Install client service dependencies
pip install -r client_service/requirements.txt

# Install development dependencies
pip install -r requirements.txt
```

### 3. Start Infrastructure Services

Still need Docker for infrastructure:

```bash
docker-compose up -d rabbitmq ipfs vault
```

### 4. Update .env for Local Development

For local development, update `.env` to use `localhost` instead of service names:

```bash
RABBITMQ_HOST=localhost
IPFS_HOST=localhost
VAULT_ADDR=http://localhost:8200
```

## Project Structure

```
blockchain-fl/
├── main_service/          # Main aggregator service
│   ├── requirements.txt   # Main service dependencies
│   ├── Dockerfile         # Main service container
│   ├── workers/          # Worker processes
│   ├── api/              # REST API
│   ├── blockchain/       # Blockchain integration
│   └── storage/          # IPFS and encryption
├── client_service/       # Client training service
│   ├── requirements.txt  # Client service dependencies
│   ├── Dockerfile        # Client service container
│   ├── training/         # Training logic
│   └── queue/            # Queue consumers
├── shared/               # Shared utilities
│   ├── models/          # Data models
│   ├── utils/           # Utilities
│   └── datasets/        # Dataset interfaces
├── scripts/              # Utility scripts
├── tests/               # Tests
├── data/                # Data files (created by preprocessing)
├── docker-compose.yml    # Infrastructure and services
└── requirements.txt      # Development dependencies only
```

## Docker Services

### Infrastructure Services (Always Running)

- **rabbitmq**: Message queue
- **ipfs**: Distributed storage
- **vault**: Key management

### Application Services

- **main-service**: Aggregator service (workers, API, validation)
- **client-service**: Training service (can be scaled)

## Environment Variables

All sensitive configuration is in `.env` file:

- RabbitMQ credentials
- Vault token
- Service ports
- Client IDs and dataset paths

## Next Steps

1. Run dataset preprocessing (Step 2)
2. Start implementing shared models and utilities (Step 3)
