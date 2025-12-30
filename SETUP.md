# Setup Instructions

## Prerequisites

- Python 3.12 or 3.10-3.13
- Docker and Docker Compose
- Git
- Go 1.21+ (for blockchain service development)

## Initial Setup

### 1. Set Up Environment Variables

Create a `.env` file from the template:

```bash
cp env_template.txt .env
```

Edit `.env` and update any values if needed (defaults work for development).

**Important:** Change default passwords in production!

### 2. Start Infrastructure Services

Start RabbitMQ, IPFS, and Blockchain Service using Docker Compose:

```bash
docker-compose up -d rabbitmq ipfs blockchain-service
```

Verify services are running:

```bash
docker-compose ps
```

Access services:

- RabbitMQ Management UI: http://localhost:15672 (credentials from .env)
- IPFS API: http://localhost:5001
- Blockchain Service API: http://localhost:8080

### 3. Build and Start Application Services

Build the Docker images for all services:

```bash
docker-compose build
```

Start the services:

```bash
# Start all services
docker-compose up -d

# Or start individually
docker-compose up -d blockchain-service
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
docker-compose logs -f blockchain-service
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
python3.12 -m venv venv  # Or python3.11, python3.13, etc.
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

Still need Docker for infrastructure services (RabbitMQ and IPFS):

```bash
docker-compose up -d rabbitmq ipfs
```

### 4. Local Blockchain Service Development (Optional)

If you want to develop the blockchain service locally in Go:

```bash
cd blockchain_service

# Install Go dependencies
go mod download

# Run locally
go run main.go

# Or build and run
go build -o blockchain-service
./blockchain-service
```

The service will run on `http://localhost:8080` by default.

**Note:** For local development, you still need Docker for RabbitMQ and IPFS, but you can run the blockchain service locally instead of in Docker.

### 5. Update .env for Local Development

For local development, update `.env` to use `localhost` instead of service names:

```bash
RABBITMQ_HOST=localhost
IPFS_HOST=localhost
BLOCKCHAIN_SERVICE_URL=http://localhost:8080
```

**Encryption Key Configuration:**

- Set `ENCRYPTION_KEY` to a Base64-encoded 32-byte key
- Generate a key:

  ```bash
  # Using the provided script
  python scripts/generate_encryption_key.py

  # Or using Python directly
  python -c "import base64, os; print(base64.b64encode(os.urandom(32)).decode())"
  ```

- Add the generated key to your `.env` file: `ENCRYPTION_KEY=your-generated-key-here`

## Project Structure

```
blockchain-fl/
├── main_service/          # Main aggregator service (Python)
│   ├── requirements.txt   # Main service dependencies
│   ├── Dockerfile         # Main service container
│   ├── workers/          # Worker processes
│   ├── api/              # REST API (FastAPI)
│   ├── blockchain/       # Blockchain client (HTTP client)
│   └── storage/          # IPFS and encryption
├── blockchain_service/   # Blockchain microservice (Go)
│   ├── main.go           # REST API server
│   ├── go.mod            # Go dependencies
│   └── Dockerfile        # Blockchain service container
├── client_service/       # Client training service (Python)
│   ├── requirements.txt  # Client service dependencies
│   ├── Dockerfile        # Client service container
│   ├── training/         # Training logic
│   └── queue/            # Queue consumers
├── shared/               # Shared utilities (Python)
│   ├── models/          # Data models
│   ├── utils/           # Utilities (crypto, hashing)
│   └── datasets/        # Dataset interfaces
├── scripts/              # Utility scripts
├── tests/               # Tests
├── data/                # Data files (created at runtime)
├── docker-compose.yml    # Infrastructure and services
└── requirements.txt      # Development dependencies only
```

## Docker Services

### Infrastructure Services (Always Running)

- **rabbitmq**: Message queue
- **ipfs**: Distributed storage
- **blockchain-service**: Blockchain operations microservice (Go)
- **Encryption keys**: Managed via environment variables

### Application Services

- **main-service**: Aggregator service (Python - workers, API, validation)
- **client-service**: Training service (Python - can be scaled)

## Environment Variables

All sensitive configuration is in `.env` file:

- RabbitMQ credentials
- Encryption key (ENCRYPTION_KEY or ENCRYPTION_KEY_FILE)
- Service ports
- Client IDs and dataset paths

## Running Tests

### Prerequisites for Tests

Before running tests, ensure the required infrastructure services are running:

```bash
# Start RabbitMQ and IPFS (required for some tests)
docker-compose up -d rabbitmq ipfs
```

# Run all tests

make test

## Testing Queue Infrastructure

After setting up the environment, you can test the queue infrastructure:

### 1. Start RabbitMQ

```bash
docker-compose up -d rabbitmq
```

Verify RabbitMQ is running:

```bash
docker-compose ps rabbitmq
```

Access RabbitMQ Management UI: http://localhost:15672

- Username: `admin` (or from `.env`)
- Password: `admin` (or from `.env`)

### 2. Test Queue Publish/Subscribe

Run the test script to verify queue functionality:

```bash
# Make sure you're in the project root and virtual environment is activated
source venv/bin/activate  # If using local development

# Run the test script
python tests/test_shared/test_queue.py
# Or using pytest
pytest tests/test_shared/test_queue.py -v
```

The test script will:

- Create a test task
- Publish it to a test queue
- Consume the message
- Verify successful delivery

### 3. Testing IPFS Integration

To run IPFS-related tests (in `test_storage.py`), ensure IPFS is running:

```bash
# Start IPFS container
docker-compose up -d ipfs

# Verify IPFS is accessible
curl http://localhost:5001/api/v0/version

# Run storage tests (IPFS tests will run if IPFS is available)
pytest tests/test_main/test_storage.py -v
```

**Note:** If IPFS is not running, IPFS-related tests will be automatically skipped with a message `(IPFS not available)`. This is expected behavior and does not indicate a problem.

## Troubleshooting

### Blockchain Service Issues

- **Service not starting**: Check Go version (`go version`) - requires Go 1.21+
- **Port already in use**: Change `BLOCKCHAIN_SERVICE_PORT` in `.env`
- **Dependencies missing**: Run `go mod download` in `blockchain_service/` directory

### Python Service Issues

- **Import errors**: Make sure virtual environment is activated and dependencies are installed
- **Port conflicts**: Check if ports 8000 (main-service) are already in use
- **IPFS connection errors**: Ensure IPFS container is running (`docker-compose ps`)
- **IPFS tests skipped**: This is normal if IPFS is not running. Start IPFS with `docker-compose up -d ipfs` to run IPFS-related tests

### Queue Issues

- **Connection refused**: Ensure RabbitMQ is running (`docker-compose ps rabbitmq`)
- **Authentication failed**: Check `RABBITMQ_USER` and `RABBITMQ_PASSWORD` in `.env`
- **Queue not found**: Queues are auto-declared, but verify queue name is correct
- **Messages not being consumed**: Check consumer is running and queue name matches
- **Test script fails**: Run `python tests/test_shared/test_queue.py` or `pytest tests/test_shared/test_queue.py` with RabbitMQ running and virtual environment activated

### General Issues

- **Services not communicating**: Check network connectivity and service names in `.env`
- **Missing encryption key**: Generate and set `ENCRYPTION_KEY` in `.env`

## Next Steps

1. Verify all services are running: `docker-compose ps`
2. Check service logs: `docker-compose logs -f [service-name]`
3. Test API endpoints (see main README.md for API documentation)
4. Start implementing workers and training logic
