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
go mod tidy
go mod download

# Run locally
go run main.go

# Or build and run
go build -o blockchain-service
./blockchain-service
```

The service will run on `http://localhost:8080` by default.

**Note:** For local development, you still need Docker for RabbitMQ and IPFS, but you can run the blockchain service locally instead of in Docker.

#### Blockchain Service Overview

The blockchain service is a Go microservice that handles all Hyperledger Fabric operations. It provides a REST API for blockchain operations and can operate in two modes:

1. **Development Mode (Default)**: Uses in-memory storage - **No blockchain setup required!** This is the default mode and works out of the box. Perfect for development, testing, and research projects.

2. **Blockchain Mode (Optional)**: Connected to a **local** Hyperledger Fabric network that you run on your own machine. This is completely local - you don't register with any external network. The Fabric network runs in Docker containers on your local machine, just like RabbitMQ and IPFS.

**Important:** For most use cases (development, testing, research), you can use the default development mode. The blockchain service will automatically use in-memory storage and work perfectly fine without any blockchain setup.

#### API Endpoints

- `GET /health` - Health check
- `POST /api/v1/model/register` - Register model version
- `POST /api/v1/model/validate` - Record validation results
- `POST /api/v1/model/rollback` - Record rollback event
- `GET /api/v1/model/provenance/{version_id}` - Get provenance chain

#### Blockchain Service Configuration

**Environment Variables:**

- `BLOCKCHAIN_SERVICE_PORT` - Port to run the service on (default: 8080)
- `FABRIC_NETWORK_PROFILE` - Path to Fabric network connection profile (e.g., `connection.json`)
- `FABRIC_WALLET_PATH` - Path to Fabric wallet directory
- `FABRIC_CHANNEL_NAME` - Channel name (default: `mychannel`)
- `FABRIC_CHAINCODE_NAME` - Chaincode name (default: `model_provenance`)
- `FABRIC_USER` - User identity to use (default: `appUser`)

#### Setting Up Local Hyperledger Fabric Network (Optional)

**Important:** This is completely optional! The project works perfectly fine in development mode without any blockchain setup.

If you want to use a **real local blockchain** (running entirely on your machine, not connected to any external network), you can set up the Hyperledger Fabric Test Network:

**Fabric Test Network (Local Blockchain - Runs on Your Machine)**

The Fabric Test Network is a **local blockchain network** that runs in Docker containers on your machine. It's completely isolated and doesn't connect to any external network.

1. Clone Hyperledger Fabric samples:

   ```bash
   git clone https://github.com/hyperledger/fabric-samples.git
   cd fabric-samples
   ```

2. Start the local test network (runs in Docker containers):

   ```bash
   ./network.sh up createChannel
   ```

3. Deploy the chaincode:

   ```bash
   ./network.sh deployCC -ccn model_provenance -ccp ../blockchain-fl/blockchain_service/chaincode -ccl go
   ```

4. Set up wallet and connection profile (see Fabric documentation)

This creates a **completely local blockchain network** on your machine - no registration, no external connections, just like running RabbitMQ or IPFS locally.

#### Chaincode

The chaincode (`blockchain_service/chaincode/model_provenance.go`) implements:

- `RegisterModelUpdate` - Records model version with metadata
- `RecordValidation` - Records validation results
- `RollbackModel` - Records rollback events
- `GetModelProvenance` - Queries provenance chain
- `VerifyIntegrity` - Verifies hash integrity
- `GetValidationHistory` - Retrieves validation history

#### Testing Blockchain Service

The service can be tested independently:

```bash
# Health check
curl http://localhost:8080/health

# Register model version
curl -X POST http://localhost:8080/api/v1/model/register \
  -H "Content-Type: application/json" \
  -d '{
    "model_version_id": "v1",
    "parent_version_id": null,
    "hash": "abc123",
    "metadata": {"iteration": 1}
  }'
```

**Integration:** The main service (Python) calls this service via HTTP using the `FabricClient` class in `main_service/blockchain/fabric_client.py`.

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
- **"Running in development mode" message**: This is normal! The service works perfectly in development mode without any blockchain setup. You only need to configure Fabric if you want to use a real blockchain.
- **Fabric connection errors**: If you're not using Fabric, you can ignore these. The service automatically falls back to development mode.

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
3. Test API endpoints (see RUNNING.md for API documentation)
4. Start implementing workers and training logic

---

# Dashboard

The dashboard is a React application with Tailwind CSS for monitoring and controlling the federated learning system.

## Development

```bash
cd dashboard
npm install
npm run dev
```

The dev server runs on `http://localhost:3000` with proxy to API at `http://localhost:8000`.

**Note:** This project uses **Tailwind CSS** for styling. The build process automatically processes Tailwind classes.

## Building for Production

```bash
npm run build
```

This creates a `dist/` directory with the production build. The main service serves this directory.

## Project Structure

```
dashboard/
├── src/
│   ├── components/     # React components
│   │   ├── Header.jsx
│   │   ├── Controls.jsx
│   │   ├── StatusCard.jsx
│   │   ├── AccuracyChart.jsx
│   │   ├── VersionsTable.jsx
│   │   ├── RollbackModal.jsx
│   │   ├── ModelDetailsModal.jsx
│   │   └── ProvenanceModal.jsx
│   ├── utils/
│   │   └── api.js      # API utility functions
│   ├── App.jsx         # Main app component
│   ├── main.jsx        # Entry point
│   └── index.css       # Global styles (Tailwind directives)
├── index.html          # HTML template
├── package.json        # Dependencies (includes Tailwind CSS)
├── vite.config.js      # Vite configuration
├── tailwind.config.js  # Tailwind CSS configuration
├── postcss.config.js   # PostCSS configuration (for Tailwind)
└── dist/               # Production build (generated)
```

**Styling:** This project uses **Tailwind CSS** for all styling. All components use Tailwind utility classes instead of custom CSS.

## Features

- Real-time training status monitoring
- Start/Stop training controls
- Model version listing and details
- Rollback functionality
- Provenance chain viewing
- Accuracy history chart
- API documentation link
