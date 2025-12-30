# Running the Project

This guide explains how to run the federated learning system and use the API/dashboard to control training.

## Quick Start

### Step 1: Setup Environment

1. **Copy environment template:**

   ```bash
   cp env_template.txt .env
   ```

2. **Generate encryption key:**

   ```bash
   python scripts/generate_encryption_key.py
   # Copy the output to .env as ENCRYPTION_KEY=...
   ```

3. **Set API key in `.env`:**
   ```bash
   API_KEY=your-secret-api-key-here
   ```

### Step 2: Start Services

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps
```

**Expected services:**

- `blockchain-fl-rabbitmq` - Message queue
- `blockchain-fl-ipfs` - Distributed storage
- `blockchain-fl-blockchain-service` - Blockchain operations
- `blockchain-fl-main-service` - Main aggregator service
- `blockchain-fl-client-service-1`, `blockchain-fl-client-service-2`, ... - Client services

### Step 3: Access Dashboard

**Dashboard URL:** `http://localhost:8000/` (served by main service)

The dashboard is a **React application** that provides a web interface to:

- ✅ Start/stop training
- ✅ Monitor training progress in real-time (auto-refreshes every 5 seconds)
- ✅ View model versions and details
- ✅ Rollback to specific versions
- ✅ View provenance chains
- ✅ Access API documentation

**Alternative URLs:**

- Dashboard: `http://localhost:8000/` (main entry point)
- API Documentation: `http://localhost:8000/docs` (FastAPI Swagger UI)

**First Time Access:**

1. Open `http://localhost:8000/` in your browser
2. Enter your API key when prompted (set `API_KEY` in `.env`)
3. Click "▶ Start Training" to begin federated learning
4. Monitor progress in real-time (auto-refreshes every 5 seconds)

**Development Mode:**

For dashboard development with hot reload:

```bash
cd dashboard
npm install
npm run dev
```

This runs the Vite dev server on `http://localhost:3000` with proxy to the API at `http://localhost:8000`.

### Step 4: Monitor Training

The dashboard shows:

- **Current iteration** number
- **Best accuracy** achieved
- **Rollback count**
- **Accuracy history** chart
- **Model versions** table

### Step 5: Control Training

- **Start Training:** Click "▶ Start Training" button
- **Stop Training:** Click "⏹ Stop Training" button (TODO: implementation)
- **Rollback:** Click "↩ Rollback to Version", enter version ID and reason
- **Refresh:** Click "🔄 Refresh Status" or wait for auto-refresh (every 5 seconds)

---

## API Endpoints

### Authentication

All endpoints (except `/health`) require an API key in the `X-API-Key` header.

Set your API key in `.env`:

```bash
API_KEY=your-secret-api-key-here
```

### Training Control

#### Start Training

```bash
POST /api/v1/training/start
```

Starts a new federated learning session. Publishes initial TRAIN tasks to all clients.

**Request:**

```json
{
  "initial_weights_cid": null, // Optional: IPFS CID of initial weights (null = generate random)
  "num_iterations": null // Optional: Max iterations (null = use default from config)
}
```

**Response:**

```json
{
  "success": true,
  "message": "Training started for 2 clients",
  "iteration": 1
}
```

#### Get Training Status

```bash
GET /api/v1/training/status
```

Returns current training status including:

- Current iteration number
- Best accuracy achieved
- Rollback count
- Training status (running/stopped/completed)
- Start time

**Response:**

```json
{
  "is_training": true,
  "current_iteration": 5,
  "best_accuracy": 92.5,
  "total_iterations": 5,
  "rollback_count": 1,
  "status": "running",
  "start_time": "2024-01-15T10:30:00Z",
  "best_checkpoint_version": "version_3",
  "best_checkpoint_cid": "QmABC123..."
}
```

#### Stop Training

```bash
POST /api/v1/training/stop
```

Stops the current training session (TODO: implementation pending).

### Model Management

#### List All Model Versions

```bash
GET /api/v1/models
```

Returns list of all model versions with their metadata.

#### Get Model Version Details

```bash
GET /api/v1/models/{version_id}
```

Returns detailed information about a specific model version:

- Version ID
- Parent version ID
- IPFS CID
- Accuracy and metrics
- Iteration number
- Rollback count
- Validation history

#### Get Provenance Chain

```bash
GET /api/v1/models/{version_id}/provenance
```

Returns the complete provenance chain (lineage) for a model version, showing all parent versions back to the initial model.

#### Manual Rollback

```bash
POST /api/v1/models/{version_id}/rollback
```

Manually rollback to a specific model version (e.g., for model poisoning scenarios).

**Request:**

```json
{
  "target_version_id": "version_3", // Optional: uses path param if not provided
  "reason": "Manual rollback due to suspected model poisoning"
}
```

**Response:**

```json
{
  "success": true,
  "message": "Rollback initiated",
  "target_version_id": "version_3",
  "transaction_id": "tx_abc123"
}
```

## Using the API

### Example: Complete Training Workflow

```bash
# 1. Start training
curl -X POST http://localhost:8000/api/v1/training/start \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json"

# 2. Monitor training status (poll every few seconds)
curl http://localhost:8000/api/v1/training/status \
  -H "X-API-Key: your-api-key"

# 3. Check specific model version
curl http://localhost:8000/api/v1/models/version_5 \
  -H "X-API-Key: your-api-key"

# 4. Rollback if needed
curl -X POST http://localhost:8000/api/v1/models/version_3/rollback \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"reason": "Accuracy dropped significantly"}'

# 5. Get final training report (after completion)
curl http://localhost:8000/api/v1/training/status \
  -H "X-API-Key: your-api-key"
```

## Dashboard (Web UI)

A web dashboard is available at `http://localhost:8000/` or `http://localhost:8000/dashboard/`.

**Features:**

- ✅ Real-time training status (auto-refreshes every 5 seconds)
- ✅ Iteration counter
- ✅ Current accuracy display
- ✅ Accuracy history chart
- ✅ Rollback controls (rollback to specific version)
- ✅ Model version browser (when list endpoint is implemented)
- ✅ Training completion report
- ✅ Start/Stop training controls

**Access:**

1. Open your browser: `http://localhost:8000/`
2. Enter your API key when prompted (or set it in the JavaScript)
3. Click "Start Training" to begin federated learning
4. Monitor progress in real-time

**Note:** The dashboard uses the REST API endpoints. Make sure your API key is set correctly.

## Monitoring Training Progress

### View Logs

```bash
# All services
docker-compose logs -f

# Main service only
docker-compose logs -f main-service

# Client services only
docker-compose logs -f client-service
```

### Check Queue Status

Access RabbitMQ Management UI: http://localhost:15672

- Username: `admin` (or from `.env`)
- Password: `admin` (or from `.env`)

View queues:

- `train_queue` - TRAIN tasks for clients
- `client_updates` - Client weight updates
- `aggregate_queue` - Aggregation tasks
- `blockchain_write` - Blockchain write tasks
- `storage_write` - Storage tasks
- `validate` - Validation tasks
- `decision` - Decision tasks
- `rollback_queue` - Rollback tasks
- `training_complete_queue` - Completion notifications

## Configuration

Training behavior is controlled via environment variables in `.env`:

```bash
# Training completion criteria
TARGET_ACCURACY=95.0          # Target accuracy to achieve
MAX_ITERATIONS=100            # Maximum training iterations
MAX_ROLLBACKS=5               # Maximum rollbacks before stopping
CONVERGENCE_PATIENCE=10       # Iterations without improvement before convergence

# Rollback thresholds
ACCURACY_TOLERANCE=0.5        # Allowed accuracy drop (0.5%)
PATIENCE_THRESHOLD=3          # Consecutive bad iterations before rollback
SEVERE_DROP_THRESHOLD=2.0     # Immediate rollback threshold (2%)

# Client training
EPOCHS=100                    # Epochs per training iteration
NUM_CLIENTS=2                 # Number of client instances
```

## Troubleshooting

### Training Not Starting

1. Check all services are running: `docker-compose ps`
2. Check API key is set: `echo $API_KEY`
3. Check main service logs: `docker-compose logs main-service`
4. Verify workers are running (check logs for worker startup messages)

### No Client Updates

1. Check client services are running: `docker-compose ps client-service`
2. Check client logs: `docker-compose logs client-service`
3. Verify queue connectivity: Check RabbitMQ Management UI
4. Check if TRAIN tasks are in `train_queue`

### Training Stuck

1. Check training status: `curl http://localhost:8000/api/v1/training/status -H "X-API-Key: your-key"`
2. Check queue status in RabbitMQ Management UI
3. Review logs for errors: `docker-compose logs -f`
4. Check if validation is completing (look for DECISION tasks)

## Next Steps

1. **Start Training**: Use `/api/v1/training/start` endpoint
2. **Monitor Progress**: Poll `/api/v1/training/status` or check logs
3. **View Model Versions**: Use `/api/v1/models` endpoints
4. **Manual Intervention**: Use rollback endpoint if needed
5. **Get Final Report**: Check status after completion
