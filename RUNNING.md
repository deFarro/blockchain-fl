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
# Start all services with 2 client instances (recommended)
docker-compose up -d --scale client-service=2

# Check status
docker-compose ps
```

**Expected services:**

- `blockchain-fl-rabbitmq` - Message queue
- `blockchain-fl-ipfs` - Distributed storage
- `blockchain-fl-blockchain-service` - Blockchain operations
- `blockchain-fl-main-service` - Main aggregator service
- `blockchain-fl-client-service-1`, `blockchain-fl-client-service-2`, ... - Client services (number depends on `--scale` flag)

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

# Client services only (shows all client instances)
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
EPOCHS=10                     # Epochs per training iteration
NUM_CLIENTS=2                 # Number of client instances
```

## Benchmarking and Metrics Collection

The system includes comprehensive benchmarking and metrics collection designed for research purposes. Metrics are automatically collected during training runs and exported to CSV files for analysis.

### Overview

The metrics collection system tracks performance across multiple dimensions:

#### **Performance Metrics**
- **Latency**: Operation-level timing for all critical operations
  - Blockchain writes (register_model_update)
  - IPFS uploads/downloads
  - Aggregation operations
  - Validation operations
  - Training operations
  - Rollback operations

- **Throughput**: 
  - Operations per second
  - Data transfer rates (network I/O)
  - Model updates processed per iteration

#### **System Metrics**
- **CPU Usage**: Total CPU time used per iteration
- **Memory Usage**: Average, min, max memory consumption per iteration
- **Network I/O**: Bytes sent/received, packets sent/received per iteration
- **Disk I/O**: Read/write operations and throughput per iteration

#### **Training Metrics**
- Final accuracy
- Training duration
- Number of iterations
- Number of rollbacks
- Convergence information
- Client participation statistics

#### **Scenario Tracking**
- Blockchain enabled/disabled
- IPFS enabled/disabled
- Dataset name
- Number of clients
- Training configuration (target accuracy, max iterations, etc.)

### Architecture

The metrics system consists of:

1. **MetricsCollector** (`shared/monitoring/metrics.py`)
   - Collects operation-level timing metrics
   - Tracks counters and aggregations
   - Stores detailed timing data with metadata
   - Tracks system metrics per iteration

2. **SystemMetricsCollector** (`shared/monitoring/system_metrics.py`)
   - Collects system-level metrics (CPU, memory, network, disk)
   - Provides summary statistics per iteration

3. **MetricsExporter** (`shared/monitoring/metrics_exporter.py`)
   - Exports metrics to CSV format
   - Flattens nested data structures
   - Handles multiple timing samples
   - Includes per-iteration system metrics

4. **API Endpoint** (`/api/v1/metrics/save`)
   - Receives metrics JSON via HTTP POST
   - Saves to CSV file on local filesystem
   - Returns path to saved file

5. **Training Completion Integration**
   - Automatically collects and exports metrics when training completes
   - Includes training summary and final results

### Usage

#### Starting Training with Scenario Tracking

When starting training via the API, scenario information is automatically captured:

```bash
curl -X POST "http://localhost:8000/api/v1/training/start" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_name": "mnist",
    "initial_weights_cid": null
  }'
```

The system will automatically:
1. Set scenario information (blockchain/IPFS status, dataset, etc.)
2. Reset system metrics collection
3. Begin collecting metrics throughout training

#### Manual Metrics Export

You can manually export metrics at any time:

```bash
curl -X POST "http://localhost:8000/api/v1/metrics/save" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "metrics": {
      "scenario_info": {
        "blockchain_enabled": true,
        "ipfs_enabled": true,
        "dataset_name": "mnist",
        "num_clients": 5
      },
      "operation_metrics": {
        "blockchain_register": {
          "count": 10,
          "total_duration": 5.2,
          "avg_duration": 0.52,
          "min_duration": 0.3,
          "max_duration": 1.1
        }
      },
      "system_metrics": {
        "cpu": {"total_time_seconds": 12.5},
        "memory": {"avg_used_bytes": 524288000, "max_used_bytes": 629145600}
      }
    },
    "filename": "my_metrics.csv"
  }'
```

#### Automatic Export on Training Completion

When training completes successfully, metrics are automatically exported:

1. All collected metrics are gathered
2. Training completion information is added
3. Metrics are exported via the `/save_metrics` endpoint
4. CSV file is saved with timestamp and scenario info in filename

Example filename: `metrics_20240115_143022_bc_on_ipfs_on.csv`

### CSV Format

The exported CSV file contains flattened metrics with columns like:

- `scenario_blockchain_enabled` (true/false)
- `scenario_ipfs_enabled` (true/false)
- `scenario_dataset_name` (e.g., "mnist")
- `scenario_num_clients` (integer)
- `system_cpu_total_time_seconds` (float) - per iteration
- `system_memory_avg_used_bytes` (float) - per iteration
- `system_memory_max_used_bytes` (float) - per iteration
- `system_memory_min_used_bytes` (float) - per iteration
- `system_network_total_bytes_sent` (integer) - per iteration
- `system_network_total_bytes_recv` (integer) - per iteration
- `system_disk_total_bytes_read` (integer) - per iteration
- `system_disk_total_bytes_written` (integer) - per iteration
- `op_blockchain_register_count` (integer)
- `op_blockchain_register_avg_duration` (float)
- `op_blockchain_register_min_duration` (float)
- `op_blockchain_register_max_duration` (float)
- `timing_sample_index` (integer) - iteration index
- `timing_blockchain_register_duration` (float) - individual timing samples
- `timing_blockchain_register_model_version_id` (string) - metadata
- `timing_blockchain_register_iteration` (integer) - iteration number
- `training_completion_final_accuracy` (float)
- `training_completion_total_iterations` (integer)
- `training_completion_rollback_count` (integer)
- ... and more

**Note:** System metrics (CPU, memory, network, disk) are tracked per iteration, so each row in the CSV represents one iteration with its corresponding hardware metrics.

### Research Scenarios

#### Scenario 1: Full System (Blockchain + IPFS)
- `blockchain_enabled: true`
- `ipfs_enabled: true`
- Measures overhead of full decentralized system

#### Scenario 2: IPFS Only (No Blockchain)
- `blockchain_enabled: false`
- `ipfs_enabled: true`
- Measures IPFS overhead without blockchain

#### Scenario 3: Blockchain Only (No IPFS)
- `blockchain_enabled: true`
- `ipfs_enabled: false`
- Measures blockchain overhead without IPFS

#### Scenario 4: Baseline (No Blockchain, No IPFS)
- `blockchain_enabled: false`
- `ipfs_enabled: false`
- Baseline performance for comparison

### Metrics Collected Per Operation

#### Blockchain Operations
- `blockchain_register`: Model version registration
  - Duration per operation
  - Transaction IDs
  - Model version IDs
  - Iteration numbers

#### IPFS Operations
- `ipfs_upload`: Model weight/diff uploads
  - Duration per upload
  - File sizes
  - CID returned
- `ipfs_download`: Model weight/diff downloads
  - Duration per download
  - File sizes
  - CID requested

#### Aggregation Operations
- `fedavg_aggregation`: Client update aggregation
  - Duration per aggregation
  - Number of clients aggregated
  - Iteration number

#### Validation Operations
- `validation`: Model validation
  - Duration per validation
  - Accuracy results
  - Model version ID

#### Training Operations
- `training`: Client training
  - Duration per training round
  - Epochs trained
  - Samples processed

### Output Directory

By default, CSV files are saved to `./metrics_output/` directory. You can specify a custom directory via the `output_dir` parameter in the `/save_metrics` endpoint.

### Integration Points

#### Training Start
- Scenario info is set automatically
- System metrics collection is reset
- Metrics collection begins

#### During Training
- All operations record timing metrics
- System metrics are sampled after each operation with iteration metadata
- Operation metadata is stored
- System metrics are tracked per iteration

#### Training Completion
- Final metrics are collected
- Training summary is added
- Metrics are exported to CSV automatically

### CSV Analysis Example

```python
import pandas as pd

# Load metrics
df = pd.read_csv('metrics_20240115_143022_bc_on_ipfs_on.csv')

# Compare blockchain operations across iterations
blockchain_ops = df[df['timing_blockchain_register_duration'].notna()]
avg_duration = blockchain_ops['timing_blockchain_register_duration'].mean()

# Compare system resources per iteration
cpu_usage = df['system_cpu_total_time_seconds'].mean()
memory_usage = df['system_memory_avg_used_bytes'].mean()

# Compare scenarios
scenarios = pd.read_csv('all_metrics.csv')
scenarios.groupby(['scenario_blockchain_enabled', 'scenario_ipfs_enabled']).agg({
    'op_blockchain_register_avg_duration': 'mean',
    'system_cpu_total_time_seconds': 'mean',
    'system_memory_avg_used_bytes': 'mean',
    'training_completion_final_accuracy': 'mean'
})

# Analyze per-iteration metrics
iteration_metrics = df.groupby('timing_sample_index').agg({
    'system_cpu_total_time_seconds': 'first',
    'system_memory_avg_used_bytes': 'first',
    'timing_blockchain_register_duration': 'mean'
})
```

### Notes

- Metrics are collected in-memory during training
- CSV export happens at training completion or manually
- System metrics are sampled after operations with iteration metadata
- All timings are in seconds
- File paths are absolute paths on the host system
- The system handles missing dependencies gracefully (logs warnings)
- System metrics are tracked per iteration, ensuring each CSV row has iteration-specific hardware metrics

## Troubleshooting

### Training Not Starting

1. Check all services are running: `docker-compose ps`
2. Check API key is set: `echo $API_KEY`
3. Check main service logs: `docker-compose logs main-service`
4. Verify workers are running (check logs for worker startup messages)

### No Client Updates

1. Check client services are running: `docker-compose ps | grep client-service`
2. Verify you have multiple clients if needed: `docker-compose up -d --scale client-service=2`
3. Check client logs: `docker-compose logs client-service`
4. Verify queue connectivity: Check RabbitMQ Management UI
5. Check if TRAIN tasks are in `train_queue`

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
6. **Analyze Metrics**: Review exported CSV files in `./metrics_output/` directory for performance analysis