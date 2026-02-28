# Setup and Running

This guide walks through: **install** → **configure** → **run** → **supported scenarios**.

---

## 1. Install

### Prerequisites

- **Python** 3.12 or 3.10–3.13
- **Docker** and **Docker Compose**
- **Git**
- **Go 1.21+** (optional; only for developing the blockchain service)

### Option A: Docker (recommended)

No extra install. Clone the repo and use Docker for all services (see **Run the application** below).

### Option B: Local development

To run the main and client services on the host (with infrastructure still in Docker):

```bash
python3.12 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install --upgrade pip
pip install -r main_service/requirements.txt
pip install -r client_service/requirements.txt
pip install -r requirements.txt
```

Start only infrastructure in Docker: `docker-compose up -d rabbitmq ipfs` (and optionally `blockchain-service` or run it locally in Go).

---

## 2. Configure

1. **Copy environment template**

   ```bash
   cp env_template.txt .env
   ```

2. **Required: encryption key and API key**
   - **ENCRYPTION_KEY** — Generate and add to `.env`:
     ```bash
     python scripts/generate_encryption_key.py
     # Put the output in .env as ENCRYPTION_KEY=...
     ```
   - **API_KEY** — Set a secret value in `.env` (used for dashboard and API):
     ```bash
     API_KEY=your-secret-api-key-here
     ```

3. **Optional: training and behaviour**
   - Training: `TARGET_ACCURACY`, `MAX_ITERATIONS`, `MAX_ROLLBACKS`, `CONVERGENCE_PATIENCE`, `ACCURACY_TOLERANCE`, `PATIENCE_THRESHOLD`, `EPOCHS`, `NUM_CLIENTS`
   - Change default RabbitMQ and other passwords in production.

4. **Local development**

   If running main/client locally, set in `.env`:

   ```bash
   RABBITMQ_HOST=localhost
   IPFS_HOST=localhost
   BLOCKCHAIN_SERVICE_URL=http://localhost:8080
   ```

5. **Blockchain service**

   Default is **development mode**: local hash-linked chain, no Hyperledger Fabric required. Optional Fabric vars: `FABRIC_NETWORK_PROFILE`, `FABRIC_WALLET_PATH`, `FABRIC_CHANNEL_NAME`, `FABRIC_CHAINCODE_NAME`, `FABRIC_USER`.

---

## 3. Run the application

1. **Start infrastructure and services**

   ```bash
   docker-compose up -d rabbitmq ipfs blockchain-service
   docker-compose up -d --scale client-service=2
   ```

   Or start everything at once: `docker-compose up -d --scale client-service=2`

2. **Verify**

   ```bash
   docker-compose ps
   docker-compose logs -f   # or per service: main-service, client-service, etc.
   ```

3. **Access**

   | What       | URL                        |
   | ---------- | -------------------------- |
   | Dashboard  | http://localhost:8000/     |
   | API docs   | http://localhost:8000/docs |
   | RabbitMQ   | http://localhost:15672     |
   | IPFS API   | http://localhost:5001      |
   | Blockchain | http://localhost:8080      |

4. **First run**
   - Open http://localhost:8000/
   - Enter the API key (from `.env`)
   - Click **Start Training** to begin federated learning

**Scaling clients:** `docker-compose up -d --scale client-service=N`. Each client gets a unique `instance_id` automatically (no manual client ID config).

**Dashboard dev (hot reload):** `cd dashboard && npm install && npm run dev` — dev server at http://localhost:3000 (proxies API to 8000).

---

## 4. Supported scenarios

### Normal training

- **Start:** Dashboard “Start Training” or `POST /api/v1/training/start` with `X-API-Key`.
- **Monitor:** Dashboard (auto-refresh) or `GET /api/v1/training/status`.
- **Rollback:** Dashboard “Rollback to Version” or `POST /api/v1/models/{version_id}/rollback` with `reason`.
- **Models:** `GET /api/v1/models`, `GET /api/v1/models/{version_id}`, `GET /api/v1/models/{version_id}/provenance`.

### Unreliable client (rollback and exclusion)

- In `.env`: `ADD_UNRELIABLE_CLIENT=true`
- Start: `make up` (or docker compose with `unreliable` profile).
- One extra client submits bad updates; the system can roll back and exclude it via regression diagnosis.

### Benchmarking and metrics (research)

- Metrics are collected during training; export to CSV on completion or via `POST /api/v1/metrics/save`.
- **Scenarios:** Full (blockchain + IPFS), IPFS only, blockchain only, baseline (neither). Scenario info is stored with metrics.
- CSVs are written to `./metrics_output/` by default (configurable). Columns include scenario flags, per-operation timings, system metrics per iteration, and training completion stats.

---

## API reference (summary)

All endpoints except `/health` require header: `X-API-Key: <your API_KEY>`.

| Method | Endpoint                                 | Purpose                                                                           |
| ------ | ---------------------------------------- | --------------------------------------------------------------------------------- |
| POST   | `/api/v1/training/start`                 | Start federated training (body: optional `initial_weights_cid`, `num_iterations`) |
| GET    | `/api/v1/training/status`                | Current iteration, best accuracy, rollback count, status                          |
| POST   | `/api/v1/training/stop`                  | Stop training (if implemented)                                                    |
| GET    | `/api/v1/models`                         | List model versions                                                               |
| GET    | `/api/v1/models/{version_id}`            | Model version details                                                             |
| GET    | `/api/v1/models/{version_id}/provenance` | Provenance chain for a version                                                    |
| POST   | `/api/v1/models/{version_id}/rollback`   | Manual rollback (body: `reason`)                                                  |
| POST   | `/api/v1/metrics/save`                   | Export metrics to CSV (body: metrics JSON, optional `filename`)                   |

Full request/response examples: open http://localhost:8000/docs (Swagger UI).

---

## Dashboard

- **URL:** http://localhost:8000/ (served by main service).
- **Features:** Start/stop training, live status, accuracy chart, model versions, rollback, provenance view, link to API docs.
- **Build (production):** `cd dashboard && npm run build` → `dist/` is served by main service.

---

## Configuration (env summary)

Sensitive and common settings live in `.env`:

- **RabbitMQ:** `RABBITMQ_HOST`, `RABBITMQ_USER`, `RABBITMQ_PASSWORD`
- **Security:** `ENCRYPTION_KEY` (or `ENCRYPTION_KEY_FILE`), `API_KEY`
- **Services:** `BLOCKCHAIN_SERVICE_URL`, `IPFS_HOST`, ports
- **Training:** `TARGET_ACCURACY`, `MAX_ITERATIONS`, `MAX_ROLLBACKS`, `CONVERGENCE_PATIENCE`, `ACCURACY_TOLERANCE`, `PATIENCE_THRESHOLD`, `EPOCHS`, `NUM_CLIENTS`
- **Scenarios:** `ADD_UNRELIABLE_CLIENT` for the unreliable client profile

---

## Troubleshooting

- **Training not starting:** Ensure all services are up (`docker-compose ps`), `API_KEY` set, main-service logs show workers started.
- **No client updates:** Check client containers are running and scaled (e.g. `--scale client-service=2`). Check RabbitMQ (http://localhost:15672) for `train_queue` and `client_updates`.
- **Training stuck:** Check `GET /api/v1/training/status`, queue backlogs, and logs (`docker-compose logs -f main-service`); ensure validation/decision workers are processing.
- **Blockchain service:** "Running in development mode" is normal; no Fabric needed. Port in use → set `BLOCKCHAIN_SERVICE_PORT` in `.env`. Go deps → `cd blockchain_service && go mod download`.
- **Python/local:** Activate venv and install deps; port 8000 free; IPFS/RabbitMQ from Docker.
- **Queue:** Connection refused → start RabbitMQ. Auth failed → check `RABBITMQ_USER`/`RABBITMQ_PASSWORD`. Run queue test: `pytest tests/test_shared/test_queue.py -v` (with RabbitMQ up).
- **Missing encryption key:** Run `python scripts/generate_encryption_key.py` and set `ENCRYPTION_KEY` in `.env`.

---

## Project structure

```
blockchain-fl/
├── main_service/       # Aggregator (Python): workers, API, dashboard backend
├── blockchain_service/ # Blockchain microservice (Go): Fabric / local chain
├── client_service/     # Training service (Python), scale with --scale
├── shared/             # Models, utils, datasets, monitoring
├── dashboard/          # React UI (Vite, Tailwind)
├── scripts/            # e.g. generate_encryption_key.py, prepare_datasets.py
├── tests/
├── docker-compose.yml
├── env_template.txt
└── .env                # Your config (create from template)
```
