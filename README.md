# Blockchain-Enabled Learning Systems for Verifiable Model Provenance

> **MSc Thesis Project** — **Topic:** Blockchain-Enabled Learning Systems for Verifiable Model Provenance, Rollback, and Regulatory Auditability.

## Quick Start

1. **Setup:** `cp env_template.txt .env` and set `ENCRYPTION_KEY` and `API_KEY`
2. **Start:** `docker-compose up -d`
3. **Access Dashboard:** Open `http://localhost:8000/`
4. **Start Training:** Click "Start Training" in the dashboard

For detailed instructions, see [SETUP.md](SETUP.md).

## Contents

- [Project Overview](#project-overview) · [Research Questions](#research-questions)
- [System Architecture](#system-architecture): [Components](#system-components), [Data Flow](#data-flow-architecture), [Processing Pipeline](#processing-pipeline), [Rollback](#rollback-strategy), [Manual Rollback](#manual-rollback), [Post-Rollback](#post-rollback-workflow), [Training Completion](#training-completion-criteria), [Regression Diagnosis](#regression-diagnosis)
- [Data Structures](#data-structures) · [Language (Go vs Python)](#language-distribution-go-vs-python) · [Smart Contract](#smart-contract-design-hyperledger-fabric) · [IPFS](#ipfs-storage) · [Diff Storage](#diff-storage-encrypted-diff--blockchain-hash)
- [Dataset Separation](#dataset-separation-training-vs-test) · [Dataset Splitting](#dataset-splitting-eg-mnist) · [Security](#security-considerations) · [Queue](#queue-design)
- [Performance](#performance-rq3) · [Tech Stack & Structure](#technology-stack--project-structure) · [Research Questions](#addressing-research-questions) · [Architecture Notes](#architecture-notes-and-implementation-priorities) · [Testing](#testing-strategy)

## Project Overview

This repository contains the prototype implementation for a Master's thesis investigating the convergence of **Distributed Ledger Technology (DLT)** and **Federated Learning (FL)**.

The project aims to address the opacity of AI development by designing a blockchain-enabled provenance system. The system records training metadata, model updates, and ownership changes on an immutable ledger to support trust, reproducibility, and compliance with emerging regulations such as the **EU AI Act**.

### Key Capabilities

- **Immutable Audit Trails:** Records training events and metadata on a permissioned blockchain
- **Model Rollback:** Enables restoration of previous model states to mitigate poisoning attacks or errors
- **Hybrid Storage:** "Selective Decentralisation" — critical metadata on-chain, heavy model artefacts off-chain (IPFS)
- **Privacy-Preserving:** Integrates with Federated Learning; clients train on local data only

## Research Questions

1. **RQ1 (Architecture):** How can blockchain be integrated into learning systems to enable verifiable model provenance, rollback, and auditability while maintaining efficiency?
2. **RQ2 (Trust & Auditability):** To what extent can blockchain-based provenance improve traceability and reproducibility of ML models vs conventional systems?
3. **RQ3 (Performance):** What are the performance implications (latency, overhead, storage) of incorporating blockchain into the training process?

---

# System Architecture

This section outlines the architecture for the blockchain-enabled federated learning system, including design decisions, component roles, data flow, and implementation notes.

## System Components

### 1. Main Service (Aggregator Service)

**Responsibilities:**

- Model weight aggregation (FedAvg)
- Validation using test datasets (test datasets stored only in main service)
- Blockchain transaction coordination (calls blockchain-service)
- Off-chain storage management (IPFS)
- Task queue management
- UI/API for manual intervention

**Important:** Main service is the only component with access to test/validation datasets. Clients only have training datasets.

**Components:**

- **Queue Manager:** Manages task distribution (RabbitMQ)
- **Aggregation Worker:** Combines client updates using federated averaging
- **Blockchain Client:** HTTP client that calls blockchain-service API
- **Storage Worker:** Handles encryption/decryption and IPFS storage
- **Validation Worker:** Runs test datasets and evaluates model performance
- **Rollback Worker:** Manages model state restoration
- **API Server:** REST API for manual task creation, monitoring, and manual rollback
- **Web UI:** Dashboard for monitoring, manual intervention, and rollback management

### 2. Blockchain Service (Go Microservice)

**Responsibilities:**

- All Hyperledger Fabric operations
- Blockchain transaction creation and submission
- Chaincode invocation using official Fabric Go SDK
- Provenance chain management
- Validation and rollback event recording

**Architecture:** Separate Go microservice (isolated from Python); REST API for blockchain operations; can be developed and deployed independently.

**API Endpoints:**

- `POST /api/v1/model/register` — Register model version
- `POST /api/v1/model/validate` — Record validation results
- `POST /api/v1/model/rollback` — Record rollback event
- `GET /api/v1/model/provenance/{version_id}` — Get provenance chain
- `GET /health` — Health check

### 3. Client Service (Training Service)

**Responsibilities:**

- Local model training on **training datasets only** (no test/validation data)
- Weight update computation (diffs or full weights)
- Task consumption from queue
- Scalable deployment (multiple instances with different training datasets)

**Important:** Clients never see test/validation datasets; they only train on their local training data.

**Components:**

- **Training Engine:** PyTorch model training
- **Queue Consumer:** Reads training tasks from queue
- **Update Publisher:** Publishes weight updates to queue
- **Config Manager:** Manages dataset paths, model architecture, hyperparameters

## Data Flow Architecture

### Training Iteration Flow

Clients send weight updates (encrypted diffs) to a queue. Multiple clients can contribute simultaneously. The pipeline runs: aggregation → blockchain → storage → validation → decision.

**Iteration coordination:** Clients are passive (train only when they receive TRAIN tasks). Main service publishes TRAIN tasks only after validation passes. Late updates for past iterations are rejected (logged and ignored).

### Processing Pipeline

**Step 1: Aggregation Worker** — Reads multiple client updates from queue; applies FedAvg aggregation; publishes aggregated update to queue.

**Step 2: Blockchain Worker** — Reads aggregated update; computes hash of encrypted diff; calls blockchain-service API to create blockchain transaction; blockchain service uses Fabric Go SDK to invoke chaincode; stores version hash, parent hash, timestamp, metadata on-chain; publishes task with blockchain hash to queue.

**Step 3: Storage Worker** — Reads task with blockchain hash; encrypts aggregated diff (AES-256); stores encrypted diff on IPFS (gets CID); verifies hash of encrypted diff matches blockchain hash; pins CID for persistence; publishes task with IPFS CID to queue.

**Step 4: Validation Worker** — Reads task with IPFS CID; retrieves encrypted diff from IPFS; decrypts and loads model weights; applies diff to previous weights; runs test dataset validation (test data stored only in main service); calls blockchain-service API to record validation results on blockchain; publishes validation result to queue.

**Step 5: Decision Logic** — Reads validation result; evaluates model performance using rollback strategy. If rollback needed: rollback entire iteration or run regression diagnosis to exclude problematic client(s). If PASS: checks if training is complete (accuracy threshold, max iterations, manual trigger, etc.); if not complete, publishes TRAIN tasks for next iteration (with new aggregated weights CID); if complete, publishes TRAINING_COMPLETE with final model information. Updates model registry.

**Step 6: Training Completion Task** — Contains final model version ID, final accuracy and metrics, IPFS CID of final weights, training summary (iterations, clients, duration), metadata. Consumed by API/UI; final state recorded on blockchain.

### Rollback Strategy

Validation uses test-set accuracy (and optionally loss, precision, recall). **Tolerance + patience:** allow small accuracy drops (e.g. 0.5–1%) and a number of consecutive “bad” iterations (e.g. 3–5) before rollback. **Best checkpoint** is the last version with best accuracy; rollback targets this. **Scenarios:** Immediate rollback if drop > tolerance; patience-based if below best for N iterations; no rollback if acceptable (reset patience, update best). **Rollback depth:** To last safe checkpoint; intermediate versions discarded. **Validation criteria (summary):** (1) Accuracy comparison with previous best. (2) Tolerance threshold for small drops. (3) Patience counter for consecutive below-best iterations. (4) Best checkpoint as rollback target.

**Configuration parameters:** `accuracy_tolerance` (max allowed accuracy drop before concern), `patience_threshold` (consecutive bad iterations before rollback), `min_accuracy_threshold` (absolute minimum for immediate rollback), `metrics_to_track` (e.g. accuracy, loss, precision, recall).

**Example flow:**

```
Iteration 1: Accuracy = 90% → Best = 90%, Patience = 0, Checkpoint = v1.0
Iteration 2: Accuracy = 91% → Best = 91%, Patience = 0, Checkpoint = v1.1 (new best)
Iteration 3: Accuracy = 90.5% → Best = 91%, Patience = 1 (within tolerance, below best)
Iteration 4: Accuracy = 90.3% → Best = 91%, Patience = 2
Iteration 5: Accuracy = 90.2% → Best = 91%, Patience = 3 (threshold reached)
→ ROLLBACK to v1.1 (last best checkpoint)
```

**Benefits:** Prevents model degradation from accumulating; allows temporary accuracy dips without premature rollback; maintains best model state; configurable for different use cases.

### Manual Rollback

Use cases: suspected poisoning, security incidents, data quality issues, operational errors, research. **Access:** API `POST /api/v1/models/{version_id}/rollback` (API key) or Web UI (version history, select target, reason). **Request:** `target_version_id`, `reason`, `triggered_by`, optional `priority`. **Execution:** Manual rollback creates ROLLBACK task. Rollback worker: (1) Validates target version exists and is accessible. (2) Retrieves target version weights from IPFS. (3) Updates blockchain via RollbackModel (from/to version, reason, triggered_by, type "manual"). (4) Records reason, triggered_by, timestamp on-chain. (5) Updates current model pointer to target version. (6) Publishes new TRAIN task to resume from rolled-back state. All actions logged on blockchain. **Version selection (UI):** Dashboard shows model version history (version ID, timestamp, accuracy, validation status, client IDs, IPFS CID); user selects target version and enters reason; system validates version is accessible.

**Blockchain recording:** RollbackModel records from_version_id, to_version_id, reason, triggered_by, timestamp, type "manual". Manual rollback can be triggered anytime; takes precedence over automatic; after it, automatic logic resets. **Example scenarios:** Suspected poisoning → roll back to version before suspicious update. Security incident → roll back to last known good state.

**Benefits of manual rollback:** Rapid response to poisoning or attacks; human oversight; complete audit trail; operational flexibility; supports regulatory compliance.

Manual takes precedence over automatic; after it, patience and best checkpoint reset.

### Post-Rollback Workflow

(1) Rollback worker retrieves target weights from IPFS, updates blockchain (current pointer = rolled-back version), records rollback event, publishes ROLLBACK to queue. (2) All clients receive ROLLBACK task (target version ID, IPFS CID); they download weights, replace local state, discard changes after cutoff, acknowledge. (3) Main service publishes new TRAIN with rolled-back weights CID; all clients continue from same state (including those who contributed after cutoff — their contributions remain on-chain for audit but are excluded from current model). (4) Best checkpoint and patience reset.

**Handling clients that contributed after cutoff:** All clients (including those who contributed updates after the rolled-back version) receive the same ROLLBACK task, load rolled-back weights from IPFS, and discard local state after the cutoff. Their previous contributions remain on blockchain for audit but are excluded from the current model. No special treatment; all clients resume from the same state.

**Rollback state reset:** After rollback, automatic rollback logic resets: best checkpoint is updated to the rolled-back version, patience counter resets to 0, accuracy tracking restarts from the rolled-back version, training continues with fresh rollback state.

### Training Completion Criteria

Training completes when **any** of: (1) **Accuracy threshold** reached (e.g. 95%). (2) **Convergence** — no improvement for N iterations (e.g. 10). (3) **Max iterations** (e.g. 100). (4) **Max rollbacks** exceeded — best model marked final. (5) **Manual trigger** via API/UI. **Overtraining:** Track validation vs training accuracy; early stopping if validation plateaus or train/val gap exceeds threshold. **On completion:** TRAINING_COMPLETE published (final version ID, metrics, IPFS CID, summary, completion reason); final state on blockchain; no new TRAIN tasks. **Configuration parameters:** `target_accuracy` (minimum accuracy to achieve), `convergence_patience` (iterations without improvement before convergence), `max_iterations` (maximum training rounds), `max_rollbacks` (maximum rollbacks before stopping), `overtraining_threshold` (validation accuracy gap threshold), `early_stopping_enabled`.

**When any completion criterion is met:** (1) Decision worker detects completion. (2) Publishes TRAINING_COMPLETE to queue (final version ID, accuracy/metrics, IPFS CID, summary, completion reason). (3) Final state recorded on blockchain. (4) API/UI notified that model is ready. (5) Training stops; no new TRAIN tasks published.

**Example completion scenarios:** (1) Accuracy threshold: target 95% reached → "Accuracy threshold reached". (2) Convergence: no improvement for 10 iterations → "Convergence detected". (3) Max rollbacks: 5 → "Rollback limit reached - investigation recommended". (4) Max iterations: 100 → "Maximum iterations reached".

**Benefits:** Prevents infinite training; ensures model reaches acceptable performance or stops when not improving; clear completion criteria for research; manual trigger for operational needs; overtraining prevention via early stopping.

### Regression Diagnosis

When regression is detected, the system can test each client’s diff individually on the previous model + test set to identify which client(s) caused the drop. (1) Normal aggregation first. (2) For each contributing client, load previous weights, apply only that client's diff, validate on test set. (3) Add identified clients to `excluded_clients` (runtime); future aggregations filter them out. **Example:** Clients 1,2,3 → 95%→92%; test client 1 alone → 94.5%, client 2 → 94.8%, client 3 → 88% → exclude client 3.

**Key point:** Normal aggregation first includes all client diffs. When regression is detected, diagnosis runs and identifies which client(s) caused it; those clients are then excluded from future aggregations. Exclusion is applied automatically at rollback time; no configuration needed.

**Example (code):**

```
Iteration 24: Clients 1, 2, 3 → Regression detected (95% → 92%)
Diagnosis:
  Test Client 1 diff alone → 94.5% (OK)
  Test Client 2 diff alone → 94.8% (OK)
  Test Client 3 diff alone → 88.0% (REGRESSION!)
Result: Exclude Client 3 from future iterations
```

**Limitations:** Currently tests each client individually (future: test combinations); requires previous weights and test dataset; no automatic re-inclusion (manual re-enable if needed). **Configuration:** Unreliable clients identified at rollback and excluded automatically; no env vars required. See `docs/REGRESSION_DIAGNOSIS.md`.

## Data Structures

### On-Chain Storage (Smart Contract State)

**ModelVersion:** version_id, parent_version_id, timestamp, client_ids, aggregated_hash (SHA-256 of aggregated weights), diff_hash (SHA-256 of encrypted diff), off_chain_location (IPFS CID), hyperparameters (JSON), validation_status (pending/passed/failed), validation_metrics (accuracy, loss, etc.), block_number.

**Validation records:** Version ID, validator ID, passed status, metrics (JSON), test dataset hash, timestamp.

**Rollback events:** From version ID, to version ID, reason, triggered_by (validator/user or "automatic"), timestamp, type ("automatic" or "manual").

**Iteration:** The iteration number stored on-chain is the current training iteration when the version is stored; enables late-update rejection (update iteration < on-chain current iteration), iteration coordination across services, and replay prevention.

### Off-Chain Storage

**Encrypted diff format:** encrypted_diff (AES-256 encrypted weight diff bytes), encryption_key_hash (for key management), optional compression (gzip), format (serialization, e.g. JSON). **Storage location:** IPFS; content-addressed (CID); pin for persistence; files referenced by Content Identifier.

## Language Distribution: Go vs Python

### Go (Blockchain Service)

**Location:** `blockchain_service/`.

**Why Go:** Hyperledger Fabric has an official, well-maintained Go SDK; avoids dependency conflicts with Python packages; clean separation of concerns (blockchain logic isolated); better performance for blockchain operations.

**What it does:** Provides REST API for blockchain operations; interacts with Hyperledger Fabric using the official Go SDK; implements RegisterModelUpdate, RecordValidation, RollbackModel, GetModelProvenance; manages Fabric network connections and transactions; can be developed and deployed independently.

**Architecture:** Runs as separate Docker container (port 8080); main service calls it via HTTP; chaincode in `blockchain_service/chaincode/model_provenance.go` (Go; runs in Fabric-managed container).

### Python (Main and Client)

**Main service (~70%):** Workers (aggregation, storage, validation, rollback), FastAPI server, queue consumers/producers, blockchain HTTP client, IPFS, encryption, validation logic, Web UI backend.

**Client service (~20%):** Training engine (PyTorch), model definitions, queue consumers, weight computation.

**Shared (~10%):** Task models, utils, config, dataset interfaces.

### Communication

Main service ↔ blockchain service: HTTP REST (e.g. httpx; no direct Fabric SDK in Python). Blockchain service ↔ Fabric: official Go SDK.

**Benefits of this architecture:** No dependency conflicts (Go service isolated); uses official, maintained Fabric SDK; clean separation of concerns; blockchain service can be developed and tested independently; main service stays focused on ML workloads.

## Smart Contract Design (Hyperledger Fabric)

### Chaincode Functions

- **RegisterModelUpdate:** Records new model version with version_id, parent_version_id, aggregated hash, diff hash, metadata. Stores ModelVersion in blockchain state; creates immutable record.
- **RecordValidation:** Records validation results (version_id, pass/fail status, metrics). Updates ValidationStatus; creates ValidationRecord.
- **RollbackModel:** Executes rollback from one version to another with reason and triggered_by. Validates version exists; creates RollbackEvent; updates current model pointer.
- **GetModelProvenance:** Queries complete lineage chain for a version_id. Returns all parent versions and validation history.
- **VerifyIntegrity:** Verifies integrity by comparing provided hash with stored hash. Returns true if match, false otherwise.

### What Is Stored On-Chain

**Model version metadata:** version_id, parent_version_id, timestamp, iteration (current training iteration for coordination and late-update rejection), num_clients, client_ids, aggregated_hash, diff_hash, off_chain_location (IPFS CID), hyperparameters (JSON), validation_status, validation_metrics (JSON), block_number.

**Validation records:** version_id, validator_id, passed status, metrics (JSON), test_dataset_hash, timestamp.

**Rollback events:** from_version_id, to_version_id, reason, triggered_by, timestamp, type ("automatic" or "manual").

**Not stored on-chain:** model weights, weight diffs, raw training data, full binaries (size and cost prohibitive).

### Benefits of Blockchain Here

**Immutable audit trail (RQ2):** Every version permanently recorded; regulatory compliance (e.g. EU AI Act). **Provenance (RQ2):** Full lineage (v1→v2→v3→rollback to v2); reproducibility. **Integrity (RQ1):** Hash of encrypted diff on-chain; verify off-chain data untampered. **Decentralized trust (RQ1):** Multiple parties can verify; auditors need not trust central authority. **Automation (RQ1):** Rollback and validation rules in code; access control via permissions.

## IPFS Storage

### Setup and Operations

**Local IPFS:** Run `ipfs daemon`; API typically http://localhost:5001. **Storage operations:** (1) Encrypt diff (AES-256-GCM). (2) Upload encrypted diff to IPFS → get CID. (3) Pin CID for persistence. (4) Store CID in blockchain metadata. (5) Verify hash of encrypted diff matches blockchain hash.

**Retrieval operations:** (1) Get CID from blockchain metadata. (2) Retrieve encrypted diff from IPFS. (3) Verify integrity (hash comparison with on-chain hash). (4) Decrypt diff. (5) Apply to model weights.

Python uses HTTP API (e.g. httpx) to IPFS daemon (default http://localhost:5001).

### Benefits

Content-addressed (CID = content hash); decentralized; immutable (new content = new CID); open source; local node for development.

## Diff Storage: Encrypted Diff + Blockchain Hash

### Why Not JWT

JWT is for auth; payloads are base64, not encrypted; size limits unsuitable for weight diffs; verification is signature-based, not blockchain-hash.

### Recommended Approach

**Process:** Client computes weight diff → encrypt (AES-256-GCM) → hash encrypted bytes (SHA-256) → store hash on blockchain, encrypted diff on IPFS (pin CID). **Verification process:** (1) Get IPFS CID from blockchain metadata. (2) Retrieve encrypted diff from IPFS. (3) Compute hash of encrypted diff. (4) Query blockchain for stored hash. (5) Compare hashes; if match, decrypt and use diff; if mismatch, integrity violation.

**Benefits:** Blockchain hash provides integrity (like a signature); encryption provides confidentiality; no size limitations for diffs; off-chain storage efficient for large payloads; aligns with proposal's hybrid storage approach.

## Dataset Separation: Training vs Test

### Principle

Global test/validation datasets exist **only** in the main service. Clients have **training data only** and never see the global test set. Training only requires training data (gradients from training loss); test data is for evaluation. Clients may optionally split their local training data into train/val for early stopping or local monitoring — that subset remains part of their training data, not the global test set.

**What clients have:** Training datasets; local data for training; different clients can have different training data. **What main service has:** Test/validation datasets only; used for evaluation, never for training.

### How Clients Train Without Test Data

Training only requires training data: gradients are computed from training loss on training batches; weight updates come from those gradients. Test data is used for evaluation after updates, not for gradient computation. This is standard in ML (train/test split). Clients may optionally split their own training data (e.g. 80/20) for local validation, early stopping, or monitoring — that subset is still part of their training data, not the global test set.

### Complete Training Iteration (Data Perspective)

**Step 1 — Main service → clients:** Task: "Train with current weights"; payload: weights_location (IPFS CID).

**Step 2 — Clients (training):** Load local training dataset and weights from IPFS; train model; compute diff; publish diff to queue.

**Step 3 — Main service (aggregation):** Collect diffs from multiple clients; aggregate (FedAvg); publish aggregated diff.

**Step 4 — Main service (validation):** Load test dataset (main service only); apply aggregated diff to model; evaluate on test data; record result on blockchain.

**Step 5 — Decision:** Evaluate validation results using rollback strategy. If accuracy acceptable: continue training (publish next TRAIN) or complete (TRAINING_COMPLETE). If rollback needed: rollback to last best checkpoint.

### Benefits of This Separation

- **Prevents data leakage:** Clients cannot use test data for training; test data remains independent; critical for fair evaluation.
- **Supports federated learning:** Each client trains on their own data; no central training dataset; true federated scenario.
- **Enables reproducibility (RQ2):** Test dataset is fixed and versioned; same test data for all validations; reproducible evaluation.
- **Regulatory compliance:** Clear separation of training vs evaluation data; audit trail shows which data was used when; supports EU AI Act requirements.

### Test Dataset Versioning

Version and hash test datasets; record test dataset hash and version in RecordValidation on-chain so reproducibility and auditability are clear.

## Dataset Splitting (e.g. MNIST)

**MNIST:** 60,000 training samples, 10,000 test. **Split configuration:** e.g. 2 clients → 30,000 training samples each; 4 clients → 15,000 each; main service holds full test set (10,000), never split. Dataset-agnostic design allows other datasets (CIFAR-10, custom) via the same interface.

**IID (default):** Shuffle full training set, split evenly into N parts. Each client gets a random sample with all classes. Pros: simpler, good for baseline. Cons: less realistic than non-IID in many FL scenarios.

**Non-IID (optional):** Split by class or other criteria (e.g. Client 1: mostly classes 0–4, Client 2: mostly 5–9). Pros: more realistic, tests FL robustness, better for research. Cons: more complex, can be harder to train.

**Dataset abstraction:** Interface: `load_training_data()`, `load_test_data()`, `split_for_federation(num_clients, split_type='iid')`. Implementations: e.g. MNISTDataset; extendable to CIFAR10, custom. **Client configuration:** Each client loads its assigned portion (e.g. `data/mnist/train/client_X.pt`); client ID determines which split. **Main service:** Loads full test set (e.g. `test/test.pt`); never touches training data.

**IID split algorithm:** (1) Load full training set. (2) Shuffle randomly (with seed). (3) Split into N equal parts. (4) Save each part to e.g. `data/mnist/train/client_X.pt`. (5) Each client gets a random sample containing all classes. **Non-IID (class-based example):** (1) Group training samples by class (0–9 for MNIST). (2) Distribute classes among clients (e.g. Client 1: 0–4, Client 2: 5–9). (3) Save each client's portion; include class distribution in config. **Benefits:** Dataset-agnostic; supports IID and non-IID for research; easy to scale client count; clear train/test separation.

### Data Preprocessing and Preparation

**Script:** `scripts/prepare_datasets.py` — one-time preparation of split data and configs.

**Usage:**  
`python scripts/prepare_datasets.py --dataset mnist --num_clients 2 --split_type iid --output_dir data/mnist`  
`python scripts/prepare_datasets.py --dataset mnist --num_clients 4 --split_type non_iid --output_dir data/mnist`

**Parameters:** `--dataset` (mnist, cifar10, …), `--num_clients`, `--split_type` (iid, non_iid), `--output_dir`, optional `--seed`.

**Output structure:**

```
data/mnist/
├── train/
│   ├── client_0.pt, client_1.pt, ...
│   └── metadata.json
├── test/
│   └── test.pt
└── config/
    ├── client_0_config.json, client_1_config.json, ...
    └── main_service_config.json
```

**Client config (e.g. client_X_config.json):** client_id, dataset_path, dataset_size, split_type, classes_distribution (non-IID), dataset_hash. **Main service config:** test_dataset_path, test_dataset_size, test_dataset_hash, dataset_version.

**Process (IID):** Load full training set → shuffle (seed) → split into N parts → save client_*.pt → hashes and configs → save test set and main config. **Process (Non-IID):** Group by class → distribute classes across clients → save per-client files and configs with class distribution.

**Dataset metadata (metadata.json):** dataset_name, total_training_samples, total_test_samples, num_clients, split_type, split_seed, client_splits (client IDs and sample counts), created_at, hashes for integrity.

**Startup:** Clients read their config, load dataset, verify hash. Main service reads main config, loads test set, verifies hash. One-time preprocessing; reproducible with seed; easy to scale clients or switch IID/non-IID; dataset-agnostic (works with any dataset implementing the interface).

**Client startup:** Client reads its config file, loads dataset from path in config, verifies dataset hash, and is ready to participate. **Main service startup:** Reads main config, loads test dataset, verifies test dataset hash, ready to validate.

**Benefits of preprocessing:** One-time step before training; clients know their data at startup; reproducible splits (seed); integrity verification (hash checking); easy to switch IID/non-IID or scale client count; dataset-agnostic; clear separation of concerns (preprocessing vs training).

**Preprocessing process (IID):** Load full training set → shuffle with seed → split into N equal parts → save each part to `train/client_X.pt` → compute hashes → generate client and main configs. **(Non-IID):** Load full training set → group by class → distribute classes among clients → save per-client files → compute statistics and configs with class distribution.

**Example workflow:** (1) Run `python scripts/prepare_datasets.py --dataset mnist --num_clients 2 --split_type iid`. (2) Client 0 loads `client_0_config.json` and `train/client_0.pt`; Client 1 loads `client_1_config.json` and `train/client_1.pt`; main service loads `main_service_config.json` and `test/test.pt`. (3) All verify hashes and are ready to train or validate.

## Security Considerations

### Encryption Strategy

- **Diff encryption:** AES-256-GCM for authenticated encryption. Key management via `ENCRYPTION_KEY` environment variable (Base64-encoded 32-byte key). Each version can use a unique key.
- **Integrity verification:** Hash of encrypted diff stored on-chain; IPFS CID provides additional integrity. Verification: retrieve from IPFS → compute hash → compare with on-chain hash before decrypt.
- **Access control:** Permissioned blockchain (Hyperledger Fabric); API keys for queue access; client IDs in configuration; IPFS content is encrypted so only authorized parties can decrypt.

## Queue Design

### Task Types

TRAIN (client training), AGGREGATE, BLOCKCHAIN_WRITE, STORAGE_WRITE, VALIDATE, ROLLBACK, DECISION, TRAINING_COMPLETE (final task when model is ready).

### Task Message Format

**Common fields:** task_id, task_type, model_version_id, parent_version_id, payload (task-specific), metadata (created_at, priority, retry_count).

**Payload by type:** TRAIN — weights_cid. AGGREGATE — client_updates. After blockchain — blockchain_hash. After storage — ipfs_cid. After validation — validation_result. TRAINING_COMPLETE — final_model_info containing: final_model_version_id, final_accuracy, final_metrics (loss, precision, recall, etc.), final_weights_cid (IPFS CID of final model weights), training_summary (total_iterations, clients_participated, training_duration, total_rounds), metadata (hyperparameters_used, dataset_info, completion_reason).

## Performance (RQ3)

**Metrics to track:** End-to-end latency (client update → validation complete), blockchain transaction latency, storage operation latency, queue processing time, throughput (updates per second).

**Optimization strategies:** Batch blockchain transactions; parallel validation workers; caching of frequently accessed model weights; compression of weight diffs.

## Technology Stack & Project Structure

### Stack

- **Languages:** Python 3.10+ (main service, client service, workers), Go 1.19+ (blockchain service and Fabric chaincode)
- **Queue:** RabbitMQ
- **Blockchain:** Hyperledger Fabric 2.5+
- **Storage:** IPFS (Python: httpx for IPFS API; run IPFS daemon locally or connect to network)
- **ML:** PyTorch
- **API:** FastAPI
- **UI:** Optional React for monitoring

### Project Structure

- **main_service/** (Python): workers/ (aggregation_worker, storage_worker, validation_worker, rollback_worker), api/ (server, routes), blockchain/ (HTTP client for blockchain-service), requirements.txt, Dockerfile
- **blockchain_service/** (Go): main.go (REST API), go.mod, go.sum, chaincode/model_provenance.go, Dockerfile, README
- **client_service/** (Python): training/ (trainer, model), queue/ (consumer), config, requirements.txt, Dockerfile
- **shared/** (Python): models/ (task), utils/ (crypto, hashing), storage/ (ipfs_client, encryption), datasets/ (dataset_interface, mnist_dataset), config/ (settings), logger/
- **scripts/:** generate_encryption_key.py, prepare_datasets.py
- **data/** (runtime), **tests/**, **docker-compose.yml**

---

## Additional Detail

### Rollback Strategy (Detail)

**Accuracy comparison:** Compare current model accuracy with previous best; track accuracy history for all model versions. Accuracy is the primary metric for rollback decisions.

**Tolerance and patience:** Allow accuracy to drop by a small amount (e.g. 0.5–1%) before concern. Track consecutive iterations where accuracy is below best; increment patience each time. If patience exceeds threshold (e.g. 3–5), trigger rollback. Maintain reference to the last "best" model version as rollback target.

**Rollback decision logic:** (1) **Immediate rollback:** If accuracy drops by more than tolerance (e.g. >2%), rollback to previous best. (2) **Patience-based:** If within tolerance but below best for N consecutive iterations, rollback after patience threshold. (3) **No rollback:** If accuracy improves or acceptable, reset patience and update best checkpoint. When rollback is triggered, roll back to last safe checkpoint (may be several iterations back); discard intermediate versions; training resumes from best checkpoint.

### Post-Rollback Workflow (Detail)

1. **Rollback execution:** Rollback worker retrieves target weights from IPFS, updates blockchain state (current model pointer = rolled-back version), records rollback event, publishes rollback notification to queue.
2. **Client notification:** All clients receive ROLLBACK task (target version ID, IPFS CID of rolled-back weights). Clients read task, download weights from IPFS, replace local weights, discard local changes after cutoff, acknowledge.
3. **Training continuation:** Main service publishes new TRAIN task with rolled-back weights CID. All clients load rolled-back weights and continue from same state. Synchronization: main service can wait for client acknowledgements before new round.
4. **Clients that contributed after cutoff:** They receive rollback notification, discard local model state, load rolled-back weights from IPFS, continue from rolled-back state; their previous contributions remain on blockchain for audit.
5. **State reset:** Best checkpoint = rolled-back version; patience = 0; accuracy tracking restarts.

### Training Completion (Detail)

**Criteria (expanded):** (1) **Accuracy threshold:** Target accuracy (e.g. 95%) reached; configurable. (2) **Convergence:** No improvement for N consecutive iterations (e.g. 10); configurable. (3) **Max iterations:** Predefined max rounds (e.g. 100). (4) **Max rollbacks:** Max rollbacks exceeded; best model marked final; may indicate need for investigation. (5) **Manual trigger:** Operator triggers via API/UI. (6) **Overtraining prevention:** Monitor validation vs training accuracy; early stopping if validation plateaus or train/val gap exceeds threshold.

**Process when completion is met:** (1) Decision worker detects completion. (2) Publishes TRAINING_COMPLETE (final version ID, accuracy/metrics, IPFS CID, summary, completion reason). (3) Final state recorded on blockchain. (4) API/UI notified. (5) Training stops; no new TRAIN tasks. (6) Model marked ready.

### Dataset Distribution (Summary)

**Clients:** Training datasets only; local data for model training; different clients can have different training data; optional local train/val split from their own data for early stopping or monitoring (subset of training data, not global test set).

**Main service:** Test/validation datasets only; used to evaluate model performance; never used for training. Test dataset is fixed and versioned for reproducibility; record test dataset hash/version on-chain when validating.

### Data Preprocessing (Detail)

**Script:** `scripts/prepare_datasets.py` — loads full dataset, splits training data (IID or non-IID, number of clients), saves per-client files and test set, generates configs and metadata.

**IID process:** Load full training set → shuffle with seed → split into N equal parts → save `train/client_X.pt` → compute hashes → generate client configs (client_id, dataset_path, dataset_size, split_type, dataset_hash) and main config (test_dataset_path, test_dataset_size, test_dataset_hash, dataset_version) → save test set and metadata (dataset_name, total_training_samples, total_test_samples, num_clients, split_type, split_seed, client_splits, created_at, hashes).

**Non-IID process:** Load full training set → group by class → distribute classes among clients (e.g. class-based or shard-based) → save per-client files → compute statistics (class distribution) → generate configs with class distribution → save test set and metadata.

**Client startup:** Read config → load dataset from path in config → verify dataset hash → ready to train. **Main service startup:** Read main config → load test set → verify test dataset hash → ready to validate.

**Benefits:** One-time preprocessing; reproducible (seed); integrity (hash check); easy to change IID/non-IID or number of clients; dataset-agnostic; clear separation of concerns.

### Queue Task Payloads (Reference)

**TRAIN:** weights_cid (IPFS CID of current weights). **AGGREGATE:** client_updates (list of client diffs). **BLOCKCHAIN_WRITE:** after write, payload includes blockchain_hash. **STORAGE_WRITE:** after storage, payload includes ipfs_cid. **VALIDATE:** after validation, payload includes validation_result. **DECISION:** payload includes validation result for decision logic. **TRAINING_COMPLETE:** final_model_info (final_model_version_id, final_accuracy, final_metrics, final_weights_cid, training_summary, metadata, completion_reason).

### Smart Contract Data (Reference)

**ModelVersion (on-chain):** version_id, parent_version_id, timestamp, iteration, num_clients, client_ids, aggregated_hash, diff_hash, off_chain_location, hyperparameters, validation_status, validation_metrics, block_number. **ValidationRecord:** version_id, validator_id, passed, metrics, test_dataset_hash, timestamp. **RollbackEvent:** from_version_id, to_version_id, reason, triggered_by, timestamp, type. **Not on-chain:** model weights, diffs, raw training data, full binaries.

### IPFS and Diff Storage (Detail)

**IPFS storage:** (1) Encrypt diff (AES-256-GCM). (2) Upload to IPFS → get CID. (3) Pin CID. (4) Store CID in blockchain metadata. (5) Verify hash of encrypted diff matches blockchain hash. **Retrieval:** Get CID from metadata → fetch from IPFS → verify hash → decrypt → apply to weights. **IPFS benefits:** Content-addressed (CID = content hash); decentralized; immutable; open source; local node for development (e.g. `ipfs daemon`, API http://localhost:5001).

**Diff storage (why not JWT):** JWT is for authentication; payloads are base64, not encrypted; size limits unsuitable for weight diffs. **Process:** Client computes diff → encrypt (AES-256-GCM) → hash encrypted bytes → store hash on-chain, encrypted diff on IPFS. **Verification:** Get CID → retrieve from IPFS → compute hash → compare with on-chain hash → decrypt if match. **Benefits:** Blockchain hash = integrity; encryption = confidentiality; no size limit; hybrid storage.

### Security (Detail)

**Encryption:** AES-256-GCM for diffs; key via ENCRYPTION_KEY (Base64 32-byte); per-version keys possible. **Integrity:** On-chain hash of encrypted diff; IPFS CID; verify after retrieval before decrypt. **Access:** Permissioned Fabric; API keys for queue; client IDs in config; IPFS content encrypted so only authorized parties decrypt.

### Configuration Parameters (Reference)

**Rollback:** accuracy_tolerance, patience_threshold, min_accuracy_threshold, metrics_to_track. **Training completion:** target_accuracy, convergence_patience, max_iterations, max_rollbacks, overtraining_threshold, early_stopping_enabled. **Dataset preparation:** --dataset, --num_clients, --split_type, --output_dir, --seed (optional).

### Processing Pipeline (Step-by-step)

**Step 1 — Aggregation:** Read client updates from queue → FedAvg → publish aggregated update. **Step 2 — Blockchain:** Read aggregated update → hash encrypted diff → call blockchain-service → Fabric invokes chaincode → store version/parent/hash/timestamp/metadata on-chain → publish task with blockchain hash. **Step 3 — Storage:** Read task → encrypt diff → upload to IPFS → get CID → verify hash → pin CID → publish task with CID. **Step 4 — Validation:** Read task → get diff from IPFS → decrypt → apply to previous weights → run test set (main service only) → record validation on blockchain → publish result. **Step 5 — Decision:** Read result → apply rollback strategy → if rollback: rollback or regression diagnosis; if pass: check completion → publish next TRAIN or TRAINING_COMPLETE → update registry. **Step 6 — Completion:** TRAINING_COMPLETE contains final version, metrics, CID, summary; consumed by API/UI; final state on blockchain.

### Example Workflows

**Preprocessing:** Run `prepare_datasets.py --dataset mnist --num_clients 2 --split_type iid`. Clients load `config/client_X_config.json` and `train/client_X.pt`; main loads `config/main_service_config.json` and `test/test.pt`; all verify hashes. **One training iteration:** Main sends TRAIN (weights CID) → clients train and publish diffs → aggregation → blockchain → storage → validation on test set → decision → next TRAIN or complete. **Rollback:** Validation detects regression → rollback worker loads target weights from IPFS → updates blockchain → publishes ROLLBACK → all clients load rolled-back weights → main publishes new TRAIN with rolled-back CID.

### References and Related Docs

- [SETUP.md](SETUP.md) — Setup and run instructions.
- `docs/REGRESSION_DIAGNOSIS.md` — Regression diagnosis in depth.
- `blockchain_service/README.md` — Blockchain service and chaincode.
- `env_template.txt` / `.env` — Environment variables (ENCRYPTION_KEY, API_KEY, queue, IPFS, etc.).

### API and Environment Quick Reference

**Blockchain service API:** POST /api/v1/model/register (register model version), POST /api/v1/model/validate (record validation), POST /api/v1/model/rollback (record rollback), GET /api/v1/model/provenance/{version_id} (provenance chain), GET /health. **Main service API:** POST /api/v1/models/{version_id}/rollback for manual rollback (API key required). **Environment:** ENCRYPTION_KEY (Base64 32-byte), API_KEY (for API auth), queue URL (RabbitMQ), IPFS API endpoint (e.g. http://localhost:5001), blockchain service URL.

### Glossary

**Aggregation (FedAvg):** Federated averaging — combine client weight updates by averaging. **CID:** Content Identifier (IPFS). **Diff:** Weight update (delta between old and new model weights). **Iteration:** One round of client training → aggregation → validation → decision. **Main service:** Aggregator; holds test set; runs workers; coordinates blockchain and IPFS. **Blockchain service:** Go microservice; all Fabric operations. **Client service:** Training service; local data only; publishes diffs. **Provenance:** Lineage of model versions (who, when, what). **Rollback:** Restore model to a previous version. **Tolerance/patience:** Rollback strategy parameters (allow small accuracy drop; allow N bad iterations before rollback). **Validation:** Evaluate model on test set; record result on blockchain.

### Extended Rollback and Completion Reference

**Rollback (automatic):** Compare current accuracy to best; if drop > tolerance or patience exceeded, set current model pointer to last best version; record rollback event on-chain; publish ROLLBACK task so all clients load rolled-back weights; publish new TRAIN from rolled-back state. **Rollback (manual):** Operator calls API or UI with target version and reason; same execution path; type "manual" on-chain. **Completion:** When accuracy threshold, convergence, max iterations, max rollbacks, or manual trigger is met, publish TRAINING_COMPLETE with final version, metrics, CID, summary; record final state on blockchain; stop publishing TRAIN tasks.

### Data Flow (Text Diagram)

```
Clients --[encrypted diffs]--> Queue --> Aggregation Worker --> Queue
  --> Blockchain Worker --> Fabric (on-chain) --> Queue
  --> Storage Worker --> IPFS (encrypted diff, CID) --> Queue
  --> Validation Worker (test set in main service) --> Queue
  --> Decision (rollback strategy, completion check)
  --> [if pass] TRAIN for next iteration or TRAINING_COMPLETE
  --> [if rollback] ROLLBACK task --> clients load weights --> new TRAIN from rolled-back state
```

---

## Addressing Research Questions

**RQ1 (Architecture):** Queue-based microservices with specialized workers. Evidence: system design and implementation. Metrics: system availability, fault tolerance.

**RQ2 (Traceability & Reproducibility):** Complete provenance chain on blockchain. Evidence: smart contract queries, audit trail logs. Metrics: provenance query time, lineage completeness.

**RQ3 (Performance):** Benchmark latency, throughput, storage overhead. Evidence: latency measurements, throughput analysis. See Performance section for metrics and optimization ideas.

## Architecture Notes and Implementation Priorities

**Design choices:** (1) Keep sequential pipeline where necessary (e.g. apply diff then validate); parallelise where possible (e.g. multiple blockchain or storage operations). (2) Rollback: worker queries blockchain for previous version, retrieves weights from IPFS, records rollback event, posts new TRAIN task with previous weights. (3) Test dataset: store test dataset hash on-chain, version test datasets, record which dataset was used for each validation.

**Implementation phases:**

- **Phase 1 (MVP):** Basic queue, client training service, simple aggregation, basic blockchain integration, IPFS storage.
- **Phase 2 (Core):** Validation worker, rollback mechanism, full smart contract, API/UI for monitoring.
- **Phase 3 (Research):** Performance benchmarking, provenance queries, comparison with baseline, documentation.

## Testing Strategy

- **Unit tests:** Individual workers, encryption, hashing.
- **Integration tests:** Queue → worker → blockchain flow.
- **End-to-end:** Full training iteration.
- **Performance:** Latency and throughput measurements.
- **Security:** Encryption and integrity verification.
