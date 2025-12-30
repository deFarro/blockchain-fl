# Blockchain-Enabled Learning Systems for Verifiable Model Provenance

> **MSc Thesis Project** > **Topic:** Blockchain-Enabled Learning Systems for Verifiable Model Provenance, Rollback, and Regulatory Auditability.

## Quick Start

1. **Setup:** `cp env_template.txt .env` and set `ENCRYPTION_KEY` and `API_KEY`
2. **Start:** `docker-compose up -d`
3. **Access Dashboard:** Open `http://localhost:8000/`
4. **Start Training:** Click "Start Training" in the dashboard

For detailed instructions, see [SETUP.md](SETUP.md) and [RUNNING.md](RUNNING.md).

## Project Overview

This repository contains the prototype implementation for a Master's thesis investigating the convergence of **Distributed Ledger Technology (DLT)** and **Federated Learning (FL)**.

The project aims to address the opacity of AI development by designing a blockchain-enabled provenance system. The system records training metadata, model updates, and ownership changes on an immutable ledger to support trust, reproducibility, and compliance with emerging regulations such as the **EU AI Act**.

### Key Capabilities

- **Immutable Audit Trails:** Records training events and metadata on a permissioned blockchain
- **Model Rollback:** Enables the restoration of previous model states to mitigate poisoning attacks or errors.
- **Hybrid Storage:** Utilises a "Selective Decentralisation" architecture where critical metadata is stored on-chain, while heavy model artefacts are stored off-chain.
- **Privacy-Preserving:** Integrates with Federated Learning workflows to simulate training without exposing raw local data.

---

## Research Questions

This prototype is designed specifically to answer the following research questions defined in the thesis proposal:

1.  **RQ1 (Architecture):** How can blockchain technology be integrated into learning systems to enable verifiable model provenance, rollback, and auditability while maintaining efficiency?
2.  **RQ2 (Trust & Auditability):** To what extent can blockchain-based provenance tracking improve the traceability and reproducibility of ML models compared with conventional systems?
3.  **RQ3 (Performance):** What are the performance implications (latency, overhead, storage) of incorporating blockchain into the training process?

---

# System Architecture Design

## Overview

This document outlines the detailed architecture for the blockchain-enabled federated learning system, addressing the research questions defined in the proposal. It consolidates all design decisions, language choices, dataset separation, and implementation details.

## System Components

### 1. Main Service (Aggregator Service)

**Responsibilities:**

- Model weight aggregation (FedAvg)
- Validation using test datasets (test datasets stored only in main service)
- Blockchain transaction coordination (calls blockchain-service)
- Off-chain storage management (IPFS)
- Task queue management
- UI/API for manual intervention

**Important:** Main service is the **only** component with access to test/validation datasets. Clients only have training datasets.

**Components:**

- **Queue Manager**: Manages task distribution (RabbitMQ)
- **Aggregation Worker**: Combines client updates using federated averaging
- **Blockchain Client**: HTTP client that calls blockchain-service API
- **Storage Worker**: Handles encryption/decryption and IPFS storage
- **Validation Worker**: Runs test datasets and evaluates model performance
- **Rollback Worker**: Manages model state restoration
- **API Server**: REST API for manual task creation, monitoring, and manual rollback
- **Web UI**: Dashboard for monitoring, manual intervention, and rollback management

### 2. Blockchain Service (Go Microservice)

**Responsibilities:**

- All Hyperledger Fabric operations
- Blockchain transaction creation and submission
- Chaincode invocation using official Fabric Go SDK
- Provenance chain management
- Validation and rollback event recording

**Architecture:**

- Separate Go microservice (isolated from Python dependencies)
- REST API for blockchain operations
- Uses official Hyperledger Fabric Go SDK
- Can be developed and deployed independently

**API Endpoints:**

- `POST /api/v1/model/register` - Register model version
- `POST /api/v1/model/validate` - Record validation results
- `POST /api/v1/model/rollback` - Record rollback event
- `GET /api/v1/model/provenance/{version_id}` - Get provenance chain
- `GET /health` - Health check

### 3. Client Service (Training Service)

**Responsibilities:**

- Local model training on **training datasets only** (no test/validation data)
- Weight update computation (diffs or full weights)
- Task consumption from queue
- Scalable deployment (multiple instances with different training datasets)

**Important:** Clients **never** see test/validation datasets. They only train on their local training data.

**Components:**

- **Training Engine**: PyTorch model training
- **Queue Consumer**: Reads training tasks from queue
- **Update Publisher**: Publishes weight updates to queue
- **Config Manager**: Manages dataset paths, model architecture, hyperparameters

## Data Flow Architecture

### Training Iteration Flow

Clients send weight updates (encrypted diffs) to a queue, which feeds into the Aggregation Worker. Multiple clients can contribute updates simultaneously.

### Processing Pipeline

**Step 1: Aggregation Worker**

- Reads multiple client updates from queue
- Applies FedAvg aggregation
- Publishes aggregated update to queue

**Step 2: Blockchain Worker**

- Reads aggregated update
- Computes hash of encrypted diff
- Calls blockchain-service API to create blockchain transaction
- Blockchain service uses Fabric Go SDK to invoke chaincode
- Stores version hash, parent hash, timestamp, metadata on-chain
- Publishes task with blockchain hash to queue

**Step 3: Storage Worker**

- Reads task with blockchain hash
- Encrypts aggregated diff (AES-256)
- Stores encrypted diff on IPFS (gets CID)
- Verifies hash of encrypted diff matches blockchain hash
- Pins CID to ensure persistence
- Publishes task with IPFS CID to queue

**Step 4: Validation Worker**

- Reads task with IPFS CID
- Retrieves encrypted diff from IPFS
- Decrypts and loads model weights
- Applies diff to previous weights
- Runs test dataset validation (test data stored only in main service)
- Calls blockchain-service API to record validation results on blockchain
- Publishes validation result to queue

**Step 5: Main Service Decision Logic**

- Reads validation result
- Evaluates model performance using rollback strategy (see Rollback Strategy section below)
- **Iteration Coordination**: Main service controls when clients start next iteration
  - Clients are passive: They only train when they receive TRAIN tasks
  - Main service is active: Publishes TRAIN tasks only after validation passes
  - **Late Updates**: Updates for past iterations are rejected (logged and ignored)
- If rollback needed:
  - Option 1: Rollback entire iteration (standard approach)
  - Option 2: Run regression diagnosis to identify problematic client(s) and exclude them (see Regression Diagnosis section)
- If PASS: Checks if training is complete (based on criteria like accuracy threshold, max iterations, or manual trigger)
  - If training continues: Publishes TRAIN tasks for **next iteration** (with new aggregated weights CID)
  - If training complete: Publishes TRAINING_COMPLETE task with final model information
- Updates model registry

### Rollback Strategy

The main service determines if a diff is good by evaluating model performance on the test dataset. The primary metric is accuracy, with support for other metrics (loss, precision, recall, etc.).

**Validation Criteria:**

**1. Accuracy Comparison:**

- Compare current model accuracy with previous best accuracy
- Track accuracy history for all model versions
- Accuracy is the primary metric for rollback decisions

**2. Tolerance and Patience Mechanism:**

The system uses a tolerance-based approach with patience to allow temporary accuracy drops:

- **Tolerance Threshold:** Allow accuracy to drop by a small amount (e.g., 0.5-1%) before considering rollback

  - Example: If best accuracy is 95%, allow down to 94% before triggering concern
  - Configurable per deployment

- **Patience Counter:** Track consecutive iterations where accuracy is below the best

  - Start with patience = 0 when accuracy drops below best
  - Increment patience for each consecutive bad iteration
  - If patience exceeds threshold (e.g., 3-5 iterations), trigger rollback

- **Best Checkpoint Tracking:** Maintain reference to the last "safe" or "best" model version
  - Store version ID of model with best accuracy
  - This becomes the rollback target if patience is exceeded

**3. Rollback Decision Logic:**

**Scenario 1: Immediate Rollback (Severe Drop)**

- If accuracy drops by more than tolerance threshold (e.g., >2% drop)
- Rollback immediately to previous best checkpoint

**Scenario 2: Patience-Based Rollback (Gradual Degradation)**

- If accuracy drops within tolerance but stays below best for N consecutive iterations
- Rollback to last best checkpoint after patience threshold is reached
- Allows model to recover from temporary dips (common in training)

**Scenario 3: No Rollback (Acceptable Performance)**

- If accuracy improves or stays within acceptable range
- Reset patience counter
- Update best checkpoint if new best accuracy achieved
- Continue training

**4. Rollback Depth:**

When rollback is triggered:

- Rollback to the last "safe" checkpoint (model version with best accuracy)
- This may be several iterations back if patience allowed multiple bad iterations
- All intermediate versions between best and current are discarded
- Training resumes from the best checkpoint

**5. Configuration Parameters:**

- `accuracy_tolerance`: Maximum allowed accuracy drop before concern (default: 0.5%)
- `patience_threshold`: Number of consecutive bad iterations before rollback (default: 3)
- `min_accuracy_threshold`: Absolute minimum accuracy below which immediate rollback (default: configurable)
- `metrics_to_track`: List of metrics to evaluate (accuracy, loss, precision, recall, etc.)

**6. Example Flow:**

```
Iteration 1: Accuracy = 90% → Best = 90%, Patience = 0, Checkpoint = v1.0
Iteration 2: Accuracy = 91% → Best = 91%, Patience = 0, Checkpoint = v1.1 (new best)
Iteration 3: Accuracy = 90.5% → Best = 91%, Patience = 1 (within tolerance, but below best)
Iteration 4: Accuracy = 90.3% → Best = 91%, Patience = 2 (still below best)
Iteration 5: Accuracy = 90.2% → Best = 91%, Patience = 3 (threshold reached)
→ ROLLBACK to v1.1 (last best checkpoint)
```

**7. Benefits:**

- Prevents model degradation from accumulating over multiple iterations
- Allows temporary accuracy dips (common in training) without premature rollback
- Maintains best model state automatically
- Configurable thresholds for different use cases
- Supports research on rollback strategies (RQ1)

### Manual Rollback

In addition to automatic rollback based on accuracy metrics, the system supports manual rollback for security and operational reasons.

**Use Cases for Manual Rollback:**

- **Model Poisoning Detection:** Suspected malicious updates from clients
- **Security Incidents:** Detected attacks or unauthorized changes
- **Data Quality Issues:** Discovery of corrupted or incorrect training data
- **Operational Errors:** Human error in configuration or deployment
- **Research Requirements:** Intentional rollback for experimentation

**Manual Rollback Process:**

**1. Access via API or UI:**

- **API Endpoint:** POST `/api/v1/models/{version_id}/rollback`

  - Requires authentication (API key)
  - Accepts target version ID and reason
  - Returns rollback task ID

- **Web UI:** Dashboard with rollback interface
  - Display model version history with accuracy metrics
  - Select target version to rollback to
  - Enter reason for rollback
  - Submit rollback request

**2. Rollback Request Parameters:**

- `target_version_id`: Version to rollback to (must be a valid previous version)
- `reason`: Human-readable reason for rollback (e.g., "Suspected model poisoning", "Security incident", "Manual intervention")
- `triggered_by`: User ID or identifier of person initiating rollback
- `priority`: Optional priority level (normal, high, urgent)

**3. Authorization and Security:**

- Only authorized users can trigger manual rollback
- API key authentication required for API access
- Role-based access control (admin, operator roles)
- All manual rollback actions are logged and recorded on blockchain
- Audit trail includes who, when, why, and to which version

**4. Rollback Execution:**

- Manual rollback request creates a ROLLBACK task in queue
- Rollback worker processes the task:
  1. Validates target version exists and is accessible
  2. Retrieves target version weights from IPFS
  3. Updates blockchain with rollback event (via RollbackModel smart contract function)
  4. Records reason, triggered_by, and timestamp on-chain
  5. Updates current model pointer to target version
  6. Publishes new training task to resume from rolled-back state

**5. Blockchain Recording:**

- RollbackModel smart contract function records:
  - From Version ID (current version)
  - To Version ID (target version)
  - Reason (from manual request)
  - Triggered By (user identifier)
  - Timestamp
  - Type: "manual" (distinguished from automatic rollback)

**6. Integration with Automatic Rollback:**

- Manual rollback can be triggered at any time, even during automatic rollback evaluation
- Manual rollback takes precedence over automatic rollback
- After manual rollback, automatic rollback logic resets (patience counter, best checkpoint)
- System resumes normal training from rolled-back state

**7. Version Selection Interface:**

- UI displays model version history with:
  - Version ID
  - Timestamp
  - Accuracy metrics
  - Validation status
  - Client IDs that contributed
  - IPFS CID for weights
- User can select any previous version as rollback target
- System validates version is accessible and valid

**8. Example Manual Rollback Scenarios:**

**Scenario 1: Suspected Poisoning**

- Operator notices unusual behavior in model predictions
- Reviews recent client contributions
- Identifies suspicious client update
- Manually rolls back to version before suspicious update
- Reason: "Suspected model poisoning from client_3"

**Scenario 2: Security Incident**

- Security team detects unauthorized access
- Manually rolls back to last known good state
- Reason: "Security incident - unauthorized access detected"

**Scenario 3: Data Quality Issue**

- Discovery that training data was corrupted
- Rollback to version before corrupted data was used
- Reason: "Data quality issue - corrupted training data"

**9. Benefits:**

- Enables rapid response to security threats (model poisoning, attacks)
- Provides human oversight and intervention capability
- Maintains complete audit trail of all rollback actions
- Supports operational flexibility and research needs
- Critical for production deployments and regulatory compliance

### Post-Rollback Workflow

When a rollback occurs (automatic or manual), the system must ensure training continues correctly from the rolled-back state.

**1. Rollback Execution:**

- Rollback worker completes rollback:
  1. Retrieves target version weights from IPFS
  2. Updates blockchain state (current model pointer = rolled-back version)
  3. Records rollback event on blockchain
  4. Publishes rollback notification to queue

**2. Client Notification and Weight Update:**

- **All clients receive notification** about rollback:

  - Rollback event published to queue as ROLLBACK task
  - Contains target version ID and IPFS CID of rolled-back weights
  - All clients (including those that contributed after cutoff) must update

- **Client Response:**
  - Clients read ROLLBACK task from queue
  - Download rolled-back weights from IPFS using provided CID
  - Replace current local weights with rolled-back weights
  - Discard any local changes made after the cutoff point
  - Acknowledge rollback completion

**3. Training Continuation:**

- After rollback, system publishes new TRAIN task to queue:

  - Task includes IPFS CID of rolled-back weights
  - All clients receive the same task
  - Clients load rolled-back weights and continue training from that point
  - Ensures all clients start from the same state

- **Synchronization:**
  - All clients must acknowledge rollback before new training round begins
  - Main service waits for all active clients to confirm rollback
  - Prevents clients from using outdated weights

**4. Handling Clients That Contributed After Cutoff:**

- **Clients that contributed updates after the rolled-back version:**

  - Receive rollback notification
  - Must discard their local model state (if they have one)
  - Must load rolled-back weights from IPFS
  - Continue training from rolled-back state
  - Their previous contributions are effectively discarded (but remain on blockchain for audit)

- **No Special Treatment:**
  - All clients treated equally after rollback
  - No distinction between clients that contributed before/after cutoff
  - Ensures consistent state across all clients

**5. Rollback State Reset:**

- After rollback, automatic rollback logic resets:
  - Best checkpoint updated to rolled-back version
  - Patience counter reset to 0
  - Accuracy tracking restarts from rolled-back version
  - Training continues with fresh rollback state

### Training Completion Criteria

The system determines when training is complete based on multiple criteria. Training can be considered ready when any of these conditions are met:

**1. Accuracy Threshold Reached:**

- Target accuracy achieved (e.g., 95% accuracy)
- Configurable per deployment
- Once reached, training stops and model is marked as ready
- Final model version recorded on blockchain

**2. Convergence Detection:**

- Model accuracy has not improved for N consecutive iterations (e.g., 10 iterations)
- Indicates model has reached its maximum performance
- Prevents unnecessary training iterations
- Configurable convergence threshold

**3. Maximum Iterations Reached:**

- Predefined maximum number of training rounds completed
- Prevents infinite training
- Configurable limit (e.g., 100 rounds)
- Model marked as ready even if accuracy threshold not reached

**4. Rollback Limit Reached:**

- Maximum number of rollbacks exceeded (e.g., 5 rollbacks)
- Indicates model cannot be improved further
- Clients may be contributing conflicting updates
- Training stops and current best model is marked as final
- Can indicate need for investigation (data quality, client issues)

### Regression Diagnosis

When a model regression is detected (accuracy drops), the system can diagnose which client(s) caused the regression by testing each client's diff individually.

**How It Works:**

1. **Normal Aggregation (First)**: Aggregate ALL client diffs using FedAvg (no filtering overhead)

2. **If Regression Detected**: Run regression diagnosis:

   - For each client that contributed to the problematic iteration:
     - Load previous model weights (before problematic iteration)
     - Apply **only this client's diff**
     - Validate on test dataset
     - Compare accuracy with baseline
   - Identify which client(s) caused the regression

3. **Exclude from Future Iterations**: When `enable_client_exclusion=True` and problematic clients identified:
   - Add problematic client IDs to `excluded_clients` list
   - **Future aggregations** will filter out excluded clients
   - Only "good" clients participate in future iterations
   - System continues training without problematic clients

**Key Point**: Exclusion only happens AFTER regression diagnosis. Normal aggregation includes ALL clients (optimized for common case - no regression).

**Example:**

```
Iteration 24: Clients 1, 2, 3 → Regression detected (95% → 92%)

Diagnosis:
  Test Client 1 diff alone → 94.5% (OK)
  Test Client 2 diff alone → 94.8% (OK)
  Test Client 3 diff alone → 88.0% (REGRESSION!)

Result: Exclude Client 3 from future iterations
```

**Configuration:**

- `enable_client_exclusion`: Enable/disable client exclusion
- `excluded_clients`: List of client IDs to exclude from aggregation

**Limitations:**

- Currently tests each client individually (future: test combinations)
- Requires access to previous weights and test dataset
- No automatic re-inclusion (manual re-enable required)

See `docs/REGRESSION_DIAGNOSIS.md` for detailed documentation.

**5. Manual Completion Trigger:**

- Operator manually triggers training completion via API/UI
- Useful for research, testing, or operational needs
- Final model state recorded regardless of metrics

**6. Overtraining Prevention:**

- **Validation Accuracy Monitoring:**

  - Track validation accuracy vs training iterations
  - If validation accuracy plateaus or decreases while training continues, consider stopping
  - Early stopping based on validation metrics

- **Gap Detection:**

  - Monitor gap between training accuracy and validation accuracy
  - Large gap indicates overfitting
  - Trigger early stopping if gap exceeds threshold

- **Convergence Indicators:**
  - Loss stops decreasing significantly
  - Accuracy improvement per iteration becomes negligible
  - Model performance stabilizes

**7. Training Completion Process:**

When any completion criterion is met:

1. Decision worker detects completion condition
2. Publishes TRAINING_COMPLETE task to queue
3. Task contains:
   - Final model version ID
   - Final accuracy and metrics
   - IPFS CID of final model weights
   - Training summary (iterations, clients, duration)
   - Completion reason (which criterion was met)
4. Final state recorded on blockchain
5. API/UI notified that model is ready
6. Training stops, no new training tasks published

**8. Configuration Parameters:**

- `target_accuracy`: Minimum accuracy to achieve (default: configurable)
- `convergence_patience`: Iterations without improvement before convergence (default: 10)
- `max_iterations`: Maximum training rounds (default: 100)
- `max_rollbacks`: Maximum rollbacks before stopping (default: 5)
- `overtraining_threshold`: Validation accuracy gap threshold (default: configurable)
- `early_stopping_enabled`: Enable early stopping based on validation metrics (default: true)

**9. Example Completion Scenarios:**

**Scenario 1: Accuracy Threshold**

- Training reaches 95% accuracy (target: 95%)
- Training stops, model marked as ready
- Completion reason: "Accuracy threshold reached"

**Scenario 2: Convergence**

- Accuracy stays at 92% for 10 consecutive iterations
- No improvement detected
- Training stops, model marked as ready
- Completion reason: "Convergence detected"

**Scenario 3: Rollback Limit**

- 5 rollbacks occurred (max: 5)
- Model cannot be improved further
- Training stops, best model marked as final
- Completion reason: "Rollback limit reached - investigation recommended"

**Scenario 4: Maximum Iterations**

- 100 training rounds completed (max: 100)
- Training stops regardless of accuracy
- Completion reason: "Maximum iterations reached"

**10. Benefits:**

- Prevents infinite training loops
- Ensures model reaches acceptable performance
- Detects when further training is not beneficial
- Provides clear completion criteria for research
- Supports operational needs with manual triggers
- Prevents overtraining through early stopping

**Step 6: Training Completion Task**

- Contains final model information:
  - Final model version ID
  - Final accuracy and metrics
  - IPFS CID of final model weights
  - Training summary (total iterations, clients participated, training duration)
  - Metadata (hyperparameters used, dataset information)
- Can be consumed by API/UI to notify that model is ready
- Final state recorded on blockchain

## Data Structures

### On-Chain Storage (Smart Contract State)

ModelVersion structure includes: version_id (unique identifier), parent_version_id (previous version for lineage), timestamp (Unix timestamp), client_ids (which clients contributed), aggregated_hash (SHA-256 of aggregated weights), diff_hash (SHA-256 of encrypted diff), off_chain_location (IPFS CID), hyperparameters (training config as JSON), validation_status (pending/passed/failed), validation_metrics (accuracy, loss, etc.), and block_number (for audit trail).

### Off-Chain Storage

Encrypted diff format includes: encrypted_diff (AES-256 encrypted weight diff bytes), encryption_key_hash (hash of key for key management), compression (optional gzip), and format (serialization format: JSON).

**Storage Location:** IPFS (InterPlanetary File System)

- Content-addressed storage (CID-based)
- Encrypted diffs stored on IPFS network
- Can run local IPFS node for development
- Provides decentralized, immutable storage
- Files referenced by Content Identifier (CID)
- Supports pinning to ensure data persistence

## Language Distribution: Go vs Python

### What's Written in Go?

**Blockchain Service** - A separate microservice that handles all Hyperledger Fabric operations.

**Location:** `blockchain_service/`

**Why Go?**

- Hyperledger Fabric has an official, well-maintained Go SDK
- Avoids dependency conflicts with Python packages
- Clean separation of concerns (blockchain logic isolated)
- Better performance for blockchain operations

**What it does:**

- Provides REST API for blockchain operations
- Interacts with Hyperledger Fabric using the official Go SDK
- Implements blockchain operations: RegisterModelUpdate, RecordValidation, RollbackModel, GetModelProvenance
- Manages Fabric network connections and transactions
- Can be developed and deployed independently

**Architecture:**

- Runs as a separate Docker container
- Exposes HTTP REST API (port 8080)
- Main service calls it via HTTP requests
- Can be optionally disabled/mocked for development

**Hyperledger Fabric Chaincode:**

- Chaincode (smart contract) is also written in Go
- Location: `main_service/blockchain/chaincode/model_provenance.go`
- Defines data structures and smart contract functions
- Runs in a secure Docker container managed by Fabric

### What's Written in Python?

**Main Service** (~70% of codebase):

- All workers (aggregation, storage, validation, rollback)
- API server (FastAPI)
- Queue consumers/producers
- Blockchain client (HTTP client that calls blockchain-service)
- Storage management (IPFS)
- Encryption/decryption
- Validation logic
- Web UI backend

**Client Service** (~20% of codebase):

- Training engine
- Model definitions
- Queue consumers
- Weight computation

**Shared** (~10% of codebase):

- Task models
- Utilities
- Configuration management
- Dataset interfaces

### Communication Between Services

**Python Main Service ↔ Go Blockchain Service:**

- Main service calls blockchain service via HTTP REST API
- Uses `httpx` (async HTTP client, already included with FastAPI)
- No direct Fabric SDK dependencies in Python
- Clean microservices architecture

**Blockchain Service ↔ Hyperledger Fabric:**

- Blockchain service uses the official Hyperledger Fabric Go SDK
- Handles all Fabric network connections, transactions, and chaincode invocations
- Isolated from Python dependency conflicts

**Benefits of This Architecture:**

- No dependency conflicts (Go service is isolated)
- Uses official, maintained Fabric SDK
- Clean separation of concerns
- Can develop/test blockchain service independently
- Main service stays focused on ML workloads

## Smart Contract Design

### Chaincode Functions (Hyperledger Fabric)

The smart contract implements the following functions:

**RegisterModelUpdate:** Records a new model version with version ID, parent version ID, aggregated hash, diff hash, and metadata. Stores ModelVersion in blockchain state and creates an immutable record.

**RecordValidation:** Records validation results with version ID, pass/fail status, and metrics. Updates ValidationStatus and creates ValidationRecord.

**RollbackModel:** Executes rollback from one version to another with a reason. Validates version exists, creates RollbackEvent, and updates current model pointer.

**GetModelProvenance:** Queries complete lineage chain for a version ID. Returns all parent versions and validation history.

**VerifyIntegrity:** Verifies integrity by comparing provided hash with stored hash. Returns true if match, false otherwise.

### What to Store On-Chain

**Model Version Metadata (Critical for Provenance):**

- Version ID (unique identifier)
- Parent Version ID (previous version for lineage)
- Timestamp (when created)
- **Iteration (training iteration number - e.g., 1, 2, 3...) - This is the CURRENT iteration when stored**
- Num Clients (number of clients that participated in aggregation)
- Client IDs (list of client IDs that contributed to this version)
- Aggregated Hash (SHA-256 of aggregated weights)
- Diff Hash (SHA-256 of encrypted diff)
- Off-Chain Location (IPFS CID)
- Hyperparameters (JSON string of training config)
- Validation Status (pending/passed/failed)
- Validation Metrics (JSON string of metrics)
- Block Number (for audit trail)

**Note**: The iteration number stored on-chain represents the current/latest iteration. This allows:

- Late update rejection (check if update iteration < on-chain current iteration)
- Iteration coordination across services
- Preventing replay attacks
- Ensuring all services agree on the current iteration

**Validation Records:**

- Version ID, Validator ID, Passed status, Metrics (JSON: accuracy, loss, etc.), Test Dataset Hash, Timestamp

**Rollback Events:**

- From Version ID, To Version ID, Reason, Triggered By (Validator ID, user ID, or "automatic"), Timestamp, Rollback Type ("automatic" or "manual")

### What NOT to Store On-Chain

- Model weights (too large, expensive)
- Weight diffs (even compressed, too large)
- Raw training data (privacy + size)
- Full model binaries (storage cost prohibitive)

### Benefits of Blockchain in This Context

**1. Immutable Audit Trail (Addresses RQ2):** Every model version is permanently recorded, cannot be deleted or modified retroactively, critical for regulatory compliance (EU AI Act).

**2. Provenance Tracking (Addresses RQ2):** Complete lineage (v1 → v2 → v3 → rollback to v2), can query any version's history, enables reproducibility.

**3. Integrity Verification (Addresses RQ1):** Hash of encrypted diff stored on-chain, can verify off-chain data hasn't been tampered with, detects storage corruption or malicious modification.

**4. Decentralized Trust (Addresses RQ1):** No single point of failure, multiple parties can verify independently, auditors can query blockchain without trusting central authority.

**5. Smart Contract Automation (Addresses RQ1):** Automated rollback logic, validation rules enforced by code, access control via permissions.

## IPFS Storage Implementation

### IPFS Setup

**Local IPFS Node:**

- Run IPFS daemon locally (command: ipfs daemon)
- Default API endpoint: http://localhost:5001
- Can connect to IPFS network or run in isolated mode

**Storage Operations:**

1. Encrypt diff (AES-256-GCM)
2. Upload encrypted diff to IPFS → Get CID
3. Pin CID to ensure persistence
4. Store CID in blockchain metadata
5. Verify hash of encrypted diff matches blockchain hash

**Retrieval Operations:**

1. Get CID from blockchain metadata
2. Retrieve encrypted diff from IPFS using CID
3. Verify integrity (hash comparison)
4. Decrypt diff
5. Apply to model weights

**IPFS Libraries:**

- Python: httpx (async HTTP client, already included with FastAPI)
- Direct HTTP API calls to IPFS daemon using async/await

### IPFS Benefits

- Content-addressed: CID is hash of content (integrity built-in)
- Decentralized: No single point of failure
- Immutable: Content cannot be changed (new content = new CID)
- Open source: No vendor lock-in
- Local development: Can run IPFS node locally

## Diff Storage: Encrypted Diff + Blockchain Hash Verification

### Why Not JWT?

JWT is not ideal because: JWTs are designed for authentication/authorization, not data storage; JWT payloads are base64-encoded (not encrypted) - anyone can read them; JWT size limits would be problematic for model weight diffs; JWT verification relies on signatures, not blockchain hashes.

### Recommended Approach

**Process:**

1. Client computes weight diff (delta between old/new weights)
2. Encrypt diff (AES-256-GCM) - provides confidentiality and authenticated encryption
3. Compute hash of encrypted diff (SHA-256 of encrypted bytes)
4. Store hash on blockchain - immutable record, part of smart contract state
5. Store encrypted diff on IPFS - upload to IPFS network, get Content Identifier (CID), pin CID to ensure persistence

**Verification Process:**

1. Get IPFS CID from blockchain metadata
2. Retrieve encrypted diff from IPFS using CID
3. Compute hash of encrypted diff
4. Query blockchain for stored hash
5. Compare hash of encrypted diff with blockchain hash
6. If match: decrypt and use diff
7. If mismatch: integrity violation detected

**Benefits:**

- Blockchain hash provides integrity verification (like JWT signature)
- Encryption provides confidentiality (better than JWT)
- No size limitations
- Off-chain storage is efficient for large diffs
- Aligns with proposal's "hybrid storage" approach

## Dataset Separation: Training vs Test

### Key Principle

**Global test/validation datasets are ONLY in the main service. Clients NEVER see the global test data.**

**Important:** Clients can train effectively without the global test data because:

- Training only requires training data (to compute gradients/weight updates)
- Test data is only for evaluation, not for training
- Clients can optionally split their local training data into train/val for local monitoring
- The global test set remains independent for fair evaluation

This separation:

- Prevents data leakage (clients can't overfit to global test data)
- Ensures fair evaluation (test data is independent)
- Aligns with federated learning best practices
- Supports research question RQ2 (reproducibility)

### Dataset Distribution

**Client Service (Training Only):**

**What clients have:**

- Training datasets only
- Local data for model training
- Different clients can have different training datasets
- Clients compute weight updates (diffs) from training

**What clients DON'T have:**

- Global test datasets (main service has these)
- Global validation datasets (main service has these)

**What clients CAN have (optional):**

- Local validation split from their training data (for early stopping, monitoring)
  - This is a subset of their training data, not the global test set
  - Used only for local training optimization
  - Does not affect the independence of the global test set

**Main Service (Test/Validation Only):**

**What main service has:**

- Test/validation datasets only
- Used to evaluate model performance
- Never used for training

**What main service DOESN'T have:**

- Training datasets (clients have these)
- Client-specific data

### How Clients Train Without Test Data

**Why This Works:**

**Training only requires training data** because:

1. Gradient computation only needs training examples - loss is computed on training batches, gradients are computed from training loss, weight updates come from gradients
2. Test data is for evaluation, not training - test data is used to measure model performance, it doesn't contribute to weight updates, it's only needed after training is complete
3. This is standard in machine learning - train/test split is fundamental to ML, models are trained on training data, performance is evaluated on test data, same principle applies to federated learning

**Optional: Local Validation Split**

Clients can optionally split their training data for local monitoring. They can split their own training data (e.g., 80/20 split) and use the validation portion for early stopping (stop if validation loss stops improving), local monitoring (track training progress), and hyperparameter tuning (if needed).

**Key Point:** Local validation is a subset of the client's training data, NOT the global test set. This is different from the global test set which is an independent dataset in main service.

### Complete Training Iteration

**Step 1:** Main Service → Client

- Task: "Train with current weights"
- Payload: weights_location (IPFS CID)

**Step 2:** Client (Training)

- Loads training dataset (local)
- Loads weights from IPFS storage
- Trains model
- Computes diff
- Publishes diff to queue

**Step 3:** Main Service (Aggregation)

- Collects diffs from multiple clients
- Aggregates (FedAvg)
- Publishes aggregated diff

**Step 4:** Main Service (Validation)

- Loads TEST dataset (main service only!)
- Applies aggregated diff
- Evaluates on test data
- Records result on blockchain

**Step 5:** Decision

- Evaluates validation results using rollback strategy (see Rollback Strategy section)
- If accuracy acceptable: Continue training
- If rollback needed: Rollback to last best checkpoint

### Dataset Splitting Strategy

The system is designed to be dataset-agnostic, starting with MNIST as the initial dataset. The training dataset is split among clients to simulate federated learning scenarios.

**MNIST Dataset:**

- 60,000 training samples
- 10,000 test samples

**Split Configuration:**

- 2 clients: Each client gets 30,000 training samples (60,000 / 2)
- 4 clients: Each client gets 15,000 training samples (60,000 / 4)
- Main service: Full test set of 10,000 samples (never split)

**Two Split Types Supported:**

**1. IID (Independent and Identically Distributed) Split (Default):**

- Randomly shuffle the full training dataset
- Split evenly into N equal parts (N = number of clients)
- Each client gets a random sample containing all classes (digits 0-9)
- Pros: Simpler, good for initial testing and baseline performance
- Cons: Less realistic (real-world federated scenarios are often non-IID)

**2. Non-IID Split (Optional, for research):**

- Split by class or other criteria to simulate realistic federated scenarios
- Common approaches:
  - Class-based split: Each client gets 2-3 classes primarily (e.g., Client 1: mostly digits 0-4, Client 2: mostly digits 5-9)
  - Quantity-based non-IID: Each client gets different amounts of each class
  - Shard-based: Divide dataset into shards, assign shards to clients
- Pros: More realistic, tests federated learning robustness, better for research
- Cons: More complex, potentially harder to train

**Implementation Approach:**

**Dataset Abstraction Layer:**

- Create a dataset interface that supports multiple datasets
- `load_training_data()` - Returns full training data
- `load_test_data()` - Returns full test data
- `split_for_federation(num_clients, split_type='iid')` - Splits training data for clients

**Dataset Implementations:**

- `MNISTDataset` - Implements interface for MNIST (initial dataset)
- Can extend to `CIFAR10Dataset`, `CustomDataset`, etc. for future datasets

**Client Configuration:**

- Each client loads its assigned portion of the training dataset
- Dataset path/config specifies which portion to load
- Client ID determines which split to use
- Example: Client 1 loads `data/mnist/client_1_train.pt` (30,000 samples for 2-client setup)

**Main Service Configuration:**

- Loads full test dataset (10,000 samples for MNIST)
- Never touches training data
- Example: `test_dataset_path: "data/mnist/test.pt"`

**IID Split Algorithm:**

1. Load full training dataset (60,000 samples for MNIST)
2. Shuffle randomly
3. Split into N equal parts (N = number of clients)
4. Save each part to separate files
5. Each client loads its assigned part

**Non-IID Split Algorithm (Class-based example):**

1. Group training samples by class (0-9 for MNIST)
2. Distribute classes among clients (e.g., Client 1: classes 0-4, Client 2: classes 5-9)
3. Each client gets samples primarily from their assigned classes
4. Save each client's portion to separate files

**Benefits:**

- Dataset-agnostic design allows switching to other datasets (CIFAR-10, custom datasets)
- Supports both IID and non-IID scenarios for comprehensive research
- Easy to scale number of clients (2, 4, 8, etc.)
- Enables comparison of IID vs non-IID performance
- Maintains clear separation between training and test data

### Data Preprocessing and Dataset Preparation

Before training begins, datasets must be preprocessed and split for federated learning. This is a one-time setup step that prepares data files for each client and the main service.

**Preprocessing Script:**

A helper script (`scripts/prepare_datasets.py`) handles dataset preparation:

- Loads the full dataset (e.g., MNIST)
- Splits training data according to configuration (IID or non-IID, number of clients)
- Saves each client's portion to separate files
- Saves test set for main service
- Generates configuration files for clients and main service
- Creates dataset metadata (hashes, statistics)

**Script Usage:**

Command-line interface for dataset preparation:

- `python scripts/prepare_datasets.py --dataset mnist --num_clients 2 --split_type iid --output_dir data/mnist`
- `python scripts/prepare_datasets.py --dataset mnist --num_clients 4 --split_type non_iid --output_dir data/mnist`

**Parameters:**

- `--dataset`: Dataset name (mnist, cifar10, etc.)
- `--num_clients`: Number of clients to split data for
- `--split_type`: iid or non_iid
- `--output_dir`: Directory to save split datasets
- `--seed`: Random seed for reproducibility (optional)

**Output Structure:**

After preprocessing, the following structure is created:

```
data/mnist/
├── train/
│   ├── client_0.pt          # Client 0's training data (30,000 samples for 2 clients)
│   ├── client_1.pt          # Client 1's training data (30,000 samples for 2 clients)
│   └── metadata.json        # Dataset metadata (splits, hashes, statistics)
├── test/
│   └── test.pt              # Full test set (10,000 samples) for main service
└── config/
    ├── client_0_config.json # Client 0 configuration
    ├── client_1_config.json # Client 1 configuration
    └── main_service_config.json # Main service configuration
```

**Client Configuration Files:**

Each client configuration file (`client_X_config.json`) contains:

- `client_id`: Unique client identifier
- `dataset_path`: Path to client's training data file
- `dataset_size`: Number of samples in client's dataset
- `split_type`: IID or non-IID
- `classes_distribution`: Distribution of classes in client's data (for non-IID)
- `dataset_hash`: SHA-256 hash of client's dataset file (for integrity)

**Main Service Configuration:**

Main service configuration file (`main_service_config.json`) contains:

- `test_dataset_path`: Path to test dataset file
- `test_dataset_size`: Number of test samples (10,000 for MNIST)
- `test_dataset_hash`: SHA-256 hash of test dataset file
- `dataset_version`: Version identifier for reproducibility

**Preprocessing Process:**

**1. IID Split Process:**

1. Load full training dataset (60,000 samples for MNIST)
2. Shuffle randomly with seed for reproducibility
3. Split into N equal parts (N = number of clients)
4. Save each part to `data/mnist/train/client_X.pt`
5. Compute hash for each client file
6. Generate client configuration files
7. Save test set to `data/mnist/test/test.pt`
8. Generate main service configuration

**2. Non-IID Split Process:**

1. Load full training dataset
2. Group samples by class (0-9 for MNIST)
3. Distribute classes among clients according to strategy:
   - Class-based: Client 1 gets classes 0-4, Client 2 gets classes 5-9
   - Or other distribution strategy
4. Save each client's portion to separate files
5. Compute statistics (class distribution per client)
6. Generate client configuration files with class distribution
7. Save test set (unchanged, contains all classes)
8. Generate main service configuration

**3. Dataset Metadata:**

Metadata file (`metadata.json`) contains:

- `dataset_name`: Name of dataset (mnist)
- `total_training_samples`: Total training samples (60,000)
- `total_test_samples`: Total test samples (10,000)
- `num_clients`: Number of clients
- `split_type`: IID or non-IID
- `split_seed`: Random seed used for splitting
- `client_splits`: List of client IDs and their sample counts
- `created_at`: Timestamp of preprocessing
- `hashes`: Dictionary of file hashes for integrity verification

**Client Startup:**

When a client service starts:

1. Reads its configuration file (`client_X_config.json`)
2. Loads dataset from path specified in configuration
3. Verifies dataset hash matches configuration (integrity check)
4. Knows exactly which data it has (size, class distribution)
5. Ready to participate in federated training

**Main Service Startup:**

When main service starts:

1. Reads main service configuration file
2. Loads test dataset from specified path
3. Verifies test dataset hash matches configuration
4. Knows test dataset version for reproducibility
5. Ready to validate models

**Benefits:**

- One-time preprocessing step before training
- Clients know their data at startup (no runtime discovery)
- Reproducible splits (seed-based)
- Integrity verification (hash checking)
- Easy to switch between IID and non-IID
- Easy to scale number of clients (re-run preprocessing)
- Dataset-agnostic (works with any dataset that implements the interface)
- Clear separation of concerns (preprocessing vs training)

**Example Workflow:**

1. **Preprocessing (one-time):**

   ```bash
   python scripts/prepare_datasets.py --dataset mnist --num_clients 2 --split_type iid
   ```

2. **Client 0 starts:**

   - Reads `data/mnist/config/client_0_config.json`
   - Loads `data/mnist/train/client_0.pt` (30,000 samples)
   - Verifies hash matches config
   - Ready to train

3. **Client 1 starts:**

   - Reads `data/mnist/config/client_1_config.json`
   - Loads `data/mnist/train/client_1.pt` (30,000 samples)
   - Verifies hash matches config
   - Ready to train

4. **Main service starts:**
   - Reads `data/mnist/config/main_service_config.json`
   - Loads `data/mnist/test/test.pt` (10,000 samples)
   - Verifies hash matches config
   - Ready to validate

### Benefits of This Separation

**1. Prevents Data Leakage:** Clients can't accidentally use test data for training, ensures test data remains independent, critical for fair evaluation.

**2. Supports Federated Learning:** Each client trains on their own data, no central training dataset, true federated scenario.

**3. Enables Reproducibility (RQ2):** Test dataset is fixed and versioned, same test data used for all validations, can reproduce evaluation results.

**4. Regulatory Compliance:** Clear separation of training vs evaluation data, audit trail shows which data was used when, supports EU AI Act requirements.

### Test Dataset Versioning

**Important:** Test datasets should be versioned and their hashes stored on-chain. When validating, record which test dataset was used including test dataset hash and test dataset version. Store this on blockchain via RecordValidation function.

This ensures:

- Reproducibility (same test dataset = comparable results)
- Auditability (can verify which test data was used)
- Research validity (consistent evaluation methodology)

## Security Considerations

### Encryption Strategy

**1. Diff Encryption:**

- Use AES-256-GCM for authenticated encryption
- Key management: Encryption keys stored via ENCRYPTION_KEY environment variable (Base64-encoded 32-byte keys)
- Each version can have unique encryption key

**2. Integrity Verification:**

- On-chain: Hash of encrypted diff
- IPFS CID: Content hash (additional integrity layer)
- Verification: Decrypt → Compute hash → Compare with on-chain hash
- Prevents tampering with IPFS storage

**3. Access Control:**

- Permissioned blockchain (Hyperledger Fabric)
- Simple authentication for queue access (API keys)
- Basic client identification (client IDs in configuration)
- IPFS content is encrypted (only authorized parties can decrypt)

## Queue Design

### Task Types

- TRAIN: Client training task
- AGGREGATE: Aggregation task
- BLOCKCHAIN_WRITE: Blockchain transaction
- STORAGE_WRITE: Off-chain storage
- VALIDATE: Model validation
- ROLLBACK: Model rollback
- DECISION: Post-validation decision
- TRAINING_COMPLETE: Final task indicating training is done and model is ready

### Task Message Format

Task messages include: task_id (unique identifier), task_type (one of the types above), model_version_id, parent_version_id, payload (task-specific data including weights_cid for training tasks, client_updates for aggregation, blockchain_hash for storage tasks, ipfs_cid for validation tasks, validation_result for decision tasks, final_model_info for training completion tasks), and metadata (created_at timestamp, priority, retry_count).

For TRAINING_COMPLETE tasks, the payload includes final_model_info containing: final_model_version_id, final_accuracy, final_metrics (loss, precision, recall, etc.), final_weights_cid (IPFS CID of final model weights), training_summary (total_iterations, clients_participated, training_duration, total_rounds), and metadata (hyperparameters_used, dataset_info, completion_reason).

## Performance Optimization

### For RQ3 (Performance Analysis)

**Metrics to Track:**

- End-to-end latency (client update → validation complete)
- Blockchain transaction latency
- Storage operation latency
- Queue processing time
- Throughput (updates per second)

**Optimization Strategies:**

- Batch blockchain transactions (multiple updates in one transaction)
- Parallel validation workers
- Caching of frequently accessed model weights
- Compression of weight diffs

## Implementation Recommendations

### Technology Stack

- **Language:**
  - Python 3.10+ (main service, client service, workers)
  - Go 1.19+ (Hyperledger Fabric chaincode/smart contracts only)
- **Queue:** RabbitMQ
- **Blockchain:** Hyperledger Fabric 2.5+
- **Storage:** IPFS (InterPlanetary File System)
  - Python library: httpx (async HTTP client for IPFS API)
  - IPFS daemon: Run locally or connect to network
- **ML Framework:** PyTorch
- **Federated Learning:** PySyft
- **API:** FastAPI
- **UI:** React (optional, for monitoring)

### Project Structure

- main_service/ (Python)
  - workers/ (aggregation_worker.py, storage_worker.py, validation_worker.py, rollback_worker.py)
  - api/ (server.py, routes.py)
  - blockchain/ (fabric_client.py - HTTP client for blockchain-service, chaincode/model_provenance.go)
  - storage/ (ipfs_client.py, encryption.py)
  - requirements.txt
  - Dockerfile
- blockchain_service/ (Go)
  - main.go (REST API server)
  - go.mod, go.sum
  - Dockerfile
  - README.md
- client_service/ (Python)
  - training/ (trainer.py, model.py)
  - queue/ (consumer.py)
  - config.py
  - requirements.txt
  - Dockerfile
- shared/ (Python)
  - models/ (task.py)
  - utils/ (crypto.py, hashing.py)
  - datasets/ (dataset_interface.py, mnist_dataset.py)
  - config.py
  - logger.py
- scripts/
  - generate_encryption_key.py
- data/ (created at runtime)
  - mnist/ (downloaded by torchvision)
- tests/
- docker-compose.yml

## Addressing Research Questions

### RQ1: Architecture Integration

**Answer:** Queue-based microservices architecture with specialized workers

**Evidence:** System design document, implementation code

**Metrics:** System availability, fault tolerance

### RQ2: Traceability & Reproducibility

**Answer:** Complete provenance chain stored on blockchain

**Evidence:** Smart contract queries, audit trail logs

**Metrics:** Provenance query time, lineage completeness

### RQ3: Performance Implications

**Answer:** Comprehensive performance benchmarking

**Evidence:** Latency measurements, throughput analysis, storage overhead

**Metrics:** See "Performance Optimization" section above

## Architecture Improvements

### Issues Identified and Solutions

**Issue 1: Sequential Processing Bottleneck**

- Solution: Keep sequential where necessary (Apply diff → Validate), but parallelize where possible (Multiple blockchain writes, multiple storage operations)

**Issue 2: Missing Aggregation Step**

- Solution: Add aggregation worker before blockchain write, combine multiple client updates into one aggregated update

**Issue 3: Rollback Mechanism**

- Solution: Rollback worker queries blockchain for previous version, retrieves previous weights from IPFS storage, updates blockchain (rollback event), posts new training task with previous weights

**Issue 4: Test Dataset Management**

- Solution: Store test dataset hash on-chain, version test datasets, record which dataset was used for each validation

### Implementation Priorities

**Phase 1 (MVP):**

1. Basic queue system
2. Client training service
3. Simple aggregation
4. Blockchain integration (basic)
5. IPFS storage

**Phase 2 (Core Features):**

1. Validation worker
2. Rollback mechanism
3. Smart contract (full functionality)
4. API/UI for monitoring

**Phase 3 (Research):**

1. Performance benchmarking
2. Provenance queries
3. Comparison with baseline
4. Documentation

### Testing Strategy

- **Unit Tests:** Individual workers, encryption, hashing
- **Integration Tests:** Queue → Worker → Blockchain flow
- **End-to-End Tests:** Full training iteration
- **Performance Tests:** Latency, throughput measurements
- **Security Tests:** Encryption, integrity verification
