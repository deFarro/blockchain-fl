# Blockchain-Enabled Learning Systems for Verifiable Model Provenance

> **MSc Thesis Project** > **Topic:** Blockchain-Enabled Learning Systems for Verifiable Model Provenance, Rollback, and Regulatory Auditability.

## Project Overview

This repository contains the prototype implementation for a Master's thesis investigating the convergence of **Distributed Ledger Technology (DLT)** and **Federated Learning (FL)**.

The project aims to address the opacity of AI development by designing a blockchain-enabled provenance system. The system records training metadata, model updates, and ownership changes on an immutable ledger to support trust, reproducibility, and compliance with emerging regulations such as the **EU AI Act**.

### Key Capabilities

- **Immutable Audit Trails:** Records training events and metadata on a permissioned blockchain (Hyperledger Fabric)
- **Model Rollback:** Enables the restoration of previous model states to mitigate poisoning attacks or errors.
- **Hybrid Storage:** Utilises a "Selective Decentralisation" architecture where critical metadata is stored on-chain, while heavy model artefacts are stored off-chain (e.g., IPFS or local secure storage).
- **Privacy-Preserving:** Integrates with Federated Learning workflows (PySyft) to simulate training without exposing raw local data.

---

## Research Questions

This prototype is designed specifically to answer the following research questions defined in the thesis proposal:

1.  **RQ1 (Architecture):** How can blockchain technology be integrated into learning systems to enable verifiable model provenance, rollback, and auditability while maintaining efficiency?
2.  **RQ2 (Trust & Auditability):** To what extent can blockchain-based provenance tracking improve the traceability and reproducibility of ML models compared with conventional systems?
3.  **RQ3 (Performance):** What are the performance implications (latency, overhead, storage) of incorporating blockchain into the training process?

---

## Technical Architecture

The solution follows a **Design Science Research (DSR)** methodology. The architecture consists of three primary layers:

1.  **Application Layer (AI Client):**
    - Python-based Federated Learning client.
    - Handles local model training and communicates updates to the aggregator.
2.  **Blockchain Layer (Trust Anchor):**
    - **Hyperledger Fabric** (Permissioned Blockchain).
    - **Smart Contracts (Chaincode):** Handles model registration, version hashing, and rollback logic.
3.  **Storage Layer:**
    - **Off-chain:** Stores large model binaries/weights/diffs.
    - **On-chain:** Stores SHA-256 hashes of the models and metadata (hyperparameters, timestamps).

---
