"""FastAPI server for main service API."""

import asyncio
import threading
import time
import uvicorn
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from shared.logger import setup_logger
from shared.config import settings
from main_service.api import routes
from main_service.workers.aggregation_worker import AggregationWorker
from main_service.workers.decision_worker import DecisionWorker
from main_service.workers.validation_worker import ValidationWorker
from main_service.workers.blockchain_worker import BlockchainWorker
from main_service.workers.storage_worker import StorageWorker
from main_service.workers.rollback_worker import RollbackWorker
from main_service.blockchain.fabric_client import FabricClient

logger = setup_logger(__name__)


def _get_current_iteration_from_blockchain(
    blockchain_worker: Optional[BlockchainWorker] = None,
) -> Optional[int]:
    """
    Get the current iteration from the latest model version on blockchain.

    Args:
        blockchain_worker: Optional blockchain worker instance to get latest version ID

    Returns:
        Current iteration number, or None if no versions exist
    """
    try:
        # Use a new event loop for this synchronous call
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def get_iteration():
            # Get latest version from blockchain worker if available
            latest_version_id = None
            if blockchain_worker:
                latest_version_id = blockchain_worker.get_latest_model_version_id()

            if not latest_version_id:
                return None

            # Query blockchain for latest version
            async with FabricClient() as blockchain_client:
                provenance = await blockchain_client.get_model_provenance(
                    latest_version_id
                )

            # Extract iteration from metadata
            metadata = provenance.get("metadata", {})
            if isinstance(metadata, dict):
                iteration = metadata.get("iteration")
                if iteration is not None:
                    return int(iteration)

            # Fallback: check top-level iteration field
            iteration = provenance.get("iteration")
            if iteration is not None:
                return int(iteration)

            return None

        iteration: Optional[int] = loop.run_until_complete(get_iteration())
        loop.close()
        return iteration
    except Exception as e:
        logger.warning(f"Failed to get current iteration from blockchain: {str(e)}")
        return None


# Global worker instances (to stop them on shutdown)
workers: Dict[str, Any] = {}
worker_threads: Dict[str, threading.Thread] = {}

# Create FastAPI application
app = FastAPI(
    title="Blockchain-Enabled Federated Learning - Main Service API",
    description="REST API for model management, training control, and monitoring",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers first (they take precedence over static file mounts)
app.include_router(routes.router)

# Serve dashboard (built React app) - mount after API routes
# FastAPI will check API routes first, then fall back to static files
dashboard_path = Path(__file__).parent.parent.parent / "dashboard" / "dist"
if dashboard_path.exists() and (dashboard_path / "index.html").exists():
    # Mount the entire dist directory at root
    # This serves index.html and all assets (CSS, JS, images, etc.)
    # API routes defined above will take precedence
    app.mount(
        "/",
        StaticFiles(directory=str(dashboard_path), html=True),
        name="dashboard",
    )
    logger.info(f"Serving dashboard from {dashboard_path}")
else:
    logger.warning(f"Dashboard not found at {dashboard_path}. API only mode.")


@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    logger.info("Main service API server starting up")

    # Start all workers in background threads
    try:

        def start_worker_in_thread(worker_name: str, worker_instance, queue_name: str):
            """Start a worker in a background thread."""

            def worker_thread():
                try:
                    logger.info(f"Starting {worker_name} worker thread")
                    worker_instance.start(queue_name)
                except Exception as e:
                    logger.error(
                        f"Error in {worker_name} worker: {str(e)}", exc_info=True
                    )

            thread = threading.Thread(
                target=worker_thread, daemon=True, name=f"{worker_name}-worker"
            )
            thread.start()
            return thread

        # Start decision worker (processes validation results and publishes train tasks)
        decision_worker = DecisionWorker()
        workers["decision"] = decision_worker
        worker_threads["decision"] = start_worker_in_thread(
            "decision", decision_worker, "decision_queue"
        )

        # Start validation worker (validates aggregated models)
        validation_worker = ValidationWorker()
        workers["validation"] = validation_worker
        worker_threads["validation"] = start_worker_in_thread(
            "validation", validation_worker, "validate"
        )

        # Start aggregation worker (aggregates client updates)
        aggregation_worker = AggregationWorker()
        workers["aggregation"] = aggregation_worker

        # Create a worker thread that continuously processes client updates
        def aggregation_worker_thread():
            """Continuously process client updates and aggregate them by iteration."""
            try:
                logger.info(
                    "Starting aggregation worker thread (will process iterations as updates arrive)"
                )

                # Wait a bit for blockchain worker to be initialized
                time.sleep(1)

                # Get initial iteration from blockchain (will get blockchain worker dynamically)
                current_iteration = _get_current_iteration_from_blockchain(
                    workers.get("blockchain") if "blockchain" in workers else None
                )
                if current_iteration is None:
                    current_iteration = 1  # Start from 1 if no versions exist
                else:
                    # Start from next iteration after the latest on blockchain
                    current_iteration += 1
                    logger.info(
                        f"Starting aggregation from iteration {current_iteration} "
                        f"(latest on blockchain: {current_iteration - 1})"
                    )

                max_iteration = 100  # Safety limit
                sync_counter = 0  # Counter to periodically sync with blockchain

                while True:
                    try:
                        # Periodically sync with blockchain to handle rollbacks
                        # Check every 5 iterations to avoid too frequent blockchain queries
                        sync_counter += 1
                        if sync_counter >= 5:
                            sync_counter = 0
                            # Get blockchain worker reference dynamically (in case it wasn't available at startup)
                            blockchain_worker_ref = (
                                workers.get("blockchain")
                                if "blockchain" in workers
                                else None
                            )
                            blockchain_iteration = (
                                _get_current_iteration_from_blockchain(
                                    blockchain_worker_ref
                                )
                            )
                            if blockchain_iteration is not None:
                                # If blockchain is ahead or we're way behind, sync up
                                expected_next = blockchain_iteration + 1
                                if current_iteration < expected_next - 5:
                                    logger.info(
                                        f"Syncing aggregation iteration: {current_iteration} -> {expected_next} "
                                        f"(blockchain iteration: {blockchain_iteration})"
                                    )
                                    current_iteration = expected_next
                                elif current_iteration < expected_next:
                                    # Just slightly behind, catch up
                                    current_iteration = expected_next

                        # Try to process the current iteration
                        # This will wait for enough client updates for this iteration
                        logger.debug(
                            f"Attempting to aggregate iteration {current_iteration}"
                        )
                        success = aggregation_worker.process_client_updates(
                            queue_name="client_updates",
                            iteration=current_iteration,
                            timeout=settings.aggregation_timeout,
                            min_clients=settings.min_clients_for_aggregation,
                        )

                        if success:
                            logger.info(
                                f"Successfully aggregated iteration {current_iteration}"
                            )
                            current_iteration += 1
                            # Don't sleep - immediately try next iteration
                        else:
                            # Not enough updates yet, wait a bit before retrying
                            time.sleep(2)

                        # Safety check
                        if current_iteration > max_iteration:
                            logger.warning(
                                f"Reached max iteration {max_iteration}, stopping aggregation worker"
                            )
                            break

                    except Exception as e:
                        logger.error(
                            f"Error in aggregation worker loop: {str(e)}", exc_info=True
                        )
                        time.sleep(5)  # Wait before retrying on error
            except KeyboardInterrupt:
                logger.info("Aggregation worker thread interrupted")
            except Exception as e:
                logger.error(
                    f"Fatal error in aggregation worker thread: {str(e)}", exc_info=True
                )

        worker_threads["aggregation"] = threading.Thread(
            target=aggregation_worker_thread, daemon=True, name="aggregation-worker"
        )
        worker_threads["aggregation"].start()

        # Start storage worker (stores encrypted diffs to IPFS)
        storage_worker = StorageWorker()
        workers["storage"] = storage_worker
        worker_threads["storage"] = start_worker_in_thread(
            "storage", storage_worker, "storage_write"
        )

        # Start blockchain worker (records updates on blockchain)
        blockchain_worker = BlockchainWorker()
        workers["blockchain"] = blockchain_worker
        worker_threads["blockchain"] = start_worker_in_thread(
            "blockchain", blockchain_worker, "blockchain_write"
        )

        # Start separate blockchain worker instance for TRAINING_COMPLETE tasks
        # (needs separate connection to avoid RabbitMQ connection conflicts)
        blockchain_completion_worker = BlockchainWorker()
        workers["blockchain_completion"] = blockchain_completion_worker
        worker_threads["blockchain_completion"] = start_worker_in_thread(
            "blockchain_completion",
            blockchain_completion_worker,
            "training_complete_queue",
        )

        # Start rollback worker (handles rollback tasks)
        rollback_worker = RollbackWorker()
        workers["rollback"] = rollback_worker
        worker_threads["rollback"] = start_worker_in_thread(
            "rollback", rollback_worker, "rollback_queue"
        )

        logger.info("All workers started successfully")

    except Exception as e:
        logger.error(f"Error starting workers: {str(e)}", exc_info=True)
        logger.warning(
            "Continuing without workers - API will work but training won't process"
        )


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown."""
    logger.info("Main service API server shutting down")

    # Stop all workers
    for worker_name, worker in workers.items():
        try:
            logger.info(f"Stopping {worker_name} worker...")
            worker.stop()
        except Exception as e:
            logger.error(f"Error stopping {worker_name} worker: {str(e)}")

    logger.info("All workers stopped")


if __name__ == "__main__":
    port = settings.api_port
    host = settings.api_host

    logger.info(f"Starting API server on {host}:{port}")
    # Disable access logs to reduce verbosity (application logs still work)
    uvicorn.run(
        app,
        host=host,
        port=port,
        access_log=False,  # Disable HTTP access logs to reduce noise from polling
    )
