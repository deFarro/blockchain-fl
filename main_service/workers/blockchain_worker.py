"""Blockchain worker that records model updates on blockchain."""

import asyncio
import json
import time
import uuid
from typing import Optional, Dict, Any
from shared.queue.consumer import QueueConsumer
from shared.queue.publisher import QueuePublisher
from shared.queue.connection import QueueConnection
from shared.models.task import (
    Task,
    TaskType,
    TaskMetadata,
    BlockchainWriteTaskPayload,
    StorageWriteTaskPayload,
)
from main_service.blockchain.fabric_client import FabricClient
from shared.utils.hashing import compute_hash
from shared.logger import setup_logger
from shared.monitoring.metrics import get_metrics_collector
from shared.storage.ipfs_client import IPFSClient

logger = setup_logger(__name__)


class BlockchainWorker:
    """Worker that records model updates on blockchain via blockchain service API."""

    def __init__(self, connection: Optional[QueueConnection] = None):
        """
        Initialize blockchain worker.

        Args:
            connection: QueueConnection instance (creates new one if None)
        """
        self.connection = connection or QueueConnection()
        self.consumer = QueueConsumer(connection=self.connection)
        self.publisher = QueuePublisher(connection=self.connection)
        self.running = False
        # Track current model version for parent_version_id
        self.current_model_version_id: Optional[str] = None
        # Track latest model version ID (alias for current_model_version_id)
        self.latest_model_version_id: Optional[str] = None

        logger.info("Blockchain worker initialized")

    def _generate_model_version_id(self, iteration: int) -> str:
        """
        Generate a unique model version ID.

        Args:
            iteration: Training iteration number

        Returns:
            Unique version ID
        """
        timestamp = int(time.time())
        unique_id = str(uuid.uuid4())[:8]
        return f"model_v{iteration}_{timestamp}_{unique_id}"

    def _compute_diff_hash(self, aggregated_diff_str: str) -> str:
        """
        Compute hash of aggregated diff (before encryption).

        This hash represents the integrity of the diff content and will be
        stored on blockchain. The storage worker will encrypt the diff and
        store it on IPFS. The blockchain hash serves as a content integrity
        check (verifying the diff hasn't been tampered with).

        Args:
            aggregated_diff_str: Aggregated diff as JSON string

        Returns:
            Hash of unencrypted diff (SHA-256)
        """
        # Convert to bytes
        diff_bytes = aggregated_diff_str.encode("utf-8")

        # Compute hash of unencrypted diff
        diff_hash = compute_hash(diff_bytes)

        logger.debug(f"Computed hash of aggregated diff: {diff_hash[:16]}...")
        return diff_hash

    async def _register_on_blockchain(
        self,
        model_version_id: str,
        parent_version_id: Optional[str],
        blockchain_hash: str,
        metadata: Dict[str, Any],
    ) -> str:
        """
        Register model update on blockchain via blockchain service API.

        Args:
            model_version_id: Unique version identifier
            parent_version_id: Parent version ID (None for initial version)
            blockchain_hash: Hash of encrypted diff
            metadata: Additional metadata (iteration, num_clients, client_ids, etc.)

        Returns:
            Transaction ID from blockchain service
        """
        start_time = time.time()
        logger.info(
            f"Registering model version {model_version_id} on blockchain "
            f"(parent: {parent_version_id})"
        )

        try:
            async with FabricClient() as blockchain_client:
                transaction_id = await blockchain_client.register_model_update(
                    model_version_id=model_version_id,
                    parent_version_id=parent_version_id,
                    hash_value=blockchain_hash,
                    metadata=metadata,
                )

            # Type assertion: register_model_update returns str (validated in fabric_client)
            if not isinstance(transaction_id, str):
                raise ValueError(f"Expected str, got {type(transaction_id)}")

            duration = time.time() - start_time
            get_metrics_collector().record_timing(
                "blockchain_register",
                duration,
                metadata={
                    "model_version_id": model_version_id,
                    "iteration": metadata.get("iteration"),
                    "transaction_id": transaction_id,
                },
            )

            logger.info(
                f"Model version {model_version_id} registered on blockchain: "
                f"tx_id={transaction_id} (duration: {duration:.3f}s)"
            )
            return transaction_id
        except Exception as e:
            duration = time.time() - start_time
            get_metrics_collector().record_timing(
                "blockchain_register",
                duration,
                metadata={
                    "model_version_id": model_version_id,
                    "status": "error",
                    "error": str(e),
                },
            )
            raise

    async def _process_blockchain_write(
        self,
        aggregated_diff_str: str,
        iteration: int,
        num_clients: int,
        client_ids: list,
    ) -> Dict[str, Any]:
        """
        Process blockchain write operation.

        Args:
            aggregated_diff_str: Aggregated diff as JSON string
            iteration: Training iteration number
            num_clients: Number of clients that participated
            client_ids: List of client IDs that contributed

        Returns:
            Dictionary with model_version_id, parent_version_id, blockchain_hash, transaction_id
        """
        # Generate model version ID
        model_version_id = self._generate_model_version_id(iteration)

        # Get parent version ID (from previous iteration)
        parent_version_id = self.current_model_version_id

        # Compute hash of unencrypted diff (content integrity check)
        blockchain_hash = self._compute_diff_hash(aggregated_diff_str)

        # Get parent metadata to carry forward rollback_count and validation_history
        parent_metadata = {}
        rollback_count = 0
        validation_history = []

        if parent_version_id:
            try:
                async with FabricClient() as blockchain_client:
                    parent_provenance = await blockchain_client.get_model_provenance(
                        parent_version_id
                    )
                    parent_metadata = parent_provenance.get("metadata", {})
                    rollback_count = parent_metadata.get("rollback_count", 0)
                    validation_history = parent_metadata.get("validation_history", [])
            except Exception as e:
                logger.warning(
                    f"Could not fetch parent metadata for {parent_version_id}: {e}. "
                    "Starting with fresh metadata."
                )

        # Prepare metadata for blockchain
        # Carry forward rollback_count and validation_history from parent
        metadata = {
            "iteration": iteration,
            "num_clients": num_clients,
            "client_ids": client_ids,
            "diff_hash": blockchain_hash,  # Hash of encrypted diff
            "rollback_count": rollback_count,  # Carried forward from parent
            "validation_history": validation_history,  # Carried forward from parent
            # ipfs_cid will be added later by storage worker
        }

        # Register on blockchain
        transaction_id = await self._register_on_blockchain(
            model_version_id=model_version_id,
            parent_version_id=parent_version_id,
            blockchain_hash=blockchain_hash,
            metadata=metadata,
        )

        # Update current and latest model version
        self.current_model_version_id = model_version_id
        self.latest_model_version_id = model_version_id

        return {
            "model_version_id": model_version_id,
            "parent_version_id": parent_version_id,
            "blockchain_hash": blockchain_hash,
            "transaction_id": transaction_id,
        }

    def _publish_storage_task(
        self,
        aggregated_diff_cid: str,
        blockchain_hash: str,
        model_version_id: str,
        parent_version_id: Optional[str],
    ) -> None:
        """
        Publish STORAGE_WRITE task to queue.

        Args:
            aggregated_diff_cid: IPFS CID of aggregated diff (not the diff itself)
            blockchain_hash: Hash from blockchain transaction
            model_version_id: Model version identifier
            parent_version_id: Parent version ID
        """
        # Create payload - ensure we only include the CID, not the actual diff
        payload_dict = StorageWriteTaskPayload(
            aggregated_diff_cid=aggregated_diff_cid,
            blockchain_hash=blockchain_hash,
            model_version_id=model_version_id,
        ).model_dump()

        # Verify payload doesn't contain large data
        payload_size = len(json.dumps(payload_dict).encode("utf-8"))
        if payload_size > 10000:  # More than 10KB is suspicious
            logger.error(
                f"WARNING: Storage task payload is suspiciously large: {payload_size} bytes. "
                f"Payload keys: {list(payload_dict.keys())}"
            )

        storage_task = Task(
            task_id=f"storage-{model_version_id}-{int(time.time())}",
            task_type=TaskType.STORAGE_WRITE,
            payload=payload_dict,
            metadata=TaskMetadata(source="blockchain_worker"),
            model_version_id=model_version_id,
            parent_version_id=parent_version_id,
        )

        queue_name = "storage_write"

        # Log message size before publishing
        task_dict = storage_task.to_dict()
        message_size = len(json.dumps(task_dict).encode("utf-8"))
        logger.debug(
            f"Publishing STORAGE_WRITE task: message_size={message_size} bytes, "
            f"payload_size={payload_size} bytes, aggregated_diff_cid={aggregated_diff_cid}"
        )

        if message_size > 100000:  # More than 100KB is definitely wrong
            logger.error(
                f"ERROR: Task message is too large: {message_size} bytes! "
                f"This should only contain metadata and CIDs. "
                f"Task payload: {json.dumps(payload_dict, indent=2)[:500]}"
            )
            raise ValueError(
                f"Task message too large ({message_size} bytes). "
                f"Only metadata and IPFS CIDs should be sent through RabbitMQ."
            )

        self.publisher.publish_task(storage_task, queue_name)

        logger.info(
            f"Published STORAGE_WRITE task for version {model_version_id} "
            f"to queue '{queue_name}'"
        )

    def _handle_blockchain_write_task(self, task: Task) -> bool:
        """
        Handle a BLOCKCHAIN_WRITE task.

        Args:
            task: BLOCKCHAIN_WRITE task to process

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Processing BLOCKCHAIN_WRITE task: {task.task_id}")

            # Parse payload (not using BlockchainWriteTaskPayload since aggregation worker
            # publishes a simplified payload)
            payload = task.payload

            # Create event loop once for all async operations
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                # No event loop in current thread, create new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Get aggregated diff CID from IPFS (aggregation worker uploads it to avoid frame size limits)
            aggregated_diff_cid = payload.get("aggregated_diff_cid")
            if not aggregated_diff_cid:
                raise ValueError(
                    "Missing 'aggregated_diff_cid' in payload. "
                    "Only IPFS CIDs should be sent through RabbitMQ."
                )

            # Download aggregated diff from IPFS (needed for hash computation)
            logger.info(
                f"Downloading aggregated diff from IPFS: CID={aggregated_diff_cid}"
            )

            async def download_from_ipfs():
                async with IPFSClient() as ipfs_client:
                    aggregated_diff_bytes = await ipfs_client.get_bytes(
                        aggregated_diff_cid
                    )
                    return aggregated_diff_bytes.decode("utf-8")

            aggregated_diff_str = loop.run_until_complete(download_from_ipfs())
            logger.info(
                f"Downloaded aggregated diff from IPFS: CID={aggregated_diff_cid}, "
                f"size={len(aggregated_diff_str.encode('utf-8'))} bytes"
            )

            iteration = payload.get("iteration")
            if iteration is None:
                raise ValueError("Missing 'iteration' in payload")

            num_clients = payload.get("num_clients", 0)
            client_ids = payload.get("client_ids", [])

            # Run async blockchain processing (reuse the same event loop)
            result = loop.run_until_complete(
                self._process_blockchain_write(
                    aggregated_diff_str=aggregated_diff_str,
                    iteration=iteration,
                    num_clients=num_clients,
                    client_ids=client_ids,
                )
            )

            # Publish STORAGE_WRITE task with IPFS CID (not the diff itself)
            # The storage worker will download from IPFS using this CID
            self._publish_storage_task(
                aggregated_diff_cid=aggregated_diff_cid,
                blockchain_hash=result["blockchain_hash"],
                model_version_id=result["model_version_id"],
                parent_version_id=result["parent_version_id"],
            )

            logger.info(
                f"BLOCKCHAIN_WRITE task completed: "
                f"version={result['model_version_id']}, "
                f"tx_id={result['transaction_id']}"
            )
            return True

        except Exception as e:
            logger.error(
                f"Error processing BLOCKCHAIN_WRITE task {task.task_id}: {str(e)}",
                exc_info=True,
            )
            return False

    def start(self, queue_name: str = "blockchain_write") -> None:
        """
        Start blockchain worker.

        Args:
            queue_name: Queue name to consume from
        """
        logger.info(f"Starting blockchain worker (consuming from {queue_name})")
        self.running = True

        def task_handler(task: Task) -> None:
            """Handle incoming tasks."""
            if task.task_type == TaskType.BLOCKCHAIN_WRITE:
                success = self._handle_blockchain_write_task(task)
                if not success:
                    logger.error(
                        f"Failed to process BLOCKCHAIN_WRITE task {task.task_id}"
                    )
            else:
                logger.warning(
                    f"Received unexpected task type: {task.task_type} "
                    f"(expected BLOCKCHAIN_WRITE)"
                )

        self.consumer.consume_tasks(queue_name, task_handler)

    def get_latest_model_version_id(self) -> Optional[str]:
        """
        Get the latest model version ID.

        Returns:
            Latest model version ID, or None if no versions exist
        """
        return self.latest_model_version_id

    def stop(self) -> None:
        """Stop blockchain worker."""
        self.running = False
        logger.info("Stopping blockchain worker")
