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
        logger.info(
            f"Registering model version {model_version_id} on blockchain "
            f"(parent: {parent_version_id})"
        )

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

        logger.info(
            f"✓ Model version {model_version_id} registered on blockchain: "
            f"tx_id={transaction_id}"
        )
        return transaction_id

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

        # Prepare metadata for blockchain
        metadata = {
            "iteration": iteration,
            "num_clients": num_clients,
            "client_ids": client_ids,
            "diff_hash": blockchain_hash,  # Hash of encrypted diff
            # ipfs_cid will be added later by storage worker
        }

        # Register on blockchain
        transaction_id = await self._register_on_blockchain(
            model_version_id=model_version_id,
            parent_version_id=parent_version_id,
            blockchain_hash=blockchain_hash,
            metadata=metadata,
        )

        # Update current model version
        self.current_model_version_id = model_version_id

        return {
            "model_version_id": model_version_id,
            "parent_version_id": parent_version_id,
            "blockchain_hash": blockchain_hash,
            "transaction_id": transaction_id,
        }

    def _publish_storage_task(
        self,
        aggregated_diff_str: str,
        blockchain_hash: str,
        model_version_id: str,
        parent_version_id: Optional[str],
    ) -> None:
        """
        Publish STORAGE_WRITE task to queue.

        Args:
            aggregated_diff_str: Aggregated diff as JSON string
            blockchain_hash: Hash from blockchain transaction
            model_version_id: Model version identifier
            parent_version_id: Parent version ID
        """
        storage_task = Task(
            task_id=f"storage-{model_version_id}-{int(time.time())}",
            task_type=TaskType.STORAGE_WRITE,
            payload=StorageWriteTaskPayload(
                aggregated_diff=aggregated_diff_str,
                blockchain_hash=blockchain_hash,
                model_version_id=model_version_id,
            ).model_dump(),
            metadata=TaskMetadata(source="blockchain_worker"),
            model_version_id=model_version_id,
            parent_version_id=parent_version_id,
        )

        queue_name = "storage_write"
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

            aggregated_diff_str = payload.get("aggregated_diff")
            if not aggregated_diff_str:
                raise ValueError("Missing 'aggregated_diff' in payload")

            iteration = payload.get("iteration")
            if iteration is None:
                raise ValueError("Missing 'iteration' in payload")

            num_clients = payload.get("num_clients", 0)
            client_ids = payload.get("client_ids", [])

            # Run async blockchain processing
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(
                self._process_blockchain_write(
                    aggregated_diff_str=aggregated_diff_str,
                    iteration=iteration,
                    num_clients=num_clients,
                    client_ids=client_ids,
                )
            )

            # Publish STORAGE_WRITE task
            self._publish_storage_task(
                aggregated_diff_str=aggregated_diff_str,
                blockchain_hash=result["blockchain_hash"],
                model_version_id=result["model_version_id"],
                parent_version_id=result["parent_version_id"],
            )

            logger.info(
                f"✓ BLOCKCHAIN_WRITE task completed: "
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

    def stop(self) -> None:
        """Stop blockchain worker."""
        self.running = False
        logger.info("Stopping blockchain worker")
