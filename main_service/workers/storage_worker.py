"""Storage worker that encrypts diffs and stores them on IPFS."""

import asyncio
import json
import time
from typing import Optional, Dict, Any
from shared.queue.consumer import QueueConsumer
from shared.queue.publisher import QueuePublisher
from shared.queue.connection import QueueConnection
from shared.models.task import (
    Task,
    TaskType,
    TaskMetadata,
    StorageWriteTaskPayload,
    ValidateTaskPayload,
)
from shared.storage.encryption import EncryptionService
from main_service.storage.ipfs_client import IPFSClient
from shared.utils.hashing import compute_hash, verify_hash
from shared.logger import setup_logger
from shared.monitoring.metrics import get_metrics_collector

logger = setup_logger(__name__)


class StorageWorker:
    """Worker that encrypts aggregated diffs and stores them on IPFS."""

    def __init__(self, connection: Optional[QueueConnection] = None):
        """
        Initialize storage worker.

        Args:
            connection: QueueConnection instance (creates new one if None)
        """
        self.connection = connection or QueueConnection()
        self.consumer = QueueConsumer(connection=self.connection)
        self.publisher = QueuePublisher(connection=self.connection)
        self.encryption_service = EncryptionService()
        self.running = False

        logger.info("Storage worker initialized")

    async def _encrypt_and_store(
        self,
        aggregated_diff_str: str,
        blockchain_hash: str,
        model_version_id: str,
    ) -> str:
        """
        Encrypt aggregated diff and store on IPFS.

        Args:
            aggregated_diff_str: Aggregated diff as JSON string
            blockchain_hash: Hash from blockchain transaction (hash of unencrypted diff for content integrity)
            model_version_id: Model version identifier

        Returns:
            IPFS CID of stored encrypted diff

        Raises:
            ValueError: If hash verification fails
            RuntimeError: If IPFS operations fail
        """
        # Convert diff string to bytes
        diff_bytes = aggregated_diff_str.encode("utf-8")

        # Verify hash of unencrypted diff matches blockchain hash (content integrity check)
        diff_hash = compute_hash(diff_bytes)
        logger.debug(f"Computed hash of unencrypted diff: {diff_hash}")

        if diff_hash != blockchain_hash:
            error_msg = (
                f"Hash mismatch! Diff hash: {diff_hash}, "
                f"Blockchain hash: {blockchain_hash}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("✓ Hash verification passed - diff content matches blockchain hash")

        # Encrypt the diff
        logger.info(f"Encrypting aggregated diff for version {model_version_id}")
        encrypted_diff = self.encryption_service.encrypt_diff(diff_bytes)
        logger.debug(
            f"Encrypted {len(diff_bytes)} bytes to {len(encrypted_diff)} bytes"
        )

        # Upload to IPFS
        upload_start = time.time()
        logger.info(f"Uploading encrypted diff to IPFS for version {model_version_id}")
        async with IPFSClient() as ipfs_client:
            cid = await ipfs_client.add_bytes(encrypted_diff, pin=True)
            upload_duration = time.time() - upload_start

            get_metrics_collector().record_timing(
                "ipfs_upload",
                upload_duration,
                metadata={
                    "model_version_id": model_version_id,
                    "size_bytes": len(encrypted_diff),
                    "cid": str(cid),
                },
            )

            logger.info(
                f"✓ Uploaded to IPFS: CID={cid} (duration: {upload_duration:.3f}s, "
                f"size: {len(encrypted_diff)} bytes)"
            )

            # Verify pinning
            pin_info = await ipfs_client.pin_ls(cid)
            if cid not in str(pin_info):
                logger.warning(f"CID {cid} may not be pinned correctly")
            else:
                logger.debug(f"✓ CID {cid} is pinned")

        return str(cid)  # Ensure CID is returned as string

    def _handle_storage_task(self, task: Task) -> bool:
        """
        Handle a STORAGE_WRITE task.

        Args:
            task: STORAGE_WRITE task to process

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Processing STORAGE_WRITE task: {task.task_id}")

            # Parse payload
            payload = StorageWriteTaskPayload(**task.payload)

            # Extract aggregated_diff and blockchain_hash from payload
            aggregated_diff_str = payload.aggregated_diff
            blockchain_hash = payload.blockchain_hash
            model_version_id = payload.model_version_id

            # Run async encryption and storage
            loop = asyncio.get_event_loop()
            cid = loop.run_until_complete(
                self._encrypt_and_store(
                    aggregated_diff_str, blockchain_hash, model_version_id
                )
            )

            # Publish VALIDATE task with IPFS CID
            # Use task.parent_version_id as the parent for the validate task
            self._publish_validate_task(model_version_id, cid, task.parent_version_id)

            logger.info(
                f"✓ Storage task completed: version={model_version_id}, CID={cid}"
            )
            return True

        except Exception as e:
            logger.error(
                f"Error processing STORAGE_WRITE task {task.task_id}: {str(e)}",
                exc_info=True,
            )
            return False

    def _publish_validate_task(
        self, model_version_id: str, ipfs_cid: str, parent_version_id: Optional[str]
    ) -> None:
        """
        Publish VALIDATE task to queue.

        Args:
            model_version_id: Model version identifier
            ipfs_cid: IPFS CID of encrypted diff
            parent_version_id: Parent model version ID
        """
        validate_task = Task(
            task_id=f"validate-{model_version_id}-{int(time.time())}",
            task_type=TaskType.VALIDATE,
            payload=ValidateTaskPayload(
                ipfs_cid=ipfs_cid,
                model_version_id=model_version_id,
                parent_version_id=parent_version_id,
            ).model_dump(),
            metadata=TaskMetadata(source="storage_worker"),
            model_version_id=model_version_id,
            parent_version_id=parent_version_id,
        )

        queue_name = "validate"
        self.publisher.publish_task(validate_task, queue_name)

        logger.info(
            f"Published VALIDATE task for version {model_version_id} "
            f"to queue '{queue_name}'"
        )

    def start(self, queue_name: str = "storage_write") -> None:
        """
        Start consuming storage tasks from queue.

        Args:
            queue_name: Name of the queue to consume from
        """
        self.running = True
        logger.info(f"Starting storage worker, consuming from queue: {queue_name}")

        def task_handler(task: Task) -> None:
            """Handle received task."""
            try:
                if task.task_type == TaskType.STORAGE_WRITE:
                    success = self._handle_storage_task(task)
                    if not success:
                        logger.error(
                            f"Failed to process STORAGE_WRITE task {task.task_id}"
                        )
                else:
                    logger.warning(
                        f"Received unexpected task type: {task.task_type}, ignoring"
                    )
            except Exception as e:
                logger.error(f"Error handling task: {str(e)}", exc_info=True)

        try:
            # Use consume_tasks for Task-based message handling
            self.consumer.consume_tasks(queue_name, task_handler)
        except KeyboardInterrupt:
            logger.info("Storage worker interrupted by user")
        except Exception as e:
            logger.error(f"Fatal error in storage worker: {str(e)}", exc_info=True)
        finally:
            self.running = False
            logger.info("Storage worker stopped")

    def stop(self) -> None:
        """Stop the storage worker."""
        self.running = False
        logger.info("Stopping storage worker")
