"""Rollback worker that handles rollback operations."""

import asyncio
from typing import Optional, cast
from shared.queue.consumer import QueueConsumer
from shared.queue.publisher import QueuePublisher
from shared.queue.connection import QueueConnection
from shared.models.task import (
    Task,
    TaskType,
    TaskMetadata,
    RollbackTaskPayload,
)
from main_service.blockchain.fabric_client import FabricClient
from shared.storage.ipfs_client import IPFSClient
from shared.logger import setup_logger

logger = setup_logger(__name__)


class RollbackWorker:
    """Worker that handles rollback operations."""

    def __init__(self, connection: Optional[QueueConnection] = None):
        """
        Initialize rollback worker.

        Args:
            connection: QueueConnection instance (creates new one if None)
        """
        self.connection = connection or QueueConnection()
        self.consumer = QueueConsumer(connection=self.connection)
        self.publisher = QueuePublisher(connection=self.connection)
        self.running = False

        logger.info("Rollback worker initialized")

    async def _verify_rollback_target(self, target_weights_cid: str) -> bool:
        """
        Verify that rollback target weights are accessible from IPFS.

        Args:
            target_weights_cid: IPFS CID of target weights

        Returns:
            True if weights are accessible, False otherwise
        """
        try:
            logger.info(f"Verifying rollback target weights: CID={target_weights_cid}")

            async with IPFSClient() as ipfs_client:
                weights_data = await ipfs_client.get_bytes(target_weights_cid)
                if weights_data:
                    logger.info(
                        f"✓ Rollback target weights verified: {len(weights_data)} bytes"
                    )
                    return True
                else:
                    logger.error(
                        f"✗ Rollback target weights not found: CID={target_weights_cid}"
                    )
                    return False

        except Exception as e:
            logger.error(
                f"Error verifying rollback target weights {target_weights_cid}: {str(e)}",
                exc_info=True,
            )
            return False

    async def _record_rollback_on_blockchain(
        self, target_version_id: str, reason: str
    ) -> Optional[str]:
        """
        Record rollback event on blockchain.

        Args:
            target_version_id: Version ID to rollback to
            reason: Reason for rollback

        Returns:
            Transaction ID if successful, None otherwise
        """
        try:
            logger.info(
                f"Recording rollback on blockchain: target_version={target_version_id}"
            )

            async with FabricClient() as blockchain_client:
                transaction_id_raw = await blockchain_client.rollback_model(
                    target_version_id=target_version_id,
                    reason=reason,
                )
                transaction_id: str = cast(str, transaction_id_raw)

            if transaction_id:
                logger.info(
                    f"✓ Rollback recorded on blockchain: tx_id={transaction_id}"
                )
                return transaction_id
            else:
                logger.error("✗ Failed to record rollback on blockchain")
                return None

        except Exception as e:
            logger.error(
                f"Error recording rollback on blockchain: {str(e)}",
                exc_info=True,
            )
            return None

    async def _process_rollback(
        self,
        target_version_id: str,
        target_weights_cid: str,
        reason: str,
        cutoff_version_id: Optional[str],
    ) -> bool:
        """
        Process rollback operation.

        Args:
            target_version_id: Version ID to rollback to
            target_weights_cid: IPFS CID of target weights
            reason: Reason for rollback
            cutoff_version_id: Version ID after which updates should be discarded

        Returns:
            True if successful, False otherwise
        """
        logger.info(
            f"Processing rollback: target_version={target_version_id}, "
            f"reason={reason}, cutoff_version={cutoff_version_id}"
        )

        # Verify rollback target is accessible
        is_accessible = await self._verify_rollback_target(target_weights_cid)
        if not is_accessible:
            logger.error(
                f"Rollback target {target_weights_cid} is not accessible. "
                "Rollback cannot proceed."
            )
            return False

        # Record rollback on blockchain
        transaction_id = await self._record_rollback_on_blockchain(
            target_version_id, reason
        )
        if not transaction_id:
            logger.warning(
                "Failed to record rollback on blockchain, but continuing with rollback"
            )

        logger.info(
            f"✓ Rollback processed successfully: target_version={target_version_id}, "
            f"tx_id={transaction_id}"
        )
        return True

    def _handle_rollback_task(self, task: Task) -> bool:
        """
        Handle a ROLLBACK task.

        Args:
            task: ROLLBACK task to process

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Processing ROLLBACK task: {task.task_id}")

            # Parse payload
            payload = RollbackTaskPayload(**task.payload)

            target_version_id = payload.target_version_id
            target_weights_cid = payload.target_weights_cid
            reason = payload.reason
            cutoff_version_id = payload.cutoff_version_id

            # Run async rollback processing
            loop = asyncio.get_event_loop()
            success = loop.run_until_complete(
                self._process_rollback(
                    target_version_id,
                    target_weights_cid,
                    reason,
                    cutoff_version_id,
                )
            )

            if success:
                logger.info(
                    f"✓ Rollback task completed: target_version={target_version_id}"
                )
            else:
                logger.error(
                    f"✗ Rollback task failed: target_version={target_version_id}"
                )

            return success

        except Exception as e:
            logger.error(
                f"Error processing ROLLBACK task {task.task_id}: {str(e)}",
                exc_info=True,
            )
            return False

    def start(self, queue_name: str = "rollback_queue") -> None:
        """
        Start rollback worker.

        Args:
            queue_name: Queue name to consume from
        """
        logger.info(f"Starting rollback worker (consuming from {queue_name})")
        self.running = True

        def task_handler(task: Task) -> None:
            """Handle incoming tasks."""
            if task.task_type == TaskType.ROLLBACK:
                success = self._handle_rollback_task(task)
                if not success:
                    logger.error(f"Failed to process ROLLBACK task {task.task_id}")
            else:
                logger.warning(
                    f"Received unexpected task type: {task.task_type} (expected ROLLBACK)"
                )

        self.consumer.consume_tasks(queue_name, task_handler)

    def stop(self) -> None:
        """Stop rollback worker."""
        self.running = False
        logger.info("Stopping rollback worker")
