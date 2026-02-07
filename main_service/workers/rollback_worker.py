"""Rollback worker that handles rollback operations."""

import asyncio
from typing import Optional, List, cast, Dict, Any
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
from shared.config import settings
from shared.utils.training import publish_train_task

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
                        f"Rollback target weights verified: {len(weights_data)} bytes"
                    )
                    return True
                else:
                    logger.error(
                        f"Rollback target weights not found: CID={target_weights_cid}"
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
                logger.info(f"Rollback recorded on blockchain: tx_id={transaction_id}")
                return transaction_id
            else:
                logger.error("Failed to record rollback on blockchain")
                return None

        except Exception as e:
            logger.error(
                f"Error recording rollback on blockchain: {str(e)}",
                exc_info=True,
            )
            return None

    async def _check_and_resume_training(
        self, target_version_id: str, target_weights_cid: str
    ) -> None:
        """
        Check if training should resume after rollback based on accuracy.

        Args:
            target_version_id: Version ID that was rolled back to
            target_weights_cid: IPFS CID of rolled-back weights
        """
        try:
            # Get accuracy of rolled-back version from blockchain
            async with FabricClient() as blockchain_client:
                provenance = await blockchain_client.get_model_provenance(
                    target_version_id
                )

                # Get latest iteration from blockchain (needed to determine next iteration)
                latest_iteration = await self._get_latest_iteration_from_blockchain(
                    blockchain_client
                )

            # Extract accuracy from validation_metrics or metadata
            validation_metrics = provenance.get("validation_metrics", {})
            accuracy = None
            if isinstance(validation_metrics, dict):
                accuracy = validation_metrics.get("accuracy")

            # Also check metadata for accuracy
            if accuracy is None:
                metadata = provenance.get("metadata", {})
                if isinstance(metadata, dict):
                    validation_history = metadata.get("validation_history", [])
                    if isinstance(validation_history, list) and validation_history:
                        # Get accuracy from most recent validation record
                        for record in reversed(validation_history):
                            if isinstance(record, dict):
                                accuracy = record.get("accuracy")
                                if accuracy is not None:
                                    break

            # Get iteration from metadata
            metadata = provenance.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            rolled_back_iteration = metadata.get("iteration")
            if rolled_back_iteration is None:
                rolled_back_iteration = provenance.get("iteration", 1)

            # Determine next iteration based on latest iteration on blockchain
            if latest_iteration is None:
                # No versions exist, start from iteration 1
                next_iteration = 1
            else:
                # Use latest iteration + 1 as the next iteration
                next_iteration = latest_iteration + 1
                logger.info(
                    f"Latest iteration on blockchain: {latest_iteration}, "
                    f"rolled-back iteration: {rolled_back_iteration}, "
                    f"next iteration will be: {next_iteration}"
                )

            target_accuracy = settings.target_accuracy

            if accuracy is None or accuracy < target_accuracy:
                # Accuracy is below target, resume training
                # Use next_iteration (which accounts for existing iterations on ledger)
                logger.info(
                    f"Accuracy is below target ({target_accuracy:.2f}%). "
                    f"Resuming training from iteration {next_iteration} "
                    f"(parent will be rolled-back version {target_version_id}, iteration {rolled_back_iteration})."
                )
                publish_train_task(
                    self.publisher,
                    next_iteration,
                    target_weights_cid,
                    source="rollback_worker",
                )
            else:
                logger.info(
                    f"Accuracy ({accuracy:.2f}%) meets or exceeds target ({target_accuracy:.2f}%). "
                    "Training will not resume automatically."
                )

        except Exception as e:
            logger.error(
                f"Error checking if training should resume after rollback: {str(e)}",
                exc_info=True,
            )
            # Don't fail the rollback if this check fails

    async def _get_latest_iteration_from_blockchain(
        self, blockchain_client: FabricClient
    ) -> Optional[int]:
        """
        Get the latest iteration number from blockchain by listing all models.

        Args:
            blockchain_client: FabricClient instance

        Returns:
            Latest iteration number, or None if no versions exist
        """
        try:
            # List all models to find the maximum iteration
            models_response = await blockchain_client.list_models()
            versions = models_response.get("versions", [])

            if not versions:
                return None

            max_iteration = None
            for version in versions:
                metadata = version.get("metadata", {})
                if not isinstance(metadata, dict):
                    metadata = {}
                iteration = metadata.get("iteration")
                if iteration is None:
                    iteration = version.get("iteration")

                if iteration is not None:
                    try:
                        iter_num = int(iteration)
                        if max_iteration is None or iter_num > max_iteration:
                            max_iteration = iter_num
                    except (ValueError, TypeError):
                        continue

            return max_iteration
        except Exception as e:
            logger.warning(
                f"Failed to get latest iteration from blockchain: {str(e)}. "
                "Will use rolled-back iteration + 1 as fallback."
            )
            return None

    async def _process_rollback(
        self,
        target_version_id: str,
        target_weights_cid: str,
        reason: str,
        cutoff_version_id: Optional[str],
        excluded_client_ids: Optional[List[str]] = None,
    ) -> bool:
        """
        Process rollback operation.

        Args:
            target_version_id: Version ID to rollback to
            target_weights_cid: IPFS CID of target weights
            reason: Reason for rollback
            cutoff_version_id: Version ID after which updates should be discarded
            excluded_client_ids: Client IDs to exclude from aggregation (identified as unreliable)

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
            f"Rollback processed successfully: target_version={target_version_id}, "
            f"tx_id={transaction_id}"
        )

        # After successful rollback, exclude unreliable clients (identified at rollback time)
        if excluded_client_ids:
            for cid in excluded_client_ids:
                if cid and cid not in settings.excluded_clients:
                    settings.excluded_clients.append(cid)
            logger.info(
                f"Rollback exclusion applied: added {excluded_client_ids} to excluded_clients"
            )

        # After successful rollback, check if training should continue
        await self._check_and_resume_training(target_version_id, target_weights_cid)

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
            excluded_client_ids = getattr(payload, "excluded_client_ids", None)

            # Run async rollback processing
            # Create new event loop if one doesn't exist in this thread
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                # No event loop in current thread, create new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            success = loop.run_until_complete(
                self._process_rollback(
                    target_version_id,
                    target_weights_cid,
                    reason,
                    cutoff_version_id,
                    excluded_client_ids=excluded_client_ids,
                )
            )

            if success:
                logger.info(
                    f"Rollback task completed: target_version={target_version_id}"
                )
            else:
                logger.error(
                    f"Rollback task failed: target_version={target_version_id}"
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
