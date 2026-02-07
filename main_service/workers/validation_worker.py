"""Validation worker that evaluates models on test dataset."""

import asyncio
import json
import time
from typing import Optional, Dict, Any, cast
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from shared.queue.consumer import QueueConsumer
from shared.queue.publisher import QueuePublisher
from shared.queue.connection import QueueConnection
from shared.models.task import (
    Task,
    TaskType,
    TaskMetadata,
    ValidateTaskPayload,
    DecisionTaskPayload,
)
from shared.storage.encryption import EncryptionService
from shared.storage.ipfs_client import IPFSClient
from main_service.blockchain.fabric_client import FabricClient
from shared.datasets import get_dataset, DatasetInterface
from shared.models.model import SimpleCNN
from shared.config import settings
from shared.logger import setup_logger
from shared.monitoring.metrics import get_metrics_collector

logger = setup_logger(__name__)


class ValidationWorker:
    """Worker that validates models on test dataset."""

    def __init__(self, connection: Optional[QueueConnection] = None):
        """
        Initialize validation worker.

        Args:
            connection: QueueConnection instance (creates new one if None)
        """
        self.connection = connection or QueueConnection()
        self.consumer = QueueConsumer(connection=self.connection)
        self.publisher = QueuePublisher(connection=self.connection)
        self.encryption_service = EncryptionService()
        self.dataset: DatasetInterface = get_dataset()
        self.model = SimpleCNN(
            num_classes=self.dataset.get_num_classes(),
            in_channels=self.dataset.get_in_channels(),
        )
        self.test_loader: Optional[DataLoader] = None
        self.running = False

        try:
            self._load_test_dataset()
        except Exception as e:
            logger.warning(
                f"Failed to prefetch test dataset: {e}. "
                "It will be loaded on first validation.",
                exc_info=True,
            )

        logger.info("Validation worker initialized")

    def _load_test_dataset(self) -> DataLoader:
        """
        Load test dataset for validation.

        Returns:
            DataLoader for test dataset
        """
        if self.test_loader is None:
            logger.info("Loading test dataset...")
            test_dataset = self.dataset.load_test_data()

            self.test_loader = DataLoader(
                test_dataset,
                batch_size=64,
                shuffle=False,
                num_workers=0,
            )
            logger.info(f"Test dataset loaded: {len(cast(Any, test_dataset))} samples")

        return self.test_loader

    async def _retrieve_and_decrypt_diff(self, ipfs_cid: str) -> Dict[str, Any]:
        """
        Retrieve encrypted diff from IPFS and decrypt it.

        Args:
            ipfs_cid: IPFS CID of encrypted diff

        Returns:
            Decrypted diff as dictionary
        """
        logger.info(f"Retrieving encrypted diff from IPFS: CID={ipfs_cid}")

        # Retrieve from IPFS
        async with IPFSClient() as ipfs_client:
            encrypted_diff = await ipfs_client.get_bytes(ipfs_cid)
            logger.debug(f"Retrieved {len(encrypted_diff)} bytes from IPFS")

        # Decrypt
        logger.info("Decrypting diff...")
        decrypted_diff_bytes = self.encryption_service.decrypt_diff(encrypted_diff)
        logger.debug(f"Decrypted {len(decrypted_diff_bytes)} bytes")

        # Deserialize JSON
        diff_str = decrypted_diff_bytes.decode("utf-8")
        diff_dict: Dict[str, Any] = json.loads(diff_str)

        logger.info("Diff retrieved and decrypted successfully")
        return diff_dict

    async def _load_previous_weights(
        self, parent_version_id: Optional[str]
    ) -> Dict[str, Any]:
        """
        Load previous model weights from parent version.

        Args:
            parent_version_id: Parent version ID (if None, use initial weights)

        Returns:
            Previous weights as dictionary
        """
        if parent_version_id is None:
            # Use initial (random) weights
            logger.info("Using initial model weights")
            return self.model.get_weights()

        # Get parent version's IPFS CID from blockchain
        logger.info(f"Loading weights from parent version {parent_version_id}")
        try:
            async with FabricClient() as blockchain_client:
                parent_provenance = await blockchain_client.get_model_provenance(
                    parent_version_id
                )

            # Extract IPFS CID from metadata
            parent_metadata = parent_provenance.get("metadata", {})
            parent_weights_cid = parent_metadata.get("ipfs_cid")

            if not parent_weights_cid:
                logger.warning(
                    f"Parent version {parent_version_id} has no ipfs_cid in metadata. "
                    "Using initial weights."
                )
                return self.model.get_weights()

            # Download and decrypt parent weights from IPFS
            logger.info(
                f"Downloading parent weights from IPFS: CID={parent_weights_cid}"
            )
            async with IPFSClient() as ipfs_client:
                encrypted_weights = await ipfs_client.get_bytes(parent_weights_cid)

            # Decrypt weights
            decrypted_weights = self.encryption_service.decrypt_diff(encrypted_weights)

            # Deserialize weights
            weights_json = decrypted_weights.decode("utf-8")
            weights_dict: Dict[str, Any] = json.loads(weights_json)

            logger.info(
                f"Successfully loaded weights from parent version {parent_version_id}"
            )
            return weights_dict

        except Exception as e:
            logger.error(
                f"Failed to load weights from parent version {parent_version_id}: {str(e)}. "
                "Using initial weights.",
                exc_info=True,
            )
            return self.model.get_weights()

    def _evaluate_model(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on test dataset.

        Args:
            test_loader: DataLoader for test dataset

        Returns:
            Dictionary of metrics (accuracy, loss, etc.)
        """
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0

        with torch.no_grad():
            for data, target in test_loader:
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                loss = F.nll_loss(output, target, reduction="sum")
                total_loss += loss.item()

        accuracy = 100.0 * correct / total
        avg_loss = total_loss / total

        metrics = {
            "accuracy": accuracy,
            "loss": avg_loss,
            "correct": correct,
            "total": total,
        }

        logger.info(
            f"Model evaluation: accuracy={accuracy:.2f}%, loss={avg_loss:.4f}, "
            f"correct={correct}/{total}"
        )

        return metrics

    async def _validate_model(
        self,
        ipfs_cid: str,
        model_version_id: str,
        parent_version_id: Optional[str],
    ) -> Dict[str, Any]:
        """
        Validate model by applying diff and evaluating on test dataset.

        Args:
            ipfs_cid: IPFS CID of encrypted diff
            model_version_id: Model version identifier
            parent_version_id: Parent version ID

        Returns:
            Validation results dictionary
        """
        logger.info(f"Validating model version {model_version_id}")

        # Retrieve and decrypt diff
        diff_dict = await self._retrieve_and_decrypt_diff(ipfs_cid)

        # Load previous weights
        previous_weights = await self._load_previous_weights(parent_version_id)

        # Apply diff to model
        logger.info("Applying diff to model weights...")
        self.model.set_weights(previous_weights)
        self.model.apply_weight_diff(diff_dict)
        logger.info("Diff applied to model")

        # Load test dataset
        test_loader = self._load_test_dataset()

        # Evaluate model
        eval_start = time.time()
        logger.info("Evaluating model on test dataset...")
        metrics = self._evaluate_model(test_loader)
        eval_duration = time.time() - eval_start

        metrics_collector = get_metrics_collector()

        # Get iteration from model version (will be set later in _handle_validation_task)
        # For now, we'll set it in the task handler after we get it from blockchain
        metrics_collector.record_timing(
            "model_validation",
            eval_duration,
            metadata={
                "model_version_id": model_version_id,
                "accuracy": metrics.get("accuracy", 0.0),
                "loss": metrics.get("loss", 0.0),
            },
        )
        # Collect system metrics sample during validation
        metrics_collector.collect_system_sample()

        validation_result = {
            "model_version_id": model_version_id,
            "parent_version_id": parent_version_id,
            "ipfs_cid": ipfs_cid,
            "metrics": metrics,
        }

        logger.info(
            f"Validation complete for version {model_version_id}: "
            f"accuracy={metrics.get('accuracy', 0.0):.2f}% "
            f"(duration: {eval_duration:.3f}s)"
        )
        return validation_result

    def _publish_decision_task(
        self,
        validation_result: Dict[str, Any],
        model_version_id: str,
        should_rollback: bool,
        rollback_reason: Optional[str] = None,
    ) -> None:
        """
        Publish DECISION task with validation results.

        Args:
            validation_result: Validation results dictionary
            model_version_id: Model version identifier
            should_rollback: Whether rollback is needed
            rollback_reason: Reason for rollback if needed
        """
        decision_task = Task(
            task_id=f"decision-{model_version_id}-{int(time.time() * 1000)}",
            task_type=TaskType.DECISION,
            payload=DecisionTaskPayload(
                validation_result=validation_result,
                model_version_id=model_version_id,
                should_rollback=should_rollback,
                rollback_reason=rollback_reason,
            ).model_dump(),
            metadata=TaskMetadata(source="validation_worker"),
            model_version_id=model_version_id,
            parent_version_id=validation_result.get("parent_version_id"),
        )

        self.publisher.publish_task(task=decision_task, queue_name="decision_queue")
        logger.info(f"Published DECISION task for version {model_version_id}")

    def _handle_validate_task(self, task: Task) -> bool:
        """
        Handle a VALIDATE task.

        Args:
            task: VALIDATE task to process

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Processing VALIDATE task: {task.task_id}")

            # Parse payload
            payload = ValidateTaskPayload(**task.payload)

            # Extract task data
            ipfs_cid = payload.ipfs_cid
            model_version_id = payload.model_version_id
            parent_version_id = payload.parent_version_id

            # Run async validation
            # Create new event loop if one doesn't exist in this thread
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                # No event loop in current thread, create new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            validation_result = loop.run_until_complete(
                self._validate_model(ipfs_cid, model_version_id, parent_version_id)
            )

            # Get iteration from model version metadata
            iteration = None
            try:

                async def get_iteration_async():
                    async with FabricClient() as blockchain_client:
                        provenance = await blockchain_client.get_model_provenance(
                            model_version_id
                        )
                        metadata = provenance.get("metadata", {})
                        if isinstance(metadata, dict):
                            iter_val = metadata.get("iteration")
                            if iter_val is not None:
                                return int(iter_val)
                        # Fallback: check top-level iteration field
                        iter_val = provenance.get("iteration")
                        if iter_val is not None:
                            return int(iter_val)
                        return None

                iteration = loop.run_until_complete(get_iteration_async())
                if iteration is not None:
                    validation_result["iteration"] = iteration
                    logger.debug(
                        f"Found iteration {iteration} for model version {model_version_id}"
                    )
                    # Update the validation timing metadata with iteration
                    metrics_collector = get_metrics_collector()
                    # Find the last model_validation timing and update its metadata
                    if (
                        "model_validation" in metrics_collector.operation_metadata
                        and metrics_collector.operation_metadata["model_validation"]
                    ):
                        last_metadata = metrics_collector.operation_metadata[
                            "model_validation"
                        ][-1]
                        last_metadata["iteration"] = iteration
                    metrics_collector.set_pending_iteration(iteration)
                    metrics_collector.collect_system_sample()
            except Exception as e:
                logger.warning(
                    f"Failed to get iteration for model version {model_version_id}: {e}. "
                    "Decision worker will use current_iteration state."
                )

            # Export metrics for this iteration incrementally
            if iteration is not None:
                try:
                    from shared.monitoring.metrics_exporter import MetricsExporter
                    from shared.monitoring.metrics import get_metrics_collector

                    metrics_collector = get_metrics_collector()
                    exporter = MetricsExporter()
                    # Initialize CSV file if not already done (idempotent)
                    if MetricsExporter.get_active_csv_path() is None:
                        scenario_info = metrics_collector.scenario_info
                        if not scenario_info:
                            # Try to infer from settings
                            from shared.config import settings

                            scenario_info = {
                                "blockchain_enabled": bool(
                                    settings.blockchain_service_url
                                ),
                                "ipfs_enabled": bool(settings.ipfs_host),
                                "num_clients": settings.num_clients,
                                "target_accuracy": settings.target_accuracy,
                                "max_iterations": settings.max_iterations,
                                "max_rollbacks": settings.max_rollbacks,
                            }
                        exporter.initialize_csv_file(scenario_info)

                    # Append metrics for this iteration
                    exporter.append_iteration_metrics(iteration, metrics_collector)
                except Exception as e:
                    logger.warning(
                        f"Failed to export incremental metrics for iteration {iteration}: {e}",
                        exc_info=True,
                    )

            # Record validation on blockchain (async)
            metrics = validation_result.get("metrics", {})
            accuracy = metrics.get("accuracy", 0.0)
            try:
                # Reuse the same event loop
                loop = asyncio.get_event_loop()

                async def record_validation_async():
                    async with FabricClient() as blockchain_client:
                        ipfs_cid = validation_result.get("ipfs_cid")
                        await blockchain_client.record_validation(
                            model_version_id=model_version_id,
                            accuracy=accuracy,
                            metrics=metrics,
                            ipfs_cid=ipfs_cid,
                        )

                loop.run_until_complete(record_validation_async())
                logger.info(
                    f"Validation recorded on blockchain for version {model_version_id}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to record validation on blockchain: {e}", exc_info=True
                )
                # Continue anyway - validation result is still published

            # For now, always pass validation (rollback logic will be in decision worker)
            # TODO: Add basic rollback check here if needed
            should_rollback = False
            rollback_reason = None

            # Publish DECISION task
            self._publish_decision_task(
                validation_result, model_version_id, should_rollback, rollback_reason
            )

            logger.info(
                f"Validation task completed: version={model_version_id}, "
                f"accuracy={validation_result['metrics']['accuracy']:.2f}%"
            )
            return True

        except Exception as e:
            logger.error(
                f"Error processing VALIDATE task {task.task_id}: {str(e)}",
                exc_info=True,
            )
            return False

    def start(self, queue_name: str = "validate_queue") -> None:
        """
        Start validation worker.

        Args:
            queue_name: Queue name to consume from
        """
        logger.info(f"Starting validation worker (consuming from {queue_name})")
        self.running = True

        def task_handler(task: Task) -> None:
            """Handle incoming tasks."""
            if task.task_type == TaskType.VALIDATE:
                success = self._handle_validate_task(task)
                if not success:
                    logger.error(f"Failed to process VALIDATE task {task.task_id}")
            else:
                logger.warning(
                    f"Received unexpected task type: {task.task_type} (expected VALIDATE)"
                )

        self.consumer.consume_tasks(queue_name, task_handler)

    def stop(self) -> None:
        """Stop validation worker."""
        self.running = False
        logger.info("Stopping validation worker")
