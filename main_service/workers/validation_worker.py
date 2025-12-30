"""Validation worker that evaluates models on test dataset."""

import asyncio
import json
import time
from typing import Optional, Dict, Any, cast
import torch
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
from main_service.storage.ipfs_client import IPFSClient
from main_service.blockchain.fabric_client import FabricClient
from shared.datasets import get_dataset, DatasetInterface
from client_service.training.model import SimpleCNN
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
        self.model = SimpleCNN(num_classes=self.dataset.get_num_classes())
        self.test_loader: Optional[DataLoader] = None
        self.running = False

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
            logger.info(
                f"✓ Test dataset loaded: {len(cast(Any, test_dataset))} samples"
            )

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

        logger.info("✓ Diff retrieved and decrypted successfully")
        return diff_dict

    async def _load_previous_weights(
        self, parent_version_id: Optional[str]
    ) -> Dict[str, Any]:
        """
        Load previous model weights.

        Args:
            parent_version_id: Parent version ID (if None, use initial weights)

        Returns:
            Previous weights as dictionary
        """
        if parent_version_id is None:
            # Use initial (random) weights
            logger.info("Using initial model weights")
            return self.model.get_weights()

        # TODO: Load weights from IPFS using parent_version_id
        # For now, use initial weights
        logger.warning(
            f"Loading weights from parent version {parent_version_id} not yet implemented. Using initial weights."
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
        import torch.nn.functional as F

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
        logger.info("✓ Diff applied to model")

        # Load test dataset
        test_loader = self._load_test_dataset()

        # Evaluate model
        eval_start = time.time()
        logger.info("Evaluating model on test dataset...")
        metrics = self._evaluate_model(test_loader)
        eval_duration = time.time() - eval_start

        get_metrics_collector().record_timing(
            "model_validation",
            eval_duration,
            metadata={
                "model_version_id": model_version_id,
                "accuracy": metrics.get("accuracy", 0.0),
                "loss": metrics.get("loss", 0.0),
            },
        )

        validation_result = {
            "model_version_id": model_version_id,
            "parent_version_id": parent_version_id,
            "ipfs_cid": ipfs_cid,
            "metrics": metrics,
        }

        logger.info(
            f"✓ Validation complete for version {model_version_id}: "
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
        logger.info(f"✓ Published DECISION task for version {model_version_id}")

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
            loop = asyncio.get_event_loop()
            validation_result = loop.run_until_complete(
                self._validate_model(ipfs_cid, model_version_id, parent_version_id)
            )

            # Record validation on blockchain (async)
            metrics = validation_result.get("metrics", {})
            accuracy = metrics.get("accuracy", 0.0)
            try:
                loop = asyncio.get_event_loop()

                async def record_validation_async():
                    async with FabricClient() as blockchain_client:
                        await blockchain_client.record_validation(
                            model_version_id=model_version_id,
                            accuracy=accuracy,
                            metrics=metrics,
                        )

                loop.run_until_complete(record_validation_async())
                logger.info(
                    f"✓ Validation recorded on blockchain for version {model_version_id}"
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
                f"✓ Validation task completed: version={model_version_id}, "
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
