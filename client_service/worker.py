"""Client service worker that consumes training tasks and publishes updates."""

import json
from typing import Optional, Dict, Any
from client_service.config import config
from client_service.training.trainer import Trainer
from client_service.training.model import SimpleCNN
from shared.queue.consumer import QueueConsumer
from shared.queue.publisher import QueuePublisher
from shared.queue.connection import QueueConnection
from shared.models.task import Task, TaskType, TrainTaskPayload
from shared.logger import setup_logger

logger = setup_logger(__name__)


class ClientWorker:
    """Worker that processes training tasks from the queue."""

    def __init__(
        self,
        trainer: Optional[Trainer] = None,
        connection: Optional[QueueConnection] = None,
    ):
        """
        Initialize client worker.

        Args:
            trainer: Trainer instance (creates new one if None)
            connection: QueueConnection instance (creates new one if None)
        """
        self.trainer = trainer or Trainer()
        self.connection = connection or QueueConnection()
        self.consumer = QueueConsumer(connection=self.connection)
        self.publisher = QueuePublisher(connection=self.connection)
        self.client_id = config.get_client_id()
        self.running = False

        logger.info(f"Client worker initialized for client_id={self.client_id}")

    def _load_weights_from_cid(
        self, weights_cid: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Load model weights from IPFS CID.

        Args:
            weights_cid: IPFS CID of model weights (None for initial training)

        Returns:
            Dictionary of weights or None if CID is None
        """
        if weights_cid is None:
            return None

        # TODO: Implement IPFS client to fetch weights
        # For now, return None (will train from scratch)
        logger.warning(
            f"Weights CID provided ({weights_cid}) but IPFS client not implemented yet. "
            "Training from scratch."
        )
        return None

    def _handle_train_task(self, task: Task) -> bool:
        """
        Handle a TRAIN task.

        Args:
            task: TRAIN task to process

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Processing TRAIN task: {task.task_id}")

            # Parse payload
            payload = TrainTaskPayload(**task.payload)

            # Check if this task is for this client
            task_client_id = payload.client_id
            if task_client_id is not None:
                # Extract numeric client ID from string (e.g., "client_0" -> 0)
                if task_client_id.startswith("client_"):
                    task_client_id_num = int(task_client_id.split("_")[1])
                else:
                    task_client_id_num = int(task_client_id)

                if task_client_id_num != self.client_id:
                    logger.info(
                        f"Task {task.task_id} is for client {task_client_id_num}, "
                        f"but this is client {self.client_id}. Skipping."
                    )
                    return True  # Not an error, just not for us

            # Load previous weights if provided
            previous_weights = self._load_weights_from_cid(payload.weights_cid)

            # Train the model
            logger.info(
                f"Starting training for iteration {payload.iteration}, "
                f"client_id={self.client_id}"
            )

            weight_diff, metrics, initial_weights = self.trainer.train(
                previous_weights=previous_weights
            )

            logger.info(
                f"Training completed: loss={metrics['loss']:.4f}, "
                f"accuracy={metrics['accuracy']:.2f}%"
            )

            # Serialize weight diff to bytes (for now, JSON)
            # TODO: Encrypt and upload to IPFS
            weight_diff_bytes = self.trainer.get_model().weights_to_bytes()
            weight_diff_str = weight_diff_bytes.decode("utf-8")

            # Create client update payload (simple dict, not a Task)
            # The aggregation worker will collect these and create an AGGREGATE task
            client_update = {
                "client_id": f"client_{self.client_id}",
                "iteration": payload.iteration,
                "weight_diff": weight_diff_str,  # Serialized as JSON string
                "metrics": metrics,
                "task_id": task.task_id,  # Reference to original task
                "model_version_id": task.model_version_id,
                "parent_version_id": task.parent_version_id,
            }

            # Publish to client_updates queue (main service aggregation worker will collect these)
            queue_name = "client_updates"
            self.publisher.publish_dict(client_update, queue_name)

            logger.info(
                f"Published weight update to queue '{queue_name}' for task {task.task_id}"
            )

            return True

        except Exception as e:
            logger.error(
                f"Error processing TRAIN task {task.task_id}: {str(e)}", exc_info=True
            )
            return False

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
            from shared.models.task import RollbackTaskPayload

            payload = RollbackTaskPayload(**task.payload)

            # Load weights from the target version
            logger.info(
                f"Rolling back to version {payload.target_version_id}, "
                f"weights CID: {payload.target_weights_cid}"
            )

            # Load weights from IPFS
            target_weights = self._load_weights_from_cid(payload.target_weights_cid)

            if target_weights is not None:
                # Set model to rolled-back weights
                self.trainer.set_model_weights(target_weights)
                logger.info(f"Model rolled back to version {payload.target_version_id}")
            else:
                logger.warning(
                    f"Could not load weights from CID {payload.target_weights_cid}. "
                    "Model may need to be reinitialized."
                )

            return True

        except Exception as e:
            logger.error(
                f"Error processing ROLLBACK task {task.task_id}: {str(e)}",
                exc_info=True,
            )
            return False

    def _handle_task(self, task: Task) -> bool:
        """
        Handle a task based on its type.

        Args:
            task: Task to process

        Returns:
            True if successful, False otherwise
        """
        if task.task_type == TaskType.TRAIN:
            return self._handle_train_task(task)
        elif task.task_type == TaskType.ROLLBACK:
            return self._handle_rollback_task(task)
        else:
            logger.warning(f"Unknown task type: {task.task_type}. Skipping.")
            return True  # Not an error, just not handled

    def start(self, queue_name: str = "train_tasks"):
        """
        Start consuming tasks from the queue.

        Args:
            queue_name: Name of the queue to consume from
        """
        self.running = True
        logger.info(f"Starting client worker for client_id={self.client_id}")

        def task_handler(task: Task):
            """Handle received task."""
            try:
                success = self._handle_task(task)
                if not success:
                    logger.error(f"Failed to process task {task.task_id}")
            except Exception as e:
                logger.error(
                    f"Unexpected error handling task {task.task_id}: {str(e)}",
                    exc_info=True,
                )

        try:
            # Start consuming
            logger.info(f"Consuming tasks from queue: {queue_name}")
            self.consumer.consume_tasks(queue_name, task_handler)
        except KeyboardInterrupt:
            logger.info("Received interrupt signal. Stopping worker...")
            self.stop()
        except Exception as e:
            logger.error(f"Error in worker loop: {str(e)}", exc_info=True)
            raise

    def stop(self):
        """Stop the worker."""
        self.running = False
        logger.info("Stopping client worker...")
        self.consumer.stop()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
