"""Client service worker that consumes training tasks and publishes updates."""

import asyncio
import json
import os
import sys
import uuid
import time
from typing import Optional, Dict, Any
from client_service.config import config
from client_service.training.trainer import Trainer
from client_service.training.model import SimpleCNN
from shared.queue.consumer import QueueConsumer
from shared.queue.publisher import QueuePublisher
from shared.queue.connection import QueueConnection
from shared.models.task import Task, TaskType, TrainTaskPayload, RollbackTaskPayload
from shared.storage.ipfs_client import IPFSClient
from shared.storage.encryption import EncryptionService
from shared.config import settings
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
        self.trainer = trainer or Trainer(epochs=config.epochs)
        self.connection = connection or QueueConnection()
        self.consumer = QueueConsumer(connection=self.connection)
        self.publisher = QueuePublisher(connection=self.connection)

        # Generate a unique client instance ID for this client instance
        self.instance_id = str(uuid.uuid4())[
            :8
        ]  # Use first 8 chars of UUID for readability
        self.running = False

        logger.info(f"Client worker initialized with instance_id={self.instance_id}")

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

        try:
            logger.info(f"Loading weights from IPFS CID: {weights_cid}")

            # Suppress progress output during IPFS download
            # Redirect stdout/stderr to suppress progress bars
            old_stdout = sys.stdout
            old_stderr = sys.stderr

            try:
                # Redirect to devnull to suppress progress bars
                devnull = open(os.devnull, "w")
                sys.stdout = devnull
                sys.stderr = devnull

                # Retrieve encrypted weights from IPFS
                async def fetch_and_decrypt():
                    async with IPFSClient() as ipfs_client:
                        encrypted_weights = await ipfs_client.get_bytes(weights_cid)
                        logger.debug(
                            f"Retrieved {len(encrypted_weights)} bytes from IPFS"
                        )

                    # Decrypt weights
                    encryption_service = EncryptionService()
                    decrypted_weights = encryption_service.decrypt_diff(
                        encrypted_weights
                    )
                    logger.debug(f"Decrypted {len(decrypted_weights)} bytes")

                    # Deserialize weights
                    weights_json = decrypted_weights.decode("utf-8")
                    weights_dict = json.loads(weights_json)
                    return weights_dict

                # Run async function
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    # No event loop in current thread, create new one
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                weights_dict: Dict[str, Any] = loop.run_until_complete(
                    fetch_and_decrypt()
                )
            finally:
                # Restore stdout/stderr
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                if "devnull" in locals():
                    devnull.close()

            logger.info(f"Successfully loaded weights from IPFS CID: {weights_cid}")
            return weights_dict

        except Exception as e:
            logger.error(
                f"Failed to load weights from IPFS CID {weights_cid}: {str(e)}",
                exc_info=True,
            )
            logger.warning("Training from scratch due to IPFS retrieval failure")
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
            receive_time = time.time()
            logger.info(
                f"Processing TRAIN task: {task.task_id} (received at {receive_time:.3f})"
            )

            # Parse payload
            payload = TrainTaskPayload(**task.payload)

            # Universal tasks: all clients process the same training task
            # All clients train on the same iteration with the same weights
            # Clients use their own instance_id when sending updates

            # Load previous weights if provided
            previous_weights = self._load_weights_from_cid(payload.weights_cid)

            # Train the model
            logger.info(
                f"Starting training for iteration {payload.iteration}, "
                f"instance_id={self.instance_id}"
            )

            weight_diff, metrics, initial_weights = self.trainer.train(
                previous_weights=previous_weights, instance_id=self.instance_id
            )

            logger.info(
                f"Training completed: loss={metrics['loss']:.4f}, "
                f"accuracy={metrics['accuracy']:.2f}%"
            )

            # Serialize weight diff to bytes
            weight_diff_bytes = self.trainer.get_model().weights_to_bytes()

            # Upload weight diff to IPFS to avoid RabbitMQ frame size limits
            # Weight diffs can be very large (several MB), so we store them in IPFS
            # and only send the CID through RabbitMQ
            logger.info(
                f"Uploading weight diff to IPFS (size: {len(weight_diff_bytes)} bytes) "
                f"for iteration {payload.iteration}"
            )

            # Upload to IPFS (synchronous wrapper for async operation)
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                # No event loop in current thread, create new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            async def upload_to_ipfs():
                from shared.storage.ipfs_client import IPFSClient

                async with IPFSClient() as ipfs_client:
                    cid = await ipfs_client.add_bytes(weight_diff_bytes, pin=True)
                    return cid

            weight_diff_cid = loop.run_until_complete(upload_to_ipfs())
            logger.info(
                f"Uploaded weight diff to IPFS: CID={weight_diff_cid} "
                f"(size: {len(weight_diff_bytes)} bytes)"
            )

            # Create client update payload (simple dict, not a Task)
            # The aggregation worker will collect these and create an AGGREGATE task
            client_update = {
                "client_id": self.instance_id,  # Use instance_id as the client identifier
                "iteration": payload.iteration,
                "weight_diff_cid": weight_diff_cid,  # IPFS CID instead of full diff
                "metrics": metrics,
                "task_id": task.task_id,  # Reference to original task
                "model_version_id": task.model_version_id,
                "parent_version_id": task.parent_version_id,
            }

            # Publish to client_updates queue (main service aggregation worker will collect these)
            queue_name = "client_updates"
            try:
                self.publisher.publish_dict(client_update, queue_name)
                logger.info(
                    f"Published weight update to queue '{queue_name}' for task {task.task_id} "
                    f"(iteration={payload.iteration}, instance_id={self.instance_id})"
                )
            except Exception as e:
                logger.error(
                    f"Failed to publish weight update to queue '{queue_name}': {str(e)}",
                    exc_info=True,
                )
                raise

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

    def start(self, queue_name: str = "train_queue"):
        """
        Start consuming tasks from the queue.

        Args:
            queue_name: Name of the queue to consume from
        """
        self.running = True
        logger.info(f"Starting client worker for instance_id={self.instance_id}")

        def task_handler(task: Task):
            """Handle received task."""
            try:
                success = self._handle_task(task)
                if not success:
                    # Raise exception to trigger nack and requeue
                    # This ensures failed tasks are not removed from the queue
                    raise Exception(f"Task {task.task_id} processing failed")
            except Exception as e:
                logger.error(
                    f"Error handling task {task.task_id}: {str(e)}",
                    exc_info=True,
                )
                # Re-raise to trigger nack and requeue in consumer
                raise

        try:
            # Start consuming from fanout exchange
            # All clients receive the same message simultaneously via their own queues
            # Use higher prefetch_count to allow clients to hold multiple messages,
            # preventing one client from monopolizing the queue
            num_clients = config.num_clients
            prefetch_count = max(
                num_clients, 5
            )  # At least as many as clients, minimum 5

            logger.info(
                f"Consuming tasks from queue: {queue_name} (prefetch_count={prefetch_count}, use_fanout=True)"
            )
            # Use fanout exchange so all clients receive the same message simultaneously
            # Each client gets its own queue bound to the fanout exchange
            # Use unique instance_id to ensure each client instance has a unique queue
            consumer_id = f"client_{self.instance_id}"
            logger.info(
                f"Using unique consumer_id: {consumer_id} (instance_id={self.instance_id})"
            )
            self.consumer.consume_tasks(
                queue_name,
                task_handler,
                prefetch_count=prefetch_count,
                use_fanout=True,
                consumer_id=consumer_id,
            )
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
