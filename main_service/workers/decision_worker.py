"""Decision worker that makes rollback/training decisions based on validation results."""

import time
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from shared.queue.consumer import QueueConsumer
from shared.queue.publisher import QueuePublisher
from shared.queue.connection import QueueConnection
from shared.models.task import (
    Task,
    TaskType,
    TaskMetadata,
    DecisionTaskPayload,
    TrainTaskPayload,
    RollbackTaskPayload,
    TrainingCompleteTaskPayload,
)
from shared.config import settings
from shared.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class ModelState:
    """Tracks model state for decision making."""

    current_iteration: int = 0
    best_accuracy: float = 0.0
    best_checkpoint_version: Optional[str] = None
    best_checkpoint_cid: Optional[str] = None
    patience_counter: int = 0
    rollback_count: int = 0
    accuracy_history: list[float] = field(default_factory=list)
    training_start_time: Optional[float] = None


class DecisionWorker:
    """Worker that makes rollback/training decisions based on validation results."""

    def __init__(self, connection: Optional[QueueConnection] = None):
        """
        Initialize decision worker.

        Args:
            connection: QueueConnection instance (creates new one if None)
        """
        self.connection = connection or QueueConnection()
        self.consumer = QueueConsumer(connection=self.connection)
        self.publisher = QueuePublisher(connection=self.connection)
        self.state = ModelState()
        self.running = False

        # Configuration parameters
        self.accuracy_tolerance = getattr(
            settings, "accuracy_tolerance", 0.5
        )  # 0.5% tolerance
        self.patience_threshold = getattr(
            settings, "patience_threshold", 3
        )  # 3 iterations
        self.severe_drop_threshold = getattr(
            settings, "severe_drop_threshold", 2.0
        )  # 2% severe drop
        self.target_accuracy = getattr(settings, "target_accuracy", 95.0)
        self.convergence_patience = getattr(
            settings, "convergence_patience", 10
        )  # 10 iterations
        self.max_iterations = getattr(settings, "max_iterations", 100)
        self.max_rollbacks = getattr(settings, "max_rollbacks", 5)
        self.num_clients = getattr(settings, "num_clients", 2)

        logger.info("Decision worker initialized")

    def _evaluate_rollback(
        self, current_accuracy: float, validation_result: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Evaluate if rollback is needed based on accuracy.

        Args:
            current_accuracy: Current model accuracy
            validation_result: Full validation result dictionary

        Returns:
            Tuple of (should_rollback, rollback_reason)
        """
        # Initialize best accuracy if this is the first iteration
        if self.state.best_accuracy == 0.0:
            self.state.best_accuracy = current_accuracy
            self.state.best_checkpoint_version = validation_result.get(
                "model_version_id"
            )
            self.state.best_checkpoint_cid = validation_result.get("ipfs_cid")
            logger.info(
                f"Initial best accuracy set: {current_accuracy:.2f}% "
                f"(version: {self.state.best_checkpoint_version})"
            )
            return False, None

        # Check for severe drop (immediate rollback)
        accuracy_drop = self.state.best_accuracy - current_accuracy
        if accuracy_drop > self.severe_drop_threshold:
            reason = (
                f"Severe accuracy drop: {accuracy_drop:.2f}% "
                f"(best: {self.state.best_accuracy:.2f}%, current: {current_accuracy:.2f}%)"
            )
            logger.warning(f"⚠ {reason}")
            return True, reason

        # Check if accuracy improved or is new best
        if current_accuracy > self.state.best_accuracy:
            # New best accuracy - update checkpoint
            self.state.best_accuracy = current_accuracy
            self.state.best_checkpoint_version = validation_result.get(
                "model_version_id"
            )
            self.state.best_checkpoint_cid = validation_result.get("ipfs_cid")
            self.state.patience_counter = 0  # Reset patience
            logger.info(
                f"New best accuracy: {current_accuracy:.2f}% "
                f"(version: {self.state.best_checkpoint_version})"
            )
            return False, None

        # Check if accuracy is within tolerance
        if accuracy_drop <= self.accuracy_tolerance:
            # Within tolerance - acceptable, reset patience
            self.state.patience_counter = 0
            logger.debug(
                f"Accuracy within tolerance: {current_accuracy:.2f}% "
                f"(drop: {accuracy_drop:.2f}%, best: {self.state.best_accuracy:.2f}%)"
            )
            return False, None

        # Accuracy dropped beyond tolerance but not severe
        # Increment patience counter
        self.state.patience_counter += 1
        logger.warning(
            f"Accuracy below best: {current_accuracy:.2f}% "
            f"(drop: {accuracy_drop:.2f}%, best: {self.state.best_accuracy:.2f}%, "
            f"patience: {self.state.patience_counter}/{self.patience_threshold})"
        )

        # Check if patience threshold exceeded
        if self.state.patience_counter >= self.patience_threshold:
            reason = (
                f"Patience threshold exceeded: {self.state.patience_counter} consecutive "
                f"iterations below best accuracy (best: {self.state.best_accuracy:.2f}%, "
                f"current: {current_accuracy:.2f}%)"
            )
            logger.warning(f"⚠ {reason}")
            return True, reason

        return False, None

    def _check_training_completion(
        self, current_accuracy: float, validation_result: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if training should be completed.

        Args:
            current_accuracy: Current model accuracy
            validation_result: Full validation result dictionary

        Returns:
            Tuple of (should_complete, completion_reason)
        """
        # Check accuracy threshold
        if current_accuracy >= self.target_accuracy:
            return (
                True,
                f"Target accuracy reached: {current_accuracy:.2f}% >= {self.target_accuracy}%",
            )

        # Check max iterations
        if self.state.current_iteration >= self.max_iterations:
            return (
                True,
                f"Maximum iterations reached: {self.state.current_iteration} >= {self.max_iterations}",
            )

        # Check max rollbacks
        if self.state.rollback_count >= self.max_rollbacks:
            return True, (
                f"Maximum rollbacks reached: {self.state.rollback_count} >= {self.max_rollbacks}. "
                "Investigation recommended."
            )

        # Check convergence (no improvement for N iterations)
        if len(self.state.accuracy_history) >= self.convergence_patience:
            recent_accuracies = self.state.accuracy_history[
                -self.convergence_patience :
            ]
            if all(acc <= self.state.best_accuracy for acc in recent_accuracies):
                return True, (
                    f"Convergence detected: No improvement for {self.convergence_patience} "
                    f"consecutive iterations (best: {self.state.best_accuracy:.2f}%)"
                )

        return False, None

    def _publish_train_tasks(
        self, iteration: int, weights_cid: Optional[str] = None
    ) -> None:
        """
        Publish a universal TRAIN task for all clients.

        Args:
            iteration: Training iteration number
            weights_cid: IPFS CID of model weights (if None, clients start from scratch)
        """
        logger.info(f"Publishing universal TRAIN task for iteration {iteration}")

        # Publish a single universal task - all clients will process it
        # Clients use their own instance_id when sending updates
        train_task = Task(
            task_id=f"train-iter{iteration}-{int(time.time() * 1000)}",
            task_type=TaskType.TRAIN,
            payload=TrainTaskPayload(
                weights_cid=weights_cid,
                iteration=iteration,
            ).model_dump(),
            metadata=TaskMetadata(source="decision_worker"),
            model_version_id=None,
            parent_version_id=None,
        )

        # Use fanout exchange so all clients receive the message simultaneously
        self.publisher.publish_task(
            task=train_task, queue_name="train_queue", use_fanout=True
        )
        logger.info(
            f"Published universal TRAIN task for iteration {iteration} via fanout exchange (all clients will receive simultaneously)"
        )

    def _publish_rollback_task(
        self, target_version_id: str, target_weights_cid: str, reason: str
    ) -> None:
        """
        Publish ROLLBACK task.

        Args:
            target_version_id: Version ID to rollback to
            target_weights_cid: IPFS CID of rolled-back weights
            reason: Reason for rollback
        """
        logger.info(f"Publishing ROLLBACK task to version {target_version_id}")

        rollback_task = Task(
            task_id=f"rollback-{target_version_id}-{int(time.time() * 1000)}",
            task_type=TaskType.ROLLBACK,
            payload=RollbackTaskPayload(
                target_version_id=target_version_id,
                target_weights_cid=target_weights_cid,
                reason=reason,
                cutoff_version_id=None,  # TODO: Set cutoff version
            ).model_dump(),
            metadata=TaskMetadata(source="decision_worker"),
            model_version_id=target_version_id,
            parent_version_id=None,
        )

        self.publisher.publish_task(task=rollback_task, queue_name="rollback_queue")
        logger.info(f"Published ROLLBACK task to version {target_version_id}")

    def _publish_training_complete_task(
        self, validation_result: Dict[str, Any], completion_reason: str
    ) -> None:
        """
        Publish TRAINING_COMPLETE task.

        Args:
            validation_result: Final validation result
            completion_reason: Reason for training completion
        """
        model_version_id = validation_result.get("model_version_id")
        metrics = validation_result.get("metrics", {})
        ipfs_cid = validation_result.get("ipfs_cid")

        # Validate required fields
        if not model_version_id:
            raise ValueError("model_version_id is required in validation_result")
        if not ipfs_cid:
            raise ValueError("ipfs_cid is required in validation_result")

        logger.info(f"Publishing TRAINING_COMPLETE task for version {model_version_id}")

        # Calculate training duration
        training_duration = None
        if self.state.training_start_time:
            training_duration = time.time() - self.state.training_start_time

        training_summary = {
            "total_iterations": self.state.current_iteration,
            "clients_participated": self.num_clients,
            "training_duration_seconds": training_duration,
            "total_rounds": self.state.current_iteration,
            "rollback_count": self.state.rollback_count,
        }

        metadata = {
            "hyperparameters_used": {},  # TODO: Add hyperparameters
            "dataset_info": {},  # TODO: Add dataset info
            "completion_reason": completion_reason,
        }

        complete_task = Task(
            task_id=f"training-complete-{model_version_id}-{int(time.time() * 1000)}",
            task_type=TaskType.TRAINING_COMPLETE,
            payload=TrainingCompleteTaskPayload(
                final_model_version_id=str(model_version_id),
                final_accuracy=metrics.get("accuracy", 0.0),
                final_metrics=metrics,
                final_weights_cid=str(ipfs_cid),
                training_summary=training_summary,
                metadata=metadata,
            ).model_dump(),
            metadata=TaskMetadata(source="decision_worker"),
            model_version_id=str(model_version_id),
            parent_version_id=validation_result.get("parent_version_id"),
        )

        self.publisher.publish_task(
            task=complete_task, queue_name="training_complete_queue"
        )
        logger.info(f"Published TRAINING_COMPLETE task for version {model_version_id}")

    def _handle_decision_task(self, task: Task) -> bool:
        """
        Handle a DECISION task.

        Args:
            task: DECISION task to process

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Processing DECISION task: {task.task_id}")

            # Parse payload
            payload = DecisionTaskPayload(**task.payload)

            validation_result = payload.validation_result
            model_version_id = payload.model_version_id
            should_rollback = payload.should_rollback
            rollback_reason = payload.rollback_reason

            # Extract metrics
            metrics = validation_result.get("metrics", {})
            current_accuracy = metrics.get("accuracy", 0.0)

            # Get iteration from validation result (from model version metadata)
            # If not present, increment current_iteration as fallback
            validated_iteration = validation_result.get("iteration")
            if validated_iteration is not None:
                # Use iteration from validation result
                self.state.current_iteration = validated_iteration
                logger.debug(
                    f"Using iteration {validated_iteration} from validation result"
                )
            else:
                # Fallback: increment current iteration
                self.state.current_iteration += 1
                logger.warning(
                    "Iteration not found in validation result, incrementing current_iteration"
                )

            # Update state
            if self.state.training_start_time is None:
                self.state.training_start_time = time.time()

            self.state.accuracy_history.append(current_accuracy)

            logger.info(
                f"Iteration {self.state.current_iteration}: accuracy={current_accuracy:.2f}%, "
                f"best={self.state.best_accuracy:.2f}%, "
                f"patience={self.state.patience_counter}/{self.patience_threshold}, "
                f"rollbacks={self.state.rollback_count}/{self.max_rollbacks}"
            )

            # Evaluate rollback (if not already determined)
            if not should_rollback:
                should_rollback, rollback_reason = self._evaluate_rollback(
                    current_accuracy, validation_result
                )

            # Handle rollback
            if should_rollback:
                if self.state.rollback_count >= self.max_rollbacks:
                    # Max rollbacks reached - complete training
                    logger.warning(
                        f"Max rollbacks reached. Completing training with best checkpoint."
                    )
                    # Use best checkpoint for final model
                    final_result = validation_result.copy()
                    final_result["model_version_id"] = (
                        self.state.best_checkpoint_version
                    )
                    final_result["ipfs_cid"] = self.state.best_checkpoint_cid
                    final_result["metrics"]["accuracy"] = self.state.best_accuracy
                    self._publish_training_complete_task(
                        final_result, "Maximum rollbacks reached"
                    )
                    # Update shared state
                    return True

                # Validate best checkpoint exists
                if not self.state.best_checkpoint_version:
                    raise ValueError(
                        "Cannot rollback: best_checkpoint_version is not set"
                    )
                if not self.state.best_checkpoint_cid:
                    raise ValueError("Cannot rollback: best_checkpoint_cid is not set")

                # Publish rollback task
                self.state.rollback_count += 1
                # Update shared state
                self._publish_rollback_task(
                    target_version_id=self.state.best_checkpoint_version,
                    target_weights_cid=self.state.best_checkpoint_cid,
                    reason=rollback_reason or "Automatic rollback",
                )

                # Reset patience after rollback
                self.state.patience_counter = 0

                # Publish TRAIN tasks to continue from rolled-back state
                self._publish_train_tasks(
                    iteration=self.state.current_iteration,
                    weights_cid=self.state.best_checkpoint_cid,
                )

                logger.info(
                    f"Rollback completed. Continuing training from iteration {self.state.current_iteration}"
                )
                return True

            # Check training completion
            should_complete, completion_reason = self._check_training_completion(
                current_accuracy, validation_result
            )

            if should_complete:
                # Validate completion reason
                if not completion_reason:
                    completion_reason = "Training completed (unknown reason)"

                # Publish training complete task
                self._publish_training_complete_task(
                    validation_result, completion_reason
                )
                # Update shared state
                logger.info(f"Training completed: {completion_reason}")
                return True

            # Continue training - publish TRAIN tasks for next iteration
            next_iteration = self.state.current_iteration + 1
            self._publish_train_tasks(
                iteration=next_iteration,
                weights_cid=validation_result.get("ipfs_cid"),
            )

            logger.info(
                f"Training continues. Published TRAIN tasks for iteration {next_iteration} "
                f"(validated iteration {self.state.current_iteration})"
            )
            return True

        except Exception as e:
            logger.error(
                f"Error processing DECISION task {task.task_id}: {str(e)}",
                exc_info=True,
            )
            return False

    def start(self, queue_name: str = "decision_queue") -> None:
        """
        Start decision worker.

        Args:
            queue_name: Queue name to consume from
        """
        logger.info(f"Starting decision worker (consuming from {queue_name})")
        self.running = True

        def task_handler(task: Task) -> None:
            """Handle incoming tasks."""
            if task.task_type == TaskType.DECISION:
                success = self._handle_decision_task(task)
                if not success:
                    logger.error(f"Failed to process DECISION task {task.task_id}")
            else:
                logger.warning(
                    f"Received unexpected task type: {task.task_type} (expected DECISION)"
                )

        self.consumer.consume_tasks(queue_name, task_handler)

    def stop(self) -> None:
        """Stop decision worker."""
        self.running = False
        logger.info("Stopping decision worker")
