"""Shared utilities for training operations."""

import time
from typing import Optional
from shared.queue.publisher import QueuePublisher
from shared.models.task import Task, TaskType, TaskMetadata, TrainTaskPayload
from shared.logger import setup_logger

logger = setup_logger(__name__)


def publish_train_task(
    publisher: QueuePublisher,
    iteration: int,
    weights_cid: Optional[str] = None,
    source: str = "system",
) -> None:
    """
    Publish a universal TRAIN task for all clients.
    
    This is a shared utility function used by decision worker, rollback worker,
    and API routes to publish training tasks.

    Args:
        publisher: QueuePublisher instance
        iteration: Training iteration number
        weights_cid: IPFS CID of model weights (if None, clients start from scratch)
        source: Source identifier for the task (e.g., "decision_worker", "rollback_worker", "api")
    """
    logger.info(f"Publishing universal TRAIN task for iteration {iteration} (source: {source})")

    train_task = Task(
        task_id=f"train-iter{iteration}-{int(time.time() * 1000)}",
        task_type=TaskType.TRAIN,
        payload=TrainTaskPayload(
            weights_cid=weights_cid,
            iteration=iteration,
        ).model_dump(),
        metadata=TaskMetadata(source=source),
        model_version_id=None,
        parent_version_id=None,
    )

    # Use fanout exchange so all clients receive the message simultaneously
    publisher.publish_task(
        task=train_task, queue_name="train_queue", use_fanout=True
    )
    logger.info(
        f"Published universal TRAIN task for iteration {iteration} via fanout exchange "
        f"(all clients will receive simultaneously)"
    )
