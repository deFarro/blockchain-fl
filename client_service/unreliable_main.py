"""
Unreliable client entrypoint: consumes TRAIN tasks and posts weight diffs to simulate
a malicious client. Iteration 1: zero diffs (no effect on training). From iteration 2:
random diffs to test rollback and client exclusion.

Run only when ADD_UNRELIABLE_CLIENT=true via: docker-compose --profile unreliable up.
"""

import asyncio
import json
import sys
import time
from typing import Dict, Any, Optional

import torch

from client_service.config import config
from shared.queue.consumer import QueueConsumer
from shared.queue.publisher import QueuePublisher
from shared.queue.connection import QueueConnection
from shared.models.task import Task, TaskType, TrainTaskPayload
from shared.storage.ipfs_client import IPFSClient
from shared.datasets import get_dataset
from shared.models.model import SimpleCNN
from shared.logger import setup_logger

logger = setup_logger(__name__)

# Fixed client ID; main service identifies and excludes this client at rollback time (individual-diff diagnosis)
UNRELIABLE_CLIENT_ID = "unreliable"


def _random_weights_dict(num_classes: int = 10, in_channels: int = 1) -> Dict[str, Any]:
    """Build a weight dict with same structure as SimpleCNN, filled with random values."""
    model = SimpleCNN(num_classes=num_classes, in_channels=in_channels)
    weights = model.get_weights()
    for name in weights:
        weights[name] = torch.randn_like(weights[name])
    return weights


def _zero_weights_dict(num_classes: int = 10, in_channels: int = 1) -> Dict[str, Any]:
    """Build a weight dict with same structure as SimpleCNN, filled with zeros (zero diff = no effect on aggregation)."""
    model = SimpleCNN(num_classes=num_classes, in_channels=in_channels)
    weights = model.get_weights()
    for name in weights:
        weights[name] = torch.zeros_like(weights[name])
    return weights


def _weights_dict_to_bytes(weights: Dict[str, Any]) -> bytes:
    """Serialize weights to JSON bytes (same format as SimpleCNN.weights_to_bytes())."""
    serializable = {}
    for name, tensor in weights.items():
        serializable[name] = tensor.cpu().numpy().tolist()
    return json.dumps(serializable).encode("utf-8")


class UnreliableClientWorker:
    """Worker that receives TRAIN tasks and publishes random weight diffs."""

    def __init__(self):
        self.connection = QueueConnection()
        self.consumer = QueueConsumer(connection=self.connection)
        self.publisher = QueuePublisher(connection=self.connection)
        dataset_loader = get_dataset(
            dataset_name=getattr(config, "dataset_name", "mnist"),
            data_dir=str(config.data_dir),
            seed=config.dataset_seed,
        )
        self.num_classes = dataset_loader.get_num_classes()
        self.in_channels = dataset_loader.get_in_channels()
        self.running = False

    def _handle_train_task(self, task: Task) -> bool:
        try:
            payload = TrainTaskPayload(**task.payload)
            # Iteration 1: zero diffs so this client does not affect the first round; from iteration 2: random diffs
            if payload.iteration == 1:
                weights = _zero_weights_dict(
                    num_classes=self.num_classes, in_channels=self.in_channels
                )
                logger.info(
                    f"Unreliable client: generating zero weight diff for iteration 1 (no effect on training)"
                )
            else:
                weights = _random_weights_dict(
                    num_classes=self.num_classes, in_channels=self.in_channels
                )
                logger.info(
                    f"Unreliable client: generating random weights for iteration {payload.iteration}"
                )
            weight_diff_bytes = _weights_dict_to_bytes(weights)

            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            async def upload():
                async with IPFSClient() as ipfs_client:
                    return await ipfs_client.add_bytes(weight_diff_bytes, pin=True)

            weight_diff_cid = loop.run_until_complete(upload())
            logger.info(
                f"Unreliable client: uploaded diff to IPFS CID={weight_diff_cid} "
                f"(iteration {payload.iteration})"
            )

            client_update = {
                "client_id": UNRELIABLE_CLIENT_ID,
                "iteration": payload.iteration,
                "weight_diff_cid": weight_diff_cid,
                "metrics": {
                    "loss": 0.0,
                    "accuracy": 0.0,
                    "samples": 10_000,
                    "epochs": 0,
                },
                "task_id": task.task_id,
                "model_version_id": task.model_version_id,
                "parent_version_id": task.parent_version_id,
            }

            self.publisher.publish_dict(client_update, "client_updates")
            logger.info(
                f"Unreliable client: published update for task {task.task_id} iteration {payload.iteration}"
            )
            return True
        except Exception as e:
            logger.error(
                f"Unreliable client error handling TRAIN task: {e}", exc_info=True
            )
            return False

    def _handle_task(self, task: Task) -> bool:
        if task.task_type == TaskType.TRAIN:
            return self._handle_train_task(task)
        if task.task_type == TaskType.ROLLBACK:
            logger.info(f"Unreliable client: ignoring ROLLBACK task {task.task_id}")
            return True
        logger.warning(f"Unreliable client: ignoring task type {task.task_type}")
        return True

    def start(self, queue_name: str = "train_queue") -> None:
        self.running = True
        logger.info(
            f"Unreliable client worker starting (client_id={UNRELIABLE_CLIENT_ID})"
        )

        def task_handler(task: Task):
            try:
                success = self._handle_task(task)
                if not success:
                    raise RuntimeError(f"Task {task.task_id} handling failed")
            except Exception as e:
                logger.error(f"Unreliable client task error: {e}", exc_info=True)
                raise

        num_clients = config.num_clients
        prefetch_count = max(num_clients, 5)
        consumer_id = f"client_{UNRELIABLE_CLIENT_ID}"

        self.consumer.consume_tasks(
            queue_name,
            task_handler,
            prefetch_count=prefetch_count,
            use_fanout=True,
            consumer_id=consumer_id,
        )

    def stop(self) -> None:
        self.running = False
        self.consumer.stop()


def main() -> None:
    logger.info("=" * 60)
    logger.info("Unreliable Client Service (random weight diffs)")
    logger.info("=" * 60)
    logger.info(f"Client ID: {UNRELIABLE_CLIENT_ID}")
    logger.info("=" * 60)

    worker = UnreliableClientWorker()

    def on_signal(sig, frame):
        logger.info("Shutdown signal received")
        worker.stop()
        sys.exit(0)

    import signal

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    try:
        worker.start(queue_name="train_queue")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
