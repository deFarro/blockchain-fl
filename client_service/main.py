"""Main entry point for client service."""

import sys
import signal
from client_service.config import config
from client_service.worker import ClientWorker
from shared.logger import setup_logger

logger = setup_logger(__name__)


def main():
    """Main function to run the client service."""
    logger.info("=" * 60)
    logger.info("Client Service Starting")
    logger.info("=" * 60)
    logger.info(f"Number of clients: {config.num_clients}")
    logger.info(f"Dataset split type: {config.split_type}")
    logger.info(f"Dataset seed: {config.dataset_seed}")
    logger.info(f"RabbitMQ: {config.rabbitmq_host}:{config.rabbitmq_port}")
    logger.info(
        "Note: Each client instance will generate a unique instance_id on startup"
    )
    logger.info("=" * 60)

    # Create worker (this generates the instance_id)
    worker = ClientWorker()

    # Prefetch training dataset on startup using the worker's instance_id
    logger.info("Prefetching training dataset on startup...")
    try:
        train_loader, train_dataset = worker.trainer.load_dataset(
            instance_id=worker.instance_id
        )
        logger.info(
            f"Training dataset prefetched successfully: "
            f"{len(train_dataset)} samples (instance_id={worker.instance_id})"
        )
    except Exception as e:
        logger.warning(
            f"Failed to prefetch training dataset: {e}. "
            "It will be loaded when training starts.",
            exc_info=True,
        )

    # Set up signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        worker.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Start worker (this blocks)
        worker.start(queue_name="train_queue")
    except Exception as e:
        logger.error(f"Fatal error in client service: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
