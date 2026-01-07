"""Aggregation worker that implements FedAvg algorithm."""

import json
import time
import threading
import queue as thread_queue
from typing import Dict, Any, List, Optional
from collections import defaultdict
import torch
import pika
from pika.channel import Channel
from shared.config import settings
from shared.queue.consumer import QueueConsumer
from shared.queue.publisher import QueuePublisher
from shared.queue.connection import QueueConnection
from shared.models.task import Task, TaskType, TaskMetadata, AggregateTaskPayload
from shared.logger import setup_logger
from shared.monitoring.metrics import get_metrics_collector
from shared.models.model import SimpleCNN

logger = setup_logger(__name__)


class AggregationWorker:
    """Worker that aggregates client weight updates using FedAvg."""

    def __init__(self, connection: Optional[QueueConnection] = None):
        """
        Initialize aggregation worker.

        Args:
            connection: QueueConnection instance (creates new one if None)
        """
        self.connection = connection or QueueConnection()
        self.consumer = QueueConsumer(connection=self.connection)
        self.publisher = QueuePublisher(connection=self.connection)
        self.running = False
        # Track current iteration being processed (None = not processing any iteration)
        self.current_iteration: Optional[int] = None
        # Track completed iterations (to reject late updates)
        self.completed_iterations: set[int] = set()

        logger.info("Aggregation worker initialized")

    def _deserialize_weight_diff(self, weight_diff_str: str) -> Dict[str, Any]:
        """
        Deserialize weight diff from JSON string.

        Args:
            weight_diff_str: JSON string containing weight diff

        Returns:
            Dictionary of weight differences
        """
        weight_diff_dict = json.loads(weight_diff_str)

        # Convert lists back to tensors
        weight_diff = {}
        for name, tensor_list in weight_diff_dict.items():
            weight_diff[name] = torch.tensor(tensor_list)

        return weight_diff

    def _fedavg_aggregate(
        self,
        client_updates: List[Dict[str, Any]],
        sample_counts: Optional[List[int]] = None,
        exclude_clients: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Aggregate client weight updates using Federated Averaging (FedAvg).

        Args:
            client_updates: List of client updates, each containing 'weight_diff' (JSON string)
            sample_counts: Optional list of sample counts per client (for weighted averaging)
            exclude_clients: Optional list of client IDs to exclude from aggregation.
                            Only used when re-aggregating after regression diagnosis.

        Returns:
            Aggregated weight diff as dictionary
        """
        agg_start = time.time()
        if not client_updates:
            raise ValueError("Cannot aggregate empty client updates list")

        if len(client_updates) == 1:
            # Single client, no aggregation needed
            logger.info("Only one client update, returning as-is")
            return self._deserialize_weight_diff(client_updates[0]["weight_diff"])

        # Filter excluded clients only if explicitly provided (for re-aggregation after diagnosis)
        if exclude_clients:
            original_count = len(client_updates)
            client_updates = [
                update
                for update in client_updates
                if update.get("client_id") not in exclude_clients
            ]
            if len(client_updates) < original_count:
                logger.info(
                    f"Excluding {original_count - len(client_updates)} client(s) from aggregation: "
                    f"{exclude_clients}"
                )
            if not client_updates:
                raise ValueError(
                    "All clients were excluded! Cannot aggregate with zero clients."
                )

        # Deserialize all weight diffs
        weight_diffs = []
        for update in client_updates:
            weight_diff = self._deserialize_weight_diff(update["weight_diff"])
            weight_diffs.append(weight_diff)

        # Get sample counts (use metrics['samples'] if available, otherwise equal weights)
        if sample_counts is None:
            sample_counts = []
            for update in client_updates:
                metrics = update.get("metrics", {})
                samples = metrics.get("samples", 1)  # Default to 1 if not available
                sample_counts.append(samples)

        # Normalize sample counts to weights
        total_samples = sum(sample_counts)
        if total_samples == 0:
            # Fallback to equal weights
            weights = [1.0 / len(client_updates)] * len(client_updates)
            logger.warning("Total sample count is 0, using equal weights")
        else:
            weights = [count / total_samples for count in sample_counts]

        # Aggregate weights using weighted average
        aggregated = {}
        for name in weight_diffs[0].keys():
            # Initialize aggregated weight
            aggregated[name] = torch.zeros_like(weight_diffs[0][name])

            # Weighted sum
            for weight_diff, weight in zip(weight_diffs, weights):
                aggregated[name] += weight_diff[name] * weight

        agg_duration = time.time() - agg_start
        get_metrics_collector().record_timing(
            "fedavg_aggregation",
            agg_duration,
            metadata={
                "num_clients": len(client_updates),
                "total_samples": total_samples,
                "excluded_clients": len(exclude_clients) if exclude_clients else 0,
            },
        )

        logger.info(
            f"Aggregated {len(client_updates)} client updates "
            f"(total samples: {total_samples}, weights: {[f'{w:.3f}' for w in weights]}, "
            f"duration: {agg_duration:.3f}s)"
        )

        return aggregated

    def _collect_client_updates(
        self, queue_name: str, iteration: int, timeout: int, min_clients: int
    ) -> List[Dict[str, Any]]:
        """
        Collect client updates from queue for a specific iteration.

        This is a simplified implementation that collects updates by consuming
        messages and filtering by iteration. In production, you'd want a more
        sophisticated batching mechanism.

        Args:
            queue_name: Name of the queue to consume from
            iteration: Training iteration number
            timeout: Maximum time to wait for updates (seconds)
            min_clients: Minimum number of clients required

        Returns:
            List of client updates
        """
        updates: List[Dict[str, Any]] = []
        start_time = time.time()

        # Set current iteration being processed
        # Note: The actual current iteration should be read from blockchain
        # This is just a local cache for late update rejection
        self.current_iteration = iteration

        logger.info(
            f"Collecting client updates for iteration {iteration} "
            f"(min_clients={min_clients}, timeout={timeout}s)"
        )

        def message_handler(message: Dict[str, Any]):
            """Handle received client update message."""
            msg_iteration = message.get("iteration")
            client_id = message.get("client_id", "unknown")

            # Ensure both are integers for comparison
            try:
                msg_iteration = (
                    int(msg_iteration) if msg_iteration is not None else None
                )
            except (ValueError, TypeError):
                logger.warning(
                    f"Invalid iteration in message from {client_id}: {msg_iteration}"
                )
                msg_iteration = None

            logger.debug(
                f"Processing message from {client_id}: iteration={msg_iteration}, "
                f"target_iteration={iteration}, match={msg_iteration == iteration}"
            )

            if msg_iteration == iteration:
                # This is the iteration we're collecting for
                updates.append(message)
                logger.info(
                    f"Received update from {client_id} "
                    f"for iteration {iteration} ({len(updates)}/{min_clients})"
                )
            elif msg_iteration is not None and msg_iteration < iteration:
                # Late update for a past iteration - reject it
                # TODO: Also check against on-chain current iteration for authoritative source
                if msg_iteration in self.completed_iterations:
                    logger.warning(
                        f"Rejecting late update from {client_id} for iteration {msg_iteration} "
                        f"(iteration {msg_iteration} already completed). "
                        f"Client should wait for next TRAIN task. "
                        f"Current iteration on-chain should be checked for authoritative rejection."
                    )
                else:
                    logger.warning(
                        f"Rejecting late update from {client_id} for iteration {msg_iteration} "
                        f"(currently processing iteration {iteration}). "
                        f"Client should wait for next TRAIN task. "
                        f"Current iteration on-chain should be checked for authoritative rejection."
                    )
            elif msg_iteration is not None and msg_iteration > iteration:
                # Future iteration - ignore for now (will be processed later)
                logger.debug(
                    f"Ignoring update from {client_id} for future iteration {msg_iteration} "
                    f"(currently processing iteration {iteration})"
                )
            else:
                logger.warning(
                    f"Received update from {client_id} with invalid iteration: {msg_iteration}"
                )

        # Use thread-safe queue for message collection
        message_queue: thread_queue.Queue = thread_queue.Queue()
        stop_flag = threading.Event()
        error_occurred = threading.Event()
        error_info: List[Exception] = []
        # Store consumer reference so main thread can stop it
        collection_consumer_ref: List[Optional[QueueConsumer]] = [None]

        def consume_messages():
            """Consume messages in a separate thread and put them in a queue."""
            # Create a NEW connection and consumer for this collection call
            # This avoids reentrancy issues when multiple collections happen concurrently
            collection_connection = QueueConnection()
            collection_consumer = QueueConsumer(connection=collection_connection)
            collection_consumer_ref[0] = collection_consumer

            try:

                def queue_handler(
                    message: Dict[str, Any],
                    channel: Optional[Channel] = None,
                    delivery_tag: Optional[int] = None,
                ):
                    """Handler that puts messages in thread-safe queue."""
                    # Check if we should stop
                    if stop_flag.is_set():
                        logger.debug("Stop flag set, ignoring message")
                        if channel and delivery_tag:
                            # Reject and requeue so other consumers can process
                            channel.basic_nack(delivery_tag=delivery_tag, requeue=True)
                        return

                    msg_iteration = message.get("iteration")
                    client_id = message.get("client_id", "unknown")

                    # Ensure iteration is an integer for comparison
                    try:
                        msg_iteration = (
                            int(msg_iteration) if msg_iteration is not None else None
                        )
                    except (ValueError, TypeError):
                        logger.warning(
                            f"Invalid iteration in message from {client_id}: {msg_iteration}"
                        )
                        msg_iteration = None

                    logger.debug(
                        f"Received message from queue: client={client_id}, iteration={msg_iteration}, target_iteration={iteration}"
                    )

                    # Only process messages for the current iteration
                    # Reject and requeue messages for other iterations
                    if msg_iteration != iteration:
                        if channel and delivery_tag:
                            if msg_iteration is not None and msg_iteration < iteration:
                                # Late update for past iteration - reject without requeue
                                logger.debug(
                                    f"Rejecting late update from {client_id} for iteration {msg_iteration} "
                                    f"(currently processing iteration {iteration})"
                                )
                                channel.basic_nack(
                                    delivery_tag=delivery_tag, requeue=False
                                )
                            elif (
                                msg_iteration is not None and msg_iteration > iteration
                            ):
                                # Future iteration - requeue for later processing
                                logger.debug(
                                    f"Requeuing message from {client_id} for future iteration {msg_iteration} "
                                    f"(currently processing iteration {iteration})"
                                )
                                channel.basic_nack(
                                    delivery_tag=delivery_tag, requeue=True
                                )
                            else:
                                # Invalid iteration - reject without requeue
                                logger.warning(
                                    f"Rejecting message from {client_id} with invalid iteration: {msg_iteration}"
                                )
                                channel.basic_nack(
                                    delivery_tag=delivery_tag, requeue=False
                                )
                        return

                    # Message matches current iteration - queue it for processing
                    try:
                        message_queue.put(message, timeout=1.0)
                        logger.debug(f"Message queued successfully from {client_id}")
                        # Acknowledge message only after successfully queuing
                        if channel and delivery_tag:
                            channel.basic_ack(delivery_tag=delivery_tag)
                    except thread_queue.Full:
                        logger.warning(
                            f"Message queue full, dropping message from {client_id}"
                        )
                        # Reject and requeue if queue is full
                        if channel and delivery_tag:
                            channel.basic_nack(delivery_tag=delivery_tag, requeue=True)

                # Consume messages (blocking call) with manual acknowledgment
                # This will block until stop_flag is set or connection is closed
                collection_consumer.consume_dict(
                    queue_name, queue_handler, auto_ack=False
                )
            except Exception as e:
                error_occurred.set()
                error_info.append(e)
                logger.error(f"Error in consumer thread: {str(e)}", exc_info=True)
            finally:
                # Clean up the collection-specific connection
                # Note: stop() may have already been called from main thread
                # but it's idempotent, so calling it again is safe
                try:
                    # Stop consuming (idempotent - safe to call multiple times)
                    # This ensures consuming is stopped even if main thread's stop() failed
                    collection_consumer.stop()
                    # Close connection after consumer is stopped
                    # The connection will be closed gracefully
                    collection_connection.close()
                except Exception as e:
                    logger.debug(f"Error closing collection connection: {str(e)}")
                finally:
                    # Always set stop flag, even if cleanup failed
                    stop_flag.set()  # Signal that consumption has stopped

        # Start consumer in a non-daemon thread (so it can be properly joined)
        consumer_thread = threading.Thread(
            target=consume_messages, name="AggregationConsumer", daemon=False
        )
        consumer_thread.start()

        # Calculate deadline for precise timeout handling
        deadline = start_time + timeout

        try:
            # Collect messages until we have enough or timeout
            deadline_passed = False
            while not stop_flag.is_set():
                current_time = time.time()

                # Check timeout using deadline (more precise than elapsed time)
                if current_time >= deadline and not deadline_passed:
                    elapsed = current_time - start_time
                    deadline_passed = True
                    logger.warning(
                        f"Timeout reached ({timeout}s, elapsed: {elapsed:.2f}s) "
                        f"while collecting updates. "
                        f"Got {len(updates)} updates (required: {min_clients}). "
                        f"Will continue processing queued messages..."
                    )
                    # Don't break immediately - allow processing of messages already in queue

                if len(updates) >= min_clients:
                    elapsed = current_time - start_time
                    logger.info(
                        f"Collected {len(updates)} updates (required: {min_clients}) "
                        f"in {elapsed:.2f}s"
                    )
                    break

                # Check for errors
                if error_occurred.is_set():
                    if error_info:
                        raise error_info[0]
                    break

                # Calculate remaining time for queue.get timeout
                if deadline_passed:
                    # After deadline, use short timeout to check queue one more time
                    # This allows processing messages that were already queued
                    remaining_time = 0.5  # Short timeout to check queue
                else:
                    remaining_time = max(0.1, deadline - current_time)
                    remaining_time = min(
                        remaining_time, 0.5
                    )  # Cap at 0.5s for responsiveness

                # Try to get a message from queue
                try:
                    message = message_queue.get(timeout=remaining_time)
                    # Process the message if it's for the current iteration
                    # Even if deadline passed, we still want to process queued messages
                    message_handler(message)
                except thread_queue.Empty:
                    # No message available
                    if deadline_passed:
                        # Deadline passed and no more messages in queue - we're done
                        elapsed = time.time() - start_time
                        logger.warning(
                            f"No more messages in queue after timeout. "
                            f"Got {len(updates)} updates (required: {min_clients}) "
                            f"after {elapsed:.2f}s"
                        )
                        break
                    # Continue waiting if deadline hasn't passed
                    continue

        except Exception as e:
            logger.error(f"Error collecting client updates: {str(e)}", exc_info=True)
            raise
        finally:
            # Stop consumer gracefully
            logger.debug("Stopping consumer...")
            stop_flag.set()

            # Stop the consumer from the main thread (consume_dict is blocking)
            # We need to stop it from outside the consumer thread
            if collection_consumer_ref[0] is not None:
                try:
                    collection_consumer_ref[0].stop()
                except Exception as e:
                    logger.debug(f"Error stopping consumer: {str(e)}")

            # Wait for consumer thread to finish (with timeout)
            # The consumer thread will stop when consume_dict() is interrupted
            consumer_thread.join(timeout=5.0)
            if consumer_thread.is_alive():
                logger.warning("Consumer thread did not stop within timeout")
            else:
                logger.debug("Consumer thread stopped successfully")

            # Check for errors in consumer thread
            if error_occurred.is_set() and error_info:
                logger.error(
                    f"Consumer thread encountered error: {error_info[0]}", exc_info=True
                )

            # Log final statistics
            elapsed_total = time.time() - start_time
            logger.info(
                f"Collection completed: {len(updates)} updates collected in {elapsed_total:.2f}s "
                f"(timeout: {timeout}s, required: {min_clients})"
            )

        return updates

    def _handle_aggregation(
        self, client_updates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Handle aggregation of client updates.

        Args:
            client_updates: List of client updates to aggregate

        Returns:
            Aggregated weight diff
        """
        if not client_updates:
            raise ValueError("No client updates to aggregate")

        # Extract iteration from first update
        iteration = client_updates[0]["iteration"]

        # Verify all updates are for the same iteration
        for update in client_updates:
            if update["iteration"] != iteration:
                raise ValueError(
                    f"Mismatched iterations: expected {iteration}, "
                    f"got {update['iteration']}"
                )

        logger.info(
            f"Aggregating {len(client_updates)} client updates for iteration {iteration}"
        )

        # Aggregate using FedAvg
        aggregated_diff = self._fedavg_aggregate(client_updates)

        # Serialize aggregated diff
        aggregated_dict = {}
        for name, tensor in aggregated_diff.items():
            aggregated_dict[name] = tensor.numpy().tolist()

        aggregated_str = json.dumps(aggregated_dict)

        return {
            "aggregated_diff": aggregated_str,
            "iteration": iteration,
            "num_clients": len(client_updates),
            "client_ids": [update["client_id"] for update in client_updates],
        }

    def _publish_aggregated_update(self, aggregated_data: Dict[str, Any]) -> None:
        """
        Publish aggregated update to queue.

        Args:
            aggregated_data: Aggregated data containing diff, iteration, etc.
        """
        # Create AGGREGATE task
        # Create task for blockchain worker
        # The blockchain worker will:
        # 1. Compute hash of aggregated_diff
        # 2. Store iteration, num_clients, client_ids in blockchain metadata
        # 3. Register on blockchain with this metadata
        aggregate_task = Task(
            task_id=f"aggregate-{aggregated_data['iteration']}-{int(time.time())}",
            task_type=TaskType.BLOCKCHAIN_WRITE,  # Next step: blockchain write
            payload={
                "aggregated_diff": aggregated_data["aggregated_diff"],
                # These will be included in blockchain metadata by blockchain worker:
                "iteration": aggregated_data["iteration"],
                "num_clients": aggregated_data["num_clients"],
                "client_ids": aggregated_data["client_ids"],
            },
            metadata=TaskMetadata(source="aggregation_worker"),
            model_version_id=None,  # Will be generated by blockchain worker
            parent_version_id=None,  # Will be set by blockchain worker based on previous version
        )

        # Publish to blockchain_write queue
        queue_name = "blockchain_write"
        self.publisher.publish_task(aggregate_task, queue_name)

        # Mark this iteration as completed
        iteration_num = aggregated_data["iteration"]
        self.completed_iterations.add(iteration_num)
        if self.current_iteration == iteration_num:
            self.current_iteration = None  # No longer processing this iteration

        logger.info(
            f"Published aggregated update for iteration {iteration_num} "
            f"to queue '{queue_name}'"
        )

    def process_client_updates(
        self,
        queue_name: str = "client_updates",
        iteration: Optional[int] = None,
        timeout: Optional[int] = None,
        min_clients: Optional[int] = None,
    ) -> bool:
        """
        Process client updates from queue and aggregate them.

        Args:
            queue_name: Name of the queue to consume from
            iteration: Specific iteration to process (None for any)
            timeout: Timeout in seconds (uses config if None)
            min_clients: Minimum clients required (uses config if None)

        Returns:
            True if successful, False otherwise
        """
        timeout = timeout or settings.aggregation_timeout
        min_clients = min_clients or settings.min_clients_for_aggregation

        try:
            # Collect client updates
            if iteration is not None:
                client_updates = self._collect_client_updates(
                    queue_name, iteration, timeout, min_clients
                )
            else:
                # For now, consume one message at a time
                # In production, you'd want a more sophisticated batching mechanism
                logger.warning(
                    "Iteration not specified. This is a simplified implementation. "
                    "In production, use a proper message batching mechanism."
                )
                return False

            # Filter excluded clients if exclusion is enabled
            original_count = len(client_updates)
            if settings.enable_client_exclusion and settings.excluded_clients:
                excluded_clients = settings.excluded_clients
                client_updates = [
                    update
                    for update in client_updates
                    if update.get("client_id") not in excluded_clients
                ]
                excluded_count = original_count - len(client_updates)
                if excluded_count > 0:
                    logger.info(
                        f"Filtered out {excluded_count} excluded client(s) from aggregation: "
                        f"{excluded_clients}. Remaining: {len(client_updates)}/{original_count}"
                    )

            # Proceed with aggregation if we have at least one client update
            # Even a single client update can be applied to the current model state
            if len(client_updates) == 0:
                logger.warning(
                    f"No client updates remaining after filtering excluded clients. "
                    f"Cannot aggregate with zero clients."
                )
                return False

            # Aggregate updates
            aggregated_data = self._handle_aggregation(client_updates)

            # Publish aggregated update
            self._publish_aggregated_update(aggregated_data)

            return True

        except Exception as e:
            logger.error(f"Error processing client updates: {str(e)}", exc_info=True)
            return False

    def start(self, queue_name: str = "client_updates"):
        """
        Start consuming client updates and aggregating them.

        Args:
            queue_name: Name of the queue to consume from
        """
        self.running = True
        logger.info(f"Starting aggregation worker, consuming from queue: {queue_name}")

        # TODO: Implement proper message batching and iteration tracking
        # For now, this is a placeholder that processes updates one at a time
        # In production, you'd want:
        # - Track iterations and batch updates by iteration
        # - Wait for minimum number of clients
        # - Handle timeouts properly
        # - Use a more sophisticated message consumption pattern

        logger.warning(
            "Aggregation worker start() is a placeholder. "
            "Use process_client_updates() with specific iteration for now."
        )

    def stop(self):
        """Stop the worker."""
        self.running = False
        logger.info("Stopping aggregation worker...")
        self.consumer.stop()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
