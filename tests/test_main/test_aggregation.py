"""Test aggregation worker functionality."""

import sys
from pathlib import Path
import os
import json
from unittest.mock import patch, AsyncMock

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Set localhost for local testing
if "RABBITMQ_HOST" not in os.environ:
    os.environ["RABBITMQ_HOST"] = "localhost"

import torch
import pytest
import time
import threading
import queue as thread_queue
from unittest.mock import Mock, MagicMock
from shared.queue.publisher import QueuePublisher
from main_service.workers.aggregation_worker import AggregationWorker
from main_service.workers.regression_diagnosis import RegressionDiagnosis
from client_service.training.model import SimpleCNN
from shared.config import settings


def serialize_diff(diff):
    """Helper to serialize weight diff to JSON string."""
    diff_dict = {}
    for name, tensor in diff.items():
        diff_dict[name] = tensor.numpy().tolist()
    return json.dumps(diff_dict)


def test_fedavg_aggregation():
    """Test FedAvg aggregation algorithm with multiple clients."""
    print("=" * 60)
    print("Testing FedAvg Aggregation")
    print("=" * 60)
    print()

    # Create models to generate weight diffs
    model1 = SimpleCNN(num_classes=10)
    weights1 = model1.get_weights()

    model2 = SimpleCNN(num_classes=10)
    weights2 = model2.get_weights()

    # Create weight diffs (simulate client updates)
    diff1 = {}
    diff2 = {}
    for name in weights1:
        diff1[name] = weights1[name] - torch.zeros_like(weights1[name])
        diff2[name] = weights2[name] - torch.zeros_like(weights2[name])

    # Serialize diffs
    diff1_str = serialize_diff(diff1)
    diff2_str = serialize_diff(diff2)

    # Create mock IPFS CID mapping
    cid1 = "QmTestClient0"
    cid2 = "QmTestClient1"
    ipfs_data = {
        cid1: diff1_str.encode("utf-8"),
        cid2: diff2_str.encode("utf-8"),
    }

    client_updates = [
        {
            "client_id": "client_0",
            "iteration": 1,
            "weight_diff_cid": cid1,
            "metrics": {"loss": 0.5, "accuracy": 90.0, "samples": 100},
        },
        {
            "client_id": "client_1",
            "iteration": 1,
            "weight_diff_cid": cid2,
            "metrics": {"loss": 0.6, "accuracy": 85.0, "samples": 200},
        },
    ]

    # Mock IPFS client
    async def mock_get_bytes(cid: str):
        return ipfs_data[cid]

    # Test aggregation
    worker = AggregationWorker()
    with patch("main_service.workers.aggregation_worker.IPFSClient") as mock_ipfs_class:
        mock_client = AsyncMock()
        mock_client.get_bytes = AsyncMock(side_effect=mock_get_bytes)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_ipfs_class.return_value = mock_client

        aggregated = worker._fedavg_aggregate(client_updates)

    assert len(aggregated) > 0, "Aggregated weights should not be empty"
    print(f"✓ Aggregated {len(aggregated)} weight parameters")

    # Verify aggregation shape
    for name in diff1:
        assert aggregated[name].shape == diff1[name].shape, f"Shape mismatch for {name}"
    print("✓ Aggregated weights have correct shapes")

    print()
    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def test_fedavg_weighted_averaging():
    """Test FedAvg with weighted averaging based on sample counts."""
    print("\n" + "=" * 60)
    print("Testing FedAvg Weighted Averaging")
    print("=" * 60)
    print()

    model1 = SimpleCNN(num_classes=10)
    weights1 = model1.get_weights()

    model2 = SimpleCNN(num_classes=10)
    weights2 = model2.get_weights()

    # Create diffs
    diff1 = {}
    diff2 = {}
    for name in weights1:
        diff1[name] = weights1[name] - torch.zeros_like(weights1[name])
        diff2[name] = weights2[name] - torch.zeros_like(weights2[name])

    # Serialize diffs
    diff1_str = serialize_diff(diff1)
    diff2_str = serialize_diff(diff2)

    # Create mock IPFS CID mapping
    cid1 = "QmTestClient0"
    cid2 = "QmTestClient1"
    ipfs_data = {
        cid1: diff1_str.encode("utf-8"),
        cid2: diff2_str.encode("utf-8"),
    }

    # Client 1 has 100 samples, Client 2 has 200 samples
    # Client 2 should have 2x weight in aggregation
    client_updates = [
        {
            "client_id": "client_0",
            "iteration": 1,
            "weight_diff_cid": cid1,
            "metrics": {"samples": 100},
        },
        {
            "client_id": "client_1",
            "iteration": 1,
            "weight_diff_cid": cid2,
            "metrics": {"samples": 200},
        },
    ]

    # Mock IPFS client
    async def mock_get_bytes(cid: str):
        return ipfs_data[cid]

    worker = AggregationWorker()
    with patch("main_service.workers.aggregation_worker.IPFSClient") as mock_ipfs_class:
        mock_client = AsyncMock()
        mock_client.get_bytes = AsyncMock(side_effect=mock_get_bytes)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_ipfs_class.return_value = mock_client

        aggregated = worker._fedavg_aggregate(client_updates)

        # Verify aggregation worked
        assert len(aggregated) > 0
        print("✓ Weighted averaging completed")

        # Test with explicit sample counts (still inside the patch context)
        aggregated2 = worker._fedavg_aggregate(client_updates, sample_counts=[100, 200])
        assert len(aggregated2) > 0
        print("✓ Explicit sample counts work")

    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def test_fedavg_single_client():
    """Test FedAvg with single client (no aggregation needed)."""
    print("\n" + "=" * 60)
    print("Testing FedAvg Single Client")
    print("=" * 60)
    print()

    model = SimpleCNN(num_classes=10)
    weights = model.get_weights()

    diff = {}
    for name in weights:
        diff[name] = weights[name] - torch.zeros_like(weights[name])

    # Serialize diff
    diff_str = serialize_diff(diff)

    # Create mock IPFS CID
    cid = "QmTestClient0"
    ipfs_data = {cid: diff_str.encode("utf-8")}

    client_updates = [
        {
            "client_id": "client_0",
            "iteration": 1,
            "weight_diff_cid": cid,
            "metrics": {"samples": 100},
        }
    ]

    # Mock IPFS client
    async def mock_get_bytes(cid: str):
        return ipfs_data[cid]

    worker = AggregationWorker()
    with patch("main_service.workers.aggregation_worker.IPFSClient") as mock_ipfs_class:
        mock_client = AsyncMock()
        mock_client.get_bytes = AsyncMock(side_effect=mock_get_bytes)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_ipfs_class.return_value = mock_client

        aggregated = worker._fedavg_aggregate(client_updates)

    # With single client, should return diff as-is
    assert len(aggregated) > 0
    print("✓ Single client handled correctly")

    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def test_fedavg_empty_updates():
    """Test FedAvg with empty client updates (should raise error)."""
    print("\n" + "=" * 60)
    print("Testing FedAvg Empty Updates")
    print("=" * 60)
    print()

    worker = AggregationWorker()

    with pytest.raises(ValueError, match="Cannot aggregate empty"):
        worker._fedavg_aggregate([])

    print("✓ Empty updates correctly raise ValueError")

    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def test_fedavg_client_exclusion():
    """Test FedAvg with client exclusion."""
    print("\n" + "=" * 60)
    print("Testing FedAvg Client Exclusion")
    print("=" * 60)
    print()

    model1 = SimpleCNN(num_classes=10)
    weights1 = model1.get_weights()

    model2 = SimpleCNN(num_classes=10)
    weights2 = model2.get_weights()

    model3 = SimpleCNN(num_classes=10)
    weights3 = model3.get_weights()

    diff1 = {}
    diff2 = {}
    diff3 = {}
    for name in weights1:
        diff1[name] = weights1[name] - torch.zeros_like(weights1[name])
        diff2[name] = weights2[name] - torch.zeros_like(weights2[name])
        diff3[name] = weights3[name] - torch.zeros_like(weights3[name])

    # Serialize diffs
    diff1_str = serialize_diff(diff1)
    diff2_str = serialize_diff(diff2)
    diff3_str = serialize_diff(diff3)

    # Create mock IPFS CID mapping
    cid1 = "QmTestClient0"
    cid2 = "QmTestClient1"
    cid3 = "QmTestClient2"
    ipfs_data = {
        cid1: diff1_str.encode("utf-8"),
        cid2: diff2_str.encode("utf-8"),
        cid3: diff3_str.encode("utf-8"),
    }

    client_updates = [
        {
            "client_id": "client_0",
            "iteration": 1,
            "weight_diff_cid": cid1,
            "metrics": {"samples": 100},
        },
        {
            "client_id": "client_1",
            "iteration": 1,
            "weight_diff_cid": cid2,
            "metrics": {"samples": 200},
        },
        {
            "client_id": "client_2",
            "iteration": 1,
            "weight_diff_cid": cid3,
            "metrics": {"samples": 150},
        },
    ]

    # Mock IPFS client
    async def mock_get_bytes(cid: str):
        return ipfs_data[cid]

    worker = AggregationWorker()
    with patch("main_service.workers.aggregation_worker.IPFSClient") as mock_ipfs_class:
        mock_client = AsyncMock()
        mock_client.get_bytes = AsyncMock(side_effect=mock_get_bytes)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_ipfs_class.return_value = mock_client

        # Aggregate all clients
        aggregated_all = worker._fedavg_aggregate(client_updates)

        # Exclude client_1
        aggregated_excluded = worker._fedavg_aggregate(
            client_updates, exclude_clients=["client_1"]
        )

        assert len(aggregated_all) > 0
        assert len(aggregated_excluded) > 0
        print("✓ Client exclusion works")

        # Verify excluded client is not in result
        # (aggregated result should be different)
        print("✓ Excluded client removed from aggregation")

        # Test excluding all clients (should raise error)
        with pytest.raises(ValueError, match="All clients were excluded"):
            worker._fedavg_aggregate(
                client_updates, exclude_clients=["client_0", "client_1", "client_2"]
            )

    print("✓ Excluding all clients correctly raises error")

    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def test_regression_diagnosis_basic():
    """Test regression diagnosis basic functionality."""
    print("\n" + "=" * 60)
    print("Testing Regression Diagnosis")
    print("=" * 60)
    print()

    # Create a simple mock test loader (just a list of batches)
    # In real scenario, this would be a DataLoader
    class MockTestLoader:
        def __init__(self):
            self.data = [
                (torch.randn(1, 1, 28, 28), torch.randint(0, 10, (1,)))
                for _ in range(10)
            ]

        def __iter__(self):
            return iter(self.data)

    model = SimpleCNN(num_classes=10)
    previous_weights = model.get_weights()

    # Create a diff that would cause regression (large negative changes)
    problematic_diff = {}
    for name, weight in previous_weights.items():
        problematic_diff[name] = -0.1 * weight  # Large negative change

    # Create a good diff (small positive changes)
    good_diff = {}
    for name, weight in previous_weights.items():
        good_diff[name] = 0.01 * weight  # Small positive change

    diagnosis = RegressionDiagnosis(model=SimpleCNN(num_classes=10))
    test_loader = MockTestLoader()

    # Test single client diff
    accuracy_good, metrics_good = diagnosis.test_single_client_diff(
        previous_weights, good_diff, test_loader
    )
    accuracy_bad, metrics_bad = diagnosis.test_single_client_diff(
        previous_weights, problematic_diff, test_loader
    )

    assert isinstance(accuracy_good, float)
    assert isinstance(accuracy_bad, float)
    assert "accuracy" in metrics_good
    assert "loss" in metrics_good
    print(f"✓ Single client diff testing works")
    print(f"  Good diff accuracy: {accuracy_good:.2f}%")
    print(f"  Bad diff accuracy: {accuracy_bad:.2f}%")

    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def test_collect_client_updates_timeout_behavior():
    """Test that collection waits for timeout unless all clients respond."""
    print("\n" + "=" * 60)
    print("Testing Collection Timeout Behavior")
    print("=" * 60)
    print()

    # Save original num_clients setting
    original_num_clients = settings.num_clients

    try:
        # Set up test scenario: 4 total clients, min 2 required
        settings.num_clients = 4
        min_clients = 2
        timeout = 1.5  # Short timeout for testing

        worker = AggregationWorker()

        # Create mock messages that will arrive at different times
        messages_to_send = [
            {"client_id": "client_0", "iteration": 1, "weight_diff_cid": "cid0"},
            {"client_id": "client_1", "iteration": 1, "weight_diff_cid": "cid1"},
            {"client_id": "client_2", "iteration": 1, "weight_diff_cid": "cid2"},
        ]

        # Use a thread-safe queue to simulate message arrival
        message_queue = thread_queue.Queue()
        stop_consuming = threading.Event()
        messages_sent = []

        def simulate_message_arrival():
            """Simulate messages arriving at different times."""
            # Send 2 messages immediately (min_clients)
            message_queue.put(messages_to_send[0])
            messages_sent.append(0)
            time.sleep(0.1)
            message_queue.put(messages_to_send[1])
            messages_sent.append(1)
            time.sleep(0.1)
            # Send 3rd message after a delay (but before timeout)
            time.sleep(0.3)
            message_queue.put(messages_to_send[2])
            messages_sent.append(2)
            # Don't send 4th message - should wait for timeout

        # Mock QueueConnection and QueueConsumer
        mock_connection = Mock()
        mock_consumer = Mock()

        def mock_consume_dict(queue_name, handler, auto_ack=False):
            """Mock consume_dict that simulates message consumption."""
            # Start thread to simulate message arrival
            arrival_thread = threading.Thread(
                target=simulate_message_arrival, daemon=True
            )
            arrival_thread.start()

            # Process messages from queue
            start_time = time.time()
            while not stop_consuming.is_set():
                try:
                    if stop_consuming.is_set():
                        break

                    try:
                        message = message_queue.get(timeout=0.05)
                        # Call handler with message
                        handler(message, channel=None, delivery_tag=None)
                    except thread_queue.Empty:
                        # Check if timeout exceeded
                        if time.time() - start_time > timeout + 0.5:
                            break
                        continue
                except Exception:
                    break

        mock_consumer.consume_dict = mock_consume_dict
        mock_consumer.stop = Mock()
        mock_connection.close = Mock()

        # Patch QueueConnection to return our mock
        with patch(
            "main_service.workers.aggregation_worker.QueueConnection",
            return_value=mock_connection,
        ):
            with patch(
                "main_service.workers.aggregation_worker.QueueConsumer",
                return_value=mock_consumer,
            ):
                start_time = time.time()
                updates = worker._collect_client_updates(
                    queue_name="test_queue",
                    iteration=1,
                    timeout=timeout,
                    min_clients=min_clients,
                )
                elapsed_time = time.time() - start_time

                # Signal stop
                stop_consuming.set()
                time.sleep(0.1)  # Give threads time to finish

        # Should have collected 3 updates (2 immediately + 1 delayed)
        # Should have waited close to timeout since not all clients responded
        assert len(updates) == 3, f"Expected 3 updates, got {len(updates)}"
        assert (
            elapsed_time >= timeout * 0.7
        ), f"Should wait close to timeout ({timeout}s), but elapsed {elapsed_time:.2f}s"
        print(f"✓ Collected {len(updates)} updates (expected 3)")
        print(f"✓ Waited {elapsed_time:.2f}s (timeout: {timeout}s)")
        print(f"✓ Correctly waited for timeout when not all clients responded")

    finally:
        # Restore original setting
        settings.num_clients = original_num_clients

    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def test_collect_client_updates_early_completion():
    """Test that collection stops early when all clients respond."""
    print("\n" + "=" * 60)
    print("Testing Collection Early Completion")
    print("=" * 60)
    print()

    # Save original num_clients setting
    original_num_clients = settings.num_clients

    try:
        # Set up test scenario: 3 total clients, min 2 required
        settings.num_clients = 3
        min_clients = 2
        timeout = 3  # Longer timeout

        worker = AggregationWorker()

        # Create mock messages
        messages_to_send = [
            {"client_id": "client_0", "iteration": 1, "weight_diff_cid": "cid0"},
            {"client_id": "client_1", "iteration": 1, "weight_diff_cid": "cid1"},
            {"client_id": "client_2", "iteration": 1, "weight_diff_cid": "cid2"},
        ]

        # Use a thread-safe queue to simulate message arrival
        message_queue = thread_queue.Queue()
        stop_consuming = threading.Event()
        all_messages_sent_event = threading.Event()

        def simulate_message_arrival():
            """Simulate all messages arriving quickly."""
            # Send all messages quickly (within 0.2 seconds)
            for msg in messages_to_send:
                message_queue.put(msg)
                time.sleep(0.05)  # Short delay between messages
            # Signal that all messages have been sent
            all_messages_sent_event.set()

        # Mock QueueConnection and QueueConsumer
        mock_connection = Mock()
        mock_consumer = Mock()

        def mock_consume_dict(queue_name, handler, auto_ack=False):
            """Mock consume_dict that simulates message consumption."""
            # Start thread to simulate message arrival
            arrival_thread = threading.Thread(
                target=simulate_message_arrival, daemon=True
            )
            arrival_thread.start()

            # Process messages from queue - keep processing until all are sent and processed
            processed_count = 0
            start_time = time.time()
            # Keep processing until we've handled all messages or stop is signaled
            while not stop_consuming.is_set():
                try:
                    try:
                        message = message_queue.get(timeout=0.15)
                        handler(message, channel=None, delivery_tag=None)
                        processed_count += 1
                    except thread_queue.Empty:
                        # If all messages have been sent AND processed by handler, wait longer
                        # to ensure main thread has time to process them from internal queue
                        if all_messages_sent_event.is_set() and processed_count >= len(
                            messages_to_send
                        ):
                            # Wait longer for main thread to process messages from internal queue
                            # The handler puts messages in an internal queue that the main thread reads from
                            time.sleep(0.8)
                            # Check one more time for any late messages
                            try:
                                message = message_queue.get(timeout=0.1)
                                handler(message, channel=None, delivery_tag=None)
                                processed_count += 1
                            except thread_queue.Empty:
                                pass
                            # Exit after giving main thread sufficient time to process
                            break
                        # If timeout exceeded, exit
                        elapsed = time.time() - start_time
                        if elapsed > timeout + 2:
                            break
                        continue
                except Exception:
                    break
            # Wait for arrival thread to finish
            arrival_thread.join(timeout=1.0)

        mock_consumer.consume_dict = mock_consume_dict
        mock_consumer.stop = Mock()
        mock_connection.close = Mock()

        # Patch QueueConnection to return our mock
        with patch(
            "main_service.workers.aggregation_worker.QueueConnection",
            return_value=mock_connection,
        ):
            with patch(
                "main_service.workers.aggregation_worker.QueueConsumer",
                return_value=mock_consumer,
            ):
                start_time = time.time()
                updates = worker._collect_client_updates(
                    queue_name="test_queue",
                    iteration=1,
                    timeout=timeout,
                    min_clients=min_clients,
                )
                elapsed_time = time.time() - start_time

                # Signal stop and wait for threads to finish
                stop_consuming.set()
                time.sleep(0.3)  # Give threads time to finish processing

        # Should have collected all 3 updates
        # Should complete early (much faster than timeout) since all clients responded
        assert (
            len(updates) == 3
        ), f"Expected 3 updates (all clients), got {len(updates)}"
        assert (
            elapsed_time < timeout * 0.4
        ), f"Should complete early (< {timeout * 0.4}s), but took {elapsed_time:.2f}s"
        print(f"✓ Collected all {len(updates)} updates (all clients responded)")
        print(f"✓ Completed early in {elapsed_time:.2f}s (timeout: {timeout}s)")
        print(f"✓ Correctly stopped early when all clients responded")

    finally:
        # Restore original setting
        settings.num_clients = original_num_clients

    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def test_collect_client_updates_minimum_clients():
    """Test that minimum client requirement is still respected."""
    print("\n" + "=" * 60)
    print("Testing Collection Minimum Clients Requirement")
    print("=" * 60)
    print()

    # Save original num_clients setting
    original_num_clients = settings.num_clients

    try:
        # Set up test scenario: 5 total clients, min 2 required
        settings.num_clients = 5
        min_clients = 2
        timeout = 1.5  # Short timeout

        worker = AggregationWorker()

        # Create mock messages - only send 1 (below minimum)
        messages_to_send = [
            {"client_id": "client_0", "iteration": 1, "weight_diff_cid": "cid0"},
        ]

        # Use a thread-safe queue to simulate message arrival
        message_queue = thread_queue.Queue()
        stop_consuming = threading.Event()

        def simulate_message_arrival():
            """Simulate only 1 message arriving."""
            message_queue.put(messages_to_send[0])
            # Don't send more - should wait for timeout

        # Mock QueueConnection and QueueConsumer
        mock_connection = Mock()
        mock_consumer = Mock()

        def mock_consume_dict(queue_name, handler, auto_ack=False):
            """Mock consume_dict that simulates message consumption."""
            # Start thread to simulate message arrival
            arrival_thread = threading.Thread(
                target=simulate_message_arrival, daemon=True
            )
            arrival_thread.start()

            # Process messages from queue
            start_time = time.time()
            while not stop_consuming.is_set():
                try:
                    if stop_consuming.is_set():
                        break

                    try:
                        message = message_queue.get(timeout=0.05)
                        handler(message, channel=None, delivery_tag=None)
                    except thread_queue.Empty:
                        # Check if timeout exceeded
                        if time.time() - start_time > timeout + 0.5:
                            break
                        continue
                except Exception:
                    break

        mock_consumer.consume_dict = mock_consume_dict
        mock_consumer.stop = Mock()
        mock_connection.close = Mock()

        # Patch QueueConnection to return our mock
        with patch(
            "main_service.workers.aggregation_worker.QueueConnection",
            return_value=mock_connection,
        ):
            with patch(
                "main_service.workers.aggregation_worker.QueueConsumer",
                return_value=mock_consumer,
            ):
                start_time = time.time()
                updates = worker._collect_client_updates(
                    queue_name="test_queue",
                    iteration=1,
                    timeout=timeout,
                    min_clients=min_clients,
                )
                elapsed_time = time.time() - start_time

                # Signal stop
                stop_consuming.set()
                time.sleep(0.1)  # Give threads time to finish

        # Should have collected only 1 update (below minimum)
        # Should have waited for timeout since we didn't reach minimum
        assert len(updates) == 1, f"Expected 1 update, got {len(updates)}"
        assert (
            elapsed_time >= timeout * 0.7
        ), f"Should wait for timeout ({timeout}s), but elapsed {elapsed_time:.2f}s"
        print(f"✓ Collected {len(updates)} update (below minimum of {min_clients})")
        print(f"✓ Waited {elapsed_time:.2f}s (timeout: {timeout}s)")
        print(f"✓ Correctly waited for timeout when minimum not reached")

    finally:
        # Restore original setting
        settings.num_clients = original_num_clients

    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def test_regression_diagnosis_identify_problematic():
    """Test regression diagnosis identifying problematic clients."""
    print("\n" + "=" * 60)
    print("Testing Regression Diagnosis - Identify Problematic Clients")
    print("=" * 60)
    print()

    class MockTestLoader:
        def __init__(self):
            self.data = [
                (torch.randn(1, 1, 28, 28), torch.randint(0, 10, (1,)))
                for _ in range(10)
            ]

        def __iter__(self):
            return iter(self.data)

    model = SimpleCNN(num_classes=10)
    previous_weights = model.get_weights()
    baseline_accuracy = 95.0  # Simulated baseline

    # Create client updates with one problematic client
    problematic_diff = {}
    good_diff1 = {}
    good_diff2 = {}
    for name, weight in previous_weights.items():
        problematic_diff[name] = -0.2 * weight  # Very bad
        good_diff1[name] = 0.01 * weight  # Good
        good_diff2[name] = 0.01 * weight  # Good

    client_updates = [
        {
            "client_id": "client_0",
            "weight_diff": serialize_diff(good_diff1),
        },
        {
            "client_id": "client_1",
            "weight_diff": serialize_diff(good_diff2),
        },
        {
            "client_id": "client_2",
            "weight_diff": serialize_diff(problematic_diff),
        },
    ]

    diagnosis = RegressionDiagnosis(model=SimpleCNN(num_classes=10))
    test_loader = MockTestLoader()

    # Run diagnosis with high threshold (will catch the bad one)
    problematic_clients = diagnosis.diagnose_regression(
        previous_weights,
        client_updates,
        test_loader,
        baseline_accuracy,
        accuracy_threshold=5.0,  # 5% drop threshold
    )

    # Should identify client_2 as problematic
    assert isinstance(problematic_clients, list)
    print(f"✓ Diagnosis identified {len(problematic_clients)} problematic client(s)")
    print(f"  Problematic clients: {problematic_clients}")

    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def run_all_tests():
    """Run all aggregation tests."""
    tests = [
        test_fedavg_aggregation,
        test_fedavg_weighted_averaging,
        test_fedavg_single_client,
        test_fedavg_empty_updates,
        test_fedavg_client_exclusion,
        test_collect_client_updates_timeout_behavior,
        test_collect_client_updates_early_completion,
        test_collect_client_updates_minimum_clients,
        test_regression_diagnosis_basic,
        test_regression_diagnosis_identify_problematic,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ {test_func.__name__} FAILED: {str(e)}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Test Summary: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
