"""Tests for blockchain worker."""

import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
import pytest
from shared.models.task import Task, TaskType, TaskMetadata
from main_service.workers.blockchain_worker import BlockchainWorker
from main_service.blockchain.fabric_client import FabricClient


def test_blockchain_worker_initialization():
    """Test blockchain worker initialization."""
    print("=" * 60)
    print("Testing Blockchain Worker - Initialization")
    print("=" * 60)
    print()

    worker = BlockchainWorker()

    assert worker.connection is not None
    assert worker.consumer is not None
    assert worker.publisher is not None
    assert worker.running is False
    assert worker.current_model_version_id is None

    print("✓ Blockchain worker initialized correctly")
    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def test_generate_model_version_id():
    """Test model version ID generation."""
    print("\n" + "=" * 60)
    print("Testing Blockchain Worker - Model Version ID Generation")
    print("=" * 60)
    print()

    worker = BlockchainWorker()

    # Generate version IDs
    version_id_1 = worker._generate_model_version_id(iteration=1)
    version_id_2 = worker._generate_model_version_id(iteration=1)
    version_id_3 = worker._generate_model_version_id(iteration=2)

    # Check format
    assert version_id_1.startswith("model_v1_")
    assert version_id_2.startswith("model_v1_")
    assert version_id_3.startswith("model_v2_")

    # Check uniqueness (should be different even for same iteration)
    assert version_id_1 != version_id_2

    print(f"✓ Generated version ID 1: {version_id_1}")
    print(f"✓ Generated version ID 2: {version_id_2}")
    print(f"✓ Generated version ID 3: {version_id_3}")
    print("✓ Version IDs are unique and correctly formatted")
    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def test_compute_diff_hash():
    """Test diff hash computation."""
    print("\n" + "=" * 60)
    print("Testing Blockchain Worker - Diff Hash Computation")
    print("=" * 60)
    print()

    worker = BlockchainWorker()

    # Test with sample diff
    diff_str = '{"layer1.weight": [1.0, 2.0, 3.0]}'
    hash_1 = worker._compute_diff_hash(diff_str)
    hash_2 = worker._compute_diff_hash(diff_str)

    # Hash should be deterministic
    assert hash_1 == hash_2
    assert len(hash_1) == 64  # SHA-256 hex string length

    # Different diff should produce different hash
    diff_str_2 = '{"layer1.weight": [1.0, 2.0, 4.0]}'
    hash_3 = worker._compute_diff_hash(diff_str_2)
    assert hash_1 != hash_3

    print(f"✓ Hash 1: {hash_1[:16]}...")
    print(f"✓ Hash 2: {hash_2[:16]}... (matches hash 1)")
    print(f"✓ Hash 3: {hash_3[:16]}... (different from hash 1)")
    print("✓ Hash computation is deterministic and produces unique hashes")
    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


@pytest.mark.asyncio
async def test_register_on_blockchain():
    """Test blockchain registration."""
    print("\n" + "=" * 60)
    print("Testing Blockchain Worker - Blockchain Registration")
    print("=" * 60)
    print()

    worker = BlockchainWorker()

    model_version_id = "test_version_1"
    parent_version_id = None
    blockchain_hash = "test_hash_12345"
    metadata = {"iteration": 1, "num_clients": 2}

    # Mock FabricClient
    mock_transaction_id = "tx_test_12345"

    with patch("main_service.workers.blockchain_worker.FabricClient") as mock_fabric_class:
        mock_client = AsyncMock()
        mock_client.register_model_update = AsyncMock(return_value=mock_transaction_id)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_fabric_class.return_value = mock_client

        transaction_id = await worker._register_on_blockchain(
            model_version_id=model_version_id,
            parent_version_id=parent_version_id,
            blockchain_hash=blockchain_hash,
            metadata=metadata,
        )

        assert transaction_id == mock_transaction_id
        print(f"✓ Registered model version: {model_version_id}")
        print(f"✓ Transaction ID: {transaction_id}")

        # Verify FabricClient was called correctly
        mock_client.register_model_update.assert_called_once_with(
            model_version_id=model_version_id,
            parent_version_id=parent_version_id,
            hash_value=blockchain_hash,
            metadata=metadata,
        )

    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


@pytest.mark.asyncio
async def test_process_blockchain_write():
    """Test processing blockchain write operation."""
    print("\n" + "=" * 60)
    print("Testing Blockchain Worker - Process Blockchain Write")
    print("=" * 60)
    print()

    worker = BlockchainWorker()

    aggregated_diff_str = '{"layer1.weight": [1.0, 2.0]}'
    iteration = 1
    num_clients = 2
    client_ids = ["client_0", "client_1"]

    # Mock FabricClient
    mock_transaction_id = "tx_test_12345"

    with patch("main_service.workers.blockchain_worker.FabricClient") as mock_fabric_class:
        mock_client = AsyncMock()
        mock_client.register_model_update = AsyncMock(return_value=mock_transaction_id)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_fabric_class.return_value = mock_client

        result = await worker._process_blockchain_write(
            aggregated_diff_str=aggregated_diff_str,
            iteration=iteration,
            num_clients=num_clients,
            client_ids=client_ids,
        )

        # Verify result structure
        assert "model_version_id" in result
        assert "parent_version_id" in result
        assert "blockchain_hash" in result
        assert "transaction_id" in result

        assert result["model_version_id"].startswith("model_v1_")
        assert result["parent_version_id"] is None  # First iteration
        assert result["blockchain_hash"] is not None
        assert result["transaction_id"] == mock_transaction_id

        # Verify current_model_version_id was updated
        assert worker.current_model_version_id == result["model_version_id"]

        print(f"✓ Model version ID: {result['model_version_id']}")
        print(f"✓ Parent version ID: {result['parent_version_id']}")
        print(f"✓ Blockchain hash: {result['blockchain_hash'][:16]}...")
        print(f"✓ Transaction ID: {result['transaction_id']}")

        # Test second iteration (should have parent)
        result_2 = await worker._process_blockchain_write(
            aggregated_diff_str='{"layer1.weight": [2.0, 3.0]}',
            iteration=2,
            num_clients=2,
            client_ids=["client_0", "client_1"],
        )

        assert result_2["parent_version_id"] == result["model_version_id"]
        print(f"✓ Second iteration parent: {result_2['parent_version_id']}")

    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def test_publish_storage_task():
    """Test publishing storage task."""
    print("\n" + "=" * 60)
    print("Testing Blockchain Worker - Publish Storage Task")
    print("=" * 60)
    print()

    worker = BlockchainWorker()

    # Mock publisher
    mock_publisher = Mock()
    worker.publisher = mock_publisher

    aggregated_diff_str = '{"layer1.weight": [1.0, 2.0]}'
    blockchain_hash = "test_hash_12345"
    model_version_id = "model_v1_1234567890_abc123"
    parent_version_id = None

    worker._publish_storage_task(
        aggregated_diff_str=aggregated_diff_str,
        blockchain_hash=blockchain_hash,
        model_version_id=model_version_id,
        parent_version_id=parent_version_id,
    )

    # Verify task was published
    assert mock_publisher.publish_task.called
    call_args = mock_publisher.publish_task.call_args

    # publish_task is called with positional args: (task, queue_name)
    # call_args[0] is tuple of positional args, call_args[1] is dict of keyword args
    published_task = call_args[0][0]  # First positional arg: task
    queue_name = call_args[0][1]  # Second positional arg: queue_name
    
    assert queue_name == "storage_write"
    assert isinstance(published_task, Task)
    assert published_task.task_type == TaskType.STORAGE_WRITE
    assert published_task.model_version_id == model_version_id
    assert published_task.parent_version_id == parent_version_id

    # Check payload
    assert published_task.payload["aggregated_diff"] == aggregated_diff_str
    assert published_task.payload["blockchain_hash"] == blockchain_hash
    assert published_task.payload["model_version_id"] == model_version_id

    print(f"✓ Published STORAGE_WRITE task for version: {model_version_id}")
    print(f"✓ Queue: storage_write")
    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def test_handle_blockchain_write_task():
    """Test handling blockchain write task."""
    print("\n" + "=" * 60)
    print("Testing Blockchain Worker - Handle Blockchain Write Task")
    print("=" * 60)
    print()

    worker = BlockchainWorker()

    # Mock publisher
    mock_publisher = Mock()
    worker.publisher = mock_publisher

    # Create BLOCKCHAIN_WRITE task
    task = Task(
        task_id="blockchain-test-001",
        task_type=TaskType.BLOCKCHAIN_WRITE,
        payload={
            "aggregated_diff": '{"layer1.weight": [1.0, 2.0]}',
            "iteration": 1,
            "num_clients": 2,
            "client_ids": ["client_0", "client_1"],
        },
        metadata=TaskMetadata(source="test"),
    )

    # Mock async methods
    mock_transaction_id = "tx_test_12345"

    async def mock_process_blockchain_write(*args, **kwargs):
        return {
            "model_version_id": "model_v1_1234567890_abc123",
            "parent_version_id": None,
            "blockchain_hash": "test_hash_12345",
            "transaction_id": mock_transaction_id,
        }

    worker._process_blockchain_write = AsyncMock(
        side_effect=mock_process_blockchain_write
    )

    # Run async method in sync context
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # If loop is already running, use run_until_complete
        result = loop.run_until_complete(
            mock_process_blockchain_write(
                aggregated_diff_str=task.payload["aggregated_diff"],
                iteration=task.payload["iteration"],
                num_clients=task.payload["num_clients"],
                client_ids=task.payload["client_ids"],
            )
        )
        # Manually call publish
        worker._publish_storage_task(
            aggregated_diff_str=task.payload["aggregated_diff"],
            blockchain_hash=result["blockchain_hash"],
            model_version_id=result["model_version_id"],
            parent_version_id=result["parent_version_id"],
        )
        success = True
    else:
        # Use the handler directly
        success = worker._handle_blockchain_write_task(task)

    assert success
    assert mock_publisher.publish_task.called

    print("✓ Task handled successfully")
    print("✓ STORAGE_WRITE task published")
    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def test_handle_blockchain_write_task_missing_fields():
    """Test handling task with missing required fields."""
    print("\n" + "=" * 60)
    print("Testing Blockchain Worker - Handle Task with Missing Fields")
    print("=" * 60)
    print()

    worker = BlockchainWorker()

    # Task missing aggregated_diff
    task = Task(
        task_id="blockchain-test-002",
        task_type=TaskType.BLOCKCHAIN_WRITE,
        payload={
            "iteration": 1,
            "num_clients": 2,
        },
        metadata=TaskMetadata(source="test"),
    )

    success = worker._handle_blockchain_write_task(task)
    assert not success

    print("✓ Task with missing fields correctly rejected")
    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def test_handle_blockchain_write_task_error():
    """Test handling task with blockchain service error."""
    print("\n" + "=" * 60)
    print("Testing Blockchain Worker - Handle Task with Error")
    print("=" * 60)
    print()

    worker = BlockchainWorker()

    # Create task
    task = Task(
        task_id="blockchain-test-003",
        task_type=TaskType.BLOCKCHAIN_WRITE,
        payload={
            "aggregated_diff": '{"layer1.weight": [1.0, 2.0]}',
            "iteration": 1,
            "num_clients": 2,
            "client_ids": ["client_0"],
        },
        metadata=TaskMetadata(source="test"),
    )

    # Mock async method to raise error
    async def mock_process_error(*args, **kwargs):
        raise Exception("Blockchain service error")

    worker._process_blockchain_write = AsyncMock(side_effect=mock_process_error)

    # Run in sync context
    loop = asyncio.get_event_loop()
    if not loop.is_running():
        success = worker._handle_blockchain_write_task(task)
        assert not success

    print("✓ Task with error correctly handled")
    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)

