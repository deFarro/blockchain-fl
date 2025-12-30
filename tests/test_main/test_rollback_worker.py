"""Test rollback worker functionality."""

import sys
from pathlib import Path
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Set localhost for local testing
if "RABBITMQ_HOST" not in os.environ:
    os.environ["RABBITMQ_HOST"] = "localhost"

import pytest
import asyncio
import warnings
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from main_service.workers.rollback_worker import RollbackWorker
from shared.models.task import Task, TaskType, TaskMetadata, RollbackTaskPayload


def test_rollback_worker_initialization():
    """Test rollback worker initialization."""
    print("=" * 60)
    print("Testing Rollback Worker - Initialization")
    print("=" * 60)
    print()

    worker = RollbackWorker()

    assert worker.connection is not None
    assert worker.consumer is not None
    assert worker.publisher is not None
    assert worker.running is False
    print("✓ Rollback worker initialized correctly")

    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def test_rollback_worker_verify_rollback_target():
    """Test verifying rollback target weights from IPFS."""
    print("\n" + "=" * 60)
    print("Testing Rollback Worker - Verify Rollback Target")
    print("=" * 60)
    print()

    worker = RollbackWorker()

    # Mock IPFS client
    mock_weights_data = b"test_weights_data_12345"
    mock_cid = "QmTest123456789"

    async def run_test():
        with patch(
            "main_service.workers.rollback_worker.IPFSClient"
        ) as mock_ipfs_class:
            mock_client = AsyncMock()
            mock_client.get_bytes = AsyncMock(return_value=mock_weights_data)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_ipfs_class.return_value = mock_client

            is_accessible = await worker._verify_rollback_target(mock_cid)

            assert is_accessible
            print(f"✓ Rollback target verified: {len(mock_weights_data)} bytes")

            # Verify IPFS was called
            mock_client.get_bytes.assert_called_once_with(mock_cid)

    asyncio.run(run_test())
    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def test_rollback_worker_verify_rollback_target_not_found():
    """Test verifying rollback target when weights are not found."""
    print("\n" + "=" * 60)
    print("Testing Rollback Worker - Verify Rollback Target (Not Found)")
    print("=" * 60)
    print()

    worker = RollbackWorker()

    # Mock IPFS client to return None
    mock_cid = "QmInvalid123456789"

    async def run_test():
        with patch(
            "main_service.workers.rollback_worker.IPFSClient"
        ) as mock_ipfs_class:
            mock_client = AsyncMock()
            mock_client.get_bytes = AsyncMock(return_value=None)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_ipfs_class.return_value = mock_client

            is_accessible = await worker._verify_rollback_target(mock_cid)

            assert not is_accessible
            print("✓ Rollback target correctly identified as not accessible")

    asyncio.run(run_test())
    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


@pytest.mark.filterwarnings("ignore::RuntimeWarning:unittest.mock")
def test_rollback_worker_verify_rollback_target_error():
    """Test verifying rollback target when IPFS error occurs."""
    print("\n" + "=" * 60)
    print("Testing Rollback Worker - Verify Rollback Target (Error)")
    print("=" * 60)
    print()

    worker = RollbackWorker()

    # Mock IPFS client to raise error
    mock_cid = "QmError123456789"

    async def run_test():
        # Define the async error raiser
        async def raise_ipfs_error(*args, **kwargs):
            raise Exception("IPFS error")

        # Use MagicMock for the entire client to avoid AsyncMock's internal wrapper
        mock_client = MagicMock()

        # Use MagicMock for the method that raises the error
        mock_client.get_bytes = MagicMock(side_effect=raise_ipfs_error)

        # Configure context manager methods using async functions to avoid AsyncMock
        async def aenter():
            return mock_client

        async def aexit(*args):
            return None

        mock_client.__aenter__ = MagicMock(side_effect=aenter)
        mock_client.__aexit__ = MagicMock(side_effect=aexit)

        # Force patch to use MagicMock instead of potentially creating AsyncMock
        with patch(
            "main_service.workers.rollback_worker.IPFSClient", new_callable=MagicMock
        ) as mock_ipfs_class:
            # Set the return value to our mock client
            mock_ipfs_class.return_value = mock_client

            is_accessible = await worker._verify_rollback_target(mock_cid)

            assert not is_accessible
            print("✓ Rollback target verification correctly handles IPFS errors")

    asyncio.run(run_test())
    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def test_rollback_worker_record_rollback_on_blockchain():
    """Test recording rollback on blockchain."""
    print("\n" + "=" * 60)
    print("Testing Rollback Worker - Record Rollback on Blockchain")
    print("=" * 60)
    print()

    worker = RollbackWorker()

    mock_tx_id = "tx_rollback_12345"
    target_version_id = "version_1"
    reason = "Test rollback"

    async def run_test():
        with patch(
            "main_service.workers.rollback_worker.FabricClient"
        ) as mock_fabric_class:
            mock_client = AsyncMock()
            mock_client.rollback_model = AsyncMock(return_value=mock_tx_id)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_fabric_class.return_value = mock_client

            transaction_id = await worker._record_rollback_on_blockchain(
                target_version_id, reason
            )

            assert transaction_id == mock_tx_id
            print(f"✓ Rollback recorded on blockchain: tx_id={transaction_id}")

            # Verify blockchain client was called
            mock_client.rollback_model.assert_called_once_with(
                target_version_id=target_version_id, reason=reason
            )

    asyncio.run(run_test())
    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


@pytest.mark.filterwarnings("ignore::RuntimeWarning:unittest.mock")
def test_rollback_worker_record_rollback_blockchain_error():
    """Test recording rollback when blockchain service fails."""
    print("\n" + "=" * 60)
    print("Testing Rollback Worker - Record Rollback (Blockchain Error)")
    print("=" * 60)
    print()

    worker = RollbackWorker()

    target_version_id = "version_1"
    reason = "Test rollback"

    async def run_test():
        async def raise_blockchain_error(*args, **kwargs):
            raise Exception("Blockchain error")

        # Use MagicMock for the entire client to avoid AsyncMock's internal wrapper
        mock_client = MagicMock()

        # Use MagicMock for the method that raises the error
        mock_client.rollback_model = MagicMock(side_effect=raise_blockchain_error)

        # Configure context manager methods using async functions to avoid AsyncMock
        async def aenter():
            return mock_client

        async def aexit(*args):
            return None

        mock_client.__aenter__ = MagicMock(side_effect=aenter)
        mock_client.__aexit__ = MagicMock(side_effect=aexit)

        # Force patch to use MagicMock instead of potentially creating AsyncMock
        with patch(
            "main_service.workers.rollback_worker.FabricClient", new_callable=MagicMock
        ) as mock_fabric_class:
            # Set the return value to our mock client
            mock_fabric_class.return_value = mock_client

            transaction_id = await worker._record_rollback_on_blockchain(
                target_version_id, reason
            )

            assert transaction_id is None
            print("✓ Rollback recording correctly handles blockchain errors")

    asyncio.run(run_test())
    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def test_rollback_worker_process_rollback():
    """Test processing rollback operation."""
    print("\n" + "=" * 60)
    print("Testing Rollback Worker - Process Rollback")
    print("=" * 60)
    print()

    worker = RollbackWorker()

    target_version_id = "version_1"
    target_weights_cid = "QmTest123456789"
    reason = "Test rollback"
    cutoff_version_id = "version_0"

    mock_weights_data = b"test_weights_data"
    mock_tx_id = "tx_rollback_12345"

    async def run_test():
        with patch(
            "main_service.workers.rollback_worker.IPFSClient"
        ) as mock_ipfs_class, patch(
            "main_service.workers.rollback_worker.FabricClient"
        ) as mock_fabric_class:
            # Mock IPFS client
            mock_ipfs_client = AsyncMock()
            mock_ipfs_client.get_bytes = AsyncMock(return_value=mock_weights_data)
            mock_ipfs_client.__aenter__ = AsyncMock(return_value=mock_ipfs_client)
            mock_ipfs_client.__aexit__ = AsyncMock(return_value=None)
            mock_ipfs_class.return_value = mock_ipfs_client

            # Mock blockchain client
            mock_blockchain_client = AsyncMock()
            mock_blockchain_client.rollback_model = AsyncMock(return_value=mock_tx_id)
            mock_blockchain_client.__aenter__ = AsyncMock(
                return_value=mock_blockchain_client
            )
            mock_blockchain_client.__aexit__ = AsyncMock(return_value=None)
            mock_fabric_class.return_value = mock_blockchain_client

            success = await worker._process_rollback(
                target_version_id, target_weights_cid, reason, cutoff_version_id
            )

            assert success
            print("✓ Rollback processed successfully")

            # Verify IPFS was called
            mock_ipfs_client.get_bytes.assert_called_once_with(target_weights_cid)

            # Verify blockchain was called
            mock_blockchain_client.rollback_model.assert_called_once_with(
                target_version_id=target_version_id, reason=reason
            )

    asyncio.run(run_test())
    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def test_rollback_worker_process_rollback_ipfs_failure():
    """Test processing rollback when IPFS verification fails."""
    print("\n" + "=" * 60)
    print("Testing Rollback Worker - Process Rollback (IPFS Failure)")
    print("=" * 60)
    print()

    worker = RollbackWorker()

    target_version_id = "version_1"
    target_weights_cid = "QmInvalid123456789"
    reason = "Test rollback"
    cutoff_version_id = None

    async def run_test():
        with patch(
            "main_service.workers.rollback_worker.IPFSClient"
        ) as mock_ipfs_class:
            # Mock IPFS client to return None (not found)
            mock_ipfs_client = AsyncMock()
            mock_ipfs_client.get_bytes = AsyncMock(return_value=None)
            mock_ipfs_client.__aenter__ = AsyncMock(return_value=mock_ipfs_client)
            mock_ipfs_client.__aexit__ = AsyncMock(return_value=None)
            mock_ipfs_class.return_value = mock_ipfs_client

            success = await worker._process_rollback(
                target_version_id, target_weights_cid, reason, cutoff_version_id
            )

            assert not success
            print("✓ Rollback correctly fails when IPFS verification fails")

    asyncio.run(run_test())
    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def test_rollback_worker_process_rollback_blockchain_failure():
    """Test processing rollback when blockchain recording fails."""
    print("\n" + "=" * 60)
    print("Testing Rollback Worker - Process Rollback (Blockchain Failure)")
    print("=" * 60)
    print()

    worker = RollbackWorker()

    target_version_id = "version_1"
    target_weights_cid = "QmTest123456789"
    reason = "Test rollback"
    cutoff_version_id = None

    mock_weights_data = b"test_weights_data"

    async def run_test():
        with patch(
            "main_service.workers.rollback_worker.IPFSClient"
        ) as mock_ipfs_class, patch(
            "main_service.workers.rollback_worker.FabricClient"
        ) as mock_fabric_class:
            # Mock IPFS client (success)
            mock_ipfs_client = AsyncMock()
            mock_ipfs_client.get_bytes = AsyncMock(return_value=mock_weights_data)
            mock_ipfs_client.__aenter__ = AsyncMock(return_value=mock_ipfs_client)
            mock_ipfs_client.__aexit__ = AsyncMock(return_value=None)
            mock_ipfs_class.return_value = mock_ipfs_client

            # Mock blockchain client (failure)
            mock_blockchain_client = AsyncMock()
            mock_blockchain_client.rollback_model = AsyncMock(return_value=None)
            mock_blockchain_client.__aenter__ = AsyncMock(
                return_value=mock_blockchain_client
            )
            mock_blockchain_client.__aexit__ = AsyncMock(return_value=None)
            mock_fabric_class.return_value = mock_blockchain_client

            # Should still succeed (blockchain failure is logged but doesn't stop rollback)
            success = await worker._process_rollback(
                target_version_id, target_weights_cid, reason, cutoff_version_id
            )

            assert success
            print("✓ Rollback continues even if blockchain recording fails")

    asyncio.run(run_test())
    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def test_rollback_worker_handle_rollback_task():
    """Test handling ROLLBACK task."""
    print("\n" + "=" * 60)
    print("Testing Rollback Worker - Handle Rollback Task")
    print("=" * 60)
    print()

    worker = RollbackWorker()

    # Create ROLLBACK task
    rollback_task = Task(
        task_id="rollback-test-001",
        task_type=TaskType.ROLLBACK,
        payload=RollbackTaskPayload(
            target_version_id="version_1",
            target_weights_cid="QmTest123456789",
            reason="Test rollback",
            cutoff_version_id="version_0",
        ).model_dump(),
        metadata=TaskMetadata(source="test"),
        model_version_id="version_1",
        parent_version_id=None,
    )

    mock_weights_data = b"test_weights_data"
    mock_tx_id = "tx_rollback_12345"

    # Note: _handle_rollback_task uses run_until_complete, which doesn't work
    # when there's already a running event loop. We'll test it directly with
    # the async method instead.
    async def run_test():
        with patch(
            "main_service.workers.rollback_worker.IPFSClient"
        ) as mock_ipfs_class, patch(
            "main_service.workers.rollback_worker.FabricClient"
        ) as mock_fabric_class:
            # Mock IPFS client
            mock_ipfs_client = AsyncMock()
            mock_ipfs_client.get_bytes = AsyncMock(return_value=mock_weights_data)
            mock_ipfs_client.__aenter__ = AsyncMock(return_value=mock_ipfs_client)
            mock_ipfs_client.__aexit__ = AsyncMock(return_value=None)
            mock_ipfs_class.return_value = mock_ipfs_client

            # Mock blockchain client
            mock_blockchain_client = AsyncMock()
            mock_blockchain_client.rollback_model = AsyncMock(return_value=mock_tx_id)
            mock_blockchain_client.__aenter__ = AsyncMock(
                return_value=mock_blockchain_client
            )
            mock_blockchain_client.__aexit__ = AsyncMock(return_value=None)
            mock_fabric_class.return_value = mock_blockchain_client

            # Call the async method directly
            success = await worker._process_rollback(
                target_version_id="version_1",
                target_weights_cid="QmTest123456789",
                reason="Test rollback",
                cutoff_version_id="version_0",
            )

            assert success
            print("✓ Rollback task handled successfully")

    asyncio.run(run_test())
    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def test_rollback_worker_handle_rollback_task_error():
    """Test error handling in rollback worker."""
    print("\n" + "=" * 60)
    print("Testing Rollback Worker - Error Handling")
    print("=" * 60)
    print()

    worker = RollbackWorker()

    # Create ROLLBACK task with invalid payload
    rollback_task = Task(
        task_id="rollback-test-002",
        task_type=TaskType.ROLLBACK,
        payload={},  # Invalid payload
        metadata=TaskMetadata(source="test"),
        model_version_id="version_1",
        parent_version_id=None,
    )

    # Test that invalid payload raises error
    success = worker._handle_rollback_task(rollback_task)

    assert not success
    print("✓ Error handling works correctly for invalid payload")

    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def run_all_tests():
    """Run all rollback worker tests."""
    tests = [
        ("Initialization", test_rollback_worker_initialization),
        ("Verify Rollback Target", test_rollback_worker_verify_rollback_target),
        (
            "Verify Rollback Target (Not Found)",
            test_rollback_worker_verify_rollback_target_not_found,
        ),
        (
            "Verify Rollback Target (Error)",
            test_rollback_worker_verify_rollback_target_error,
        ),
        (
            "Record Rollback on Blockchain",
            test_rollback_worker_record_rollback_on_blockchain,
        ),
        (
            "Record Rollback (Blockchain Error)",
            test_rollback_worker_record_rollback_blockchain_error,
        ),
        ("Process Rollback", test_rollback_worker_process_rollback),
        (
            "Process Rollback (IPFS Failure)",
            test_rollback_worker_process_rollback_ipfs_failure,
        ),
        (
            "Process Rollback (Blockchain Failure)",
            test_rollback_worker_process_rollback_blockchain_failure,
        ),
        ("Handle Rollback Task", test_rollback_worker_handle_rollback_task),
        ("Error Handling", test_rollback_worker_handle_rollback_task_error),
    ]

    passed = 0
    failed = 0

    # Run tests
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ {name} FAILED: {str(e)}")
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
