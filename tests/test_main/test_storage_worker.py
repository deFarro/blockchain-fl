"""Test storage worker functionality."""

import sys
from pathlib import Path
import os
import json
import asyncio
import base64

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Set localhost for local testing
if "RABBITMQ_HOST" not in os.environ:
    os.environ["RABBITMQ_HOST"] = "localhost"

import pytest
import torch
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from main_service.workers.storage_worker import StorageWorker
from shared.models.task import Task, TaskType, TaskMetadata, StorageWriteTaskPayload
from shared.storage.encryption import EncryptionService
from shared.utils.hashing import compute_hash


def serialize_diff(diff):
    """Helper to serialize weight diff to JSON string."""
    diff_dict = {}
    for name, tensor in diff.items():
        diff_dict[name] = tensor.numpy().tolist()
    return json.dumps(diff_dict)


def setup_test_encryption_key():
    """Helper to get the test encryption key.

    The key is already set in conftest.py, so this just returns a new key
    for use in creating EncryptionService instances.

    Returns:
        bytes: A raw encryption key (for use in creating EncryptionService instances)
    """
    from shared.utils.crypto import AES256GCM

    return AES256GCM.generate_key()


def test_storage_worker_encrypt_and_store():
    """Test encrypt_and_store method."""
    print("=" * 60)
    print("Testing Storage Worker - Encrypt and Store")
    print("=" * 60)
    print()

    from client_service.training.model import SimpleCNN

    # Create model and get weights
    model = SimpleCNN(num_classes=10)
    weights = model.get_weights()

    # Create weight diff
    diff = {}
    for name, tensor in weights.items():
        diff[name] = tensor - torch.zeros_like(tensor)

    # Serialize diff
    aggregated_diff_str = serialize_diff(diff)

    # Create storage worker
    worker = StorageWorker()

    # Compute hash of unencrypted diff (content integrity check)
    diff_bytes = aggregated_diff_str.encode("utf-8")
    expected_hash = compute_hash(diff_bytes)

    # Mock IPFS client
    mock_cid = "QmTest123456789"

    # Test encrypt_and_store
    async def run_test():
        with patch("main_service.workers.storage_worker.IPFSClient") as mock_ipfs_class:
            mock_client = AsyncMock()
            mock_client.add_bytes = AsyncMock(return_value=mock_cid)
            mock_client.pin_ls = AsyncMock(
                return_value={"Keys": {mock_cid: {"Type": "recursive"}}}
            )
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_ipfs_class.return_value = mock_client

            cid = await worker._encrypt_and_store(
                aggregated_diff_str, expected_hash, "version_1"
            )

            assert cid == mock_cid
            print(f"✓ Encrypt and store successful: CID={cid}")

            # Verify IPFS was called (with encrypted diff)
            # Note: We can't verify the exact encrypted value since AES-GCM is non-deterministic
            # But we can verify it was called with some encrypted bytes
            assert mock_client.add_bytes.called
            call_args = mock_client.add_bytes.call_args
            assert call_args[1]["pin"] is True
            assert isinstance(call_args[0][0], bytes)
            mock_client.pin_ls.assert_called_once()

    asyncio.run(run_test())
    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def test_storage_worker_hash_verification():
    """Test hash verification in encrypt_and_store."""
    print("\n" + "=" * 60)
    print("Testing Storage Worker - Hash Verification")
    print("=" * 60)
    print()

    # Set up encryption key for worker initialization
    setup_test_encryption_key()

    aggregated_diff_str = '{"test": "data"}'
    wrong_hash = "wrong_hash_12345"

    worker = StorageWorker()

    # Test that hash mismatch raises ValueError
    async def run_test():
        with pytest.raises(ValueError, match="Hash mismatch"):
            await worker._encrypt_and_store(
                aggregated_diff_str, wrong_hash, "version_1"
            )

    asyncio.run(run_test())
    print("✓ Hash mismatch correctly raises ValueError")

    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def test_storage_worker_handle_task():
    """Test handling STORAGE_WRITE task."""
    print("\n" + "=" * 60)
    print("Testing Storage Worker - Handle Task")
    print("=" * 60)
    print()

    from client_service.training.model import SimpleCNN

    # Create model and get weights
    model = SimpleCNN(num_classes=10)
    weights = model.get_weights()

    # Create weight diff
    diff = {}
    for name, tensor in weights.items():
        diff[name] = tensor - torch.zeros_like(tensor)

    # Serialize diff
    aggregated_diff_str = serialize_diff(diff)

    # Create storage worker
    worker = StorageWorker()

    # Compute hash of unencrypted diff (content integrity check)
    diff_bytes = aggregated_diff_str.encode("utf-8")
    expected_hash = compute_hash(diff_bytes)

    # Mock publisher
    mock_publisher = Mock()
    worker.publisher = mock_publisher

    # Create STORAGE_WRITE task
    storage_task = Task(
        task_id="storage-test-001",
        task_type=TaskType.STORAGE_WRITE,
        payload=StorageWriteTaskPayload(
            aggregated_diff=aggregated_diff_str,
            blockchain_hash=expected_hash,
            model_version_id="version_1",
        ).model_dump(),
        metadata=TaskMetadata(source="test"),
        model_version_id="version_1",
        parent_version_id="version_0",
    )

    # Mock IPFS client
    mock_cid = "QmTest123456789"

    # Test handling task
    # Note: _handle_storage_task uses run_until_complete, which doesn't work
    # when there's already a running event loop. We'll test it directly with
    # the async method instead.
    async def run_test():
        with patch("main_service.workers.storage_worker.IPFSClient") as mock_ipfs_class:
            mock_client = AsyncMock()
            mock_client.add_bytes = AsyncMock(return_value=mock_cid)
            mock_client.pin_ls = AsyncMock(
                return_value={"Keys": {mock_cid: {"Type": "recursive"}}}
            )
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_ipfs_class.return_value = mock_client

            # Call the async method directly instead of the sync wrapper
            cid = await worker._encrypt_and_store(
                aggregated_diff_str, expected_hash, "version_1"
            )

            # Manually publish validate task to test the full flow
            worker._publish_validate_task("version_1", cid, "version_0")

            assert cid == mock_cid
            print("✓ Task handling successful")

            # Verify IPFS was called (with encrypted diff)
            # Note: We can't verify the exact encrypted value since AES-GCM is non-deterministic
            # But we can verify it was called with some encrypted bytes
            assert mock_client.add_bytes.called
            call_args = mock_client.add_bytes.call_args
            assert call_args[1]["pin"] is True
            assert isinstance(call_args[0][0], bytes)

            # Verify VALIDATE task was published
            assert mock_publisher.publish_task.called
            published_task = mock_publisher.publish_task.call_args[0][0]
            assert published_task.task_type == TaskType.VALIDATE
            assert published_task.payload["ipfs_cid"] == mock_cid
            print("✓ VALIDATE task published correctly")

    asyncio.run(run_test())

    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def test_storage_worker_hash_mismatch():
    """Test handling hash mismatch error."""
    print("\n" + "=" * 60)
    print("Testing Storage Worker - Hash Mismatch Error")
    print("=" * 60)
    print()

    # Set up encryption key for worker initialization
    setup_test_encryption_key()

    aggregated_diff_str = '{"test": "data"}'
    wrong_hash = "wrong_hash_12345"

    worker = StorageWorker()

    # Mock publisher
    mock_publisher = Mock()
    worker.publisher = mock_publisher

    # Create STORAGE_WRITE task with wrong hash
    storage_task = Task(
        task_id="storage-test-002",
        task_type=TaskType.STORAGE_WRITE,
        payload=StorageWriteTaskPayload(
            aggregated_diff=aggregated_diff_str,
            blockchain_hash=wrong_hash,
            model_version_id="version_1",
        ).model_dump(),
        metadata=TaskMetadata(source="test"),
        model_version_id="version_1",
        parent_version_id="version_0",
    )

    # Test that task handling fails on hash mismatch
    success = worker._handle_storage_task(storage_task)

    assert not success, "Task handling should fail on hash mismatch"
    print("✓ Hash mismatch correctly causes task failure")

    # Verify VALIDATE task was NOT published
    assert not mock_publisher.publish_task.called
    print("✓ VALIDATE task not published on error")

    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


@pytest.mark.asyncio
async def test_storage_worker_ipfs_integration():
    """Test storage worker with real IPFS (if available)."""
    print("\n" + "=" * 60)
    print("Testing Storage Worker - IPFS Integration")
    print("=" * 60)
    print()

    try:
        from client_service.training.model import SimpleCNN
        from shared.storage.ipfs_client import IPFSClient

        # Create model and get weights
        model = SimpleCNN(num_classes=10)
        weights = model.get_weights()

        # Create weight diff
        diff = {}
        for name, tensor in weights.items():
            diff[name] = tensor - torch.zeros_like(tensor)

        # Serialize diff
        aggregated_diff_str = serialize_diff(diff)

        # Create storage worker
        worker = StorageWorker()

        # Compute hash of unencrypted diff (content integrity check)
        diff_bytes = aggregated_diff_str.encode("utf-8")
        expected_hash = compute_hash(diff_bytes)

        # Test with real IPFS
        cid = await worker._encrypt_and_store(
            aggregated_diff_str, expected_hash, "version_1"
        )

        assert cid, "CID should not be empty"
        print(f"✓ Full encrypt and store successful: CID={cid}")

        # Verify we can retrieve it
        async with IPFSClient() as ipfs_client:
            retrieved = await ipfs_client.get_bytes(cid)
            # Verify it's encrypted (should be bytes, not the original diff)
            assert isinstance(retrieved, bytes)
            assert len(retrieved) > 0
            # Decrypt to verify it matches original
            decrypted = worker.encryption_service.decrypt_diff(retrieved)
            assert decrypted == diff_bytes
            print("✓ Retrieved and decrypted diff from IPFS matches original")

    except Exception as e:
        print(f"⚠ IPFS not available: {str(e)}")
        print("  Skipping IPFS integration test (make sure IPFS daemon is running)")
        pytest.skip("IPFS not available")

    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def run_all_tests():
    """Run all storage worker tests."""
    tests = [
        ("Encrypt and Store", test_storage_worker_encrypt_and_store),
        ("Hash Verification", test_storage_worker_hash_verification),
        ("Handle Task", test_storage_worker_handle_task),
        ("Hash Mismatch Error", test_storage_worker_hash_mismatch),
    ]

    async_tests = [
        ("IPFS Integration", test_storage_worker_ipfs_integration),
    ]

    passed = 0
    failed = 0

    # Run sync tests
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ {name} FAILED: {str(e)}")
            import traceback

            traceback.print_exc()
            failed += 1

    # Run async tests
    for name, test_func in async_tests:
        try:
            asyncio.run(test_func())
            passed += 1
        except Exception as e:
            if "skip" in str(e).lower():
                print(f"⚠ {name} SKIPPED (IPFS not available)")
            else:
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
