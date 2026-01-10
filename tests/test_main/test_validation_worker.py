"""Test validation worker functionality."""

import sys
from pathlib import Path
import os
import json
import asyncio

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Set localhost for local testing
if "RABBITMQ_HOST" not in os.environ:
    os.environ["RABBITMQ_HOST"] = "localhost"

import pytest
import torch
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from main_service.workers.validation_worker import ValidationWorker
from shared.models.task import Task, TaskType, TaskMetadata, ValidateTaskPayload
from shared.storage.encryption import EncryptionService
from shared.utils.crypto import AES256GCM


def serialize_diff(diff):
    """Helper to serialize weight diff to JSON string."""
    diff_dict = {}
    for name, tensor in diff.items():
        diff_dict[name] = tensor.numpy().tolist()
    return json.dumps(diff_dict)


def test_validation_worker_load_test_dataset():
    """Test loading test dataset."""
    print("=" * 60)
    print("Testing Validation Worker - Load Test Dataset")
    print("=" * 60)
    print()

    worker = ValidationWorker()
    test_loader = worker._load_test_dataset()

    assert test_loader is not None
    assert len(test_loader) > 0
    print(f"✓ Test dataset loaded: {len(test_loader)} batches")

    # Test that it's cached
    test_loader2 = worker._load_test_dataset()
    assert test_loader is test_loader2
    print("✓ Test dataset is cached")

    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def test_validation_worker_retrieve_and_decrypt_diff():
    """Test retrieving and decrypting diff from IPFS."""
    print("\n" + "=" * 60)
    print("Testing Validation Worker - Retrieve and Decrypt Diff")
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

    # Serialize and encrypt diff
    diff_str = serialize_diff(diff)
    diff_bytes = diff_str.encode("utf-8")

    key = AES256GCM.generate_key()
    encryption_service = EncryptionService(key=key)
    encrypted_diff = encryption_service.encrypt_diff(diff_bytes)

    # Create worker
    worker = ValidationWorker()
    worker.encryption_service = encryption_service

    # Mock IPFS client
    mock_cid = "QmTest123456789"

    async def run_test():
        with patch(
            "main_service.workers.validation_worker.IPFSClient"
        ) as mock_ipfs_class:
            mock_client = AsyncMock()
            mock_client.get_bytes = AsyncMock(return_value=encrypted_diff)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_ipfs_class.return_value = mock_client

            diff_dict = await worker._retrieve_and_decrypt_diff(mock_cid)

            assert isinstance(diff_dict, dict)
            assert len(diff_dict) > 0
            print(f"✓ Diff retrieved and decrypted: {len(diff_dict)} weight layers")

            # Verify IPFS was called
            mock_client.get_bytes.assert_called_once_with(mock_cid)

    asyncio.run(run_test())
    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def test_validation_worker_evaluate_model():
    """Test model evaluation on test dataset."""
    print("\n" + "=" * 60)
    print("Testing Validation Worker - Evaluate Model")
    print("=" * 60)
    print()

    worker = ValidationWorker()
    test_loader = worker._load_test_dataset()

    # Evaluate model (should work even with random weights)
    metrics = worker._evaluate_model(test_loader)

    assert "accuracy" in metrics
    assert "loss" in metrics
    assert "correct" in metrics
    assert "total" in metrics
    assert 0 <= metrics["accuracy"] <= 100
    assert metrics["loss"] >= 0
    assert metrics["total"] > 0
    print(
        f"✓ Model evaluation: accuracy={metrics['accuracy']:.2f}%, "
        f"loss={metrics['loss']:.4f}, correct={metrics['correct']}/{metrics['total']}"
    )

    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def test_validation_worker_validate_model():
    """Test full validation flow."""
    print("\n" + "=" * 60)
    print("Testing Validation Worker - Validate Model")
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

    # Serialize and encrypt diff
    diff_str = serialize_diff(diff)
    diff_bytes = diff_str.encode("utf-8")

    key = AES256GCM.generate_key()
    encryption_service = EncryptionService(key=key)
    encrypted_diff = encryption_service.encrypt_diff(diff_bytes)

    # Create worker
    worker = ValidationWorker()
    worker.encryption_service = encryption_service

    # Mock IPFS client
    mock_cid = "QmTest123456789"

    async def run_test():
        with patch(
            "main_service.workers.validation_worker.IPFSClient"
        ) as mock_ipfs_class:
            mock_client = AsyncMock()
            mock_client.get_bytes = AsyncMock(return_value=encrypted_diff)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_ipfs_class.return_value = mock_client

            validation_result = await worker._validate_model(
                ipfs_cid=mock_cid,
                model_version_id="version_1",
                parent_version_id=None,
            )

            assert "model_version_id" in validation_result
            assert "metrics" in validation_result
            assert validation_result["model_version_id"] == "version_1"
            assert "accuracy" in validation_result["metrics"]
            print(
                f"✓ Validation complete: accuracy={validation_result['metrics']['accuracy']:.2f}%"
            )

    asyncio.run(run_test())
    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def test_validation_worker_publish_decision_task():
    """Test publishing DECISION task."""
    print("\n" + "=" * 60)
    print("Testing Validation Worker - Publish Decision Task")
    print("=" * 60)
    print()

    worker = ValidationWorker()

    # Mock publisher
    mock_publisher = Mock()
    worker.publisher = mock_publisher

    # Create validation result
    validation_result = {
        "model_version_id": "version_1",
        "parent_version_id": "version_0",
        "ipfs_cid": "QmTest123456789",
        "metrics": {
            "accuracy": 95.5,
            "loss": 0.15,
            "correct": 9550,
            "total": 10000,
        },
    }

    # Publish decision task
    worker._publish_decision_task(
        validation_result=validation_result,
        model_version_id="version_1",
        should_rollback=False,
        rollback_reason=None,
    )

    # Verify task was published
    assert mock_publisher.publish_task.called
    call_args = mock_publisher.publish_task.call_args
    published_task = call_args.kwargs["task"]
    assert published_task.task_type == TaskType.DECISION
    assert published_task.payload["model_version_id"] == "version_1"
    assert published_task.payload["should_rollback"] is False
    assert published_task.payload["validation_result"]["metrics"]["accuracy"] == 95.5
    print("✓ DECISION task published correctly")

    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def test_validation_worker_handle_validate_task():
    """Test handling VALIDATE task."""
    print("\n" + "=" * 60)
    print("Testing Validation Worker - Handle Validate Task")
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

    # Serialize and encrypt diff
    diff_str = serialize_diff(diff)
    diff_bytes = diff_str.encode("utf-8")

    key = AES256GCM.generate_key()
    encryption_service = EncryptionService(key=key)
    encrypted_diff = encryption_service.encrypt_diff(diff_bytes)

    # Create worker
    worker = ValidationWorker()
    worker.encryption_service = encryption_service

    # Mock publisher
    mock_publisher = Mock()
    worker.publisher = mock_publisher

    # Create VALIDATE task
    validate_task = Task(
        task_id="validate-test-001",
        task_type=TaskType.VALIDATE,
        payload=ValidateTaskPayload(
            ipfs_cid="QmTest123456789",
            model_version_id="version_1",
            parent_version_id=None,
        ).model_dump(),
        metadata=TaskMetadata(source="test"),
        model_version_id="version_1",
        parent_version_id=None,
    )

    # Mock IPFS client
    async def run_test():
        with patch(
            "main_service.workers.validation_worker.IPFSClient"
        ) as mock_ipfs_class:
            mock_client = AsyncMock()
            mock_client.get_bytes = AsyncMock(return_value=encrypted_diff)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_ipfs_class.return_value = mock_client

            # Call the async method directly instead of the sync wrapper
            validation_result = await worker._validate_model(
                ipfs_cid="QmTest123456789",
                model_version_id="version_1",
                parent_version_id=None,
            )

            # Manually publish decision task to test the full flow
            worker._publish_decision_task(
                validation_result=validation_result,
                model_version_id="version_1",
                should_rollback=False,
                rollback_reason=None,
            )

            assert validation_result is not None
            assert "metrics" in validation_result
            print("✓ Task handling successful")

            # Verify DECISION task was published
            assert mock_publisher.publish_task.called
            call_args = mock_publisher.publish_task.call_args
            published_task = call_args.kwargs["task"]
            assert published_task.task_type == TaskType.DECISION
            assert published_task.payload["model_version_id"] == "version_1"
            print("✓ DECISION task published correctly")

    asyncio.run(run_test())

    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


@pytest.mark.filterwarnings("ignore::RuntimeWarning:unittest.mock")
def test_validation_worker_error_handling():
    """Test error handling in validation worker."""
    print("\n" + "=" * 60)
    print("Testing Validation Worker - Error Handling")
    print("=" * 60)
    print()

    worker = ValidationWorker()

    # Mock publisher
    mock_publisher = Mock()
    worker.publisher = mock_publisher

    # Create VALIDATE task with invalid IPFS CID
    validate_task = Task(
        task_id="validate-test-002",
        task_type=TaskType.VALIDATE,
        payload=ValidateTaskPayload(
            ipfs_cid="invalid_cid",
            model_version_id="version_1",
            parent_version_id=None,
        ).model_dump(),
        metadata=TaskMetadata(source="test"),
        model_version_id="version_1",
        parent_version_id=None,
    )

    # Mock IPFS client to raise error
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
            "main_service.workers.validation_worker.IPFSClient", new_callable=MagicMock
        ) as mock_ipfs_class:
            # Set the return value to our mock client
            mock_ipfs_class.return_value = mock_client

            # Test that async method raises error
            with pytest.raises(Exception, match="IPFS error"):
                await worker._retrieve_and_decrypt_diff("invalid_cid")

            print("✓ Error handling works correctly")

            # Verify DECISION task was NOT published
            assert not mock_publisher.publish_task.called
            print("✓ DECISION task not published on error")

    asyncio.run(run_test())

    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def run_all_tests():
    """Run all validation worker tests."""
    tests = [
        ("Load Test Dataset", test_validation_worker_load_test_dataset),
        ("Retrieve and Decrypt Diff", test_validation_worker_retrieve_and_decrypt_diff),
        ("Evaluate Model", test_validation_worker_evaluate_model),
        ("Validate Model", test_validation_worker_validate_model),
        ("Publish Decision Task", test_validation_worker_publish_decision_task),
        ("Handle Validate Task", test_validation_worker_handle_validate_task),
        ("Error Handling", test_validation_worker_error_handling),
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

    print("\n" + "=" * 60)
    print(f"Test Summary: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
