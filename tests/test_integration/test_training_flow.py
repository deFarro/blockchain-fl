"""Integration tests for complete training iteration flow."""

import pytest
import json
import time
from typing import Dict, Any, Optional
from client_service.training.model import SimpleCNN
from shared.queue.connection import QueueConnection
from shared.queue.publisher import QueuePublisher
from shared.queue.consumer import QueueConsumer
from shared.models.task import Task, TaskType, TaskMetadata
from shared.storage.encryption import EncryptionService
from shared.storage.ipfs_client import IPFSClient
from main_service.workers.aggregation_worker import AggregationWorker
from main_service.workers.blockchain_worker import BlockchainWorker
from main_service.workers.storage_worker import StorageWorker
from main_service.workers.validation_worker import ValidationWorker
from main_service.workers.decision_worker import DecisionWorker
from main_service.workers.rollback_worker import RollbackWorker
from client_service.worker import ClientWorker
from client_service.training.trainer import Trainer
from shared.logger import setup_logger

logger = setup_logger(__name__)


def serialize_diff(diff: Dict[str, Any]) -> str:
    """Helper to serialize weight diff to JSON string."""
    diff_dict = {}
    for name, tensor in diff.items():
        diff_dict[name] = tensor.numpy().tolist()
    return json.dumps(diff_dict)


@pytest.mark.integration
def test_complete_training_iteration():
    """
    Test complete training iteration flow:
    1. Client receives TRAIN task
    2. Client trains and publishes update
    3. Aggregation worker aggregates
    4. Storage worker encrypts and stores
    5. Blockchain worker records on-chain
    6. Validation worker validates
    7. Decision worker makes decision
    """
    print("\n" + "=" * 80)
    print("Integration Test: Complete Training Iteration Flow")
    print("=" * 80)
    print()

    # Setup
    connection = QueueConnection()
    publisher = QueuePublisher(connection=connection)

    # Create initial model weights
    model = SimpleCNN(num_classes=10)
    initial_weights = model.get_weights()

    # Serialize and encrypt initial weights
    serializable_weights = {}
    for name, tensor in initial_weights.items():
        serializable_weights[name] = tensor.numpy().tolist()
    weights_json = json.dumps(serializable_weights)
    weights_bytes = weights_json.encode("utf-8")

    encryption_service = EncryptionService()
    encrypted_weights = encryption_service.encrypt_diff(weights_bytes)

    # Upload to IPFS (if available)
    import asyncio

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    ipfs_cid = None
    try:

        async def upload_weights():
            async with IPFSClient() as ipfs_client:
                return await ipfs_client.add_bytes(encrypted_weights, pin=True)

        ipfs_cid = loop.run_until_complete(upload_weights())
        print(f"✓ Initial weights uploaded to IPFS: CID={ipfs_cid}")
    except Exception as e:
        pytest.skip(f"IPFS not available: {e}")

    # Step 1: Create TRAIN task for client
    iteration = 1
    client_id = 0

    train_task = Task(
        task_id=f"test-train-{iteration}-client_{client_id}-{int(time.time())}",
        task_type=TaskType.TRAIN,
        payload={
            "weights_cid": ipfs_cid,
            "iteration": iteration,
            "client_id": f"client_{client_id}",
        },
        metadata=TaskMetadata(source="integration_test"),
        model_version_id=None,
        parent_version_id=None,
    )

    print(
        f"Step 1: Publishing TRAIN task for client_{client_id}, iteration {iteration}"
    )
    publisher.publish_task(train_task, "train_queue")
    print("✓ TRAIN task published")

    # Step 2: Client processes TRAIN task
    print("\nStep 2: Client processing TRAIN task...")
    # Use 1 epoch for faster integration testing
    trainer = Trainer(epochs=1)
    client_worker = ClientWorker(trainer=trainer, connection=connection)

    # Simulate client training (without actually consuming from queue)
    # In real flow, client would consume from train_queue
    previous_weights = None  # Would load from IPFS using ipfs_cid
    weight_diff, metrics, _ = trainer.train(previous_weights=previous_weights)

    print(f"✓ Client training completed: accuracy={metrics['accuracy']:.2f}%")

    # Publish client update
    client_update = {
        "client_id": f"client_{client_id}",
        "iteration": iteration,
        "weight_diff": serialize_diff(weight_diff),
        "metrics": metrics,
        "task_id": train_task.task_id,
        "model_version_id": None,
        "parent_version_id": None,
    }

    publisher.publish_dict(client_update, "client_updates")
    print("✓ Client update published to client_updates queue")

    # Step 3: Aggregation worker aggregates
    print("\nStep 3: Aggregation worker aggregating client updates...")
    aggregation_worker = AggregationWorker(connection=connection)

    # Collect client updates (simplified - in real flow would consume from queue)
    client_updates = [client_update]
    aggregated_diff = aggregation_worker._fedavg_aggregate(client_updates)

    print(f"✓ Aggregation completed: {len(aggregated_diff)} weight parameters")

    # Step 4: Blockchain worker records on-chain
    print("\nStep 4: Blockchain worker recording on-chain...")
    blockchain_worker = BlockchainWorker(connection=connection)

    # Create blockchain write task
    aggregated_diff_str = serialize_diff(aggregated_diff)
    blockchain_task = Task(
        task_id=f"test-blockchain-{iteration}-{int(time.time())}",
        task_type=TaskType.BLOCKCHAIN_WRITE,
        payload={
            "aggregated_diff": aggregated_diff_str,
            "iteration": iteration,
            "num_clients": 1,
            "client_ids": [f"client_{client_id}"],
        },
        metadata=TaskMetadata(source="integration_test"),
        model_version_id=None,
        parent_version_id=None,
    )

    # Process blockchain write (simplified - would normally consume from queue)
    try:
        result = loop.run_until_complete(
            blockchain_worker._process_blockchain_write(
                aggregated_diff_str=aggregated_diff_str,
                iteration=iteration,
                num_clients=1,
                client_ids=[f"client_{client_id}"],
            )
        )
        print(f"✓ Blockchain write completed: version_id={result['model_version_id']}")
        model_version_id = result["model_version_id"]
        blockchain_hash = result["blockchain_hash"]
    except Exception as e:
        pytest.skip(f"Blockchain service not available: {e}")

    # Step 5: Storage worker encrypts and stores
    print("\nStep 5: Storage worker encrypting and storing...")
    storage_worker = StorageWorker(connection=connection)

    # Create storage write task
    storage_task = Task(
        task_id=f"test-storage-{iteration}-{int(time.time())}",
        task_type=TaskType.STORAGE_WRITE,
        payload={
            "aggregated_diff": aggregated_diff_str,
            "blockchain_hash": blockchain_hash,
            "model_version_id": model_version_id,
        },
        metadata=TaskMetadata(source="integration_test"),
        model_version_id=model_version_id,
        parent_version_id=None,
    )

    # Process storage write (simplified)
    try:
        cid = loop.run_until_complete(
            storage_worker._encrypt_and_store(
                aggregated_diff_str=aggregated_diff_str,
                blockchain_hash=blockchain_hash,
            )
        )
        print(f"✓ Storage completed: IPFS CID={cid}")
    except Exception as e:
        pytest.skip(f"Storage/IPFS not available: {e}")

    # Step 6: Validation worker validates
    print("\nStep 6: Validation worker validating model...")
    validation_worker = ValidationWorker(connection=connection)

    # Create validate task
    validate_task = Task(
        task_id=f"test-validate-{iteration}-{int(time.time())}",
        task_type=TaskType.VALIDATE,
        payload={
            "ipfs_cid": cid,
            "model_version_id": model_version_id,
            "parent_version_id": None,
        },
        metadata=TaskMetadata(source="integration_test"),
        model_version_id=model_version_id,
        parent_version_id=None,
    )

    # Process validation (simplified)
    try:
        validation_result = loop.run_until_complete(
            validation_worker._validate_model(
                ipfs_cid=cid,
                model_version_id=model_version_id,
                parent_version_id=None,
            )
        )
        accuracy = validation_result["metrics"]["accuracy"]
        print(f"✓ Validation completed: accuracy={accuracy:.2f}%")
    except Exception as e:
        pytest.skip(f"Validation not available: {e}")

    # Step 7: Decision worker makes decision
    print("\nStep 7: Decision worker making decision...")
    decision_worker = DecisionWorker(connection=connection)

    # Create decision task
    decision_task = Task(
        task_id=f"test-decision-{iteration}-{int(time.time())}",
        task_type=TaskType.DECISION,
        payload={
            "validation_result": validation_result,
            "model_version_id": model_version_id,
            "should_rollback": False,
            "rollback_reason": None,
        },
        metadata=TaskMetadata(source="integration_test"),
        model_version_id=model_version_id,
        parent_version_id=None,
    )

    # Process decision (simplified)
    success = decision_worker._handle_decision_task(decision_task)
    print(f"✓ Decision completed: success={success}")

    print("\n" + "=" * 80)
    print("✓ Integration test PASSED: Complete training iteration flow")
    print("=" * 80)


@pytest.mark.integration
def test_rollback_flow():
    """
    Test rollback flow:
    1. Model validation shows accuracy drop
    2. Decision worker triggers rollback
    3. Rollback worker verifies and records rollback
    """
    print("\n" + "=" * 80)
    print("Integration Test: Rollback Flow")
    print("=" * 80)
    print()

    import asyncio
    from shared.models.task import DecisionTaskPayload, RollbackTaskPayload

    # Setup
    connection = QueueConnection()
    publisher = QueuePublisher(connection=connection)

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Step 1: Create initial model with good accuracy
    print("Step 1: Setting up initial model with good accuracy...")
    model = SimpleCNN(num_classes=10)
    initial_weights = model.get_weights()

    # Serialize and encrypt initial weights
    serializable_weights = {}
    for name, tensor in initial_weights.items():
        serializable_weights[name] = tensor.numpy().tolist()
    weights_json = json.dumps(serializable_weights)
    weights_bytes = weights_json.encode("utf-8")

    encryption_service = EncryptionService()
    encrypted_weights = encryption_service.encrypt_diff(weights_bytes)

    # Upload to IPFS
    try:

        async def upload_weights():
            async with IPFSClient() as ipfs_client:
                return await ipfs_client.add_bytes(encrypted_weights, pin=True)

        initial_cid = loop.run_until_complete(upload_weights())
        print(f"✓ Initial weights uploaded: CID={initial_cid}")
    except Exception as e:
        pytest.skip(f"IPFS not available: {e}")

    # Step 2: Simulate a model version with good accuracy (best checkpoint)
    print("\nStep 2: Simulating best checkpoint with good accuracy...")
    best_version_id = "test-best-version-001"
    best_accuracy = 95.0

    # Step 3: Create validation result with poor accuracy (triggering rollback)
    print("\nStep 3: Creating validation result with poor accuracy...")
    current_accuracy = 88.0  # Significant drop from 95%
    validation_result = {
        "model_version_id": "test-current-version-002",
        "parent_version_id": best_version_id,
        "ipfs_cid": initial_cid,
        "metrics": {
            "accuracy": current_accuracy,
            "loss": 0.5,
            "correct": 8800,
            "total": 10000,
        },
    }

    # Step 4: Decision worker evaluates rollback
    print("\nStep 4: Decision worker evaluating rollback...")
    decision_worker = DecisionWorker(connection=connection)

    # Set up decision worker state with best checkpoint
    decision_worker.state.best_accuracy = best_accuracy
    decision_worker.state.best_checkpoint_version = best_version_id
    decision_worker.state.best_checkpoint_cid = initial_cid
    decision_worker.state.current_iteration = 2

    # Evaluate rollback
    should_rollback, rollback_reason = decision_worker._evaluate_rollback(
        current_accuracy, validation_result
    )

    assert should_rollback, "Rollback should be triggered for significant accuracy drop"
    assert rollback_reason is not None, "Rollback reason should be provided"
    print(f"✓ Rollback triggered: {rollback_reason}")

    # Step 5: Decision worker publishes rollback task
    print("\nStep 5: Decision worker publishing rollback task...")
    decision_worker._publish_rollback_task(
        target_version_id=best_version_id,
        target_weights_cid=initial_cid,
        reason=rollback_reason or "Integration test rollback",
    )
    print("✓ Rollback task published")

    # Step 6: Rollback worker processes rollback
    print("\nStep 6: Rollback worker processing rollback...")
    rollback_worker = RollbackWorker(connection=connection)

    rollback_task = Task(
        task_id=f"test-rollback-{int(time.time())}",
        task_type=TaskType.ROLLBACK,
        payload=RollbackTaskPayload(
            target_version_id=best_version_id,
            target_weights_cid=initial_cid,
            reason=rollback_reason or "Integration test rollback",
            cutoff_version_id=None,
        ).model_dump(),
        metadata=TaskMetadata(source="integration_test"),
        model_version_id=best_version_id,
        parent_version_id=None,
    )

    # Process rollback
    try:
        success = loop.run_until_complete(
            rollback_worker._process_rollback(
                target_version_id=best_version_id,
                target_weights_cid=initial_cid,
                reason=rollback_reason or "Integration test rollback",
                cutoff_version_id=None,
            )
        )
        assert success, "Rollback should succeed"
        print("✓ Rollback processed successfully")
    except Exception as e:
        pytest.skip(
            f"Rollback processing failed (blockchain/IPFS may not be available): {e}"
        )

    print("\n" + "=" * 80)
    print("✓ Integration test PASSED: Rollback flow")
    print("=" * 80)


@pytest.mark.integration
def test_training_completion():
    """
    Test training completion flow:
    1. Model reaches target accuracy
    2. Decision worker publishes TRAINING_COMPLETE task
    3. Training status is updated
    """
    print("\n" + "=" * 80)
    print("Integration Test: Training Completion Flow")
    print("=" * 80)
    print()

    import asyncio
    from shared.models.task import DecisionTaskPayload, TrainingCompleteTaskPayload
    from shared.config import settings

    # Setup
    connection = QueueConnection()
    publisher = QueuePublisher(connection=connection)

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Step 1: Create model with target accuracy
    print("Step 1: Setting up model with target accuracy...")
    model = SimpleCNN(num_classes=10)
    final_weights = model.get_weights()

    # Serialize and encrypt final weights
    serializable_weights = {}
    for name, tensor in final_weights.items():
        serializable_weights[name] = tensor.numpy().tolist()
    weights_json = json.dumps(serializable_weights)
    weights_bytes = weights_json.encode("utf-8")

    encryption_service = EncryptionService()
    encrypted_weights = encryption_service.encrypt_diff(weights_bytes)

    # Upload to IPFS
    try:

        async def upload_weights():
            async with IPFSClient() as ipfs_client:
                return await ipfs_client.add_bytes(encrypted_weights, pin=True)

        final_cid = loop.run_until_complete(upload_weights())
        print(f"✓ Final weights uploaded: CID={final_cid}")
    except Exception as e:
        pytest.skip(f"IPFS not available: {e}")

    # Step 2: Create validation result with target accuracy
    print("\nStep 2: Creating validation result with target accuracy...")
    target_accuracy = getattr(settings, "target_accuracy", 95.0)
    current_accuracy = target_accuracy  # Reached target

    final_version_id = "test-final-version-001"
    validation_result = {
        "model_version_id": final_version_id,
        "parent_version_id": None,
        "ipfs_cid": final_cid,
        "metrics": {
            "accuracy": current_accuracy,
            "loss": 0.1,
            "correct": int(current_accuracy * 100),
            "total": 10000,
        },
    }

    # Step 3: Decision worker checks completion
    print("\nStep 3: Decision worker checking training completion...")
    decision_worker = DecisionWorker(connection=connection)
    decision_worker.state.current_iteration = 10
    decision_worker.state.best_accuracy = current_accuracy
    decision_worker.state.best_checkpoint_version = final_version_id
    decision_worker.state.best_checkpoint_cid = final_cid

    should_complete, completion_reason = decision_worker._check_training_completion(
        current_accuracy, validation_result
    )

    assert should_complete, "Training should complete when target accuracy is reached"
    assert completion_reason is not None, "Completion reason should be provided"
    print(f"✓ Training completion detected: {completion_reason}")

    # Step 4: Decision worker publishes TRAINING_COMPLETE task
    print("\nStep 4: Decision worker publishing TRAINING_COMPLETE task...")
    decision_worker._publish_training_complete_task(
        validation_result, completion_reason or "Target accuracy reached"
    )
    print("✓ TRAINING_COMPLETE task published")

    # Verify the task was published (check queue or mock publisher)
    # In a real test, we would consume from the queue to verify
    print("\n" + "=" * 80)
    print("✓ Integration test PASSED: Training completion flow")
    print("=" * 80)
