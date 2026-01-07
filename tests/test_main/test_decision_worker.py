"""Test decision worker functionality."""

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
from unittest.mock import Mock, patch
from main_service.workers.decision_worker import DecisionWorker, ModelState
from shared.models.task import Task, TaskType, TaskMetadata, DecisionTaskPayload


def test_decision_worker_initialization():
    """Test decision worker initialization."""
    print("=" * 60)
    print("Testing Decision Worker - Initialization")
    print("=" * 60)
    print()

    worker = DecisionWorker()

    assert worker.state.current_iteration == 0
    assert worker.state.best_accuracy == 0.0
    assert worker.state.patience_counter == 0
    assert worker.state.rollback_count == 0
    assert worker.accuracy_tolerance == 0.5
    assert worker.patience_threshold == 3
    assert worker.severe_drop_threshold == 2.0
    print("✓ Decision worker initialized correctly")

    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def test_decision_worker_evaluate_rollback_first_iteration():
    """Test rollback evaluation on first iteration."""
    print("\n" + "=" * 60)
    print("Testing Decision Worker - Evaluate Rollback (First Iteration)")
    print("=" * 60)
    print()

    worker = DecisionWorker()

    validation_result = {
        "model_version_id": "version_1",
        "ipfs_cid": "QmTest123456789",
        "metrics": {"accuracy": 90.0},
    }

    should_rollback, reason = worker._evaluate_rollback(90.0, validation_result)

    assert not should_rollback
    assert reason is None
    assert worker.state.best_accuracy == 90.0
    assert worker.state.best_checkpoint_version == "version_1"
    assert worker.state.best_checkpoint_cid == "QmTest123456789"
    print("✓ First iteration correctly initializes best accuracy")

    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def test_decision_worker_evaluate_rollback_new_best():
    """Test rollback evaluation when accuracy improves."""
    print("\n" + "=" * 60)
    print("Testing Decision Worker - Evaluate Rollback (New Best)")
    print("=" * 60)
    print()

    worker = DecisionWorker()

    # Set initial best
    worker.state.best_accuracy = 90.0
    worker.state.best_checkpoint_version = "version_1"
    worker.state.best_checkpoint_cid = "QmTest111"

    validation_result = {
        "model_version_id": "version_2",
        "ipfs_cid": "QmTest222",
        "metrics": {"accuracy": 92.0},
    }

    should_rollback, reason = worker._evaluate_rollback(92.0, validation_result)

    assert not should_rollback
    assert reason is None
    assert worker.state.best_accuracy == 92.0
    assert worker.state.best_checkpoint_version == "version_2"
    assert worker.state.best_checkpoint_cid == "QmTest222"
    assert worker.state.patience_counter == 0
    print("✓ New best accuracy correctly updates checkpoint and resets patience")

    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def test_decision_worker_evaluate_rollback_within_tolerance():
    """Test rollback evaluation when accuracy is within tolerance."""
    print("\n" + "=" * 60)
    print("Testing Decision Worker - Evaluate Rollback (Within Tolerance)")
    print("=" * 60)
    print()

    worker = DecisionWorker()

    # Set best accuracy
    worker.state.best_accuracy = 95.0
    worker.state.best_checkpoint_version = "version_1"
    worker.state.patience_counter = 1  # Had some patience before

    validation_result = {
        "model_version_id": "version_2",
        "ipfs_cid": "QmTest222",
        "metrics": {"accuracy": 94.6},  # 0.4% drop (within 0.5% tolerance)
    }

    should_rollback, reason = worker._evaluate_rollback(94.6, validation_result)

    assert not should_rollback
    assert reason is None
    assert worker.state.best_accuracy == 95.0  # Best unchanged
    assert worker.state.patience_counter == 0  # Patience reset
    print("✓ Accuracy within tolerance correctly resets patience")

    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def test_decision_worker_evaluate_rollback_severe_drop():
    """Test rollback evaluation for severe accuracy drop."""
    print("\n" + "=" * 60)
    print("Testing Decision Worker - Evaluate Rollback (Severe Drop)")
    print("=" * 60)
    print()

    worker = DecisionWorker()

    # Set best accuracy
    worker.state.best_accuracy = 95.0
    worker.state.best_checkpoint_version = "version_1"
    worker.state.best_checkpoint_cid = "QmTest111"

    validation_result = {
        "model_version_id": "version_2",
        "ipfs_cid": "QmTest222",
        "metrics": {"accuracy": 92.5},  # 2.5% drop (exceeds 2.0% severe threshold)
    }

    should_rollback, reason = worker._evaluate_rollback(92.5, validation_result)

    assert should_rollback
    assert reason is not None
    assert "Severe accuracy drop" in reason
    assert "2.50" in reason or "2.5" in reason
    print("✓ Severe drop correctly triggers immediate rollback")

    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def test_decision_worker_evaluate_rollback_patience_exceeded():
    """Test rollback evaluation when patience threshold is exceeded."""
    print("\n" + "=" * 60)
    print("Testing Decision Worker - Evaluate Rollback (Patience Exceeded)")
    print("=" * 60)
    print()

    worker = DecisionWorker()

    # Set best accuracy
    worker.state.best_accuracy = 95.0
    worker.state.best_checkpoint_version = "version_1"
    worker.state.best_checkpoint_cid = "QmTest111"
    worker.state.patience_threshold = 3

    # Simulate 3 consecutive bad iterations
    for i in range(3):
        validation_result = {
            "model_version_id": f"version_{i+2}",
            "ipfs_cid": f"QmTest{i+2}",
            "metrics": {"accuracy": 94.0},  # 1% drop (beyond tolerance but not severe)
        }

        should_rollback, reason = worker._evaluate_rollback(94.0, validation_result)

        if i < 2:
            # First two iterations: patience increases, no rollback
            assert not should_rollback
            assert worker.state.patience_counter == i + 1
        else:
            # Third iteration: patience threshold exceeded, rollback
            assert should_rollback
            assert reason is not None
            assert "Patience threshold exceeded" in reason
            assert worker.state.patience_counter == 3

    print("✓ Patience threshold exceeded correctly triggers rollback")

    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def test_decision_worker_check_completion_accuracy_threshold():
    """Test training completion check for accuracy threshold."""
    print("\n" + "=" * 60)
    print("Testing Decision Worker - Check Completion (Accuracy Threshold)")
    print("=" * 60)
    print()

    worker = DecisionWorker()
    worker.target_accuracy = 95.0

    validation_result = {
        "model_version_id": "version_1",
        "metrics": {"accuracy": 96.0},
    }

    should_complete, reason = worker._check_training_completion(96.0, validation_result)

    assert should_complete
    assert reason is not None
    assert "Target accuracy reached" in reason
    print("✓ Accuracy threshold correctly triggers completion")

    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def test_decision_worker_check_completion_max_iterations():
    """Test training completion check for max iterations."""
    print("\n" + "=" * 60)
    print("Testing Decision Worker - Check Completion (Max Iterations)")
    print("=" * 60)
    print()

    worker = DecisionWorker()
    worker.max_iterations = 10
    worker.state.current_iteration = 10

    validation_result = {
        "model_version_id": "version_10",
        "metrics": {"accuracy": 90.0},
    }

    should_complete, reason = worker._check_training_completion(90.0, validation_result)

    assert should_complete
    assert reason is not None
    assert "Maximum iterations reached" in reason
    print("✓ Max iterations correctly triggers completion")

    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def test_decision_worker_check_completion_max_rollbacks():
    """Test training completion check for max rollbacks."""
    print("\n" + "=" * 60)
    print("Testing Decision Worker - Check Completion (Max Rollbacks)")
    print("=" * 60)
    print()

    worker = DecisionWorker()
    worker.max_rollbacks = 5
    worker.state.rollback_count = 5

    validation_result = {
        "model_version_id": "version_1",
        "metrics": {"accuracy": 90.0},
    }

    should_complete, reason = worker._check_training_completion(90.0, validation_result)

    assert should_complete
    assert reason is not None
    assert "Maximum rollbacks reached" in reason
    print("✓ Max rollbacks correctly triggers completion")

    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def test_decision_worker_check_completion_convergence():
    """Test training completion check for convergence."""
    print("\n" + "=" * 60)
    print("Testing Decision Worker - Check Completion (Convergence)")
    print("=" * 60)
    print()

    worker = DecisionWorker()
    worker.convergence_patience = 5
    worker.state.best_accuracy = 95.0
    # Add 5 consecutive accuracies at or below best
    worker.state.accuracy_history = [94.0, 94.5, 94.0, 94.5, 94.0]

    validation_result = {
        "model_version_id": "version_6",
        "metrics": {"accuracy": 94.5},
    }

    should_complete, reason = worker._check_training_completion(94.5, validation_result)

    assert should_complete
    assert reason is not None
    assert "Convergence detected" in reason
    print("✓ Convergence correctly triggers completion")

    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def test_decision_worker_publish_train_tasks():
    """Test publishing TRAIN tasks."""
    print("\n" + "=" * 60)
    print("Testing Decision Worker - Publish Train Tasks")
    print("=" * 60)
    print()

    worker = DecisionWorker()
    worker.num_clients = 3

    # Mock publisher - explicitly set publish_task as synchronous
    # Use side_effect with a regular function to ensure it's not treated as async
    def sync_publish_task(*args, **kwargs):
        return None

    mock_publisher = Mock(spec=["publish_task"])
    mock_publisher.publish_task = Mock(side_effect=sync_publish_task)
    worker.publisher = mock_publisher

    # Publish train tasks
    worker._publish_train_tasks(iteration=1, weights_cid="QmTest123")

    # Verify a single universal task was published (not one per client)
    assert mock_publisher.publish_task.call_count == 1

    # Check the universal task
    call = mock_publisher.publish_task.call_args_list[0]
    task = call.kwargs["task"]
    assert task.task_type == TaskType.TRAIN
    assert task.payload["iteration"] == 1
    assert task.payload["weights_cid"] == "QmTest123"
    # Universal tasks don't include client_id - clients use their own instance_id when sending updates
    
    # Verify fanout exchange is used
    assert call.kwargs.get("use_fanout") == True, "Should use fanout exchange for universal tasks"

    print("✓ Universal TRAIN task published correctly (all clients will receive via fanout)")

    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def test_decision_worker_publish_rollback_task():
    """Test publishing ROLLBACK task."""
    print("\n" + "=" * 60)
    print("Testing Decision Worker - Publish Rollback Task")
    print("=" * 60)
    print()

    worker = DecisionWorker()

    # Mock publisher - explicitly set publish_task as synchronous
    # Use side_effect with a regular function to ensure it's not treated as async
    def sync_publish_task(*args, **kwargs):
        return None

    mock_publisher = Mock(spec=["publish_task"])
    mock_publisher.publish_task = Mock(side_effect=sync_publish_task)
    worker.publisher = mock_publisher

    # Publish rollback task
    worker._publish_rollback_task(
        target_version_id="version_1",
        target_weights_cid="QmTest111",
        reason="Test rollback",
    )

    # Verify task was published
    assert mock_publisher.publish_task.called
    call_args = mock_publisher.publish_task.call_args
    task = call_args.kwargs["task"]
    assert task.task_type == TaskType.ROLLBACK
    assert task.payload["target_version_id"] == "version_1"
    assert task.payload["target_weights_cid"] == "QmTest111"
    assert task.payload["reason"] == "Test rollback"
    print("✓ ROLLBACK task published correctly")

    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def test_decision_worker_publish_training_complete_task():
    """Test publishing TRAINING_COMPLETE task."""
    print("\n" + "=" * 60)
    print("Testing Decision Worker - Publish Training Complete Task")
    print("=" * 60)
    print()

    worker = DecisionWorker()
    worker.state.training_start_time = 1000.0
    worker.state.current_iteration = 10
    worker.state.rollback_count = 2

    # Mock publisher - explicitly set publish_task as synchronous
    # Use side_effect with a regular function to ensure it's not treated as async
    def sync_publish_task(*args, **kwargs):
        return None

    mock_publisher = Mock(spec=["publish_task"])
    mock_publisher.publish_task = Mock(side_effect=sync_publish_task)
    worker.publisher = mock_publisher

    validation_result = {
        "model_version_id": "version_10",
        "ipfs_cid": "QmTest999",
        "metrics": {"accuracy": 96.0, "loss": 0.1},
    }

    # Publish training complete task
    worker._publish_training_complete_task(
        validation_result=validation_result, completion_reason="Test completion"
    )

    # Verify task was published
    assert mock_publisher.publish_task.called
    call_args = mock_publisher.publish_task.call_args
    task = call_args.kwargs["task"]
    assert task.task_type == TaskType.TRAINING_COMPLETE
    assert task.payload["final_model_version_id"] == "version_10"
    assert task.payload["final_accuracy"] == 96.0
    assert task.payload["final_weights_cid"] == "QmTest999"
    assert task.payload["training_summary"]["total_iterations"] == 10
    assert task.payload["training_summary"]["rollback_count"] == 2
    assert task.payload["metadata"]["completion_reason"] == "Test completion"
    print("✓ TRAINING_COMPLETE task published correctly")

    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def test_decision_worker_handle_decision_continue_training():
    """Test handling DECISION task that continues training."""
    print("\n" + "=" * 60)
    print("Testing Decision Worker - Handle Decision (Continue Training)")
    print("=" * 60)
    print()

    worker = DecisionWorker()
    worker.num_clients = 2

    # Mock publisher - explicitly set publish_task as synchronous
    # Use side_effect with a regular function to ensure it's not treated as async
    def sync_publish_task(*args, **kwargs):
        return None

    mock_publisher = Mock(spec=["publish_task"])
    mock_publisher.publish_task = Mock(side_effect=sync_publish_task)
    worker.publisher = mock_publisher

    # Create DECISION task with good accuracy
    decision_task = Task(
        task_id="decision-test-001",
        task_type=TaskType.DECISION,
        payload=DecisionTaskPayload(
            validation_result={
                "model_version_id": "version_1",
                "ipfs_cid": "QmTest111",
                "metrics": {"accuracy": 90.0},
            },
            model_version_id="version_1",
            should_rollback=False,
            rollback_reason=None,
        ).model_dump(),
        metadata=TaskMetadata(source="test"),
        model_version_id="version_1",
        parent_version_id=None,
    )

    # Handle decision task
    success = worker._handle_decision_task(decision_task)

    assert success
    assert worker.state.current_iteration == 1
    assert worker.state.best_accuracy == 90.0
    assert len(worker.state.accuracy_history) == 1

    # Verify universal TRAIN task was published (via fanout exchange)
    assert mock_publisher.publish_task.called
    train_calls = [
        call
        for call in mock_publisher.publish_task.call_args_list
        if call.kwargs["task"].task_type == TaskType.TRAIN
    ]
    assert len(train_calls) == 1  # Single universal task via fanout exchange
    # Verify fanout exchange is used
    assert train_calls[0].kwargs.get("use_fanout") == True, "Should use fanout exchange for universal tasks"
    print("✓ Decision task handled correctly, universal TRAIN task published via fanout exchange")

    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def test_decision_worker_handle_decision_rollback():
    """Test handling DECISION task that triggers rollback."""
    print("\n" + "=" * 60)
    print("Testing Decision Worker - Handle Decision (Rollback)")
    print("=" * 60)
    print()

    worker = DecisionWorker()
    worker.num_clients = 2

    # Set initial best
    worker.state.best_accuracy = 95.0
    worker.state.best_checkpoint_version = "version_1"
    worker.state.best_checkpoint_cid = "QmTest111"
    worker.state.patience_counter = 3  # Patience exceeded

    # Mock publisher - explicitly set publish_task as synchronous
    # Use side_effect with a regular function to ensure it's not treated as async
    def sync_publish_task(*args, **kwargs):
        return None

    mock_publisher = Mock(spec=["publish_task"])
    mock_publisher.publish_task = Mock(side_effect=sync_publish_task)
    worker.publisher = mock_publisher

    # Create DECISION task with poor accuracy
    decision_task = Task(
        task_id="decision-test-002",
        task_type=TaskType.DECISION,
        payload=DecisionTaskPayload(
            validation_result={
                "model_version_id": "version_2",
                "ipfs_cid": "QmTest222",
                "metrics": {"accuracy": 92.0},  # 3% drop
            },
            model_version_id="version_2",
            should_rollback=False,  # Will be determined by evaluation
            rollback_reason=None,
        ).model_dump(),
        metadata=TaskMetadata(source="test"),
        model_version_id="version_2",
        parent_version_id="version_1",
    )

    # Handle decision task
    success = worker._handle_decision_task(decision_task)

    assert success
    assert worker.state.rollback_count == 1
    assert worker.state.patience_counter == 0  # Reset after rollback

    # Verify ROLLBACK task was published
    rollback_calls = [
        call
        for call in mock_publisher.publish_task.call_args_list
        if call.kwargs["task"].task_type == TaskType.ROLLBACK
    ]
    assert len(rollback_calls) == 1

    # Verify universal TRAIN task was published (to continue from rollback via fanout exchange)
    train_calls = [
        call
        for call in mock_publisher.publish_task.call_args_list
        if call.kwargs["task"].task_type == TaskType.TRAIN
    ]
    assert len(train_calls) == 1  # Single universal task via fanout exchange
    # Verify fanout exchange is used
    assert train_calls[0].kwargs.get("use_fanout") == True, "Should use fanout exchange for universal tasks"
    print("✓ Rollback correctly triggered, ROLLBACK and universal TRAIN task published via fanout exchange")

    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def test_decision_worker_handle_decision_max_rollbacks():
    """Test handling DECISION task when max rollbacks reached."""
    print("\n" + "=" * 60)
    print("Testing Decision Worker - Handle Decision (Max Rollbacks)")
    print("=" * 60)
    print()

    worker = DecisionWorker()
    worker.max_rollbacks = 3
    worker.state.rollback_count = 3
    worker.state.best_accuracy = 95.0
    worker.state.best_checkpoint_version = "version_1"
    worker.state.best_checkpoint_cid = "QmTest111"

    # Mock publisher - explicitly set publish_task as synchronous
    # Use side_effect with a regular function to ensure it's not treated as async
    def sync_publish_task(*args, **kwargs):
        return None

    mock_publisher = Mock(spec=["publish_task"])
    mock_publisher.publish_task = Mock(side_effect=sync_publish_task)
    worker.publisher = mock_publisher

    # Create DECISION task that would trigger rollback
    decision_task = Task(
        task_id="decision-test-003",
        task_type=TaskType.DECISION,
        payload=DecisionTaskPayload(
            validation_result={
                "model_version_id": "version_2",
                "ipfs_cid": "QmTest222",
                "metrics": {"accuracy": 90.0},
            },
            model_version_id="version_2",
            should_rollback=True,
            rollback_reason="Test rollback",
        ).model_dump(),
        metadata=TaskMetadata(source="test"),
        model_version_id="version_2",
        parent_version_id="version_1",
    )

    # Handle decision task
    success = worker._handle_decision_task(decision_task)

    assert success

    # Verify TRAINING_COMPLETE task was published (not ROLLBACK)
    complete_calls = [
        call
        for call in mock_publisher.publish_task.call_args_list
        if call.kwargs["task"].task_type == TaskType.TRAINING_COMPLETE
    ]
    assert len(complete_calls) == 1

    # Verify ROLLBACK task was NOT published
    rollback_calls = [
        call
        for call in mock_publisher.publish_task.call_args_list
        if call.kwargs["task"].task_type == TaskType.ROLLBACK
    ]
    assert len(rollback_calls) == 0
    print("✓ Max rollbacks correctly triggers training completion instead of rollback")

    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def run_all_tests():
    """Run all decision worker tests."""
    tests = [
        ("Initialization", test_decision_worker_initialization),
        (
            "Evaluate Rollback (First Iteration)",
            test_decision_worker_evaluate_rollback_first_iteration,
        ),
        (
            "Evaluate Rollback (New Best)",
            test_decision_worker_evaluate_rollback_new_best,
        ),
        (
            "Evaluate Rollback (Within Tolerance)",
            test_decision_worker_evaluate_rollback_within_tolerance,
        ),
        (
            "Evaluate Rollback (Severe Drop)",
            test_decision_worker_evaluate_rollback_severe_drop,
        ),
        (
            "Evaluate Rollback (Patience Exceeded)",
            test_decision_worker_evaluate_rollback_patience_exceeded,
        ),
        (
            "Check Completion (Accuracy Threshold)",
            test_decision_worker_check_completion_accuracy_threshold,
        ),
        (
            "Check Completion (Max Iterations)",
            test_decision_worker_check_completion_max_iterations,
        ),
        (
            "Check Completion (Max Rollbacks)",
            test_decision_worker_check_completion_max_rollbacks,
        ),
        (
            "Check Completion (Convergence)",
            test_decision_worker_check_completion_convergence,
        ),
        ("Publish Train Tasks", test_decision_worker_publish_train_tasks),
        ("Publish Rollback Task", test_decision_worker_publish_rollback_task),
        (
            "Publish Training Complete Task",
            test_decision_worker_publish_training_complete_task,
        ),
        (
            "Handle Decision (Continue Training)",
            test_decision_worker_handle_decision_continue_training,
        ),
        ("Handle Decision (Rollback)", test_decision_worker_handle_decision_rollback),
        (
            "Handle Decision (Max Rollbacks)",
            test_decision_worker_handle_decision_max_rollbacks,
        ),
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
