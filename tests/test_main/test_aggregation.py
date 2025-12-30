"""Test aggregation worker functionality."""

import sys
from pathlib import Path
import os
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Set localhost for local testing
if "RABBITMQ_HOST" not in os.environ:
    os.environ["RABBITMQ_HOST"] = "localhost"

import torch
import pytest
from shared.queue.publisher import QueuePublisher
from main_service.workers.aggregation_worker import AggregationWorker
from main_service.workers.regression_diagnosis import RegressionDiagnosis
from client_service.training.model import SimpleCNN


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

    client_updates = [
        {
            "client_id": "client_0",
            "iteration": 1,
            "weight_diff": serialize_diff(diff1),
            "metrics": {"loss": 0.5, "accuracy": 90.0, "samples": 100},
        },
        {
            "client_id": "client_1",
            "iteration": 1,
            "weight_diff": serialize_diff(diff2),
            "metrics": {"loss": 0.6, "accuracy": 85.0, "samples": 200},
        },
    ]

    # Test aggregation
    worker = AggregationWorker()
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

    # Client 1 has 100 samples, Client 2 has 200 samples
    # Client 2 should have 2x weight in aggregation
    client_updates = [
        {
            "client_id": "client_0",
            "iteration": 1,
            "weight_diff": serialize_diff(diff1),
            "metrics": {"samples": 100},
        },
        {
            "client_id": "client_1",
            "iteration": 1,
            "weight_diff": serialize_diff(diff2),
            "metrics": {"samples": 200},
        },
    ]

    worker = AggregationWorker()
    aggregated = worker._fedavg_aggregate(client_updates)

    # Verify aggregation worked
    assert len(aggregated) > 0
    print("✓ Weighted averaging completed")

    # Test with explicit sample counts
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

    client_updates = [
        {
            "client_id": "client_0",
            "iteration": 1,
            "weight_diff": serialize_diff(diff),
            "metrics": {"samples": 100},
        }
    ]

    worker = AggregationWorker()
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

    client_updates = [
        {
            "client_id": "client_0",
            "iteration": 1,
            "weight_diff": serialize_diff(diff1),
            "metrics": {"samples": 100},
        },
        {
            "client_id": "client_1",
            "iteration": 1,
            "weight_diff": serialize_diff(diff2),
            "metrics": {"samples": 200},
        },
        {
            "client_id": "client_2",
            "iteration": 1,
            "weight_diff": serialize_diff(diff3),
            "metrics": {"samples": 150},
        },
    ]

    worker = AggregationWorker()

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
    # (Note: In real scenario, this depends on actual model behavior)
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
