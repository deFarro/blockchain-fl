"""Test client service training functionality."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import os
import torch
from client_service.training.model import SimpleCNN, create_model
from client_service.training.trainer import Trainer
from client_service.config import config

# Set localhost for local testing
if "RABBITMQ_HOST" not in os.environ:
    os.environ["RABBITMQ_HOST"] = "localhost"


def test_model_creation():
    """Test model creation and basic operations."""
    print("Testing model creation...")

    model = create_model(num_classes=10)
    print(f"✓ Model created: {model}")

    # Test forward pass
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    assert output.shape == (1, 10), f"Expected output shape (1, 10), got {output.shape}"
    print("✓ Forward pass works")

    # Test weight serialization
    weights = model.get_weights()
    assert len(weights) > 0, "Model should have weights"
    print(f"✓ Model has {len(weights)} parameter groups")

    # Test weight diff computation
    diff = model.compute_weight_diff()
    assert len(diff) == len(weights), "Diff should have same keys as weights"
    print("✓ Weight diff computation works")

    # Test weight serialization to bytes
    weights_bytes = model.weights_to_bytes()
    assert len(weights_bytes) > 0, "Serialized weights should not be empty"
    print("✓ Weight serialization works")

    # Test weight deserialization
    restored_weights = SimpleCNN.weights_from_bytes(weights_bytes)
    assert len(restored_weights) == len(weights), "Restored weights should match"
    print("✓ Weight deserialization works")


def test_trainer_basic():
    """Test basic trainer functionality."""
    print("\nTesting trainer...")

    trainer = Trainer(learning_rate=0.001, batch_size=32, epochs=1)
    print("✓ Trainer created")

    # Test dataset loading
    train_loader, train_dataset = trainer.load_dataset(instance_id="test_client_0")
    assert len(train_dataset) > 0, "Dataset should not be empty"
    print(f"✓ Dataset loaded: {len(train_dataset)} samples")

    # Test training
    weight_diff, metrics, _ = trainer.train(train_loader=train_loader)
    assert "loss" in metrics, "Metrics should include loss"
    assert "accuracy" in metrics, "Metrics should include accuracy"
    assert len(weight_diff) > 0, "Weight diff should not be empty"
    print(
        f"✓ Training completed: loss={metrics['loss']:.4f}, accuracy={metrics['accuracy']:.2f}%"
    )


def test_weight_diff():
    """Test weight diff computation during training."""
    print("\nTesting weight diff computation...")

    trainer = Trainer(learning_rate=0.001, batch_size=32, epochs=1)
    train_loader, _ = trainer.load_dataset(instance_id="test_client_0")

    # Get initial weights
    initial_weights = trainer.get_model().get_weights()

    # Train with previous weights
    weight_diff, metrics, actual_initial_weights = trainer.train(
        train_loader=train_loader, previous_weights=initial_weights
    )

    # Verify diff is not zero (model should have learned something)
    has_non_zero = False
    for name, diff_tensor in weight_diff.items():
        if torch.any(diff_tensor != 0):
            has_non_zero = True
            break

    assert has_non_zero, "Weight diff should have non-zero values after training"
    print("✓ Weight diff contains non-zero values (model learned)")

    # Test applying diff
    # Use the actual initial weights that were used during training
    model2 = create_model()
    model2.set_weights(actual_initial_weights)
    model2.apply_weight_diff(weight_diff)

    # Verify models match
    final_weights = trainer.get_model().get_weights()
    model2_weights = model2.get_weights()

    for name in final_weights:
        assert torch.allclose(
            final_weights[name], model2_weights[name]
        ), f"Weights should match after applying diff: {name}"
    print("✓ Weight diff application works correctly")


if __name__ == "__main__":
    print("=" * 60)
    print("Client Service Training Tests")
    print("=" * 60)
    print()

    try:
        # Run tests
        test_model_creation()
        test_trainer_basic()
        test_weight_diff()

        print("\n" + "=" * 60)
        print("✓ All tests PASSED")
        print("=" * 60)
        sys.exit(0)

    except Exception as e:
        print(f"\n✗ Test FAILED: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
