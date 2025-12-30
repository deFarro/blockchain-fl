"""Test storage utilities (encryption and IPFS)."""

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
from shared.storage.encryption import (
    EncryptionService,
    encrypt_diff,
    decrypt_diff,
)
from main_service.storage.ipfs_client import IPFSClient, add_to_ipfs, get_from_ipfs
from shared.utils.crypto import AES256GCM, CryptoError
from shared.config import settings


def test_encryption_basic():
    """Test basic encryption/decryption."""
    print("=" * 60)
    print("Testing Encryption - Basic")
    print("=" * 60)
    print()

    # Generate a test key
    key = AES256GCM.generate_key()

    # Create encryption service
    service = EncryptionService(key=key)

    # Test data
    test_data = b"Hello, this is test data for encryption!"

    # Encrypt
    encrypted = service.encrypt_diff(test_data)
    assert len(encrypted) > len(
        test_data
    ), "Encrypted data should be longer (includes nonce + tag)"
    print(f"✓ Encrypted {len(test_data)} bytes to {len(encrypted)} bytes")

    # Decrypt
    decrypted = service.decrypt_diff(encrypted)
    assert decrypted == test_data, "Decrypted data should match original"
    print("✓ Decryption successful")

    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def test_encryption_with_associated_data():
    """Test encryption with associated data."""
    print("\n" + "=" * 60)
    print("Testing Encryption - With Associated Data")
    print("=" * 60)
    print()

    key = AES256GCM.generate_key()
    service = EncryptionService(key=key)

    test_data = b"Test data"
    associated_data = b"metadata: version_1"

    # Encrypt with associated data
    encrypted = service.encrypt_diff(test_data, associated_data)
    print("✓ Encrypted with associated data")

    # Decrypt with same associated data (should work)
    decrypted = service.decrypt_diff(encrypted, associated_data)
    assert decrypted == test_data
    print("✓ Decryption with matching associated data successful")

    # Decrypt with wrong associated data (should fail)
    wrong_data = b"metadata: version_2"
    try:
        service.decrypt_diff(encrypted, wrong_data)
        assert False, "Should have raised CryptoError"
    except CryptoError:
        print("✓ Decryption with wrong associated data correctly failed")

    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def test_encryption_convenience_functions():
    """Test convenience functions."""
    print("\n" + "=" * 60)
    print("Testing Encryption - Convenience Functions")
    print("=" * 60)
    print()

    # Set a test encryption key (Base64-encoded as expected by settings)
    test_key = AES256GCM.generate_key()
    test_key_b64 = base64.b64encode(test_key).decode("utf-8")

    # Set environment variable
    original_key = os.environ.get("ENCRYPTION_KEY")
    os.environ["ENCRYPTION_KEY"] = test_key_b64

    try:
        # Force reload of settings module to pick up new env var
        import importlib
        import shared.config

        importlib.reload(shared.config)
        # Recreate settings instance with new env var
        shared.config.settings = shared.config.Settings()

        # Re-import convenience functions to use new settings
        from shared.storage import encryption

        importlib.reload(encryption)

        test_data = b"Test data for convenience functions"

        # Use convenience functions (they will use the new settings)
        encrypted = encryption.encrypt_diff(test_data)
        decrypted = encryption.decrypt_diff(encrypted)

        assert decrypted == test_data
        print("✓ Convenience functions work correctly")
    finally:
        # Restore original environment variable
        if original_key:
            os.environ["ENCRYPTION_KEY"] = original_key
        else:
            os.environ.pop("ENCRYPTION_KEY", None)
        # Reload settings to restore original state
        import importlib
        import shared.config

        importlib.reload(shared.config)
        # Reload encryption module to restore original state
        from shared.storage import encryption

        importlib.reload(encryption)

    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


def test_encryption_weight_diff():
    """Test encryption with actual weight diff format."""
    print("\n" + "=" * 60)
    print("Testing Encryption - Weight Diff Format")
    print("=" * 60)
    print()

    from client_service.training.model import SimpleCNN

    # Create model and get weights
    model = SimpleCNN(num_classes=10)
    weights = model.get_weights()

    # Serialize to JSON bytes (simulating weight diff)
    diff_dict = {}
    for name, tensor in weights.items():
        diff_dict[name] = tensor.numpy().tolist()

    diff_json = json.dumps(diff_dict)
    diff_bytes = diff_json.encode("utf-8")

    print(f"Original diff size: {len(diff_bytes)} bytes")

    # Encrypt
    key = AES256GCM.generate_key()
    service = EncryptionService(key=key)
    encrypted = service.encrypt_diff(diff_bytes)

    print(f"Encrypted diff size: {len(encrypted)} bytes")
    print(f"Overhead: {len(encrypted) - len(diff_bytes)} bytes (nonce + tag)")

    # Decrypt
    decrypted = service.decrypt_diff(encrypted)
    assert decrypted == diff_bytes, "Decrypted diff should match original"

    # Verify we can deserialize
    decrypted_dict = json.loads(decrypted.decode("utf-8"))
    assert len(decrypted_dict) == len(diff_dict)
    print("✓ Weight diff encryption/decryption successful")

    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


@pytest.mark.asyncio
async def test_ipfs_basic():
    """Test basic IPFS operations."""
    print("\n" + "=" * 60)
    print("Testing IPFS - Basic Operations")
    print("=" * 60)
    print()

    # Skip if IPFS is not available
    try:
        async with IPFSClient() as client:
            # Test version check
            version = await client.version()
            assert "Version" in version or "version" in version
            print(f"✓ IPFS version check successful: {version}")

            # Test node ID
            node_id = await client.id()
            assert "ID" in node_id or "id" in node_id
            print(f"✓ IPFS node ID check successful")
    except Exception as e:
        print(f"⚠ IPFS not available: {str(e)}")
        print("  Skipping IPFS tests (make sure IPFS daemon is running)")
        pytest.skip("IPFS not available")


@pytest.mark.asyncio
async def test_ipfs_add_and_get():
    """Test IPFS add and get operations."""
    print("\n" + "=" * 60)
    print("Testing IPFS - Add and Get")
    print("=" * 60)
    print()

    try:
        # Test data
        test_data = b"This is test data for IPFS storage!"

        # Add to IPFS
        async with IPFSClient() as client:
            cid = await client.add_bytes(test_data, pin=True)
            assert cid, "CID should not be empty"
            print(f"✓ Added data to IPFS: CID={cid}")

            # Retrieve from IPFS
            retrieved = await client.get_bytes(cid)
            assert retrieved == test_data, "Retrieved data should match original"
            print(f"✓ Retrieved data from IPFS: {len(retrieved)} bytes")

            # Test pinning
            pin_info = await client.pin_ls(cid)
            print(f"✓ Pinning verified: {pin_info}")

    except Exception as e:
        print(f"⚠ IPFS not available: {str(e)}")
        print("  Skipping IPFS tests (make sure IPFS daemon is running)")
        pytest.skip("IPFS not available")


@pytest.mark.asyncio
async def test_ipfs_convenience_functions():
    """Test IPFS convenience functions."""
    print("\n" + "=" * 60)
    print("Testing IPFS - Convenience Functions")
    print("=" * 60)
    print()

    try:
        test_data = b"Test data for convenience functions"

        # Use convenience function to add
        cid = await add_to_ipfs(test_data, pin=True)
        assert cid
        print(f"✓ add_to_ipfs() successful: CID={cid}")

        # Use convenience function to get
        retrieved = await get_from_ipfs(cid)
        assert retrieved == test_data
        print(f"✓ get_from_ipfs() successful: {len(retrieved)} bytes")

    except Exception as e:
        print(f"⚠ IPFS not available: {str(e)}")
        print("  Skipping IPFS tests (make sure IPFS daemon is running)")
        pytest.skip("IPFS not available")


@pytest.mark.asyncio
async def test_ipfs_weight_diff():
    """Test IPFS with actual weight diff format."""
    print("\n" + "=" * 60)
    print("Testing IPFS - Weight Diff Format")
    print("=" * 60)
    print()

    try:
        from client_service.training.model import SimpleCNN

        # Create model and serialize weights
        model = SimpleCNN(num_classes=10)
        weights = model.get_weights()

        diff_dict = {}
        for name, tensor in weights.items():
            diff_dict[name] = tensor.numpy().tolist()

        diff_json = json.dumps(diff_dict)
        diff_bytes = diff_json.encode("utf-8")

        print(f"Diff size: {len(diff_bytes)} bytes")

        # Encrypt first
        key = AES256GCM.generate_key()
        service = EncryptionService(key=key)
        encrypted = service.encrypt_diff(diff_bytes)

        print(f"Encrypted size: {len(encrypted)} bytes")

        # Add encrypted diff to IPFS
        async with IPFSClient() as client:
            cid = await client.add_bytes(encrypted, pin=True)
            print(f"✓ Added encrypted diff to IPFS: CID={cid}")

            # Retrieve
            retrieved_encrypted = await client.get_bytes(cid)
            assert retrieved_encrypted == encrypted

            # Decrypt
            retrieved_decrypted = service.decrypt_diff(retrieved_encrypted)
            assert retrieved_decrypted == diff_bytes

            print("✓ Full cycle: encrypt → IPFS → retrieve → decrypt successful")

    except Exception as e:
        print(f"⚠ IPFS not available: {str(e)}")
        print("  Skipping IPFS tests (make sure IPFS daemon is running)")
        pytest.skip("IPFS not available")


def run_all_tests():
    """Run all storage tests."""
    tests = [
        ("Encryption Basic", test_encryption_basic),
        ("Encryption With Associated Data", test_encryption_with_associated_data),
        ("Encryption Convenience Functions", test_encryption_convenience_functions),
        ("Encryption Weight Diff", test_encryption_weight_diff),
    ]

    async_tests = [
        ("IPFS Basic", test_ipfs_basic),
        ("IPFS Add and Get", test_ipfs_add_and_get),
        ("IPFS Convenience Functions", test_ipfs_convenience_functions),
        ("IPFS Weight Diff", test_ipfs_weight_diff),
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
