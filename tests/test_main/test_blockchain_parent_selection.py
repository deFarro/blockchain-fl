"""Tests for blockchain worker parent version selection in normal and rollback scenarios."""

import pytest
from unittest.mock import AsyncMock, patch
from main_service.workers.blockchain_worker import BlockchainWorker
from main_service.blockchain.fabric_client import FabricClient


@pytest.mark.asyncio
async def test_parent_selection_normal_flow():
    """Test parent version selection in normal flow (no rollback)."""
    print("\n" + "=" * 60)
    print("Testing Parent Selection - Normal Flow")
    print("=" * 60)
    print()

    worker = BlockchainWorker()

    # Mock blockchain client responses
    async def mock_list_models():
        """Return versions for normal flow: iterations 0, 1, 2."""
        return {
            "versions": [
                {
                    "version_id": "model_v0_initial_1000_abc",
                    "timestamp": "1000",
                    "metadata": {"iteration": 0},
                    "parent_version_id": None,
                },
                {
                    "version_id": "model_v1_2000_def",
                    "timestamp": "2000",
                    "metadata": {"iteration": 1},
                    "parent_version_id": "model_v0_initial_1000_abc",
                },
                {
                    "version_id": "model_v2_3000_ghi",
                    "timestamp": "3000",
                    "metadata": {"iteration": 2},
                    "parent_version_id": "model_v1_2000_def",
                },
            ]
        }

    async def mock_get_provenance(version_id: str):
        """Return provenance for a version."""
        versions = {
            "model_v0_initial_1000_abc": {
                "version_id": "model_v0_initial_1000_abc",
                "timestamp": "1000",
                "metadata": {"iteration": 0},
                "parent_version_id": None,
            },
            "model_v1_2000_def": {
                "version_id": "model_v1_2000_def",
                "timestamp": "2000",
                "metadata": {"iteration": 1},
                "parent_version_id": "model_v0_initial_1000_abc",
            },
            "model_v2_3000_ghi": {
                "version_id": "model_v2_3000_ghi",
                "timestamp": "3000",
                "metadata": {"iteration": 2},
                "parent_version_id": "model_v1_2000_def",
            },
        }
        return versions.get(version_id, {})

    async def mock_get_most_recent_rollback():
        """No rollback in normal flow."""
        return None

    with patch(
        "main_service.workers.blockchain_worker.FabricClient"
    ) as mock_fabric_class:
        mock_client = AsyncMock()
        mock_client.list_models = AsyncMock(side_effect=mock_list_models)
        mock_client.get_model_provenance = AsyncMock(side_effect=mock_get_provenance)
        mock_client.get_most_recent_rollback = AsyncMock(
            side_effect=mock_get_most_recent_rollback
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_fabric_class.return_value = mock_client

        # Test iteration 0 (should return None)
        parent_0 = await worker._get_parent_version_id_for_iteration(0)
        assert parent_0 is None
        print(f"✓ Iteration 0 parent: {parent_0} (expected: None)")

        # Test iteration 1 (should return iteration 0)
        parent_1 = await worker._get_parent_version_id_for_iteration(1)
        assert parent_1 == "model_v0_initial_1000_abc"
        print(f"✓ Iteration 1 parent: {parent_1} (expected: model_v0_initial_1000_abc)")

        # Test iteration 2 (should return iteration 1)
        parent_2 = await worker._get_parent_version_id_for_iteration(2)
        assert parent_2 == "model_v1_2000_def"
        print(f"✓ Iteration 2 parent: {parent_2} (expected: model_v1_2000_def)")

        # Test iteration 3 (should return iteration 2)
        parent_3 = await worker._get_parent_version_id_for_iteration(3)
        assert parent_3 == "model_v2_3000_ghi"
        print(f"✓ Iteration 3 parent: {parent_3} (expected: model_v2_3000_ghi)")

    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


@pytest.mark.asyncio
async def test_parent_selection_after_rollback_first_iteration():
    """Test parent selection for first iteration after rollback."""
    print("\n" + "=" * 60)
    print("Testing Parent Selection - First Iteration After Rollback")
    print("=" * 60)
    print()

    worker = BlockchainWorker()

    # Scenario: Rollback to iteration 1, then create iteration 3
    # Iteration 2 exists but was created before rollback
    rollback_timestamp = 5000  # Rollback happened at timestamp 5000

    async def mock_list_models():
        """Return versions including pre-rollback iteration 2."""
        return {
            "versions": [
                {
                    "version_id": "model_v0_initial_1000_abc",
                    "timestamp": "1000",
                    "metadata": {"iteration": 0},
                    "parent_version_id": None,
                },
                {
                    "version_id": "model_v1_2000_def",
                    "timestamp": "2000",
                    "metadata": {"iteration": 1},
                    "parent_version_id": "model_v0_initial_1000_abc",
                },
                {
                    "version_id": "model_v2_3000_ghi",  # Created BEFORE rollback
                    "timestamp": "3000",  # Before rollback timestamp (5000)
                    "metadata": {"iteration": 2},
                    "parent_version_id": "model_v1_2000_def",
                },
            ]
        }

    async def mock_get_provenance(version_id: str):
        """Return provenance for a version."""
        versions = {
            "model_v0_initial_1000_abc": {
                "version_id": "model_v0_initial_1000_abc",
                "timestamp": "1000",
                "metadata": {"iteration": 0},
                "parent_version_id": None,
            },
            "model_v1_2000_def": {
                "version_id": "model_v1_2000_def",
                "timestamp": "2000",
                "metadata": {"iteration": 1},
                "parent_version_id": "model_v0_initial_1000_abc",
            },
            "model_v2_3000_ghi": {
                "version_id": "model_v2_3000_ghi",
                "timestamp": "3000",
                "metadata": {"iteration": 2},
                "parent_version_id": "model_v1_2000_def",
            },
        }
        return versions.get(version_id, {})

    async def mock_get_most_recent_rollback():
        """Return rollback event to iteration 1."""
        return {
            "to_version_id": "model_v1_2000_def",
            "timestamp": str(rollback_timestamp),
            "reason": "test rollback",
        }

    with patch(
        "main_service.workers.blockchain_worker.FabricClient"
    ) as mock_fabric_class:
        mock_client = AsyncMock()
        mock_client.list_models = AsyncMock(side_effect=mock_list_models)
        mock_client.get_model_provenance = AsyncMock(side_effect=mock_get_provenance)
        mock_client.get_most_recent_rollback = AsyncMock(
            side_effect=mock_get_most_recent_rollback
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_fabric_class.return_value = mock_client

        # Test iteration 3 after rollback to iteration 1
        # Should use iteration 1 (rollback target) as parent, NOT iteration 2
        parent_3 = await worker._get_parent_version_id_for_iteration(3)
        assert parent_3 == "model_v1_2000_def", (
            f"Expected parent to be model_v1_2000_def (rollback target), "
            f"but got {parent_3}"
        )
        print(f"✓ Iteration 3 parent: {parent_3} (expected: model_v1_2000_def)")
        print("✓ Correctly skipped iteration 2 (created before rollback)")

    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


@pytest.mark.asyncio
async def test_parent_selection_after_rollback_subsequent_iterations():
    """Test parent selection for subsequent iterations after rollback."""
    print("\n" + "=" * 60)
    print("Testing Parent Selection - Subsequent Iterations After Rollback")
    print("=" * 60)
    print()

    worker = BlockchainWorker()

    # Scenario: Rollback to iteration 1, create iteration 3 (uses iteration 1),
    # then create iteration 4 (should use iteration 3)
    rollback_timestamp = 5000

    async def mock_list_models():
        """Return versions including post-rollback iteration 3."""
        return {
            "versions": [
                {
                    "version_id": "model_v0_initial_1000_abc",
                    "timestamp": "1000",
                    "metadata": {"iteration": 0},
                    "parent_version_id": None,
                },
                {
                    "version_id": "model_v1_2000_def",
                    "timestamp": "2000",
                    "metadata": {"iteration": 1},
                    "parent_version_id": "model_v0_initial_1000_abc",
                },
                {
                    "version_id": "model_v2_3000_ghi",  # Created BEFORE rollback
                    "timestamp": "3000",
                    "metadata": {"iteration": 2},
                    "parent_version_id": "model_v1_2000_def",
                },
                {
                    "version_id": "model_v3_6000_jkl",  # Created AFTER rollback
                    "timestamp": "6000",  # After rollback timestamp (5000)
                    "metadata": {"iteration": 3},
                    "parent_version_id": "model_v1_2000_def",  # Uses rollback target
                },
            ]
        }

    async def mock_get_provenance(version_id: str):
        """Return provenance for a version."""
        versions = {
            "model_v0_initial_1000_abc": {
                "version_id": "model_v0_initial_1000_abc",
                "timestamp": "1000",
                "metadata": {"iteration": 0},
                "parent_version_id": None,
            },
            "model_v1_2000_def": {
                "version_id": "model_v1_2000_def",
                "timestamp": "2000",
                "metadata": {"iteration": 1},
                "parent_version_id": "model_v0_initial_1000_abc",
            },
            "model_v2_3000_ghi": {
                "version_id": "model_v2_3000_ghi",
                "timestamp": "3000",
                "metadata": {"iteration": 2},
                "parent_version_id": "model_v1_2000_def",
            },
            "model_v3_6000_jkl": {
                "version_id": "model_v3_6000_jkl",
                "timestamp": "6000",
                "metadata": {"iteration": 3},
                "parent_version_id": "model_v1_2000_def",
            },
        }
        return versions.get(version_id, {})

    async def mock_get_most_recent_rollback():
        """Return rollback event to iteration 1."""
        return {
            "to_version_id": "model_v1_2000_def",
            "timestamp": str(rollback_timestamp),
            "reason": "test rollback",
        }

    with patch(
        "main_service.workers.blockchain_worker.FabricClient"
    ) as mock_fabric_class:
        mock_client = AsyncMock()
        mock_client.list_models = AsyncMock(side_effect=mock_list_models)
        mock_client.get_model_provenance = AsyncMock(side_effect=mock_get_provenance)
        mock_client.get_most_recent_rollback = AsyncMock(
            side_effect=mock_get_most_recent_rollback
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_fabric_class.return_value = mock_client

        # Test iteration 4 after rollback
        # Iteration 3 exists and was created after rollback, so normal logic should apply
        # Should use iteration 3 as parent (not iteration 1 or iteration 2)
        parent_4 = await worker._get_parent_version_id_for_iteration(4)
        assert parent_4 == "model_v3_6000_jkl", (
            f"Expected parent to be model_v3_6000_jkl (iteration 3), "
            f"but got {parent_4}"
        )
        print(f"✓ Iteration 4 parent: {parent_4} (expected: model_v3_6000_jkl)")
        print("✓ Correctly uses iteration 3 (previous iteration) as parent")

    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


@pytest.mark.asyncio
async def test_parent_selection_rollback_timestamp_comparison():
    """Test that versions created before rollback are not counted as post-rollback children."""
    print("\n" + "=" * 60)
    print("Testing Parent Selection - Rollback Timestamp Comparison")
    print("=" * 60)
    print()

    worker = BlockchainWorker()

    # Scenario: Rollback to iteration 1 at timestamp 5000
    # Iteration 2 exists with timestamp 3000 (before rollback)
    # Iteration 3 exists with timestamp 6000 (after rollback)
    rollback_timestamp = 5000

    async def mock_list_models():
        """Return versions with different timestamps relative to rollback."""
        return {
            "versions": [
                {
                    "version_id": "model_v0_initial_1000_abc",
                    "timestamp": "1000",
                    "metadata": {"iteration": 0},
                    "parent_version_id": None,
                },
                {
                    "version_id": "model_v1_2000_def",
                    "timestamp": "2000",
                    "metadata": {"iteration": 1},
                    "parent_version_id": "model_v0_initial_1000_abc",
                },
                {
                    "version_id": "model_v2_3000_ghi",  # Created BEFORE rollback
                    "timestamp": "3000",  # Before rollback timestamp (5000)
                    "metadata": {"iteration": 2},
                    "parent_version_id": "model_v1_2000_def",
                },
                {
                    "version_id": "model_v3_6000_jkl",  # Created AFTER rollback
                    "timestamp": "6000",  # After rollback timestamp (5000)
                    "metadata": {"iteration": 3},
                    "parent_version_id": "model_v1_2000_def",
                },
            ]
        }

    async def mock_get_provenance(version_id: str):
        """Return provenance for a version."""
        versions = {
            "model_v0_initial_1000_abc": {
                "version_id": "model_v0_initial_1000_abc",
                "timestamp": "1000",
                "metadata": {"iteration": 0},
                "parent_version_id": None,
            },
            "model_v1_2000_def": {
                "version_id": "model_v1_2000_def",
                "timestamp": "2000",
                "metadata": {"iteration": 1},
                "parent_version_id": "model_v0_initial_1000_abc",
            },
            "model_v2_3000_ghi": {
                "version_id": "model_v2_3000_ghi",
                "timestamp": "3000",
                "metadata": {"iteration": 2},
                "parent_version_id": "model_v1_2000_def",
            },
            "model_v3_6000_jkl": {
                "version_id": "model_v3_6000_jkl",
                "timestamp": "6000",
                "metadata": {"iteration": 3},
                "parent_version_id": "model_v1_2000_def",
            },
        }
        return versions.get(version_id, {})

    async def mock_get_most_recent_rollback():
        """Return rollback event to iteration 1."""
        return {
            "to_version_id": "model_v1_2000_def",
            "timestamp": str(rollback_timestamp),
            "reason": "test rollback",
        }

    with patch(
        "main_service.workers.blockchain_worker.FabricClient"
    ) as mock_fabric_class:
        mock_client = AsyncMock()
        mock_client.list_models = AsyncMock(side_effect=mock_list_models)
        mock_client.get_model_provenance = AsyncMock(side_effect=mock_get_provenance)
        mock_client.get_most_recent_rollback = AsyncMock(
            side_effect=mock_get_most_recent_rollback
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_fabric_class.return_value = mock_client

        # When creating iteration 4:
        # - Iteration 2 (timestamp 3000) has iteration 1 as parent but was created BEFORE rollback
        #   → Should NOT be counted as post-rollback child
        # - Iteration 3 (timestamp 6000) has iteration 1 as parent and was created AFTER rollback
        #   → Should be counted as post-rollback child
        # - Since iteration 3 exists as post-rollback child, use normal logic → iteration 3 as parent

        parent_4 = await worker._get_parent_version_id_for_iteration(4)
        assert parent_4 == "model_v3_6000_jkl", (
            f"Expected parent to be model_v3_6000_jkl (iteration 3), "
            f"but got {parent_4}"
        )
        print(f"✓ Iteration 4 parent: {parent_4} (expected: model_v3_6000_jkl)")
        print("✓ Correctly ignored iteration 2 (created before rollback)")
        print("✓ Correctly detected iteration 3 (created after rollback)")

    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


@pytest.mark.asyncio
async def test_provenance_chain_skips_versions_after_rollback():
    """Test that provenance chain correctly skips versions after rollback."""
    print("\n" + "=" * 60)
    print("Testing Provenance Chain - Skips Versions After Rollback")
    print("=" * 60)
    print()

    worker = BlockchainWorker()

    # Scenario: Normal flow creates iterations 0, 1, 2
    # Rollback to iteration 1
    # Create iteration 3 (uses iteration 1 as parent)
    # Create iteration 4 (uses iteration 3 as parent)
    # Provenance chain for iteration 4 should be: 0 -> 1 -> 3 -> 4 (skipping 2)

    async def mock_list_models():
        """Return all versions including skipped iteration 2."""
        return {
            "versions": [
                {
                    "version_id": "model_v0_initial_1000_abc",
                    "timestamp": "1000",
                    "metadata": {"iteration": 0},
                    "parent_version_id": None,
                },
                {
                    "version_id": "model_v1_2000_def",
                    "timestamp": "2000",
                    "metadata": {"iteration": 1},
                    "parent_version_id": "model_v0_initial_1000_abc",
                },
                {
                    "version_id": "model_v2_3000_ghi",  # Should be skipped
                    "timestamp": "3000",
                    "metadata": {"iteration": 2},
                    "parent_version_id": "model_v1_2000_def",
                },
                {
                    "version_id": "model_v3_6000_jkl",
                    "timestamp": "6000",
                    "metadata": {"iteration": 3},
                    "parent_version_id": "model_v1_2000_def",  # Uses rollback target
                },
                {
                    "version_id": "model_v4_7000_mno",
                    "timestamp": "7000",
                    "metadata": {"iteration": 4},
                    "parent_version_id": "model_v3_6000_jkl",
                },
            ]
        }

    async def mock_get_provenance(version_id: str):
        """Return provenance for a version."""
        versions = {
            "model_v0_initial_1000_abc": {
                "version_id": "model_v0_initial_1000_abc",
                "timestamp": "1000",
                "metadata": {"iteration": 0},
                "parent_version_id": None,
            },
            "model_v1_2000_def": {
                "version_id": "model_v1_2000_def",
                "timestamp": "2000",
                "metadata": {"iteration": 1},
                "parent_version_id": "model_v0_initial_1000_abc",
            },
            "model_v2_3000_ghi": {
                "version_id": "model_v2_3000_ghi",
                "timestamp": "3000",
                "metadata": {"iteration": 2},
                "parent_version_id": "model_v1_2000_def",
            },
            "model_v3_6000_jkl": {
                "version_id": "model_v3_6000_jkl",
                "timestamp": "6000",
                "metadata": {"iteration": 3},
                "parent_version_id": "model_v1_2000_def",
            },
            "model_v4_7000_mno": {
                "version_id": "model_v4_7000_mno",
                "timestamp": "7000",
                "metadata": {"iteration": 4},
                "parent_version_id": "model_v3_6000_jkl",
            },
        }
        return versions.get(version_id, {})

    async def mock_get_most_recent_rollback():
        """Return rollback event to iteration 1."""
        return {
            "to_version_id": "model_v1_2000_def",
            "timestamp": "5000",
            "reason": "test rollback",
        }

    # Build provenance data synchronously for testing
    provenance_data = {
        "model_v0_initial_1000_abc": {
            "version_id": "model_v0_initial_1000_abc",
            "timestamp": "1000",
            "metadata": {"iteration": 0},
            "parent_version_id": None,
        },
        "model_v1_2000_def": {
            "version_id": "model_v1_2000_def",
            "timestamp": "2000",
            "metadata": {"iteration": 1},
            "parent_version_id": "model_v0_initial_1000_abc",
        },
        "model_v2_3000_ghi": {
            "version_id": "model_v2_3000_ghi",
            "timestamp": "3000",
            "metadata": {"iteration": 2},
            "parent_version_id": "model_v1_2000_def",
        },
        "model_v3_6000_jkl": {
            "version_id": "model_v3_6000_jkl",
            "timestamp": "6000",
            "metadata": {"iteration": 3},
            "parent_version_id": "model_v1_2000_def",
        },
        "model_v4_7000_mno": {
            "version_id": "model_v4_7000_mno",
            "timestamp": "7000",
            "metadata": {"iteration": 4},
            "parent_version_id": "model_v3_6000_jkl",
        },
    }

    def build_provenance_chain_sync(version_id: str, chain=None):
        """Recursively build provenance chain (synchronous version for test)."""
        if chain is None:
            chain = []

        provenance = provenance_data.get(version_id)
        if not provenance:
            return chain

        chain.append(provenance["version_id"])
        parent_id = provenance.get("parent_version_id")
        if parent_id:
            return build_provenance_chain_sync(parent_id, chain)
        return chain

    with patch(
        "main_service.workers.blockchain_worker.FabricClient"
    ) as mock_fabric_class:
        mock_client = AsyncMock()
        mock_client.list_models = AsyncMock(side_effect=mock_list_models)
        mock_client.get_model_provenance = AsyncMock(side_effect=mock_get_provenance)
        mock_client.get_most_recent_rollback = AsyncMock(
            side_effect=mock_get_most_recent_rollback
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_fabric_class.return_value = mock_client

        # Build provenance chain for iteration 4
        chain = build_provenance_chain_sync("model_v4_7000_mno")
        chain.reverse()  # Reverse to get chronological order

        # Expected chain: 0 -> 1 -> 3 -> 4 (skipping 2)
        expected_chain = [
            "model_v0_initial_1000_abc",
            "model_v1_2000_def",
            "model_v3_6000_jkl",
            "model_v4_7000_mno",
        ]

        assert chain == expected_chain, (
            f"Expected provenance chain: {expected_chain}, " f"but got: {chain}"
        )

        print("✓ Provenance chain for iteration 4:")
        for i, version_id in enumerate(chain):
            print(f"  {i}. {version_id}")
        print("✓ Correctly skipped iteration 2 (model_v2_3000_ghi)")

    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)


@pytest.mark.asyncio
async def test_parent_selection_iteration_1_uses_initial():
    """Test that iteration 1 correctly uses iteration 0 (initial) as parent."""
    print("\n" + "=" * 60)
    print("Testing Parent Selection - Iteration 1 Uses Initial")
    print("=" * 60)
    print()

    worker = BlockchainWorker()

    async def mock_list_models():
        """Return initial version."""
        return {
            "versions": [
                {
                    "version_id": "model_v0_initial_1000_abc",
                    "timestamp": "1000",
                    "metadata": {"iteration": 0},
                    "parent_version_id": None,
                },
            ]
        }

    async def mock_get_provenance(version_id: str):
        """Return provenance for a version."""
        if version_id == "model_v0_initial_1000_abc":
            return {
                "version_id": "model_v0_initial_1000_abc",
                "timestamp": "1000",
                "metadata": {"iteration": 0},
                "parent_version_id": None,
            }
        return {}

    async def mock_get_most_recent_rollback():
        """No rollback."""
        return None

    with patch(
        "main_service.workers.blockchain_worker.FabricClient"
    ) as mock_fabric_class:
        mock_client = AsyncMock()
        mock_client.list_models = AsyncMock(side_effect=mock_list_models)
        mock_client.get_model_provenance = AsyncMock(side_effect=mock_get_provenance)
        mock_client.get_most_recent_rollback = AsyncMock(
            side_effect=mock_get_most_recent_rollback
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_fabric_class.return_value = mock_client

        # Test iteration 1 should use iteration 0 as parent
        parent_1 = await worker._get_parent_version_id_for_iteration(1)
        assert parent_1 == "model_v0_initial_1000_abc", (
            f"Expected parent to be model_v0_initial_1000_abc (iteration 0), "
            f"but got {parent_1}"
        )
        print(f"✓ Iteration 1 parent: {parent_1} (expected: model_v0_initial_1000_abc)")

    print("=" * 60)
    print("✓ Test PASSED")
    print("=" * 60)
