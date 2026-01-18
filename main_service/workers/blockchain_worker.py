"""Blockchain worker that records model updates on blockchain."""

import asyncio
import json
import time
import uuid
from typing import Optional, Dict, Any, cast
from shared.queue.consumer import QueueConsumer
from shared.queue.publisher import QueuePublisher
from shared.queue.connection import QueueConnection
from shared.models.task import (
    Task,
    TaskType,
    TaskMetadata,
    BlockchainWriteTaskPayload,
    StorageWriteTaskPayload,
)
from main_service.blockchain.fabric_client import FabricClient
from shared.utils.hashing import compute_hash
from shared.logger import setup_logger
from shared.monitoring.metrics import get_metrics_collector
from shared.storage.ipfs_client import IPFSClient

logger = setup_logger(__name__)


class BlockchainWorker:
    """Worker that records model updates on blockchain via blockchain service API."""

    def __init__(self, connection: Optional[QueueConnection] = None):
        """
        Initialize blockchain worker.

        Args:
            connection: QueueConnection instance (creates new one if None)
        """
        self.connection = connection or QueueConnection()
        self.consumer = QueueConsumer(connection=self.connection)
        self.publisher = QueuePublisher(connection=self.connection)
        self.running = False
        # Track current model version for parent_version_id
        self.current_model_version_id: Optional[str] = None
        # Track latest model version ID (alias for current_model_version_id)
        self.latest_model_version_id: Optional[str] = None

        logger.info("Blockchain worker initialized")

    async def _get_most_recent_rollback_target(
        self, blockchain_client: FabricClient
    ) -> Optional[tuple[str, int]]:
        """
        Get the most recent rollback target version ID and timestamp from the blockchain.

        Queries rollback events directly from the blockchain to find the most recent
        rollback and returns the target version ID and rollback timestamp.

        Args:
            blockchain_client: FabricClient instance

        Returns:
            Tuple of (most recent rollback target version ID, rollback timestamp),
            or None if no rollback detected
        """
        try:
            rollback_event = await blockchain_client.get_most_recent_rollback()

            if rollback_event is None:
                # No rollback events found
                logger.debug("No rollback events found on blockchain")
                return None

            logger.debug(f"Retrieved rollback event: {rollback_event}")

            # Extract the target version ID from the rollback event
            # The chaincode stores it as ToVersionID
            target_version_id = rollback_event.get(
                "to_version_id"
            ) or rollback_event.get("target_version_id")

            # Extract rollback timestamp
            rollback_timestamp_str = rollback_event.get("timestamp", "0")
            try:
                rollback_timestamp = int(rollback_timestamp_str)
            except (ValueError, TypeError):
                logger.warning(
                    f"Invalid rollback timestamp: {rollback_timestamp_str}. Using 0."
                )
                rollback_timestamp = 0

            if target_version_id:
                logger.info(
                    f"Found most recent rollback target: {target_version_id} "
                    f"(timestamp: {rollback_timestamp}, "
                    f"reason: {rollback_event.get('reason')})"
                )
                return (cast(str, target_version_id), rollback_timestamp)
            else:
                logger.warning(
                    f"Rollback event found but no target_version_id: {rollback_event}"
                )

            return None

        except Exception as e:
            logger.warning(
                f"Failed to get most recent rollback target: {str(e)}",
                exc_info=True,
            )
            return None

    async def _get_parent_version_id_for_iteration(
        self, iteration: int
    ) -> Optional[str]:
        """
        Get the parent version ID for a given iteration.

        After rollback, the parent should be the version that was rolled back to,
        not necessarily the version with iteration = (current_iteration - 1).

        Strategy:
        1. Check if there was a recent rollback (by finding the "head" of active chain)
        2. If rollback detected, use the rolled-back version as parent
        3. Otherwise, use normal logic (iteration N-1)

        Args:
            iteration: Current iteration number

        Returns:
            Parent version ID, or None if this is iteration 0 (initial version)
        """
        if iteration <= 0:
            # Iteration 0 (initial version) has no parent
            return None

        if iteration == 1:
            # Iteration 1 should have iteration 0 (initial version) as parent
            # Find the initial version (iteration 0)
            try:
                async with FabricClient() as blockchain_client:
                    models_response = await blockchain_client.list_models()
                    versions = models_response.get("versions", [])

                    # Find version with iteration 0
                    for version in versions:
                        metadata = version.get("metadata", {})
                        if not isinstance(metadata, dict):
                            metadata = {}
                        version_iteration = metadata.get("iteration")
                        if version_iteration is None:
                            version_iteration = version.get("iteration")

                        if version_iteration is not None:
                            try:
                                iter_num = int(version_iteration)
                                if iter_num == 0:
                                    initial_version_id = version.get("version_id")
                                    if initial_version_id:
                                        logger.info(
                                            f"Found initial version {initial_version_id} as parent for iteration 1"
                                        )
                                        return cast(str, initial_version_id)
                            except (ValueError, TypeError):
                                continue

                    # Also check if version_id starts with "model_v0_initial"
                    for version in versions:
                        version_id = version.get("version_id", "")
                        if version_id.startswith("model_v0_initial"):
                            logger.info(
                                f"Found initial version {version_id} (by ID pattern) as parent for iteration 1"
                            )
                            return cast(str, version_id)

                    logger.warning(
                        "No initial version (iteration 0) found on blockchain. "
                        "Iteration 1 will have no parent."
                    )
                    return None
            except Exception as e:
                logger.warning(
                    f"Failed to find initial version for iteration 1: {str(e)}. "
                    "Iteration 1 will have no parent."
                )
                return None

        try:
            # Query blockchain to find parent version
            async with FabricClient() as blockchain_client:
                # First, check if there was a recent rollback
                # Find the "head" of the active chain (version with no children)
                rollback_info = await self._get_most_recent_rollback_target(
                    blockchain_client
                )

                if rollback_info:
                    rollback_target, rollback_timestamp = rollback_info
                    # Check if this is the first iteration after rollback
                    # If a version already exists with rollback_target as parent, use normal logic
                    # Otherwise, use rollback_target as parent (first iteration after rollback)
                    try:
                        # Check if any version has rollback_target as parent
                        models_response = await blockchain_client.list_models()
                        versions = models_response.get("versions", [])

                        # Get rollback iteration to check for post-rollback children
                        rollback_provenance = (
                            await blockchain_client.get_model_provenance(
                                rollback_target
                            )
                        )
                        rollback_metadata = rollback_provenance.get("metadata", {})
                        if not isinstance(rollback_metadata, dict):
                            rollback_metadata = {}
                        rollback_iteration = rollback_metadata.get("iteration")
                        if rollback_iteration is None:
                            rollback_iteration = rollback_provenance.get("iteration")

                        if rollback_iteration is None:
                            logger.warning(
                                f"Rollback target {rollback_target} has no iteration. "
                                "Using it as parent anyway."
                            )
                            return rollback_target

                        rollback_iter_num = int(rollback_iteration)

                        # Check if there's a version with rollback_target as parent that was created AFTER the rollback
                        # We do this by comparing version timestamps to the rollback timestamp
                        rollback_target_has_post_rollback_children = False
                        for version in versions:
                            parent_id = version.get("parent_version_id")
                            if parent_id == rollback_target:
                                # Check if this version was created after the rollback
                                version_timestamp_str = version.get("timestamp", "0")
                                try:
                                    version_timestamp = int(version_timestamp_str)
                                    if version_timestamp > rollback_timestamp:
                                        # This version was created after rollback
                                        version_metadata = version.get("metadata", {})
                                        if not isinstance(version_metadata, dict):
                                            version_metadata = {}
                                        version_iteration = version_metadata.get(
                                            "iteration"
                                        )
                                        if version_iteration is None:
                                            version_iteration = version.get("iteration")

                                        if version_iteration is not None:
                                            try:
                                                version_iter_num = int(
                                                    version_iteration
                                                )
                                                if version_iter_num > rollback_iter_num:
                                                    # This is a post-rollback version
                                                    rollback_target_has_post_rollback_children = (
                                                        True
                                                    )
                                                    logger.debug(
                                                        f"Found post-rollback version: {version.get('version_id')} "
                                                        f"(iteration {version_iter_num}, timestamp {version_timestamp}) "
                                                        f"with rollback target as parent (rollback timestamp: {rollback_timestamp})"
                                                    )
                                                    break
                                            except (ValueError, TypeError):
                                                continue
                                except (ValueError, TypeError):
                                    continue

                        if not rollback_target_has_post_rollback_children:
                            # No version created after rollback yet
                            # This is the first iteration after rollback - use rollback target
                            logger.info(
                                f"First iteration after rollback: Using rolled-back version {rollback_target} "
                                f"(iteration {rollback_iter_num}) as parent for iteration {iteration}"
                            )
                            return rollback_target
                        else:
                            # A version already exists with rollback_target as parent and iteration > rollback_iteration
                            # Use normal parent selection (iteration N-1)
                            logger.debug(
                                f"Rollback target {rollback_target} already has post-rollback children. "
                                f"Using normal parent selection for iteration {iteration}."
                            )
                    except Exception as e:
                        # If we can't check, be safe and use rollback target
                        # (better to use rollback target than wrong parent)
                        logger.warning(
                            f"Failed to check if rollback target has children: {str(e)}. "
                            f"Using rollback target {rollback_target} as parent."
                        )
                        return rollback_target
                else:
                    logger.debug(
                        f"No rollback target found for iteration {iteration}. "
                        "Using normal parent selection."
                    )

                models_response = await blockchain_client.list_models()
                versions = models_response.get("versions", [])

                # Extract iterations from all versions
                version_data = []
                for version in versions:
                    metadata = version.get("metadata", {})
                    if not isinstance(metadata, dict):
                        metadata = {}
                    version_iteration = metadata.get("iteration")
                    if version_iteration is None:
                        version_iteration = version.get("iteration")

                    if version_iteration is not None:
                        try:
                            iter_num = int(version_iteration)
                            timestamp = int(version.get("timestamp", "0") or "0")
                            version_data.append(
                                {
                                    "version_id": version.get("version_id"),
                                    "iteration": iter_num,
                                    "timestamp": timestamp,
                                }
                            )
                        except (ValueError, TypeError):
                            continue

                if not version_data:
                    # Fallback to self.current_model_version_id if no versions found
                    logger.warning(
                        f"No versions found on blockchain. "
                        f"Using current_model_version_id as fallback: {self.current_model_version_id}"
                    )
                    return self.current_model_version_id

                # Strategy: After rollback, we want to use the rolled-back version as parent,
                # not necessarily the version with iteration = (current_iteration - 1)
                #
                # Find all versions with iteration < current_iteration
                # Then find the one that should be the parent based on the actual chain
                candidates = [v for v in version_data if v["iteration"] < iteration]

                if not candidates:
                    # No valid parent found, fallback
                    logger.warning(
                        f"No version found with iteration < {iteration} on blockchain. "
                        f"Using current_model_version_id as fallback: {self.current_model_version_id}"
                    )
                    return cast(Optional[str], self.current_model_version_id)

                candidates.sort(
                    key=lambda v: (v["iteration"], v["timestamp"]), reverse=True
                )

                highest_iter_candidate = candidates[0]

                parent_iteration = iteration - 1
                parent_candidates = [
                    v for v in candidates if v["iteration"] == parent_iteration
                ]

                if parent_candidates:
                    parent_candidates.sort(key=lambda v: v["timestamp"], reverse=True)
                    candidate_version_id = parent_candidates[0]["version_id"]

                    if rollback_target_has_post_rollback_children:
                        # Post-rollback children exist, use normal parent selection
                        parent_version_id = candidate_version_id
                        parent_iteration = parent_candidates[0]["iteration"]
                        logger.info(
                            f"Using normal parent selection: {parent_version_id} "
                            f"(iteration {parent_iteration}) for iteration {iteration}"
                        )
                    else:
                        # Check if candidate (iteration N-1) has any children
                        # If no children exist, it might have been rolled back from
                        candidate_has_children = False
                        try:
                            # Check all versions to see if any have candidate as parent
                            for version in versions:
                                parent_id = version.get("parent_version_id")
                                if parent_id == candidate_version_id:
                                    candidate_has_children = True
                                    break
                        except Exception:
                            pass

                        candidate_iteration = parent_candidates[0]["iteration"]

                        # Check if candidate's parent matches expected (iteration N-2)
                        is_skipping_candidate = False
                        rollback_parent_id = None
                        if not candidate_has_children:
                            try:
                                candidate_provenance = (
                                    await blockchain_client.get_model_provenance(
                                        candidate_version_id
                                    )
                                )
                                candidate_parent_id = candidate_provenance.get(
                                    "parent_version_id"
                                )
                                if candidate_parent_id:
                                    # Get candidate parent's iteration
                                    candidate_parent_provenance = (
                                        await blockchain_client.get_model_provenance(
                                            candidate_parent_id
                                        )
                                    )
                                    candidate_parent_metadata = (
                                        candidate_parent_provenance.get("metadata", {})
                                    )
                                    if not isinstance(candidate_parent_metadata, dict):
                                        candidate_parent_metadata = {}
                                    candidate_parent_iter = (
                                        candidate_parent_metadata.get("iteration")
                                    )
                                    if candidate_parent_iter is None:
                                        candidate_parent_iter = (
                                            candidate_parent_provenance.get("iteration")
                                        )

                                    expected_parent_iter = candidate_iteration - 1
                                    # Only skip if candidate's parent is NOT the expected iteration
                                    if (
                                        candidate_parent_iter is not None
                                        and int(candidate_parent_iter)
                                        != expected_parent_iter
                                    ):
                                        # Parent doesn't match expected - this suggests rollback
                                        parent_in_candidates = any(
                                            v["version_id"] == candidate_parent_id
                                            for v in candidates
                                        )
                                        if parent_in_candidates:
                                            # Use parent instead of candidate
                                            is_skipping_candidate = True
                                            rollback_parent_id = candidate_parent_id
                            except Exception:
                                pass

                        if is_skipping_candidate and rollback_parent_id:
                            # Rollback detected: candidate was rolled back from
                            try:
                                # Use the rolled-back version as parent
                                rollback_parent_provenance = (
                                    await blockchain_client.get_model_provenance(
                                        rollback_parent_id
                                    )
                                )
                                rollback_parent_metadata = (
                                    rollback_parent_provenance.get("metadata", {})
                                )
                                if not isinstance(rollback_parent_metadata, dict):
                                    rollback_parent_metadata = {}
                                rollback_parent_iter = rollback_parent_metadata.get(
                                    "iteration"
                                )
                                if rollback_parent_iter is None:
                                    rollback_parent_iter = (
                                        rollback_parent_provenance.get("iteration")
                                    )

                                if rollback_parent_iter is not None:
                                    # Verify this version exists in our candidates
                                    rollback_candidates = [
                                        v
                                        for v in candidates
                                        if v["version_id"] == rollback_parent_id
                                    ]
                                    if rollback_candidates:
                                        parent_version_id = rollback_parent_id
                                        parent_iteration = int(rollback_parent_iter)
                                        logger.info(
                                            f"Rollback detected: Using rolled-back version {parent_version_id} "
                                            f"(iteration {parent_iteration}) as parent for iteration {iteration} "
                                            f"(iteration {candidate_iteration} exists but was rolled back from)"
                                        )
                                    else:
                                        # Rollback parent not in candidates, fall through to normal logic
                                        is_skipping_candidate = False
                                else:
                                    is_skipping_candidate = False
                            except Exception as e:
                                logger.warning(
                                    f"Failed to check rollback for {candidate_version_id}: {str(e)}. "
                                    f"Using normal parent selection."
                                )
                                is_skipping_candidate = False

                    if not is_skipping_candidate:
                        # Normal case: use iteration N-1 as parent
                        try:
                            candidate_provenance = (
                                await blockchain_client.get_model_provenance(
                                    candidate_version_id
                                )
                            )
                            candidate_parent_id = candidate_provenance.get(
                                "parent_version_id"
                            )

                            if candidate_parent_id:
                                # Get candidate parent's iteration
                                candidate_parent_provenance = (
                                    await blockchain_client.get_model_provenance(
                                        candidate_parent_id
                                    )
                                )
                                candidate_parent_metadata = (
                                    candidate_parent_provenance.get("metadata", {})
                                )
                                if not isinstance(candidate_parent_metadata, dict):
                                    candidate_parent_metadata = {}
                                candidate_parent_iter = candidate_parent_metadata.get(
                                    "iteration"
                                )
                                if candidate_parent_iter is None:
                                    candidate_parent_iter = (
                                        candidate_parent_provenance.get("iteration")
                                    )

                                expected_parent_iter = parent_iteration - 1
                                if (
                                    candidate_parent_iter is not None
                                    and int(candidate_parent_iter)
                                    == expected_parent_iter
                                ):
                                    # Candidate is in the active chain, use it
                                    parent_version_id = candidate_version_id
                                    parent_iteration = parent_candidates[0]["iteration"]
                                else:
                                    # Candidate's parent doesn't match expected (chain broken)
                                    rollback_parent_candidates = [
                                        v
                                        for v in candidates
                                        if v["iteration"] == expected_parent_iter
                                    ]
                                    if rollback_parent_candidates:
                                        rollback_parent_candidates.sort(
                                            key=lambda v: v["timestamp"], reverse=True
                                        )
                                        parent_version_id = rollback_parent_candidates[
                                            0
                                        ]["version_id"]
                                        parent_iteration = rollback_parent_candidates[
                                            0
                                        ]["iteration"]
                                        logger.info(
                                            f"Chain broken: Using version {parent_version_id} "
                                            f"(iteration {parent_iteration}) as parent for iteration {iteration} "
                                            f"(iteration {parent_iteration + 1} exists but parent chain is broken)"
                                        )
                                    else:
                                        # Fallback to candidate
                                        parent_version_id = candidate_version_id
                                        parent_iteration = parent_candidates[0][
                                            "iteration"
                                        ]
                            else:
                                # Candidate has no parent (shouldn't happen for iteration > 1)
                                parent_version_id = candidate_version_id
                                parent_iteration = parent_candidates[0]["iteration"]
                        except Exception as e:
                            logger.warning(
                                f"Failed to verify parent chain for {candidate_version_id}: {str(e)}. "
                                f"Using it as parent anyway."
                            )
                            parent_version_id = candidate_version_id
                            parent_iteration = parent_candidates[0]["iteration"]
                else:
                    # No version with parent_iteration found (likely after rollback)
                    parent_version_id = highest_iter_candidate["version_id"]
                    parent_iteration = highest_iter_candidate["iteration"]

                    if parent_iteration == 0 and iteration > 1:
                        # This is suspicious - we're using initial version as parent for iteration > 1
                        # This suggests rollback detection failed
                        logger.error(
                            f"WARNING: Using initial version (iteration 0) as parent for iteration {iteration}. "
                            f"This suggests rollback detection failed. "
                            f"Please check if rollback events are being stored correctly on the blockchain."
                        )
                    else:
                        logger.info(
                            f"After rollback: Using version {parent_version_id} (iteration {parent_iteration}) "
                            f"as parent for iteration {iteration} "
                            f"(expected iteration {iteration - 1} not found)"
                        )

                if parent_version_id != self.current_model_version_id:
                    logger.info(
                        f"Found parent version {parent_version_id} (iteration {parent_iteration}) "
                        f"for iteration {iteration}. "
                        f"Updating current_model_version_id from {self.current_model_version_id}."
                    )
                    # Update current_model_version_id to keep it in sync
                    self.current_model_version_id = parent_version_id
                    self.latest_model_version_id = parent_version_id

                return cast(str, parent_version_id) if parent_version_id else None

        except Exception as e:
            logger.warning(
                f"Failed to find parent version for iteration {iteration}: {str(e)}. "
                f"Using current_model_version_id as fallback: {self.current_model_version_id}"
            )
            return self.current_model_version_id

    def _generate_model_version_id(self, iteration: int) -> str:
        """
        Generate a unique model version ID.

        Args:
            iteration: Training iteration number

        Returns:
            Unique version ID
        """
        timestamp = int(time.time())
        unique_id = str(uuid.uuid4())[:8]
        return f"model_v{iteration}_{timestamp}_{unique_id}"

    def _compute_diff_hash(self, aggregated_diff_str: str) -> str:
        """
        Compute hash of aggregated diff (before encryption).

        This hash represents the integrity of the diff content and will be
        stored on blockchain. The storage worker will encrypt the diff and
        store it on IPFS. The blockchain hash serves as a content integrity
        check (verifying the diff hasn't been tampered with).

        Args:
            aggregated_diff_str: Aggregated diff as JSON string

        Returns:
            Hash of unencrypted diff (SHA-256)
        """
        # Convert to bytes
        diff_bytes = aggregated_diff_str.encode("utf-8")

        # Compute hash of unencrypted diff
        diff_hash = compute_hash(diff_bytes)

        logger.debug(f"Computed hash of aggregated diff: {diff_hash[:16]}...")
        return diff_hash

    async def _register_on_blockchain(
        self,
        model_version_id: str,
        parent_version_id: Optional[str],
        blockchain_hash: str,
        metadata: Dict[str, Any],
    ) -> str:
        """
        Register model update on blockchain via blockchain service API.

        Args:
            model_version_id: Unique version identifier
            parent_version_id: Parent version ID (None for initial version)
            blockchain_hash: Hash of encrypted diff
            metadata: Additional metadata (iteration, num_clients, client_ids, etc.)

        Returns:
            Transaction ID from blockchain service
        """
        start_time = time.time()
        logger.info(
            f"Registering model version {model_version_id} on blockchain "
            f"(parent: {parent_version_id})"
        )

        try:
            async with FabricClient() as blockchain_client:
                transaction_id = await blockchain_client.register_model_update(
                    model_version_id=model_version_id,
                    parent_version_id=parent_version_id,
                    hash_value=blockchain_hash,
                    metadata=metadata,
                )

            # Type assertion: register_model_update returns str (validated in fabric_client)
            if not isinstance(transaction_id, str):
                raise ValueError(f"Expected str, got {type(transaction_id)}")

            duration = time.time() - start_time
            get_metrics_collector().record_timing(
                "blockchain_register",
                duration,
                metadata={
                    "model_version_id": model_version_id,
                    "iteration": metadata.get("iteration"),
                    "transaction_id": transaction_id,
                },
            )

            logger.info(
                f"Model version {model_version_id} registered on blockchain: "
                f"tx_id={transaction_id} (duration: {duration:.3f}s)"
            )
            return transaction_id
        except Exception as e:
            duration = time.time() - start_time
            get_metrics_collector().record_timing(
                "blockchain_register",
                duration,
                metadata={
                    "model_version_id": model_version_id,
                    "status": "error",
                    "error": str(e),
                },
            )
            raise

    async def _process_blockchain_write(
        self,
        aggregated_diff_str: str,
        iteration: int,
        num_clients: int,
        client_ids: list,
    ) -> Dict[str, Any]:
        """
        Process blockchain write operation.

        Args:
            aggregated_diff_str: Aggregated diff as JSON string
            iteration: Training iteration number
            num_clients: Number of clients that participated
            client_ids: List of client IDs that contributed

        Returns:
            Dictionary with model_version_id, parent_version_id, blockchain_hash, transaction_id
        """
        # Generate model version ID
        model_version_id = self._generate_model_version_id(iteration)

        # Get parent version ID
        # After rollback, we need to find the correct parent based on iteration
        # Parent should be the latest version with iteration = (current_iteration - 1)
        parent_version_id = await self._get_parent_version_id_for_iteration(iteration)

        # Compute hash of unencrypted diff (content integrity check)
        blockchain_hash = self._compute_diff_hash(aggregated_diff_str)

        # Get parent metadata to carry forward rollback_count and validation_history
        parent_metadata = {}
        rollback_count = 0
        validation_history = []

        if parent_version_id:
            try:
                async with FabricClient() as blockchain_client:
                    parent_provenance = await blockchain_client.get_model_provenance(
                        parent_version_id
                    )
                    parent_metadata = parent_provenance.get("metadata", {})
                    rollback_count = parent_metadata.get("rollback_count", 0)
                    validation_history = parent_metadata.get("validation_history", [])

                    # Check if this is a version created after a rollback
                    # If parent's iteration is less than current iteration - 1, it's a rollback
                    parent_iteration = parent_metadata.get("iteration")
                    if parent_iteration is None:
                        parent_iteration = parent_provenance.get("iteration")

                    if parent_iteration is not None:
                        parent_iter_num = int(parent_iteration)
                        # If parent iteration is less than expected (iteration - 1), rollback happened
                        if parent_iter_num < iteration - 1:
                            # Increment rollback_count for this new version after rollback
                            rollback_count += 1
                            logger.info(
                                f"Rollback detected: parent iteration {parent_iter_num} < expected {iteration - 1}. "
                                f"Incrementing rollback_count to {rollback_count}"
                            )
            except Exception as e:
                logger.warning(
                    f"Could not fetch parent metadata for {parent_version_id}: {e}. "
                    "Starting with fresh metadata."
                )

        # Prepare metadata for blockchain
        # Carry forward rollback_count and validation_history from parent
        metadata = {
            "iteration": iteration,
            "num_clients": num_clients,
            "client_ids": client_ids,
            "diff_hash": blockchain_hash,  # Hash of encrypted diff
            "rollback_count": rollback_count,  # Carried forward from parent (incremented if rollback)
            "validation_history": validation_history,  # Carried forward from parent
            # ipfs_cid will be added later by storage worker
        }

        # Register on blockchain
        transaction_id = await self._register_on_blockchain(
            model_version_id=model_version_id,
            parent_version_id=parent_version_id,
            blockchain_hash=blockchain_hash,
            metadata=metadata,
        )

        # Update current and latest model version
        self.current_model_version_id = model_version_id
        self.latest_model_version_id = model_version_id

        return {
            "model_version_id": model_version_id,
            "parent_version_id": parent_version_id,
            "blockchain_hash": blockchain_hash,
            "transaction_id": transaction_id,
        }

    def _publish_storage_task(
        self,
        aggregated_diff_cid: str,
        blockchain_hash: str,
        model_version_id: str,
        parent_version_id: Optional[str],
    ) -> None:
        """
        Publish STORAGE_WRITE task to queue.

        Args:
            aggregated_diff_cid: IPFS CID of aggregated diff (not the diff itself)
            blockchain_hash: Hash from blockchain transaction
            model_version_id: Model version identifier
            parent_version_id: Parent version ID
        """
        # Create payload - ensure we only include the CID, not the actual diff
        payload_dict = StorageWriteTaskPayload(
            aggregated_diff_cid=aggregated_diff_cid,
            blockchain_hash=blockchain_hash,
            model_version_id=model_version_id,
        ).model_dump()

        # Verify payload doesn't contain large data
        payload_size = len(json.dumps(payload_dict).encode("utf-8"))
        if payload_size > 10000:  # More than 10KB is suspicious
            logger.error(
                f"WARNING: Storage task payload is suspiciously large: {payload_size} bytes. "
                f"Payload keys: {list(payload_dict.keys())}"
            )

        storage_task = Task(
            task_id=f"storage-{model_version_id}-{int(time.time())}",
            task_type=TaskType.STORAGE_WRITE,
            payload=payload_dict,
            metadata=TaskMetadata(source="blockchain_worker"),
            model_version_id=model_version_id,
            parent_version_id=parent_version_id,
        )

        queue_name = "storage_write"

        # Log message size before publishing
        task_dict = storage_task.to_dict()
        message_size = len(json.dumps(task_dict).encode("utf-8"))
        logger.debug(
            f"Publishing STORAGE_WRITE task: message_size={message_size} bytes, "
            f"payload_size={payload_size} bytes, aggregated_diff_cid={aggregated_diff_cid}"
        )

        if message_size > 100000:  # More than 100KB is definitely wrong
            logger.error(
                f"ERROR: Task message is too large: {message_size} bytes! "
                f"This should only contain metadata and CIDs. "
                f"Task payload: {json.dumps(payload_dict, indent=2)[:500]}"
            )
            raise ValueError(
                f"Task message too large ({message_size} bytes). "
                f"Only metadata and IPFS CIDs should be sent through RabbitMQ."
            )

        self.publisher.publish_task(storage_task, queue_name)

        logger.info(
            f"Published STORAGE_WRITE task for version {model_version_id} "
            f"to queue '{queue_name}'"
        )

    def _handle_blockchain_write_task(self, task: Task) -> bool:
        """
        Handle a BLOCKCHAIN_WRITE task.

        Args:
            task: BLOCKCHAIN_WRITE task to process

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Processing BLOCKCHAIN_WRITE task: {task.task_id}")

            # Parse payload (not using BlockchainWriteTaskPayload since aggregation worker
            # publishes a simplified payload)
            payload = task.payload

            # Create event loop once for all async operations
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                # No event loop in current thread, create new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Get aggregated diff CID from IPFS (aggregation worker uploads it to avoid frame size limits)
            aggregated_diff_cid = payload.get("aggregated_diff_cid")
            if not aggregated_diff_cid:
                raise ValueError(
                    "Missing 'aggregated_diff_cid' in payload. "
                    "Only IPFS CIDs should be sent through RabbitMQ."
                )

            # Download aggregated diff from IPFS (needed for hash computation)
            logger.info(
                f"Downloading aggregated diff from IPFS: CID={aggregated_diff_cid}"
            )

            async def download_from_ipfs():
                async with IPFSClient() as ipfs_client:
                    aggregated_diff_bytes = await ipfs_client.get_bytes(
                        aggregated_diff_cid
                    )
                    return aggregated_diff_bytes.decode("utf-8")

            aggregated_diff_str = loop.run_until_complete(download_from_ipfs())
            logger.info(
                f"Downloaded aggregated diff from IPFS: CID={aggregated_diff_cid}, "
                f"size={len(aggregated_diff_str.encode('utf-8'))} bytes"
            )

            iteration = payload.get("iteration")
            if iteration is None:
                raise ValueError("Missing 'iteration' in payload")

            num_clients = payload.get("num_clients", 0)
            client_ids = payload.get("client_ids", [])

            # Run async blockchain processing (reuse the same event loop)
            result = loop.run_until_complete(
                self._process_blockchain_write(
                    aggregated_diff_str=aggregated_diff_str,
                    iteration=iteration,
                    num_clients=num_clients,
                    client_ids=client_ids,
                )
            )

            # Publish STORAGE_WRITE task with IPFS CID (not the diff itself)
            # The storage worker will download from IPFS using this CID
            self._publish_storage_task(
                aggregated_diff_cid=aggregated_diff_cid,
                blockchain_hash=result["blockchain_hash"],
                model_version_id=result["model_version_id"],
                parent_version_id=result["parent_version_id"],
            )

            logger.info(
                f"BLOCKCHAIN_WRITE task completed: "
                f"version={result['model_version_id']}, "
                f"tx_id={result['transaction_id']}"
            )
            return True

        except Exception as e:
            logger.error(
                f"Error processing BLOCKCHAIN_WRITE task {task.task_id}: {str(e)}",
                exc_info=True,
            )
            return False

    def start(self, queue_name: str = "blockchain_write") -> None:
        """
        Start blockchain worker.

        Args:
            queue_name: Queue name to consume from
        """
        logger.info(f"Starting blockchain worker (consuming from {queue_name})")
        self.running = True

        def task_handler(task: Task) -> None:
            """Handle incoming tasks."""
            if task.task_type == TaskType.BLOCKCHAIN_WRITE:
                success = self._handle_blockchain_write_task(task)
                if not success:
                    logger.error(
                        f"Failed to process BLOCKCHAIN_WRITE task {task.task_id}"
                    )
            else:
                logger.warning(
                    f"Received unexpected task type: {task.task_type} "
                    f"(expected BLOCKCHAIN_WRITE)"
                )

        self.consumer.consume_tasks(queue_name, task_handler)

    def get_latest_model_version_id(self) -> Optional[str]:
        """
        Get the latest model version ID.

        Returns:
            Latest model version ID, or None if no versions exist
        """
        return self.latest_model_version_id

    def stop(self) -> None:
        """Stop blockchain worker."""
        self.running = False
        logger.info("Stopping blockchain worker")
