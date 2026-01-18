"""API routes for main service."""

import time
import uuid
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from shared.logger import setup_logger
from shared.queue.publisher import QueuePublisher
from shared.queue.connection import QueueConnection
from shared.utils.training import publish_train_task
from shared.models.task import Task, TaskType, TaskMetadata, RollbackTaskPayload
from shared.config import settings
from shared.utils.hashing import compute_hash
from shared.storage.ipfs_client import IPFSClient
from main_service.blockchain.fabric_client import FabricClient
from main_service.services.training_service import prepopulate_initial_weights
from main_service.api.models import (
    ModelVersionResponse,
    ModelVersionListResponse,
    ProvenanceChainResponse,
    ManualRollbackRequest,
    ManualRollbackResponse,
    StartTrainingRequest,
    StartTrainingResponse,
    TrainingStatusResponse,
    ErrorResponse,
)
from main_service.api.auth import verify_api_key
from main_service.workers.blockchain_worker import BlockchainWorker

logger = setup_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["main-service"])


@router.get("/models", response_model=ModelVersionListResponse)
async def list_models(
    limit: Optional[int] = 100,
    offset: Optional[int] = 0,
    api_key: str = Depends(verify_api_key),
):
    """
    List all model versions.

    Args:
        limit: Maximum number of versions to return
        offset: Number of versions to skip
        api_key: API key for authentication

    Returns:
        List of model versions
    """
    try:
        logger.debug(f"Listing models (limit={limit}, offset={offset})")

        # Query blockchain service for model versions
        async with FabricClient() as blockchain_client:
            blockchain_response = await blockchain_client.list_models()

        # Parse response from blockchain service
        if not blockchain_response:
            logger.warning("Blockchain service returned empty response")
            blockchain_response = {}

        logger.debug(f"Blockchain response: {blockchain_response}")

        blockchain_versions = blockchain_response.get("versions")
        if blockchain_versions is None:
            logger.warning(
                "Blockchain service returned None for versions, using empty list"
            )
            blockchain_versions = []
        elif not isinstance(blockchain_versions, list):
            logger.warning(
                f"Blockchain service returned non-list for versions: {type(blockchain_versions)}, using empty list"
            )
            blockchain_versions = []

        total = blockchain_response.get("total", 0)
        if not isinstance(total, int):
            total = len(blockchain_versions)

        # Convert to ModelVersionResponse format
        versions: List[ModelVersionResponse] = []
        for bv in blockchain_versions:
            metadata = bv.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}

            # Extract accuracy from validation_metrics
            # ValidationMetrics can be at top level (from chaincode) or in metadata (fallback)
            validation_metrics = bv.get("validation_metrics") or metadata.get(
                "validation_metrics"
            )
            accuracy = None
            if isinstance(validation_metrics, dict):
                accuracy = validation_metrics.get("accuracy")
            # Also check if accuracy is directly in validation_metrics as a number
            elif validation_metrics is None and "accuracy" in bv:
                accuracy = bv.get("accuracy")

            version = ModelVersionResponse(
                version_id=bv.get("version_id", ""),
                parent_version_id=bv.get("parent_version_id"),
                hash=bv.get("hash", ""),
                iteration=(
                    metadata.get("iteration")
                    if isinstance(metadata.get("iteration"), int)
                    else None
                ),
                num_clients=(
                    metadata.get("num_clients")
                    if isinstance(metadata.get("num_clients"), int)
                    else None
                ),
                client_ids=(
                    metadata.get("client_ids")
                    if isinstance(metadata.get("client_ids"), list)
                    else None
                ),
                ipfs_cid=metadata.get("ipfs_cid"),
                timestamp=bv.get("timestamp"),
                validation_status=bv.get("validation_status")
                or metadata.get("validation_status"),
                accuracy=accuracy,
                metadata=metadata,
            )
            versions.append(version)

        # Apply offset and limit
        if offset is not None and offset > 0:
            versions = versions[offset:]
        if limit is not None and limit > 0:
            versions = versions[:limit]

        # Sort by timestamp (newest first) if available
        versions.sort(key=lambda v: v.timestamp or "", reverse=True)

        # Return response
        response = ModelVersionListResponse(versions=versions, total=total)
        logger.debug(f"Returning {len(versions)} model versions (total: {total})")
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing models: {str(e)}",
        )


@router.get("/models/{version_id}", response_model=ModelVersionResponse)
async def get_model(
    version_id: str,
    api_key: str = Depends(verify_api_key),
):
    """
    Get details of a specific model version.

    Args:
        version_id: Model version identifier
        api_key: API key for authentication

    Returns:
        Model version details
    """
    try:
        logger.info(f"Getting model version: {version_id}")

        # Query blockchain service for model version
        async with FabricClient() as blockchain_client:
            provenance = await blockchain_client.get_model_provenance(version_id)

        # Parse provenance data
        metadata = provenance.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        # Extract accuracy from validation_metrics
        # ValidationMetrics can be at top level (from chaincode) or in metadata (fallback)
        validation_metrics = provenance.get("validation_metrics") or metadata.get(
            "validation_metrics"
        )
        accuracy = None
        if isinstance(validation_metrics, dict):
            accuracy = validation_metrics.get("accuracy")

        return ModelVersionResponse(
            version_id=provenance.get("version_id", version_id),
            parent_version_id=provenance.get("parent_version_id"),
            hash=provenance.get("hash", ""),
            iteration=metadata.get("iteration") or provenance.get("iteration"),
            num_clients=metadata.get("num_clients") or provenance.get("num_clients"),
            client_ids=metadata.get("client_ids") or provenance.get("client_ids"),
            ipfs_cid=metadata.get("ipfs_cid")
            or provenance.get("ipfs_cid")
            or provenance.get("ipfsCID"),
            timestamp=provenance.get("timestamp"),
            validation_status=provenance.get("validation_status")
            or metadata.get("validation_status"),
            accuracy=accuracy,
            metadata=metadata,
        )
    except Exception as e:
        logger.error(
            f"Error getting model version {version_id}: {str(e)}", exc_info=True
        )
        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
                if "not found" in str(e).lower()
                else status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=f"Error getting model version: {str(e)}",
        )


def _provenance_to_model_version(
    provenance: dict, version_id: str
) -> ModelVersionResponse:
    """
    Convert provenance data to ModelVersionResponse.

    Args:
        provenance: Provenance data from blockchain
        version_id: Version ID (fallback if not in provenance)

    Returns:
        ModelVersionResponse object
    """
    metadata = provenance.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    # Extract accuracy from validation_metrics
    validation_metrics = provenance.get("validation_metrics") or metadata.get(
        "validation_metrics"
    )
    accuracy = None
    if isinstance(validation_metrics, dict):
        accuracy = validation_metrics.get("accuracy")

    return ModelVersionResponse(
        version_id=provenance.get("version_id", version_id),
        parent_version_id=provenance.get("parent_version_id"),
        hash=provenance.get("hash", ""),
        iteration=metadata.get("iteration") or provenance.get("iteration"),
        num_clients=metadata.get("num_clients") or provenance.get("num_clients"),
        client_ids=metadata.get("client_ids") or provenance.get("client_ids"),
        ipfs_cid=metadata.get("ipfs_cid")
        or provenance.get("ipfs_cid")
        or provenance.get("ipfsCID"),
        timestamp=provenance.get("timestamp"),
        validation_status=provenance.get("validation_status")
        or metadata.get("validation_status"),
        accuracy=accuracy,
        metadata=metadata,
    )


async def _build_provenance_chain(
    blockchain_client: FabricClient,
    version_id: str,
    visited: Optional[set] = None,
) -> List[ModelVersionResponse]:
    """
    Build provenance chain by recursively following parent_version_id links.

    Args:
        blockchain_client: Fabric client instance
        version_id: Starting version ID
        visited: Set of visited version IDs to prevent cycles

    Returns:
        List of ModelVersionResponse objects ordered from oldest to newest
    """
    if visited is None:
        visited = set()

    # Prevent infinite loops
    if version_id in visited:
        logger.warning(f"Cycle detected in provenance chain at {version_id}")
        return []

    visited.add(version_id)

    try:
        provenance = await blockchain_client.get_model_provenance(version_id)
        version = _provenance_to_model_version(provenance, version_id)

        parent_version_id = provenance.get("parent_version_id")
        if parent_version_id:
            try:
                # Recursively get parent chain
                parent_chain = await _build_provenance_chain(
                    blockchain_client, parent_version_id, visited
                )
                # Return chain with oldest first (parent chain + current version)
                return parent_chain + [version]
            except Exception as e:
                logger.warning(
                    f"Could not fetch parent chain for {version_id}, parent: {parent_version_id}: {str(e)}"
                )
                # Return at least the current version if we can't get parents
                return [version]
        else:
            # This is the initial version
            return [version]
    except Exception as e:
        logger.error(
            f"Error building provenance chain for {version_id}: {str(e)}", exc_info=True
        )
        # Return empty chain if we can't even get the current version
        return []


@router.get("/models/{version_id}/provenance", response_model=ProvenanceChainResponse)
async def get_provenance(
    version_id: str,
    include_chain: bool = False,
    api_key: str = Depends(verify_api_key),
):
    """
    Get provenance chain for a model version.

    Args:
        version_id: Model version identifier
        include_chain: Whether to include full provenance chain
        api_key: API key for authentication

    Returns:
        Provenance information
    """
    try:
        logger.info(f"Getting provenance for version: {version_id}")

        # Query blockchain service
        async with FabricClient() as blockchain_client:
            provenance = await blockchain_client.get_model_provenance(version_id)

            chain: Optional[List[ModelVersionResponse]] = None
            if include_chain:
                # Build full provenance chain by following parent_version_id
                chain = await _build_provenance_chain(blockchain_client, version_id)

        return ProvenanceChainResponse(
            version_id=provenance.get("version_id", version_id),
            parent_version_id=provenance.get("parent_version_id"),
            hash=provenance.get("hash", ""),
            metadata=provenance.get("metadata", {}),
            timestamp=provenance.get("timestamp"),
            chain=chain,
        )
    except Exception as e:
        logger.error(
            f"Error getting provenance for {version_id}: {str(e)}", exc_info=True
        )
        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
                if "not found" in str(e).lower()
                else status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=f"Error getting provenance: {str(e)}",
        )


@router.post("/models/{version_id}/rollback", response_model=ManualRollbackResponse)
async def manual_rollback(
    version_id: str,
    request: ManualRollbackRequest,
    api_key: str = Depends(verify_api_key),
):
    """
    Manually rollback to a specific model version.

    Args:
        version_id: Version ID to rollback to (same as request.target_version_id)
        request: Rollback request with reason
        api_key: API key for authentication

    Returns:
        Rollback response
    """
    try:
        logger.info(
            f"Manual rollback requested: version={version_id}, reason={request.reason}"
        )

        # Use version_id from path (request.target_version_id is optional for backward compatibility)
        target_version_id = request.target_version_id or version_id
        if request.target_version_id and version_id != target_version_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Version ID in path must match target_version_id in request",
            )

        # Get IPFS CID for the target version from blockchain
        async with FabricClient() as blockchain_client:
            provenance = await blockchain_client.get_model_provenance(target_version_id)

        metadata = provenance.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        ipfs_cid = (
            metadata.get("ipfs_cid")
            or provenance.get("ipfs_cid")
            or provenance.get("ipfsCID")
        )

        if not ipfs_cid:
            validation_history = metadata.get("validation_history", [])
            if isinstance(validation_history, list) and validation_history:
                for validation_record in reversed(validation_history):
                    if isinstance(validation_record, dict):
                        cid = validation_record.get("ipfs_cid")
                        if cid:
                            ipfs_cid = cid
                            logger.info(
                                f"Found IPFS CID in validation_history for {target_version_id}: {cid}"
                            )
                            break

        if not ipfs_cid:
            logger.error(
                f"IPFS CID not found for version {target_version_id}. "
                f"Provenance keys: {list(provenance.keys())}, "
                f"Metadata keys: {list(metadata.keys()) if isinstance(metadata, dict) else 'N/A'}"
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"IPFS CID not found for version {version_id}",
            )

        # Publish ROLLBACK task to queue
        connection = QueueConnection()
        publisher = QueuePublisher(connection=connection)

        rollback_task = Task(
            task_id=f"manual-rollback-{target_version_id}-{int(time.time())}",
            task_type=TaskType.ROLLBACK,
            payload=RollbackTaskPayload(
                target_version_id=target_version_id,
                target_weights_cid=ipfs_cid,
                reason=f"Manual rollback: {request.reason}",
                cutoff_version_id=None,  # Manual rollback doesn't specify cutoff
            ).model_dump(),
            metadata=TaskMetadata(source="api_manual_rollback"),
            model_version_id=target_version_id,
            parent_version_id=provenance.get("parent_version_id"),
        )

        publisher.publish_task(rollback_task, "rollback_queue")
        logger.info(f"Published manual rollback task for version {target_version_id}")

        return ManualRollbackResponse(
            success=True,
            message=f"Rollback task published for version {target_version_id}",
            transaction_id=None,  # Will be set by rollback worker
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error performing manual rollback: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error performing rollback: {str(e)}",
        )


@router.post("/training/start", response_model=StartTrainingResponse)
async def start_training(
    request: StartTrainingRequest,
    api_key: str = Depends(verify_api_key),
):
    """
    Start a new training session.

    Args:
        request: Training start request
        api_key: API key for authentication

    Returns:
        Start training response
    """
    try:
        logger.info("Training start requested via API")

        connection = QueueConnection()
        publisher = QueuePublisher(connection=connection)

        # Get number of clients from settings
        num_clients = settings.num_clients

        iteration = 1  # Start from iteration 1

        # Prepopulate initial weights if not provided
        initial_weights_cid = request.initial_weights_cid
        if initial_weights_cid is None:
            logger.info(
                "No initial weights provided, generating random initial weights..."
            )
            initial_weights_cid = await prepopulate_initial_weights()
            logger.info(f"Generated initial weights with CID: {initial_weights_cid}")

        # Register initial model version on blockchain (iteration 0)
        # This creates a valid checkpoint that can be rolled back to
        try:
            # Generate initial model version ID
            initial_version_id = (
                f"model_v0_initial_{int(time.time())}_{str(uuid.uuid4())[:8]}"
            )

            # For initial weights, we use the encrypted weights hash as the blockchain hash
            # Since we already have the encrypted weights in IPFS, we need to compute its hash
            # For now, we'll use a placeholder hash - in production, we'd retrieve and hash the encrypted weights
            # But since we just uploaded them, we can compute the hash from the CID metadata
            # For simplicity, we'll use the CID as a reference and compute hash from IPFS
            async with FabricClient() as blockchain_client:
                # Compute hash of encrypted weights (we need to retrieve from IPFS)
                async with IPFSClient() as ipfs_client:
                    encrypted_weights = await ipfs_client.get_bytes(initial_weights_cid)
                    initial_hash = compute_hash(encrypted_weights)

                # Register initial model version on blockchain
                initial_metadata = {
                    "iteration": 0,
                    "num_clients": 0,  # No clients participated in initial version
                    "client_ids": [],
                    "ipfs_cid": initial_weights_cid,
                    "diff_hash": initial_hash,  # Hash of encrypted initial weights
                    "rollback_count": 0,
                    "validation_history": [],
                    "is_initial": True,
                }

                transaction_id = await blockchain_client.register_model_update(
                    model_version_id=initial_version_id,
                    parent_version_id=None,  # No parent for initial version
                    hash_value=initial_hash,
                    metadata=initial_metadata,
                )

                logger.info(
                    f"Registered initial model version on blockchain: "
                    f"version_id={initial_version_id}, cid={initial_weights_cid}, "
                    f"tx_id={transaction_id}"
                )
        except Exception as e:
            logger.warning(
                f"Failed to register initial model version on blockchain: {str(e)}. "
                "Training will continue, but initial checkpoint may not be available for rollback.",
                exc_info=True,
            )

        # Publish a single universal TRAIN task for all clients
        # All clients will process the same task and send their updates
        publish_train_task(
            publisher,
            iteration,
            initial_weights_cid,
            source="api_start_training",
        )

        return StartTrainingResponse(
            success=True,
            message=f"Training started with universal task (all {num_clients} clients will process)",
            iteration=iteration,
        )
    except Exception as e:
        logger.error(f"Error starting training: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error starting training: {str(e)}",
        )


@router.post("/training/stop", response_model=dict)
async def stop_training(
    api_key: str = Depends(verify_api_key),
):
    """
    Stop the current training session.

    Args:
        api_key: API key for authentication

    Returns:
        Stop training response
    """
    try:
        logger.info("Training stop requested via API")

        # TODO: Implement training stop logic
        # This would need to communicate with decision worker to stop publishing new tasks

        return {
            "success": True,
            "message": "Training stop requested (not yet implemented)",
        }
    except Exception as e:
        logger.error(f"Error stopping training: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error stopping training: {str(e)}",
        )


@router.get("/training/status", response_model=TrainingStatusResponse)
async def get_training_status(
    api_key: str = Depends(verify_api_key),
):
    """
    Get current training status.

    Args:
        api_key: API key for authentication

    Returns:
        Training status derived from blockchain
    """
    try:
        logger.debug("Training status requested via API")

        # Get latest model version from blockchain
        # Use the shared blockchain worker instance from server startup
        # Import workers here to avoid circular import
        from main_service.api.server import workers

        latest_version_id = None
        if "blockchain" in workers:
            blockchain_worker = workers["blockchain"]
            latest_version_id = blockchain_worker.get_latest_model_version_id()
        else:
            # Fallback: create temporary instance (shouldn't happen if workers started correctly)
            logger.warning(
                "Blockchain worker not found in shared workers, creating temporary instance"
            )
            blockchain_worker = BlockchainWorker()
            latest_version_id = blockchain_worker.get_latest_model_version_id()

        if not latest_version_id:
            # No versions exist yet
            return TrainingStatusResponse(
                is_training=False,
                current_iteration=None,
                best_accuracy=None,
                total_iterations=0,
                rollback_count=0,
                status="stopped",
                start_time=None,
                best_checkpoint_version=None,
                best_checkpoint_cid=None,
            )

        # Query blockchain for latest version
        async with FabricClient() as blockchain_client:
            provenance = await blockchain_client.get_model_provenance(latest_version_id)

        metadata = provenance.get("metadata", {})
        validation_metrics = provenance.get("validation_metrics", {})

        # Extract status from metadata
        current_iteration = metadata.get("iteration")
        rollback_count = metadata.get("rollback_count", 0)
        validation_history = metadata.get("validation_history", [])

        # Find best accuracy from validation_history or current validation_metrics
        best_accuracy = None
        accuracy_history_list = []
        if validation_history:
            accuracy_history_list = [v.get("accuracy", 0.0) for v in validation_history]
            best_accuracy = (
                max(accuracy_history_list) if accuracy_history_list else None
            )
        elif validation_metrics:
            current_acc = validation_metrics.get("accuracy")
            if current_acc is not None:
                accuracy_history_list = [current_acc]
                best_accuracy = current_acc

        # Get timestamp for start_time (from first version in chain)
        # For now, use current version's timestamp
        start_time = None
        timestamp_str = provenance.get("timestamp")
        if timestamp_str:
            try:
                start_time = datetime.fromtimestamp(int(timestamp_str))
            except (ValueError, TypeError):
                pass

        # Check if training has completed by looking for completion_reason in metadata
        # or by checking blockchain worker's completion tracker
        completion_reason = None
        completion_info = None

        # First check metadata
        if isinstance(metadata, dict):
            completion_reason = metadata.get("completion_reason")

        # Also check blockchain worker's completion tracker
        # Check both blockchain worker instances (main and completion)
        # Check if training has completed for any version (not just the latest)
        try:
            from main_service.api.server import workers

            # Check main blockchain worker - first check if any completion exists
            if "blockchain" in workers:
                blockchain_worker = workers["blockchain"]
                if blockchain_worker.has_any_completion():
                    # Get latest completion info (most recent)
                    completion_info = blockchain_worker.get_latest_completion_info()
                    if completion_info:
                        completion_reason = completion_info.get("completion_reason")
                else:
                    # Fallback: check specific version
                    completion_info = blockchain_worker.get_completion_info(
                        latest_version_id
                    )
                    if completion_info:
                        completion_reason = completion_info.get("completion_reason")

            # Also check blockchain_completion worker
            if not completion_reason and "blockchain_completion" in workers:
                blockchain_completion_worker = workers["blockchain_completion"]
                if blockchain_completion_worker.has_any_completion():
                    # Get latest completion info (most recent)
                    completion_info = (
                        blockchain_completion_worker.get_latest_completion_info()
                    )
                    if completion_info:
                        completion_reason = completion_info.get("completion_reason")
                else:
                    # Fallback: check specific version
                    completion_info = blockchain_completion_worker.get_completion_info(
                        latest_version_id
                    )
                    if completion_info:
                        completion_reason = completion_info.get("completion_reason")
        except Exception as e:
            logger.debug(f"Could not check completion tracker: {str(e)}")
            pass  # Ignore errors accessing blockchain worker

        # Determine status based on validation_status and completion
        validation_status = provenance.get("validation_status", "pending")

        if completion_reason:
            # Training has completed
            training_status = "completed"
            is_training = False
        elif validation_status == "pending":
            training_status = "running"
            is_training = True
        elif validation_status == "passed":
            # Check if this might be the final version (no new versions created recently)
            # For now, assume still training if validation passed
            training_status = "running"
            is_training = True
        else:
            training_status = "stopped"
            is_training = False

        # Get best checkpoint info from metadata
        best_checkpoint_version = metadata.get("best_checkpoint_version")
        best_checkpoint_cid = metadata.get("best_checkpoint_cid")

        return TrainingStatusResponse(
            is_training=is_training,
            current_iteration=current_iteration,
            best_accuracy=best_accuracy,
            total_iterations=current_iteration if current_iteration else 0,
            rollback_count=rollback_count,
            status=training_status,
            start_time=start_time,
            best_checkpoint_version=best_checkpoint_version,
            best_checkpoint_cid=best_checkpoint_cid,
        )
    except Exception as e:
        logger.error(f"Error getting training status: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting training status: {str(e)}",
        )


@router.get("/health")
async def health_check():
    """
    Health check endpoint (no authentication required).

    Returns:
        Health status
    """
    return {"status": "healthy", "service": "main-service"}
