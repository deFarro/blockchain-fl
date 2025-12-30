"""API routes for main service."""

from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from shared.logger import setup_logger
from shared.queue.publisher import QueuePublisher
from shared.queue.connection import QueueConnection
from shared.models.task import Task, TaskType, TaskMetadata, RollbackTaskPayload
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
import time

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
        # TODO: Query blockchain service for model versions
        # For now, return empty list
        # In production, this would query the blockchain service API
        logger.info(f"Listing models (limit={limit}, offset={offset})")

        # Placeholder: In real implementation, query blockchain service
        versions: List[ModelVersionResponse] = []

        return ModelVersionListResponse(versions=versions, total=len(versions))
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
        return ModelVersionResponse(
            version_id=provenance.get("version_id", version_id),
            parent_version_id=provenance.get("parent_version_id"),
            hash=provenance.get("hash", ""),
            iteration=provenance.get("metadata", {}).get("iteration"),
            num_clients=provenance.get("metadata", {}).get("num_clients"),
            client_ids=provenance.get("metadata", {}).get("client_ids"),
            ipfs_cid=provenance.get("metadata", {}).get("ipfs_cid"),
            timestamp=provenance.get("timestamp"),
            validation_status=provenance.get("validation_status"),
            accuracy=provenance.get("validation_metrics", {}).get("accuracy"),
            metadata=provenance.get("metadata", {}),
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
            # TODO: Build full provenance chain by following parent_version_id
            chain = []

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

        ipfs_cid = provenance.get("metadata", {}).get("ipfs_cid")
        if not ipfs_cid:
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

        # Get number of clients from settings (default to 2 if not set)
        from shared.config import settings

        num_clients = getattr(settings, "num_clients", 2)

        iteration = 1  # Start from iteration 1

        # Prepopulate initial weights if not provided
        initial_weights_cid = request.initial_weights_cid
        if initial_weights_cid is None:
            logger.info(
                "No initial weights provided, generating random initial weights..."
            )
            initial_weights_cid = await prepopulate_initial_weights()
            logger.info(f"Generated initial weights with CID: {initial_weights_cid}")

        # Publish initial TRAIN tasks for all clients
        for client_id in range(num_clients):
            train_task = Task(
                task_id=f"api-train-{iteration}-client_{client_id}-{int(time.time())}",
                task_type=TaskType.TRAIN,
                payload={
                    "weights_cid": initial_weights_cid,
                    "iteration": iteration,
                    "client_id": f"client_{client_id}",
                },
                metadata=TaskMetadata(source="api_start_training"),
                model_version_id=None,  # Will be set by decision worker
                parent_version_id=None,  # Will be set by decision worker
            )

            publisher.publish_task(train_task, "train_queue")
            logger.info(
                f"Published TRAIN task for client_{client_id}, iteration {iteration}"
            )

        return StartTrainingResponse(
            success=True,
            message=f"Training started for {num_clients} clients",
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

    Note: This queries the blockchain for the latest model version and derives status.
    All training data (iterations, accuracy, rollbacks) is stored in metadata on-chain.

    Args:
        api_key: API key for authentication

    Returns:
        Training status derived from blockchain
    """
    try:
        logger.info("Training status requested via API")

        # Get latest model version from blockchain
        # For now, we'll need to track latest version ID or query all versions
        # TODO: Add blockchain service endpoint to get latest version
        # For now, try to get from blockchain worker's latest version tracking

        from main_service.workers.blockchain_worker import BlockchainWorker

        # Create a temporary blockchain worker to get latest version
        # In production, this should be a shared instance or query blockchain directly
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
                accuracy_history=None,
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

        # Determine status based on validation_status
        validation_status = provenance.get("validation_status", "pending")
        if validation_status == "pending":
            training_status = "running"
        elif validation_status == "passed":
            training_status = "running"  # Still training
        else:
            training_status = "stopped"

        # is_training: assume True if we have a version with pending/passed validation
        is_training = validation_status in ["pending", "passed"]

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
            accuracy_history=accuracy_history_list if accuracy_history_list else None,
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
