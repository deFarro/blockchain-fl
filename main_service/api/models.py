"""Pydantic models for API requests and responses."""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime


class ModelVersionResponse(BaseModel):
    """Response model for a single model version."""

    version_id: str = Field(..., description="Model version identifier")
    parent_version_id: Optional[str] = Field(
        None, description="Parent version ID (for provenance chain)"
    )
    hash: str = Field(..., description="Hash of encrypted diff")
    iteration: Optional[int] = Field(None, description="Training iteration number")
    num_clients: Optional[int] = Field(
        None, description="Number of clients that participated"
    )
    client_ids: Optional[List[str]] = Field(None, description="List of client IDs")
    ipfs_cid: Optional[str] = Field(None, description="IPFS CID of encrypted diff")
    timestamp: Optional[str] = Field(None, description="Timestamp of version creation")
    validation_status: Optional[str] = Field(
        None, description="Validation status (pending, passed, failed)"
    )
    accuracy: Optional[float] = Field(None, description="Validation accuracy")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class ModelVersionListResponse(BaseModel):
    """Response model for list of model versions."""

    versions: List[ModelVersionResponse] = Field(
        ..., description="List of model versions"
    )
    total: int = Field(..., description="Total number of versions")


class ProvenanceChainResponse(BaseModel):
    """Response model for model provenance chain."""

    version_id: str = Field(..., description="Model version identifier")
    parent_version_id: Optional[str] = Field(None, description="Parent version ID")
    hash: str = Field(..., description="Hash of encrypted diff")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    timestamp: Optional[str] = Field(None, description="Timestamp")
    chain: Optional[List[ModelVersionResponse]] = Field(
        None, description="Full provenance chain (if requested)"
    )


class ManualRollbackRequest(BaseModel):
    """Request model for manual rollback."""

    target_version_id: str = Field(..., description="Version ID to rollback to")
    reason: str = Field(..., description="Reason for rollback")


class ManualRollbackResponse(BaseModel):
    """Response model for manual rollback."""

    success: bool = Field(..., description="Whether rollback was successful")
    message: str = Field(..., description="Response message")
    transaction_id: Optional[str] = Field(None, description="Blockchain transaction ID")


class StartTrainingRequest(BaseModel):
    """Request model for starting training."""

    initial_weights_cid: Optional[str] = Field(
        None, description="IPFS CID of initial weights (None to start from scratch)"
    )
    num_iterations: Optional[int] = Field(
        None, description="Maximum number of iterations (None for unlimited)"
    )


class StartTrainingResponse(BaseModel):
    """Response model for starting training."""

    success: bool = Field(..., description="Whether training was started")
    message: str = Field(..., description="Response message")
    iteration: Optional[int] = Field(None, description="Starting iteration number")


class TrainingStatusResponse(BaseModel):
    """Response model for training status."""

    is_training: bool = Field(..., description="Whether training is currently active")
    current_iteration: Optional[int] = Field(
        None, description="Current iteration number"
    )
    best_accuracy: Optional[float] = Field(None, description="Best accuracy achieved")
    total_iterations: int = Field(..., description="Total iterations completed")
    rollback_count: int = Field(..., description="Number of rollbacks performed")
    status: str = Field(
        ..., description="Training status (running, stopped, completed)"
    )
    start_time: Optional[datetime] = Field(None, description="Training start time")


class ErrorResponse(BaseModel):
    """Response model for errors."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
