"""Task message models for queue communication."""

from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class TaskType(str, Enum):
    """Task types for queue communication."""

    TRAIN = "TRAIN"  # Client training task
    AGGREGATE = "AGGREGATE"  # Aggregation task
    BLOCKCHAIN_WRITE = "BLOCKCHAIN_WRITE"  # Blockchain transaction
    STORAGE_WRITE = "STORAGE_WRITE"  # Off-chain storage
    VALIDATE = "VALIDATE"  # Model validation
    ROLLBACK = "ROLLBACK"  # Model rollback
    DECISION = "DECISION"  # Post-validation decision
    TRAINING_COMPLETE = "TRAINING_COMPLETE"  # Final task indicating training is done


class TaskPayload(BaseModel):
    """Base class for task-specific payload data."""

    pass


class TrainTaskPayload(TaskPayload):
    """Payload for TRAIN tasks."""

    weights_cid: Optional[str] = Field(
        None, description="IPFS CID of model weights to start training from"
    )
    iteration: int = Field(..., description="Training iteration number")


class AggregateTaskPayload(TaskPayload):
    """Payload for AGGREGATE tasks."""

    client_updates: List[Dict[str, Any]] = Field(
        ..., description="List of client weight updates to aggregate"
    )
    iteration: int = Field(..., description="Training iteration number")


class BlockchainWriteTaskPayload(TaskPayload):
    """Payload for BLOCKCHAIN_WRITE tasks."""

    blockchain_hash: str = Field(
        ..., description="Hash of encrypted diff to store on-chain"
    )
    model_version_id: str = Field(..., description="Unique model version identifier")
    parent_version_id: Optional[str] = Field(
        None, description="Parent model version ID (for provenance chain)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class StorageWriteTaskPayload(TaskPayload):
    """Payload for STORAGE_WRITE tasks."""

    aggregated_diff: str = Field(
        ..., description="Aggregated weight diff as JSON string (to be encrypted)"
    )
    blockchain_hash: str = Field(
        ..., description="Expected hash of encrypted diff (from blockchain transaction)"
    )
    model_version_id: str = Field(..., description="Model version identifier")


class ValidateTaskPayload(TaskPayload):
    """Payload for VALIDATE tasks."""

    ipfs_cid: str = Field(..., description="IPFS CID of encrypted diff to validate")
    model_version_id: str = Field(..., description="Model version identifier")
    parent_version_id: Optional[str] = Field(
        None, description="Parent model version ID"
    )


class RollbackTaskPayload(TaskPayload):
    """Payload for ROLLBACK tasks."""

    target_version_id: str = Field(..., description="Model version to rollback to")
    target_weights_cid: str = Field(..., description="IPFS CID of rolled-back weights")
    reason: str = Field(..., description="Reason for rollback")
    cutoff_version_id: Optional[str] = Field(
        None, description="Version ID after which updates should be discarded"
    )


class DecisionTaskPayload(TaskPayload):
    """Payload for DECISION tasks."""

    validation_result: Dict[str, Any] = Field(
        ..., description="Validation results (accuracy, loss, etc.)"
    )
    model_version_id: str = Field(..., description="Model version that was validated")
    should_rollback: bool = Field(..., description="Whether rollback is needed")
    rollback_reason: Optional[str] = Field(
        None, description="Reason for rollback if needed"
    )


class TrainingCompleteTaskPayload(TaskPayload):
    """Payload for TRAINING_COMPLETE tasks."""

    final_model_version_id: str = Field(
        ..., description="Final model version identifier"
    )
    final_accuracy: float = Field(..., description="Final model accuracy")
    final_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Additional metrics (loss, precision, recall, etc.)",
    )
    final_weights_cid: str = Field(..., description="IPFS CID of final model weights")
    training_summary: Dict[str, Any] = Field(
        ...,
        description="Training summary (total_iterations, clients_participated, training_duration, total_rounds)",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (hyperparameters_used, dataset_info, completion_reason)",
    )


class TaskMetadata(BaseModel):
    """Task metadata."""

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Task creation timestamp",
    )
    priority: int = Field(
        default=0, description="Task priority (higher = more important)"
    )
    retry_count: int = Field(default=0, description="Number of retry attempts")
    source: Optional[str] = Field(
        None, description="Source of the task (worker, API, etc.)"
    )


class Task(BaseModel):
    """Task message model for queue communication."""

    task_id: str = Field(..., description="Unique task identifier")
    task_type: TaskType = Field(..., description="Type of task")
    payload: Dict[str, Any] = Field(..., description="Task-specific payload data")
    metadata: TaskMetadata = Field(
        default_factory=lambda: TaskMetadata(source=None), description="Task metadata"
    )
    model_version_id: Optional[str] = Field(
        None, description="Model version identifier (if applicable)"
    )
    parent_version_id: Optional[str] = Field(
        None, description="Parent model version ID (for provenance chain)"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for queue serialization."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Create task from dictionary (from queue deserialization)."""
        # Handle datetime serialization
        if "metadata" in data and "created_at" in data["metadata"]:
            if isinstance(data["metadata"]["created_at"], str):
                data["metadata"]["created_at"] = datetime.fromisoformat(
                    data["metadata"]["created_at"]
                )
        return cls(**data)
