"""Hyperledger Fabric client - calls blockchain service API."""

from typing import Optional, Dict, Any
import httpx
from shared.config import settings
from shared.logger import setup_logger

logger = setup_logger(__name__)


def _parse_json_response(response: httpx.Response) -> Dict[str, Any]:
    """
    Parse JSON response from httpx with proper typing.

    Args:
        response: httpx Response object

    Returns:
        Parsed JSON as dictionary
    """
    data: Any = response.json()
    if isinstance(data, dict):
        return data
    raise ValueError(f"Expected dict, got {type(data)}")


class FabricClient:
    """Hyperledger Fabric client that calls the blockchain service API."""

    def __init__(self, service_url: Optional[str] = None):
        """
        Initialize Fabric client.

        Args:
            service_url: Blockchain service URL (defaults to environment variable or localhost)
        """
        self.service_url = service_url or settings.blockchain_service_url
        self.client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.client = httpx.AsyncClient(base_url=self.service_url, timeout=30.0)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.client:
            await self.client.aclose()

    async def register_model_update(
        self,
        model_version_id: str,
        parent_version_id: Optional[str],
        hash_value: str,
        metadata: Dict[str, Any],
    ) -> str:
        """
        Register a new model version on blockchain.

        Args:
            model_version_id: Unique version identifier
            parent_version_id: Parent version ID (None for initial version)
            hash_value: Hash of encrypted diff
            metadata: Additional metadata

        Returns:
            Transaction ID
        """
        if not self.client:
            raise RuntimeError("FabricClient must be used as async context manager")

        payload = {
            "model_version_id": model_version_id,
            "parent_version_id": parent_version_id,
            "hash": hash_value,
            "metadata": metadata,
        }

        try:
            response = await self.client.post("/api/v1/model/register", json=payload)
            response.raise_for_status()
            result = _parse_json_response(response)
            tx_id = result.get("transaction_id")
            if not isinstance(tx_id, str):
                raise ValueError("Response missing transaction_id")
            logger.info(
                f"Registered model version {model_version_id} on blockchain: {tx_id}"
            )
            return tx_id
        except httpx.HTTPError as e:
            logger.error(f"Failed to register model update: {str(e)}")
            raise

    async def record_validation(
        self,
        model_version_id: str,
        accuracy: float,
        metrics: Dict[str, float],
    ) -> str:
        """
        Record validation results on blockchain.

        Args:
            model_version_id: Model version ID
            accuracy: Validation accuracy
            metrics: Additional metrics

        Returns:
            Transaction ID
        """
        if not self.client:
            raise RuntimeError("FabricClient must be used as async context manager")

        payload = {
            "model_version_id": model_version_id,
            "accuracy": accuracy,
            "metrics": metrics,
        }

        try:
            response = await self.client.post("/api/v1/model/validate", json=payload)
            response.raise_for_status()
            result = _parse_json_response(response)
            tx_id = result.get("transaction_id")
            if not isinstance(tx_id, str):
                raise ValueError("Response missing transaction_id")
            logger.info(
                f"Recorded validation for {model_version_id} on blockchain: {tx_id}"
            )
            return tx_id
        except httpx.HTTPError as e:
            logger.error(f"Failed to record validation: {str(e)}")
            raise

    async def rollback_model(self, target_version_id: str, reason: str) -> str:
        """
        Record rollback event on blockchain.

        Args:
            target_version_id: Version to rollback to
            reason: Rollback reason

        Returns:
            Transaction ID
        """
        if not self.client:
            raise RuntimeError("FabricClient must be used as async context manager")

        payload = {
            "target_version_id": target_version_id,
            "reason": reason,
        }

        try:
            response = await self.client.post("/api/v1/model/rollback", json=payload)
            response.raise_for_status()
            result = _parse_json_response(response)
            tx_id = result.get("transaction_id")
            if not isinstance(tx_id, str):
                raise ValueError("Response missing transaction_id")
            logger.info(
                f"Recorded rollback to {target_version_id} on blockchain: {tx_id}"
            )
            return tx_id
        except httpx.HTTPError as e:
            logger.error(f"Failed to record rollback: {str(e)}")
            raise

    async def get_model_provenance(self, model_version_id: str) -> Dict[str, Any]:
        """
        Get model provenance chain.

        Args:
            model_version_id: Model version ID

        Returns:
            Provenance information
        """
        if not self.client:
            raise RuntimeError("FabricClient must be used as async context manager")

        try:
            response = await self.client.get(
                f"/api/v1/model/provenance/{model_version_id}"
            )
            response.raise_for_status()
            return _parse_json_response(response)
        except httpx.HTTPError as e:
            logger.error(f"Failed to get provenance: {str(e)}")
            raise

    async def health_check(self) -> bool:
        """
        Check if blockchain service is healthy.

        Returns:
            True if healthy
        """
        if not self.client:
            raise RuntimeError("FabricClient must be used as async context manager")

        try:
            response = await self.client.get("/health")
            response.raise_for_status()
            return True
        except httpx.HTTPError:
            return False


# Convenience function
async def get_fabric_client(service_url: Optional[str] = None) -> FabricClient:
    """
    Get Fabric client instance.

    Args:
        service_url: Optional blockchain service URL

    Returns:
        FabricClient instance
    """
    return FabricClient(service_url)
