"""Service for training-related operations."""

import json
from typing import Dict, Any
from client_service.training.model import SimpleCNN
from shared.storage.encryption import EncryptionService
from main_service.storage.ipfs_client import IPFSClient
from shared.logger import setup_logger

logger = setup_logger(__name__)


async def prepopulate_initial_weights() -> str:
    """
    Generate initial model weights with random values, encrypt, and upload to IPFS.

    This function creates a new model with random weights, serializes them,
    encrypts them, and uploads to IPFS. Used when starting training without
    providing initial weights.

    Returns:
        IPFS CID of the encrypted initial weights

    Raises:
        Exception: If IPFS upload or encryption fails
    """
    logger.info("Prepopulating initial model weights...")

    # Create model with random weights
    model = SimpleCNN(num_classes=10)
    weights = model.get_weights()

    # Serialize weights to JSON string
    serializable_weights: Dict[str, Any] = {}
    for name, tensor in weights.items():
        serializable_weights[name] = tensor.numpy().tolist()

    weights_json = json.dumps(serializable_weights)
    weights_bytes = weights_json.encode("utf-8")

    logger.debug(f"Serialized initial weights: {len(weights_bytes)} bytes")

    # Encrypt weights
    encryption_service = EncryptionService()
    encrypted_weights = encryption_service.encrypt_diff(weights_bytes)
    logger.debug(f"Encrypted initial weights: {len(encrypted_weights)} bytes")

    # Upload to IPFS
    async with IPFSClient() as ipfs_client:
        cid = await ipfs_client.add_bytes(encrypted_weights, pin=True)

    logger.info(f"✓ Initial weights uploaded to IPFS: CID={cid}")
    return str(cid)
