"""Encryption utilities for model weight diffs using AES-256-GCM."""

from typing import Optional
from shared.utils.crypto import (
    AES256GCM,
    CryptoError,
    encrypt_data,
    decrypt_data,
)
from shared.config import settings
from shared.logger import setup_logger

logger = setup_logger(__name__)


class EncryptionService:
    """Service for encrypting and decrypting model weight diffs."""

    def __init__(self, key: Optional[bytes] = None):
        """
        Initialize encryption service.

        Args:
            key: 32-byte encryption key. If None, uses key from settings.
        """
        if key is None:
            key = settings.get_encryption_key()
        self.cipher = AES256GCM(key)
        self.key = key

    def encrypt_diff(
        self, diff_bytes: bytes, associated_data: Optional[bytes] = None
    ) -> bytes:
        """
        Encrypt weight diff bytes.

        Args:
            diff_bytes: Weight diff as bytes (e.g., JSON serialized)
            associated_data: Optional associated data for authentication

        Returns:
            Encrypted bytes (nonce + ciphertext + tag)
        """
        encrypted = self.cipher.encrypt(diff_bytes, associated_data)
        logger.debug(f"Encrypted {len(diff_bytes)} bytes to {len(encrypted)} bytes")
        return encrypted

    def decrypt_diff(
        self, encrypted_bytes: bytes, associated_data: Optional[bytes] = None
    ) -> bytes:
        """
        Decrypt weight diff bytes.

        Args:
            encrypted_bytes: Encrypted bytes (nonce + ciphertext + tag)
            associated_data: Optional associated data (must match encryption)

        Returns:
            Decrypted bytes (weight diff)

        Raises:
            CryptoError: If decryption fails
        """
        try:
            decrypted = self.cipher.decrypt(encrypted_bytes, associated_data)
            logger.debug(
                f"Decrypted {len(encrypted_bytes)} bytes to {len(decrypted)} bytes"
            )
            return decrypted
        except CryptoError as e:
            logger.error(f"Decryption failed: {str(e)}")
            raise

    def get_key(self) -> bytes:
        """
        Get the encryption key.

        Returns:
            Encryption key bytes
        """
        return self.key


# Convenience functions that use settings key
def encrypt_diff(diff_bytes: bytes, associated_data: Optional[bytes] = None) -> bytes:
    """
    Convenience function to encrypt weight diff using settings key.

    Args:
        diff_bytes: Weight diff as bytes
        associated_data: Optional associated data

    Returns:
        Encrypted bytes
    """
    service = EncryptionService()
    return service.encrypt_diff(diff_bytes, associated_data)


def decrypt_diff(
    encrypted_bytes: bytes, associated_data: Optional[bytes] = None
) -> bytes:
    """
    Convenience function to decrypt weight diff using settings key.

    Args:
        encrypted_bytes: Encrypted bytes
        associated_data: Optional associated data

    Returns:
        Decrypted bytes

    Raises:
        CryptoError: If decryption fails
    """
    service = EncryptionService()
    return service.decrypt_diff(encrypted_bytes, associated_data)
