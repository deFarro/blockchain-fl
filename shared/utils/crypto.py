"""Encryption and decryption utilities using AES-256-GCM."""

import os
from typing import Optional, Tuple
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class CryptoError(Exception):
    """Exception raised for cryptographic operations."""

    pass


class AES256GCM:
    """AES-256-GCM encryption/decryption utility."""

    KEY_SIZE = 32  # 256 bits
    NONCE_SIZE = 12  # 96 bits for GCM
    TAG_SIZE = 16  # 128 bits for authentication tag

    def __init__(self, key: Optional[bytes] = None):
        """
        Initialize AES-256-GCM cipher.

        Args:
            key: 32-byte encryption key. If None, generates a random key.
        """
        if key is None:
            key = os.urandom(self.KEY_SIZE)
        elif len(key) != self.KEY_SIZE:
            raise CryptoError(f"Key must be {self.KEY_SIZE} bytes, got {len(key)}")

        self.key = key
        self.cipher = AESGCM(key)

    @classmethod
    def generate_key(cls) -> bytes:
        """
        Generate a random 32-byte key.

        Returns:
            Random 32-byte key
        """
        return os.urandom(cls.KEY_SIZE)

    @classmethod
    def derive_key_from_password(
        cls, password: str, salt: Optional[bytes] = None
    ) -> Tuple[bytes, bytes]:
        """
        Derive encryption key from password using PBKDF2.

        Args:
            password: Password string
            salt: Salt bytes. If None, generates random salt.

        Returns:
            Tuple of (key, salt)
        """
        if salt is None:
            salt = os.urandom(16)

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=cls.KEY_SIZE,
            salt=salt,
            iterations=100000,
            backend=default_backend(),
        )
        key = kdf.derive(password.encode("utf-8"))
        return key, salt

    def encrypt(
        self, plaintext: bytes, associated_data: Optional[bytes] = None
    ) -> bytes:
        """
        Encrypt plaintext using AES-256-GCM.

        Args:
            plaintext: Data to encrypt
            associated_data: Optional associated data for authentication (not encrypted)

        Returns:
            Encrypted data in format: nonce (12 bytes) + ciphertext + tag (16 bytes)
        """
        nonce = os.urandom(self.NONCE_SIZE)
        ciphertext = self.cipher.encrypt(nonce, plaintext, associated_data)
        # Format: nonce + ciphertext (which includes tag)
        return nonce + ciphertext

    def decrypt(
        self, ciphertext: bytes, associated_data: Optional[bytes] = None
    ) -> bytes:
        """
        Decrypt ciphertext using AES-256-GCM.

        Args:
            ciphertext: Encrypted data in format: nonce (12 bytes) + ciphertext + tag (16 bytes)
            associated_data: Optional associated data for authentication (must match encryption)

        Returns:
            Decrypted plaintext

        Raises:
            CryptoError: If decryption fails (invalid key, tampered data, etc.)
        """
        if len(ciphertext) < self.NONCE_SIZE + self.TAG_SIZE:
            raise CryptoError(
                f"Ciphertext too short. Expected at least {self.NONCE_SIZE + self.TAG_SIZE} bytes"
            )

        nonce = ciphertext[: self.NONCE_SIZE]
        encrypted_data = ciphertext[self.NONCE_SIZE :]

        try:
            plaintext = self.cipher.decrypt(nonce, encrypted_data, associated_data)
            return plaintext
        except Exception as e:
            raise CryptoError(f"Decryption failed: {str(e)}") from e

    def get_key(self) -> bytes:
        """
        Get the encryption key.

        Returns:
            Encryption key bytes
        """
        return self.key


def encrypt_data(data: bytes, key: bytes) -> bytes:
    """
    Convenience function to encrypt data.

    Args:
        data: Data to encrypt
        key: 32-byte encryption key

    Returns:
        Encrypted data
    """
    cipher = AES256GCM(key)
    return cipher.encrypt(data)


def decrypt_data(ciphertext: bytes, key: bytes) -> bytes:
    """
    Convenience function to decrypt data.

    Args:
        ciphertext: Encrypted data
        key: 32-byte decryption key

    Returns:
        Decrypted data

    Raises:
        CryptoError: If decryption fails
    """
    cipher = AES256GCM(key)
    return cipher.decrypt(ciphertext)
