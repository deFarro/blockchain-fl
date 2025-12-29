"""Hash computation utilities using SHA-256."""

import hashlib
from typing import Union


def compute_hash(data: Union[bytes, str]) -> str:
    """
    Compute SHA-256 hash of data.

    Args:
        data: Data to hash (bytes or string)

    Returns:
        Hexadecimal hash string
    """
    if isinstance(data, str):
        data = data.encode("utf-8")

    sha256_hash = hashlib.sha256()
    sha256_hash.update(data)
    return sha256_hash.hexdigest()


def compute_file_hash(file_path: str, chunk_size: int = 8192) -> str:
    """
    Compute SHA-256 hash of a file.

    Args:
        file_path: Path to file
        chunk_size: Size of chunks to read (default: 8KB)

    Returns:
        Hexadecimal hash string

    Raises:
        FileNotFoundError: If file doesn't exist
        IOError: If file cannot be read
    """
    sha256_hash = hashlib.sha256()

    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256_hash.update(chunk)

    return sha256_hash.hexdigest()


def verify_hash(data: Union[bytes, str], expected_hash: str) -> bool:
    """
    Verify that data matches expected hash.

    Args:
        data: Data to verify (bytes or string)
        expected_hash: Expected hexadecimal hash string

    Returns:
        True if hash matches, False otherwise
    """
    computed_hash = compute_hash(data)
    return computed_hash.lower() == expected_hash.lower()


def verify_file_hash(file_path: str, expected_hash: str) -> bool:
    """
    Verify that file matches expected hash.

    Args:
        file_path: Path to file
        expected_hash: Expected hexadecimal hash string

    Returns:
        True if hash matches, False otherwise

    Raises:
        FileNotFoundError: If file doesn't exist
        IOError: If file cannot be read
    """
    computed_hash = compute_file_hash(file_path)
    return computed_hash.lower() == expected_hash.lower()

