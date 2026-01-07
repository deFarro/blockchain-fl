"""Async IPFS client using httpx."""

import json
import os
import sys
from typing import Optional, Dict, Any, Union, List
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


def _parse_json_string(json_str: str) -> Dict[str, Any]:
    """
    Parse JSON string with proper typing.

    Args:
        json_str: JSON string to parse

    Returns:
        Parsed JSON as dictionary
    """
    data: Any = json.loads(json_str)
    if isinstance(data, dict):
        return data
    raise ValueError(f"Expected dict, got {type(data)}")


class IPFSClient:
    """Async IPFS client using httpx for HTTP API calls."""

    def __init__(self, base_url: Optional[str] = None):
        """
        Initialize IPFS client.

        Args:
            base_url: IPFS API base URL. Defaults to settings.ipfs_url
        """
        self.base_url = base_url or settings.ipfs_url
        self.client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        """Async context manager entry."""
        # Disable progress bars and verbose output
        # Suppress progress output by redirecting stdout/stderr temporarily if needed
        # httpx doesn't show progress by default, but IPFS might
        # We'll use follow_redirects and disable any progress callbacks
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=30.0,
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.client:
            await self.client.aclose()

    async def _post(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make POST request to IPFS API.

        Args:
            endpoint: API endpoint (e.g., '/api/v0/add')
            **kwargs: Additional arguments for httpx.post

        Returns:
            JSON response as dictionary

        Raises:
            httpx.HTTPError: If request fails
        """
        if not self.client:
            raise RuntimeError("IPFSClient must be used as async context manager")

        url = f"{endpoint}"
        try:
            response = await self.client.post(url, **kwargs)
            response.raise_for_status()
            # IPFS API returns newline-delimited JSON for some endpoints
            text = response.text.strip()
            if text.startswith("{"):
                return _parse_json_response(response)
            else:
                # Handle newline-delimited JSON
                lines = [line for line in text.split("\n") if line.strip()]
                if len(lines) == 1:
                    return _parse_json_string(lines[0])
                else:
                    # If multiple lines, return the last one (most recent result)
                    return _parse_json_string(lines[-1])
        except httpx.HTTPError as e:
            logger.error(f"IPFS API error on {endpoint}: {str(e)}")
            raise

    async def add_bytes(self, data: bytes, pin: bool = True) -> str:
        """
        Add bytes data to IPFS.

        Args:
            data: Bytes to add
            pin: Whether to pin the content (default: True)

        Returns:
            IPFS Content Identifier (CID)
        """
        if not self.client:
            raise RuntimeError("IPFSClient must be used as async context manager")

        # IPFS API expects multipart/form-data with file field
        files = {"file": ("data", data, "application/octet-stream")}
        params = {"pin": str(pin).lower()}

        url = "/api/v0/add"
        try:
            response = await self.client.post(url, files=files, data=params)
            response.raise_for_status()

            # IPFS returns newline-delimited JSON
            text = response.text.strip()
            lines = [line for line in text.split("\n") if line.strip()]
            if not lines:
                raise ValueError("Empty response from IPFS")

            # Parse the last line (IPFS may return progress updates)
            result = _parse_json_string(lines[-1])

            # Access Hash key directly - IPFS always returns this
            if "Hash" not in result:
                raise ValueError("IPFS response missing Hash/CID")

            hash_value = result["Hash"]
            if not isinstance(hash_value, str):
                raise ValueError("IPFS response Hash is not a string")

            logger.info(f"Added data to IPFS: CID={hash_value}, size={len(data)} bytes")
            return hash_value
        except httpx.HTTPError as e:
            logger.error(f"Failed to add data to IPFS: {str(e)}")
            raise

    async def add_file(self, file_path: str, pin: bool = True) -> str:
        """
        Add file to IPFS.

        Args:
            file_path: Path to file
            pin: Whether to pin the content (default: True)

        Returns:
            IPFS Content Identifier (CID)
        """
        with open(file_path, "rb") as f:
            data = f.read()
        return await self.add_bytes(data, pin=pin)

    async def get_bytes(self, cid: str) -> bytes:
        """
        Retrieve bytes from IPFS by CID.

        Args:
            cid: IPFS Content Identifier

        Returns:
            Retrieved bytes

        Raises:
            httpx.HTTPError: If retrieval fails
        """
        if not self.client:
            raise RuntimeError("IPFSClient must be used as async context manager")

        url = f"/api/v0/cat?arg={cid}"
        try:
            # Use streaming to read content without progress bars
            # Read all content at once to avoid any progress indicators
            async with self.client.stream("POST", url) as response:
                response.raise_for_status()
                # Read all content without showing progress
                content = b""
                async for chunk in response.aiter_bytes():
                    content += chunk
            return content
        except httpx.HTTPError as e:
            logger.error(f"Failed to retrieve CID {cid}: {str(e)}")
            raise

    async def pin_add(self, cid: str) -> bool:
        """
        Pin content in IPFS.

        Args:
            cid: IPFS Content Identifier

        Returns:
            True if successful
        """
        if not self.client:
            raise RuntimeError("IPFSClient must be used as async context manager")

        url = f"/api/v0/pin/add?arg={cid}"
        try:
            response = await self.client.post(url)
            response.raise_for_status()
            result = _parse_json_response(response)
            # Check if pinning was successful
            pinned = result.get("Pins", [])
            if isinstance(pinned, list):
                return cid in pinned
            return False
        except httpx.HTTPError as e:
            logger.error(f"Failed to pin CID {cid}: {str(e)}")
            return False

    async def pin_rm(self, cid: str) -> bool:
        """
        Unpin content from IPFS.

        Args:
            cid: IPFS Content Identifier

        Returns:
            True if successful
        """
        if not self.client:
            raise RuntimeError("IPFSClient must be used as async context manager")

        url = f"/api/v0/pin/rm?arg={cid}"
        try:
            response = await self.client.post(url)
            response.raise_for_status()
            return True
        except httpx.HTTPError as e:
            logger.error(f"Failed to unpin CID {cid}: {str(e)}")
            return False

    async def pin_ls(self, cid: Optional[str] = None) -> Dict[str, Any]:
        """
        List pinned content.

        Args:
            cid: Optional CID to check. If None, lists all pinned content.

        Returns:
            Dictionary with pin information
        """
        if not self.client:
            raise RuntimeError("IPFSClient must be used as async context manager")

        url = "/api/v0/pin/ls"
        if cid:
            url += f"?arg={cid}"

        try:
            response = await self.client.post(url)
            response.raise_for_status()
            return _parse_json_response(response)
        except httpx.HTTPError as e:
            logger.error(f"Failed to list pins: {str(e)}")
            raise

    async def id(self) -> Dict[str, Any]:
        """
        Get IPFS node information.

        Returns:
            Dictionary with node ID and addresses
        """
        if not self.client:
            raise RuntimeError("IPFSClient must be used as async context manager")

        url = "/api/v0/id"
        try:
            response = await self.client.post(url)
            response.raise_for_status()
            return _parse_json_response(response)
        except httpx.HTTPError as e:
            logger.error(f"Failed to get IPFS node ID: {str(e)}")
            raise

    async def version(self) -> Dict[str, Any]:
        """
        Get IPFS version information.

        Returns:
            Dictionary with version info
        """
        if not self.client:
            raise RuntimeError("IPFSClient must be used as async context manager")

        url = "/api/v0/version"
        try:
            response = await self.client.post(url)
            response.raise_for_status()
            return _parse_json_response(response)
        except httpx.HTTPError as e:
            logger.error(f"Failed to get IPFS version: {str(e)}")
            raise


# Convenience function for async usage
async def add_to_ipfs(data: bytes, pin: bool = True) -> str:
    """
    Convenience function to add data to IPFS.

    Args:
        data: Bytes to add
        pin: Whether to pin the content

    Returns:
        IPFS Content Identifier (CID)
    """
    async with IPFSClient() as client:
        cid: str = await client.add_bytes(data, pin=pin)
        return cid


async def get_from_ipfs(cid: str) -> bytes:
    """
    Convenience function to retrieve data from IPFS.

    Args:
        cid: IPFS Content Identifier

    Returns:
        Retrieved bytes
    """
    async with IPFSClient() as client:
        data: bytes = await client.get_bytes(cid)
        return data
