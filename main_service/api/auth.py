"""API key authentication for FastAPI."""

from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from shared.config import settings
from shared.logger import setup_logger

logger = setup_logger(__name__)

# API Key header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """
    Verify API key from request header.

    Args:
        api_key: API key from X-API-Key header

    Returns:
        API key if valid

    Raises:
        HTTPException: If API key is missing or invalid
    """
    if not api_key:
        logger.warning("API request missing API key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide X-API-Key header.",
        )

    # Get expected API key from settings
    expected_key = settings.api_key

    if not expected_key:
        logger.warning("API key authentication not configured in settings")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key authentication not configured",
        )

    if api_key != expected_key:
        logger.warning(f"Invalid API key attempted: {api_key[:8]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    logger.debug("API key verified successfully")
    return api_key
