"""Main entry point when running as module: python -m main_service."""

import uvicorn
from shared.config import settings
from shared.logger import setup_logger
from main_service.api.server import app

logger = setup_logger(__name__)

if __name__ == "__main__":
    port = settings.api_port
    host = settings.api_host

    logger.info(f"Starting API server on {host}:{port}")
    logger.info(
        f"API key authentication: {'enabled' if settings.api_key else 'disabled (not configured)'}"
    )

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=settings.log_level.lower(),
    )

