"""Logging configuration."""

import logging
import sys
from typing import Optional
from pythonjsonlogger import jsonlogger
from shared.config import settings


def setup_logger(name: str, log_level: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger with JSON formatting.

    Args:
        name: Logger name (typically __name__)
        log_level: Logging level (defaults to settings.log_level)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    if log_level is None:
        log_level = settings.log_level

    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Remove existing handlers
    logger.handlers.clear()

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Use JSON formatter for structured logging
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance (convenience function).

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return setup_logger(name)

