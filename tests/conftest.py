"""Pytest configuration and shared fixtures."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set localhost for local testing (not in Docker)
import os

if "RABBITMQ_HOST" not in os.environ:
    os.environ["RABBITMQ_HOST"] = "localhost"
if "IPFS_HOST" not in os.environ:
    os.environ["IPFS_HOST"] = "localhost"
if "BLOCKCHAIN_SERVICE_URL" not in os.environ:
    os.environ["BLOCKCHAIN_SERVICE_URL"] = "http://localhost:8080"
