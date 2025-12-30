"""Pytest configuration and shared fixtures."""

import sys
from pathlib import Path
import os
import base64

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set localhost for local testing (not in Docker)
if "RABBITMQ_HOST" not in os.environ:
    os.environ["RABBITMQ_HOST"] = "localhost"
if "IPFS_HOST" not in os.environ:
    os.environ["IPFS_HOST"] = "localhost"
if "BLOCKCHAIN_SERVICE_URL" not in os.environ:
    os.environ["BLOCKCHAIN_SERVICE_URL"] = "http://localhost:8080"

# Set up test encryption key BEFORE any imports that might use settings
# This ensures settings picks up the key when it's first created
if "ENCRYPTION_KEY" not in os.environ:
    # Generate a random 32-byte key and encode it as Base64
    # This is done before importing anything that might use settings
    import secrets
    test_key = secrets.token_bytes(32)
    test_key_b64 = base64.b64encode(test_key).decode("utf-8")
    os.environ["ENCRYPTION_KEY"] = test_key_b64
