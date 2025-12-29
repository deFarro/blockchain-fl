#!/usr/bin/env python3
"""Generate a Base64-encoded 32-byte encryption key."""

import base64
import os
import sys


def main():
    """Generate and print a Base64-encoded encryption key."""
    key = os.urandom(32)  # 32 bytes = 256 bits
    key_b64 = base64.b64encode(key).decode("utf-8")
    
    print("Generated encryption key:")
    print(key_b64)
    print()
    print("Add this to your .env file:")
    print(f"ENCRYPTION_KEY={key_b64}")
    print()
    print("Or set it as an environment variable:")
    print(f"export ENCRYPTION_KEY={key_b64}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

