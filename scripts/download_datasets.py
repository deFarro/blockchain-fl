"""
Download a single dataset at Docker build time (driven by DATASET_NAME from .env).

Usage:
  python download_datasets.py DATASET_NAME [BASE_DIR]
  DATASET_NAME: mnist, caltech101, or usps (must match .env).
  BASE_DIR defaults to /app/data (dataset goes to BASE_DIR/DATASET_NAME).

Uses only torchvision (no project imports). Run with PYTHONPATH unset if needed.
"""

import os
import sys


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: download_datasets.py DATASET_NAME [BASE_DIR]", file=sys.stderr)
        sys.exit(1)
    dataset_name = sys.argv[1].lower()
    base_dir = sys.argv[2] if len(sys.argv) > 2 else "/app/data"
    base_dir = os.path.abspath(base_dir)
    os.makedirs(base_dir, exist_ok=True)

    if dataset_name == "mnist":
        from torchvision.datasets import MNIST

        mnist_root = os.path.join(base_dir, "mnist")
        print("Downloading MNIST (train)...", flush=True)
        MNIST(root=mnist_root, train=True, download=True)
        print("Downloading MNIST (test)...", flush=True)
        MNIST(root=mnist_root, train=False, download=True)
        print("MNIST ready at", mnist_root, flush=True)
    elif dataset_name == "caltech101":
        from torchvision.datasets import Caltech101

        caltech_root = os.path.join(base_dir, "caltech101")
        print("Downloading Caltech101...", flush=True)
        Caltech101(root=caltech_root, target_type="category", download=True)
        print("Caltech101 ready at", caltech_root, flush=True)
    elif dataset_name == "usps":
        from torchvision.datasets import USPS

        usps_root = os.path.join(base_dir, "usps")
        print("Downloading USPS (train)...", flush=True)
        USPS(root=usps_root, train=True, download=True)
        print("Downloading USPS (test)...", flush=True)
        USPS(root=usps_root, train=False, download=True)
        print("USPS ready at", usps_root, flush=True)
    else:
        print(
            f"Unknown dataset: {dataset_name}. Supported: mnist, caltech101, usps",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
