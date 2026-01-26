#!/usr/bin/env python3
"""
Generate a manifest file for static assets with content-based fingerprinting.

This script scans the static directory, calculates a hash for each file,
and generates a JSON manifest mapping original filenames to fingerprinted versions.

The manifest is used in production to enable filename-based cache busting.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def calculate_file_hash(file_path: Path) -> str:
    """Calculate a hash of a file's contents for fingerprinting.

    Args:
        file_path: Path to the file to hash

    Returns:
        str: Hex digest of the file's hash (first 8 characters)
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()[:8]  # Use first 8 chars for brevity


def generate_fingerprinted_name(file_path: str, file_hash: str) -> str:
    """Generate a fingerprinted filename by inserting the hash before the extension.

    Args:
        file_path: Original file path
        file_hash: Hash to insert

    Returns:
        str: Fingerprinted filename
    """
    name, ext = os.path.splitext(file_path)
    return f"{name}.{file_hash}{ext}"


def scan_static_directory(
    static_dir: Path, exclude_dirs: Set[str] = set()
) -> Dict[str, str]:
    """Scan the static directory and generate a manifest of fingerprinted filenames.

    Args:
        static_dir: Path to the static directory
        exclude_dirs: Set of directory names to exclude from fingerprinting

    Returns:
        Dict mapping original paths (relative to static_dir) to fingerprinted paths
    """
    if exclude_dirs is None:
        exclude_dirs = set()

    manifest = {}

    # Walk through all files in the static directory
    for root, dirs, files in os.walk(static_dir):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        for file in files:
            file_path = Path(root) / file
            if not file_path.is_file():
                continue

            # Get path relative to static directory
            rel_path = file_path.relative_to(static_dir)
            rel_path_str = str(rel_path).replace("\\", "/")  # Normalize path separators

            # Calculate hash and generate fingerprinted name
            file_hash = calculate_file_hash(file_path)
            fingerprinted_name = generate_fingerprinted_name(rel_path_str, file_hash)

            # Add to manifest
            manifest[rel_path_str] = fingerprinted_name

    return manifest


def main() -> int:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Generate static asset manifest with fingerprinted filenames"
    )
    parser.add_argument(
        "--static-dir",
        type=str,
        default="shallweswim/static",
        help="Path to the static directory (default: shallweswim/static)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="shallweswim/static/asset-manifest.json",
        help="Output path for the manifest file (default: shallweswim/static/asset-manifest.json)",
    )
    parser.add_argument(
        "--exclude-dirs",
        type=str,
        nargs="*",
        default=["plots", "tidecharts"],
        help="Directories to exclude from fingerprinting (default: plots tidecharts)",
    )
    args = parser.parse_args()

    static_dir = Path(args.static_dir)
    output_path = Path(args.output)
    exclude_dirs = set(args.exclude_dirs)

    # Validate static directory
    if not static_dir.is_dir():
        logger.error(f"Static directory not found: {static_dir}")
        return 1

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Generate manifest
        logger.info(f"Scanning static directory: {static_dir}")
        manifest = scan_static_directory(static_dir, exclude_dirs)

        # Write manifest to file
        with open(output_path, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)

        logger.info(f"Generated manifest with {len(manifest)} entries: {output_path}")
        return 0

    except Exception as e:
        logger.error(f"Error generating manifest: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
