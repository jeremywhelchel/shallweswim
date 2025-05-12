"""Asset management for the ShallWeSwim application.

This module provides functionality for managing static assets, including
fingerprinting for cache busting in production environments.
"""

import json
import logging
import os
from typing import Dict, Optional


def load_asset_manifest(manifest_path: str) -> Optional[Dict[str, str]]:
    """Load the asset manifest from a JSON file.

    Args:
        manifest_path: Path to the asset manifest JSON file

    Returns:
        Dictionary mapping original file paths to fingerprinted file paths,
        or None if the manifest could not be loaded
    """
    logging.info(f"Loading asset manifest from {manifest_path}")

    # Check if the manifest path is absolute or relative
    if not os.path.isabs(manifest_path):
        # Try to find the manifest in the current directory or parent directories
        current_dir = os.getcwd()
        while current_dir:
            full_path = os.path.join(current_dir, manifest_path)
            logging.info(f"Checking path: {full_path}")
            if os.path.exists(full_path):
                manifest_path = full_path
                logging.info(f"Found manifest at: {manifest_path}")
                break
            parent_dir = os.path.dirname(current_dir)
            if parent_dir == current_dir:
                break
            current_dir = parent_dir

    try:
        logging.info(f"Opening manifest file: {manifest_path}")
        with open(manifest_path, "r") as f:
            manifest: Dict[str, str] = json.load(f)
            logging.info(f"Loaded asset manifest with {len(manifest)} entries")
            # Log a few sample entries to help with debugging
            sample_entries = list(manifest.items())[:3]
            logging.info(f"Sample entries: {sample_entries}")
            return manifest
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Failed to load asset manifest: {e}")
        return None


class AssetManager:
    """Manages static assets and provides fingerprinting functionality."""

    def __init__(self) -> None:
        """Initialize the asset manager with an empty manifest."""
        self.manifest: Dict[str, str] = {}

    def get_fingerprinted_path(self, file_path: str) -> str:
        """Get the fingerprinted path for a static file.

        Args:
            file_path: Path to the static file, relative to the static directory

        Returns:
            Fingerprinted path if available, otherwise the original path

        Raises:
            KeyError: If the manifest is populated but the file is not in it
        """
        if self.manifest and file_path in self.manifest:
            fingerprinted_path: str = self.manifest[file_path]
            return fingerprinted_path
        elif self.manifest:
            # If we have a manifest but the file is not in it, that's an error
            # Fail fast and loud - never silently fall back
            raise KeyError(f"File not found in asset manifest: {file_path}")
        else:
            # If we don't have a manifest, just return the original path
            return file_path

    def static_url(self, file_path: str) -> str:
        """Generate a URL for a static file, using fingerprinting if available.

        Args:
            file_path: Path to the static file, relative to the static directory

        Returns:
            URL for the static file
        """
        # Get the fingerprinted path if available
        fingerprinted_path = self.get_fingerprinted_path(file_path)

        # Log which path is being used to help with debugging
        if fingerprinted_path != file_path:
            logging.debug(
                f"Using fingerprinted path: {fingerprinted_path} for {file_path}"
            )
        else:
            logging.debug(f"Using original path: {file_path}")

        # Return the URL with the fingerprinted path
        url: str = f"/static/{fingerprinted_path}"
        return url
