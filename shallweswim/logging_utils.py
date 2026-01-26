"""Logging utilities for the ShallWeSwim application.

This module provides logging configuration and utilities for the application,
including custom filters and handlers for different environments.
"""

import logging
import os

import google.cloud.logging  # type: ignore[import]

# Determine project root for relative log paths (directory containing this file)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class RelativePathFilter(logging.Filter):
    """Logging filter to add relative path attribute to LogRecords."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Calculate and add 'relativepath' to the log record.

        Args:
            record: The log record to process

        Returns:
            Always True to process the record
        """
        try:
            # Use os.path.normpath for cross-platform compatibility
            record.relativepath = os.path.relpath(
                os.path.normpath(record.pathname), PROJECT_ROOT
            )
        except ValueError:
            record.relativepath = record.pathname  # Fallback if calculation fails
        return True  # Always process the record


def _configure_local_handler(root_logger: logging.Logger) -> None:
    """Configure a stream handler for local execution using relative path.

    Args:
        root_logger: The root logger to configure
    """
    # Clear existing handlers to ensure clean setup
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    log_format = "%(levelname)s:%(relativepath)s:%(lineno)d: %(message)s"
    formatter = logging.Formatter(log_format)  # Standard formatter
    handler = logging.StreamHandler()  # Defaults to sys.stderr
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


def setup_logging() -> None:
    """Configures logging based on the environment (Cloud Run or local)."""
    root_logger = logging.getLogger()

    # Add the filter to inject 'relativepath' into all records
    # Ensure only one instance is added
    if not any(isinstance(f, RelativePathFilter) for f in root_logger.filters):
        root_logger.addFilter(RelativePathFilter())

    # Set root logger level unconditionally
    root_logger.setLevel(logging.INFO)

    # If running in Google Cloud Run, use cloud logging
    if "K_SERVICE" in os.environ:
        # Setup Google Cloud logging
        # By default this captures all logs at INFO level and higher
        log_client = google.cloud.logging.Client()  # type: ignore[no-untyped-call]
        log_client.get_default_handler()  # type: ignore[no-untyped-call]
        log_client.setup_logging()  # type: ignore[no-untyped-call]
        logging.info("Using google cloud logging")
    else:
        # Use the helper function to configure local logging
        _configure_local_handler(root_logger)
        logging.info("Using standard stream handler with relative path format")
