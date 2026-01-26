"""Background update utilities for feed data.

This module contains helper functions for the background update loop
that refreshes feed data periodically.
"""

import asyncio
import logging
from typing import Any

from shallweswim.clients.base import (
    BaseApiClient,
    BaseClientError,
    StationUnavailableError,
)
from shallweswim.core import feeds


def is_expired(feeds_dict: dict[str, feeds.Feed | None], dataset: str) -> bool:
    """Check if a dataset has expired and needs to be refreshed.

    Uses the feed's built-in expiration logic.

    Args:
        feeds_dict: Dictionary mapping dataset names to Feed objects
        dataset: The dataset to check

    Returns:
        True if the dataset is expired or missing, False otherwise
    """
    feed = feeds_dict.get(dataset)
    if feed is None:
        return True
    return feed.is_expired


async def update_dataset(
    feeds_dict: dict[str, feeds.Feed | None],
    clients: dict[str, BaseApiClient],
    dataset: str,
) -> None:
    """Update a specific dataset by fetching fresh data.

    Args:
        feeds_dict: Dictionary mapping dataset names to Feed objects
        clients: Dictionary of API clients
        dataset: The dataset to update
    """
    feed = feeds_dict.get(dataset)
    if feed is not None:
        await feed.update(clients=clients)


def handle_task_exception(
    task: asyncio.Task[Any],
    log_fn: Any,
) -> None:
    """Handle exceptions from asyncio tasks with appropriate logging levels.

    This callback logs exceptions at the appropriate level based on whether
    they're expected operational conditions or unexpected errors.

    Log levels:
    - WARNING: Expected conditions (station unavailable) - does NOT trigger alerts
    - ERROR: Unexpected conditions (API changes, bugs) - triggers alerts

    Note: This is a done_callback. Re-raising here doesn't crash the service -
    it just triggers asyncio's exception handler. The feed stays stale and
    retries on the next interval regardless.

    Args:
        task: The asyncio task that completed (successfully or with an exception)
        log_fn: Logging function that takes (message, level) parameters
    """
    try:
        # If the task raised an exception, this will re-raise it
        task.result()
    except asyncio.CancelledError:
        # Task was cancelled, which is normal during shutdown
        log_fn(f"Task {task.get_name()} was cancelled", logging.DEBUG)
    except StationUnavailableError as e:
        # Expected operational condition - station has no data
        # Log as WARNING (does NOT trigger GCP alerts)
        # Feed stays stale, will retry on next interval
        log_fn(f"Station unavailable: {e}", logging.WARNING)
    except BaseClientError as e:
        # Unexpected upstream issue - potential API change or parsing error
        # Log as ERROR (triggers GCP alerts)
        # Feed stays stale, will retry on next interval
        log_fn(f"Upstream error: {e}", logging.ERROR)
        raise  # Re-raise for extra visibility in asyncio's exception handler
    except Exception as e:
        # Internal error - bug in our code
        # Log as ERROR (triggers GCP alerts)
        log_fn(
            f"Unhandled exception in task {task.get_name()}: {e}",
            logging.ERROR,
        )
        raise  # Re-raise for extra visibility in asyncio's exception handler
