"""Update utilities for feed data.

This module contains helper functions for the serving cycle that refreshes
feed data.
"""

from shallweswim.clients.base import BaseApiClient
from shallweswim.core import feeds


async def update_dataset(
    feeds_dict: dict[feeds.FeedName, feeds.Feed | None],
    clients: dict[str, BaseApiClient],
    dataset: feeds.FeedName,
) -> None:
    """Update a specific dataset by fetching fresh data.

    Args:
        feeds_dict: Dictionary mapping dataset names to Feed objects
        clients: Dictionary of API clients
        dataset: The dataset to update
    """
    feed = feeds_dict.get(dataset)
    if feed is not None:
        await feed.update(clients=clients, feed_name=dataset)
