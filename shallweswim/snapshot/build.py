"""Build a location's in-memory snapshot from its data manager."""

import datetime

from shallweswim.core import feeds
from shallweswim.core.manager import LocationDataManager
from shallweswim.snapshot.model import (
    FeedMetadata,
    FeedSnapshot,
    LocationSnapshot,
    PlotSnapshot,
)

# The feed each plot is drawn from, mirroring the manager's plot submission.
PLOT_SOURCE_FEEDS: dict[feeds.PlotName, feeds.FeedName] = {
    feeds.PlotName.LIVE_TEMPS: feeds.FeedName.LIVE_TEMPS,
    feeds.PlotName.HISTORIC_TEMPS_2MO: feeds.FeedName.HISTORIC_TEMPS,
    feeds.PlotName.HISTORIC_TEMPS_12MO: feeds.FeedName.HISTORIC_TEMPS,
}


def _as_utc(timestamp: datetime.datetime) -> datetime.datetime:
    """Make a feed's naive UTC scheduling timestamp explicit for the manifest."""
    return timestamp.replace(tzinfo=datetime.UTC)


def build_location_snapshot(manager: LocationDataManager) -> LocationSnapshot:
    """Capture the serving state a manager currently holds.

    Feeds without data are not published; every feed with data is published
    with its served frame and status metadata, and every generated plot is
    published with the fetch timestamp of the feed it was drawn from.

    Args:
        manager: The location's data manager.

    Returns:
        The location's feeds and plots ready for serialization.

    Raises:
        ValueError: If the location timezone has no IANA name, or a plot exists
            for a feed that has no data.
    """
    timezone = getattr(manager.config.timezone, "zone", None)
    if not isinstance(timezone, str):
        raise ValueError(f"Location {manager.config.code} timezone has no zone name")

    status = manager.status
    snapshot_feeds: dict[feeds.FeedName, FeedSnapshot] = {}
    for feed_name in feeds.FeedName:
        if not manager.has_feed_data(feed_name):
            continue
        feed = manager._feeds[feed_name]
        assert feed is not None
        feed_status = status.feeds[feed_name]
        assert feed_status.fetch_timestamp is not None
        frame = manager.get_feed_values(feed_name)
        snapshot_feeds[feed_name] = FeedSnapshot(
            frame=frame,
            metadata=FeedMetadata(
                # The feed's own citation key names what capture archives for
                # it; the manager exposes no other accessor for it.
                source_identity=feed.feed_config.citation_key,
                fetch_timestamp=_as_utc(feed_status.fetch_timestamp),
                next_fetch_after=(
                    None
                    if feed_status.next_fetch_after is None
                    else _as_utc(feed_status.next_fetch_after)
                ),
                expiration_seconds=feed_status.expiration_seconds,
                record_count=len(frame),
                consecutive_failures=feed_status.consecutive_failures,
                last_error=feed_status.error,
                timezone=timezone,
                historical=feed_status.historical_temp_status,
            ),
        )

    snapshot_plots: dict[feeds.PlotName, PlotSnapshot] = {}
    for plot_name, source_feed in PLOT_SOURCE_FEEDS.items():
        data = manager.get_plot(plot_name)
        if data is None:
            continue
        if source_feed not in snapshot_feeds:
            raise ValueError(
                f"Location {manager.config.code} has plot {plot_name} "
                f"but no {source_feed} data"
            )
        snapshot_plots[plot_name] = PlotSnapshot(
            data=data,
            feed=source_feed,
            feed_fetch_timestamp=snapshot_feeds[source_feed].metadata.fetch_timestamp,
        )

    return LocationSnapshot(feeds=snapshot_feeds, plots=snapshot_plots)
