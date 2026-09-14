"""Build a location's in-memory snapshot from its data manager."""

import datetime

from shallweswim.core import feeds
from shallweswim.core.manager import LocationDataManager
from shallweswim.snapshot.model import (
    FeedFailure,
    FeedHold,
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


def is_held(feed: feeds.Feed) -> bool:
    """Whether the run left this feed on the generation it was restored from.

    A held feed holds no data in this process, carries a scheduled fetch time,
    and has recorded no failure. In a bounded run that is exactly a feed whose
    schedule `capture.restore_schedule` restored and whose `update()` then did
    nothing because the feed was not yet due: a feed that has never attempted a
    fetch in this process can only have a scheduled time from restoration, a
    successful attempt leaves data, and a failed one leaves a failure count.
    The run summary counts a held feed as published, because the generation it
    was restored from keeps serving its frame.

    Args:
        feed: A configured feed after the run's serving cycle.

    Returns:
        True if the feed was not due and was never attempted this run.
    """
    return (
        not feed.has_data
        and feed._next_fetch_after is not None
        and feed._fetch_timestamp is None
        and feed._consecutive_failures == 0
    )


def build_location_snapshot(manager: LocationDataManager) -> LocationSnapshot:
    """Capture the serving state a manager currently holds.

    Every feed configured for the location is reported exactly once: one that
    holds data with its served frame and status metadata, one the run did not
    fetch because it was not due as a hold, and one that was due and produced
    nothing with only what this run learned about the failure, so manifest
    assembly can carry the last published entry forward in either case. Feeds
    that are not configured for the location are absent. Every generated plot
    is published with the fetch timestamp of the feed it was drawn from.

    Args:
        manager: The location's data manager.

    Returns:
        The location's feeds, failures, and plots ready for serialization.

    Raises:
        ValueError: If the location timezone has no IANA name, or a plot exists
            for a feed that has no data.
    """
    timezone = getattr(manager.config.timezone, "zone", None)
    if not isinstance(timezone, str):
        raise ValueError(f"Location {manager.config.code} timezone has no zone name")

    status = manager.status
    snapshot_feeds: dict[feeds.FeedName, FeedSnapshot] = {}
    snapshot_failures: dict[feeds.FeedName, FeedFailure] = {}
    snapshot_holds: dict[feeds.FeedName, FeedHold] = {}
    for feed_name in feeds.FeedName:
        # The manager exposes no accessor for the feed objects themselves, and
        # a feed's citation key and status are what a failure record holds.
        feed = manager._feeds.get(feed_name)
        if feed is None:
            continue
        feed_status = status.feeds[feed_name]
        if not manager.has_feed_data(feed_name):
            if is_held(feed):
                snapshot_holds[feed_name] = FeedHold(
                    source_identity=feed.feed_config.citation_key
                )
                continue
            snapshot_failures[feed_name] = FeedFailure(
                source_identity=feed.feed_config.citation_key,
                consecutive_failures=feed_status.consecutive_failures,
                last_error=feed_status.error,
                next_fetch_after=(
                    None
                    if feed_status.next_fetch_after is None
                    else _as_utc(feed_status.next_fetch_after)
                ),
            )
            continue
        assert feed_status.fetch_timestamp is not None
        frame = manager.get_feed_values(feed_name)
        snapshot_feeds[feed_name] = FeedSnapshot(
            frame=frame,
            metadata=FeedMetadata(
                # The feed's own citation key names what capture archives for
                # it, and identifies the source a carried-forward entry must
                # still match.
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

    return LocationSnapshot(
        feeds=snapshot_feeds,
        plots=snapshot_plots,
        failures=snapshot_failures,
        holds=snapshot_holds,
    )
