"""Building a location snapshot from a manager holding real feeds."""

import datetime

import pandas as pd
import pytest

from shallweswim.archive.store import MemoryObjectStore
from shallweswim.clients.base import StationUnavailableError
from shallweswim.core import feeds
from shallweswim.snapshot.build import build_location_snapshot, is_held
from shallweswim.snapshot.load import load_current
from shallweswim.snapshot.model import Snapshot
from shallweswim.snapshot.publish import publish
from shallweswim.snapshot.store import SnapshotStore
from tests.snapshot_fixtures import FETCHED, fresh_manager, seeded_manager


def test_builder_captures_every_feed_and_plot() -> None:
    manager = seeded_manager()
    tides = manager._feeds[feeds.FeedName.TIDES]
    assert tides is not None
    tides._last_error = StationUnavailableError("station 1234567 returned no data")
    tides._consecutive_failures = 2
    tides._next_fetch_after = FETCHED + datetime.timedelta(minutes=2)

    snapshot = build_location_snapshot(manager)

    assert set(snapshot.feeds) == set(feeds.FeedName)
    assert snapshot.failures == {}
    for feed_name, feed_snapshot in snapshot.feeds.items():
        feed = manager._feeds[feed_name]
        assert feed is not None
        assert feed_snapshot.frame is feed.values
        metadata = feed_snapshot.metadata
        assert metadata.source_identity == feed.feed_config.citation_key
        assert metadata.fetch_timestamp == FETCHED.replace(tzinfo=datetime.UTC)
        assert metadata.timezone == "US/Eastern"
        assert metadata.record_count == len(feed.values)
        assert metadata.expiration_seconds == feed.status.expiration_seconds

    tides_metadata = snapshot.feeds[feeds.FeedName.TIDES].metadata
    assert tides_metadata.consecutive_failures == 2
    assert tides_metadata.last_error == "station 1234567 returned no data"
    assert tides_metadata.next_fetch_after == (
        FETCHED + datetime.timedelta(minutes=2)
    ).replace(tzinfo=datetime.UTC)
    live_metadata = snapshot.feeds[feeds.FeedName.LIVE_TEMPS].metadata
    assert live_metadata.consecutive_failures == 0
    assert live_metadata.last_error is None
    assert live_metadata.next_fetch_after == (
        FETCHED + datetime.timedelta(minutes=10)
    ).replace(tzinfo=datetime.UTC)
    assert live_metadata.historical is None

    historical = snapshot.feeds[feeds.FeedName.HISTORIC_TEMPS].metadata.historical
    assert historical is not None
    assert historical.required_years == [2025, 2026]
    assert historical.available_years == [2025, 2026]
    assert historical.fetched_years == [2026]
    assert historical.failed_years == {}

    assert set(snapshot.plots) == set(feeds.PlotName)
    assert snapshot.plots[feeds.PlotName.LIVE_TEMPS].data == b"<svg>live</svg>"
    assert snapshot.plots[feeds.PlotName.LIVE_TEMPS].feed is feeds.FeedName.LIVE_TEMPS
    assert (
        snapshot.plots[feeds.PlotName.HISTORIC_TEMPS_12MO].feed
        is feeds.FeedName.HISTORIC_TEMPS
    )
    for plot in snapshot.plots.values():
        assert plot.feed_fetch_timestamp == FETCHED.replace(tzinfo=datetime.UTC)


def test_builder_records_a_configured_feed_without_data_as_a_failure() -> None:
    manager = seeded_manager()
    currents = manager._feeds[feeds.FeedName.CURRENTS]
    assert currents is not None
    currents._data = None
    currents._last_error = StationUnavailableError("station TEST001 returned no data")
    currents._consecutive_failures = 3
    currents._next_fetch_after = FETCHED + datetime.timedelta(minutes=5)
    # A feed that is not configured for the location stays absent entirely.
    manager._feeds[feeds.FeedName.TIDES] = None

    snapshot = build_location_snapshot(manager)

    assert set(snapshot.feeds) == {
        feeds.FeedName.LIVE_TEMPS,
        feeds.FeedName.HISTORIC_TEMPS,
    }
    assert set(snapshot.failures) == {feeds.FeedName.CURRENTS}
    failure = snapshot.failures[feeds.FeedName.CURRENTS]
    assert failure.source_identity == currents.feed_config.citation_key
    assert failure.consecutive_failures == 3
    assert failure.last_error == "station TEST001 returned no data"
    assert failure.next_fetch_after == (
        FETCHED + datetime.timedelta(minutes=5)
    ).replace(tzinfo=datetime.UTC)


def test_builder_records_a_feed_that_was_not_due_as_held() -> None:
    """A restored feed the run never attempted is held, not failed."""
    manager = fresh_manager()
    live = manager._feeds[feeds.FeedName.LIVE_TEMPS]
    assert live is not None
    # What restore_schedule leaves behind for a feed that is not yet due.
    live._next_fetch_after = FETCHED + datetime.timedelta(minutes=5)

    snapshot = build_location_snapshot(manager)

    assert is_held(live)
    assert set(snapshot.holds) == {feeds.FeedName.LIVE_TEMPS}
    assert snapshot.holds[feeds.FeedName.LIVE_TEMPS].source_identity == (
        live.feed_config.citation_key
    )
    assert feeds.FeedName.LIVE_TEMPS not in snapshot.failures
    assert feeds.FeedName.LIVE_TEMPS not in snapshot.feeds
    # Every other feed was due and produced nothing, which is a failure.
    assert set(snapshot.failures) == set(feeds.FeedName) - {feeds.FeedName.LIVE_TEMPS}


def test_builder_keeps_a_restored_feed_that_failed_as_a_failure() -> None:
    """A due feed that attempted and failed is never mistaken for a held one."""
    manager = fresh_manager()
    live = manager._feeds[feeds.FeedName.LIVE_TEMPS]
    assert live is not None
    live._next_fetch_after = FETCHED - datetime.timedelta(minutes=5)
    # What the feed's own failure path leaves behind: a retry in the future.
    live._last_error = StationUnavailableError("station 8518750 returned no data")
    live._schedule_after_failure(FETCHED)

    snapshot = build_location_snapshot(manager)

    assert not is_held(live)
    assert snapshot.holds == {}
    assert snapshot.failures[feeds.FeedName.LIVE_TEMPS].consecutive_failures == 1


def test_builder_reports_a_location_whose_feeds_all_failed() -> None:
    manager = seeded_manager()
    for feed_name in feeds.FeedName:
        feed = manager._feeds[feed_name]
        assert feed is not None
        feed._data = None
    manager._plots = {}

    snapshot = build_location_snapshot(manager)

    assert snapshot.feeds == {}
    assert snapshot.plots == {}
    assert set(snapshot.failures) == set(feeds.FeedName)
    assert snapshot.holds == {}


def test_builder_rejects_a_plot_whose_feed_has_no_data() -> None:
    manager = seeded_manager()
    live = manager._feeds[feeds.FeedName.LIVE_TEMPS]
    assert live is not None
    live._data = None

    with pytest.raises(ValueError, match="plot live_temps but no live_temps data"):
        build_location_snapshot(manager)


@pytest.mark.asyncio
async def test_built_snapshot_publishes_and_loads_equivalently() -> None:
    manager = seeded_manager()
    snapshot = Snapshot(locations={"nyc": build_location_snapshot(manager)})
    store = SnapshotStore(MemoryObjectStore())

    result = await publish(
        store,
        snapshot,
        run_id="run-1",
        now=datetime.datetime(2026, 6, 1, 12, 1, tzinfo=datetime.UTC),
    )
    loaded = await load_current(store)

    assert result.outcome == "success"
    assert loaded is not None
    for feed_name in feeds.FeedName:
        pd.testing.assert_frame_equal(
            manager.get_feed_values(feed_name),
            loaded.frames["nyc"][feed_name],
            check_freq=False,
        )
    assert loaded.plots["nyc"] == manager._plots
    published_feed = loaded.manifest.locations["nyc"].feeds[feeds.FeedName.TIDES]
    assert published_feed.source_identity == "coops:tide:1234567"
