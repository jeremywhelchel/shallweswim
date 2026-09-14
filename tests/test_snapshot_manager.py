"""Serving one location from a loaded generation.

A `SnapshotLocationManager` must answer every route-facing call exactly as
the feed-backed `LocationDataManager` that produced the generation would, and
must report status by the feed rules from the manifest alone.
"""

import datetime
from unittest.mock import patch

import pandas as pd
import pytest

from shallweswim.archive.store import MemoryObjectStore
from shallweswim.core.feeds import FeedName, PlotName
from shallweswim.core.manager import LocationDataManager
from shallweswim.core.queries import DataUnavailableError
from shallweswim.core.serving import LocationServing
from shallweswim.snapshot.build import build_location_snapshot
from shallweswim.snapshot.load import load_current
from shallweswim.snapshot.manager import (
    SnapshotLocationManager,
    feed_status_from_manifest,
)
from shallweswim.snapshot.model import (
    FeedObject,
    LocationManifest,
    Snapshot,
)
from shallweswim.snapshot.publish import publish
from shallweswim.snapshot.store import SnapshotStore
from tests.conftest import TEST_CONFIG_FULL
from tests.snapshot_fixtures import FETCHED, FETCHED_AT, seeded_manager, tides_frame

# A local time inside every fixture frame's window.
LOCAL_T = datetime.datetime(2026, 6, 1, 4, 0)
LOADED_AT = datetime.datetime(2026, 6, 1, 12, 30, tzinfo=datetime.UTC)
# The naive UTC clock both managers measure age against.
NOW = FETCHED + datetime.timedelta(minutes=3)


def _serving(serving: LocationServing) -> LocationServing:
    """Typed pass-through: pyrefly proves the argument satisfies the Protocol."""
    return serving


async def _load_manager(manager: LocationDataManager) -> SnapshotLocationManager:
    """Publish the manager's serving state and load it back as a snapshot manager."""
    store = SnapshotStore(MemoryObjectStore())
    snapshot = Snapshot(locations={"nyc": build_location_snapshot(manager)})
    result = await publish(store, snapshot, run_id="run-1", now=LOADED_AT)
    assert result.outcome == "success"
    loaded = await load_current(store)
    assert loaded is not None
    return SnapshotLocationManager(
        TEST_CONFIG_FULL,
        loaded.manifest.locations["nyc"],
        loaded.frames["nyc"],
        loaded.plots["nyc"],
        LOADED_AT,
    )


def _empty_manager() -> SnapshotLocationManager:
    return SnapshotLocationManager(
        TEST_CONFIG_FULL, LocationManifest(feeds={}, plots={}), {}, {}, LOADED_AT
    )


def _feed_object(
    *,
    fetch_timestamp: datetime.datetime = FETCHED_AT,
    next_fetch_after: datetime.datetime | None = FETCHED_AT
    + datetime.timedelta(minutes=10),
    expiration_seconds: float | None = 600.0,
    consecutive_failures: int = 0,
    last_error: str | None = None,
) -> FeedObject:
    return FeedObject(
        key="published/objects/sha256-abc.parquet",
        size_bytes=1,
        source_identity="coops:tides:1234567",
        fetch_timestamp=fetch_timestamp,
        next_fetch_after=next_fetch_after,
        expiration_seconds=expiration_seconds,
        record_count=4,
        consecutive_failures=consecutive_failures,
        last_error=last_error,
        timezone="US/Eastern",
        historical=None,
    )


def test_both_managers_satisfy_the_serving_protocol() -> None:
    assert _serving(seeded_manager()) is not None
    assert _serving(_empty_manager()) is not None


@pytest.mark.asyncio
async def test_snapshot_manager_answers_like_the_manager_it_was_built_from() -> None:
    manager = seeded_manager()
    snapshot_manager = await _load_manager(manager)

    assert snapshot_manager.config is TEST_CONFIG_FULL
    assert snapshot_manager.loaded_at == LOADED_AT
    assert snapshot_manager.has_data is manager.has_data is True
    for feed_name in FeedName:
        assert snapshot_manager.has_feed(feed_name) is manager.has_feed(feed_name)
        assert snapshot_manager.has_feed_data(feed_name) is manager.has_feed_data(
            feed_name
        )
        pd.testing.assert_frame_equal(
            snapshot_manager.get_feed_values(feed_name),
            manager.get_feed_values(feed_name),
            check_freq=False,
        )
    assert snapshot_manager.has_feed("nope") is manager.has_feed("nope") is False
    for plot_name in PlotName:
        assert snapshot_manager.get_plot(plot_name) == manager.get_plot(plot_name)

    assert (
        snapshot_manager.get_current_temperature() == manager.get_current_temperature()
    )
    assert snapshot_manager.get_tide_info_at_time(
        LOCAL_T
    ) == manager.get_tide_info_at_time(LOCAL_T)
    predicted = snapshot_manager.predict_tide_at_time(LOCAL_T)
    assert predicted is not None
    assert predicted == manager.predict_tide_at_time(LOCAL_T)
    assert snapshot_manager.get_chart_info(LOCAL_T) == manager.get_chart_info(LOCAL_T)
    assert snapshot_manager.get_current_flow_info() == manager.get_current_flow_info()
    assert snapshot_manager.predict_flow_at_time(
        LOCAL_T
    ) == manager.predict_flow_at_time(LOCAL_T)

    with (
        patch("shallweswim.core.feeds.utc_now", return_value=NOW),
        patch("shallweswim.snapshot.manager.utc_now", return_value=NOW),
    ):
        expected = manager.status
        actual = snapshot_manager.status
    assert set(actual.feeds) == set(expected.feeds) == {name.value for name in FeedName}
    for name, status in actual.feeds.items():
        # The manifest carries no feed class name, so `name` differs by design.
        # The summary's byte count is excluded: a categorical column's reported
        # memory depends on pandas' internal caches, not on the served data.
        assert status.name == name
        exclude = {"name": True, "data_summary": {"memory_usage_bytes"}}
        assert status.model_dump(exclude=exclude) == expected.feeds[name].model_dump(
            exclude=exclude
        )


def test_status_applies_the_feed_rules_to_manifest_metadata() -> None:
    frame = tides_frame()

    fresh = feed_status_from_manifest(
        FeedName.TIDES,
        "nyc",
        _feed_object(),
        frame,
        FETCHED + datetime.timedelta(minutes=1),
    )
    assert fresh.name == "tides"
    assert fresh.location == "nyc"
    assert fresh.fetch_timestamp == FETCHED
    assert fresh.next_fetch_after == FETCHED + datetime.timedelta(minutes=10)
    assert fresh.age_seconds == 60.0
    assert fresh.seconds_until_next_fetch == 540.0
    assert fresh.is_expired is False
    assert fresh.is_healthy is True
    assert fresh.expiration_seconds == 600.0
    assert fresh.consecutive_failures == 0
    assert fresh.error is None
    assert fresh.data_summary is not None
    assert fresh.data_summary.length == len(frame)

    # Past the next fetch time but inside the health buffer.
    due = feed_status_from_manifest(
        FeedName.TIDES,
        "nyc",
        _feed_object(),
        frame,
        FETCHED + datetime.timedelta(minutes=12),
    )
    assert due.is_expired is True
    assert due.is_healthy is True
    assert due.seconds_until_next_fetch == 0.0

    # Older than the expiration interval plus the fifteen-minute buffer.
    stale = feed_status_from_manifest(
        FeedName.TIDES,
        "nyc",
        _feed_object(),
        frame,
        FETCHED + datetime.timedelta(minutes=26),
    )
    assert stale.is_expired is True
    assert stale.is_healthy is False

    # A carried-forward entry: the original fetch, this run's failure fields.
    carried = feed_status_from_manifest(
        FeedName.TIDES,
        "nyc",
        _feed_object(
            next_fetch_after=FETCHED_AT + datetime.timedelta(hours=2, minutes=1),
            consecutive_failures=3,
            last_error="boom",
        ),
        frame,
        FETCHED + datetime.timedelta(hours=2),
    )
    assert carried.age_seconds == 7200.0
    assert carried.is_expired is False
    assert carried.is_healthy is False
    assert carried.seconds_until_next_fetch == 60.0
    assert carried.consecutive_failures == 3
    assert carried.error == "boom"

    # A feed that never refreshes is never expired and always healthy.
    forever = feed_status_from_manifest(
        FeedName.TIDES,
        "nyc",
        _feed_object(next_fetch_after=None, expiration_seconds=None),
        frame,
        FETCHED + datetime.timedelta(days=400),
    )
    assert forever.is_expired is False
    assert forever.is_healthy is True
    assert forever.seconds_until_next_fetch is None

    # A refreshing feed with no scheduled fetch is due, as a fresh feed is.
    unscheduled = feed_status_from_manifest(
        FeedName.TIDES, "nyc", _feed_object(next_fetch_after=None), frame, FETCHED
    )
    assert unscheduled.is_expired is True


def test_status_property_reads_the_clock_at_call_time() -> None:
    manager = SnapshotLocationManager(
        TEST_CONFIG_FULL,
        LocationManifest(feeds={FeedName.TIDES: _feed_object()}, plots={}),
        {FeedName.TIDES: tides_frame()},
        {},
        LOADED_AT,
    )
    with patch(
        "shallweswim.snapshot.manager.utc_now",
        return_value=FETCHED + datetime.timedelta(minutes=30),
    ):
        status = manager.status
    assert set(status.feeds) == {"tides"}
    assert status.feeds["tides"].age_seconds == 1800.0
    assert status.feeds["tides"].is_healthy is False


def test_empty_location_reports_no_data_and_serves_nothing() -> None:
    manager = _empty_manager()

    assert manager.has_data is False
    assert manager.status.feeds == {}
    assert manager.has_feed("tides") is True
    assert manager.has_feed("nope") is False
    assert manager.has_feed_data(FeedName.TIDES) is False
    assert manager.get_plot(PlotName.LIVE_TEMPS) is None
    assert manager.predict_tide_at_time(LOCAL_T) is None
    with pytest.raises(KeyError):
        manager.get_feed_values("nope")
    with pytest.raises(DataUnavailableError):
        manager.get_feed_values(FeedName.TIDES)
    with pytest.raises(DataUnavailableError):
        manager.get_current_temperature()
    with pytest.raises(DataUnavailableError):
        manager.get_tide_info_at_time(LOCAL_T)
    with pytest.raises(DataUnavailableError):
        manager.get_current_flow_info()
    with pytest.raises(DataUnavailableError):
        manager.predict_flow_at_time(LOCAL_T)


def test_constructor_rejects_an_inconsistent_generation() -> None:
    manifest = LocationManifest(feeds={FeedName.TIDES: _feed_object()}, plots={})
    with pytest.raises(ValueError, match="without frames"):
        SnapshotLocationManager(TEST_CONFIG_FULL, manifest, {}, {}, LOADED_AT)
    with pytest.raises(ValueError, match="timezone-aware"):
        SnapshotLocationManager(
            TEST_CONFIG_FULL,
            manifest,
            {FeedName.TIDES: tides_frame()},
            {},
            LOADED_AT.replace(tzinfo=None),
        )
