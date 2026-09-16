"""One-shot capture job behavior over mocked clients and a memory archive.

These tests exercise ``shallweswim.update`` through the real capture hook
(``shallweswim.archive.capture``) so archived keys and rows are asserted from an
in-memory object store rather than mocked out.
"""

import asyncio
import datetime
import logging
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from types import MappingProxyType
from typing import cast
from unittest.mock import Mock

import aiohttp
import pandas as pd
import pytest
import pytz

from shallweswim import config, update
from shallweswim.archive import store as archive_store
from shallweswim.archive.observations import (
    CURRENTS_UNIT,
    TEMPERATURE_UNIT,
    read_observations,
)
from shallweswim.archive.store import MemoryObjectStore
from shallweswim.clients.base import BaseApiClient, StationUnavailableError
from shallweswim.clients.coops import CoopsApi
from shallweswim.clients.nwis import NwisApi, NwisConnectionError
from shallweswim.config import (
    CoopsTempFeedConfig,
    LocationConfig,
    NwisTempFeedConfig,
)
from shallweswim.core import backfill, feeds
from shallweswim.core import manager as manager_module
from shallweswim.core.manager import build_feeds
from shallweswim.snapshot.load import load_current
from shallweswim.snapshot.model import (
    CurrentPointer,
    FeedObject,
    LocationManifest,
    Manifest,
)
from shallweswim.snapshot.store import CURRENT_KEY, SnapshotStore
from shallweswim.types import TIDE_TYPE_CATEGORIES
from shallweswim.util import utc_now
from tests.conftest import TEST_CONFIG_FULL, TEST_CONFIG_OBSERVATION_CURRENTS
from tests.snapshot_fixtures import fresh_manager

# Fixed observation timestamps keep archived partition keys deterministic
# regardless of when the suite runs, and avoid daylight-saving transitions.
OBSERVATION_START = "2026-06-01"
OBSERVATION_YEAR = 2026

TEMPERATURE_KEY = f"archive/temperature/coops/2345678/{OBSERVATION_YEAR}.parquet"
CURRENTS_KEY = f"archive/currents/nwis/12345678%3A72255/{OBSERVATION_YEAR}.parquet"

# Historical source whose configured range ended before the current year.
RETIRED_HISTORY_CONFIG = LocationConfig(
    code="pst",
    name="Retired History Location",
    swim_location="Old Beach",
    swim_location_link="http://example.com/old",
    description="Test location whose historical source stopped reporting",
    latitude=41.0,
    longitude=-71.0,
    timezone=pytz.timezone("US/Eastern"),
    default_temperature_unit="F",
    live_temp_source=CoopsTempFeedConfig(station=5555555, name="Retired Temp"),
    historic_temp_source=CoopsTempFeedConfig(
        station=5555555, name="Retired Temp", start_year=2015, end_year=2018
    ),
    enabled=True,
)

# Historical-only source with a multi-year configured range.
MULTI_YEAR_HISTORY_CONFIG = LocationConfig(
    code="mly",
    name="Multi Year History Location",
    swim_location="Long Record Beach",
    swim_location_link="http://example.com/long",
    description="Test location with several configured historical years",
    latitude=42.0,
    longitude=-70.0,
    timezone=pytz.timezone("US/Eastern"),
    default_temperature_unit="F",
    historic_temp_source=CoopsTempFeedConfig(
        station=6666666, name="Long Record Temp", start_year=2024
    ),
    enabled=True,
)


@pytest.fixture(autouse=True)
def clear_store_cache() -> Iterator[None]:
    """Keep mocked stores from leaking between capture tests."""
    archive_store.gcs_store.cache_clear()
    yield
    archive_store.gcs_store.cache_clear()


@pytest.fixture(autouse=True)
def setup_logging_mock(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Replace job logging setup, which would otherwise drop caplog handlers."""
    mock = Mock(return_value="console")
    monkeypatch.setattr(update.logging_utils, "setup_logging", mock)
    return mock


def _temp_frame() -> pd.DataFrame:
    """Return a fixed day of hourly water temperatures, as a client returns them."""
    index = pd.date_range(
        OBSERVATION_START, periods=24, freq="h", tz="UTC", name="time"
    )
    return pd.DataFrame({"water_temp": [68.5] * len(index)}, index=index)


def _currents_frame() -> pd.DataFrame:
    """Return a fixed day of hourly NWIS velocity observations in UTC."""
    index = pd.date_range(
        OBSERVATION_START, periods=24, freq="h", tz="UTC", name="time"
    )
    return pd.DataFrame({"velocity_fps": [1.5] * len(index)}, index=index)


def _tides_frame() -> pd.DataFrame:
    """Return a minimal high/low tide frame for publishing runs."""
    index = pd.date_range(
        OBSERVATION_START, periods=4, freq="6h", tz="UTC", name="time"
    )
    return pd.DataFrame(
        {
            "prediction": [-0.5, 1.2, -0.3, 1.4],
            "type": pd.Categorical(
                ["low", "high", "low", "high"], categories=TIDE_TYPE_CATEGORIES
            ),
        },
        index=index,
    )


class MockCoopsApi(CoopsApi):
    """CO-OPS client returning fixed frames and counting calls per product."""

    def __init__(self) -> None:
        super().__init__(session=cast(aiohttp.ClientSession, None))
        self.tides_calls = 0
        self.currents_calls = 0
        self.live_temperature_calls = 0
        self.historic_temperature_calls = 0
        self.live_temperature_error: Exception | None = None
        self.historic_temperature_error: Exception | None = None
        self.tides_error: Exception | None = None
        self.currents_error: Exception | None = None

    async def tides(
        self, station: int, timezone: str, location_code: str = "unknown"
    ) -> pd.DataFrame:
        """Count tide requests; only a publishing run makes them."""
        self.tides_calls += 1
        if self.tides_error:
            raise self.tides_error
        return _tides_frame()

    async def currents(
        self,
        station: str,
        timezone: str,
        interpolate: bool = True,
        location_code: str = "unknown",
    ) -> pd.DataFrame:
        """Count prediction current requests; only a publishing run makes them."""
        self.currents_calls += 1
        if self.currents_error:
            raise self.currents_error
        index = pd.date_range(
            OBSERVATION_START, periods=4, freq="h", tz="UTC", name="time"
        )
        return pd.DataFrame({"velocity": [1.0, 1.1, 0.9, 0.8]}, index=index)

    async def temperature(
        self,
        station: int,
        product: str,
        begin_date: object,
        end_date: object,
        timezone: str,
        interval: str | None = None,
        location_code: str = "unknown",
    ) -> pd.DataFrame:
        """Return fixed temperatures, tracking live and historical calls apart."""
        if interval == "6-min":
            self.live_temperature_calls += 1
            if self.live_temperature_error:
                raise self.live_temperature_error
        else:
            self.historic_temperature_calls += 1
            if self.historic_temperature_error:
                raise self.historic_temperature_error
        return _temp_frame()


class MockNwisApi(NwisApi):
    """NWIS client returning fixed observational currents."""

    def __init__(self) -> None:
        super().__init__(session=cast(aiohttp.ClientSession, None))
        self.currents_calls = 0
        self.currents_error: Exception | None = None

    async def currents(
        self,
        site_no: str,
        parameter_cd: str,
        timezone: str,
        location_code: str = "unknown",
    ) -> pd.DataFrame:
        """Return fixed velocity observations or raise the configured error."""
        self.currents_calls += 1
        if self.currents_error:
            raise self.currents_error
        return _currents_frame()


def _install_job_environment(
    monkeypatch: pytest.MonkeyPatch,
    configs: list[LocationConfig],
) -> tuple[MockCoopsApi, MockNwisApi, MemoryObjectStore]:
    """Point the job at fake locations, mocked clients, and a memory archive."""
    monkeypatch.setenv("SHALLWESWIM_ARCHIVE_BUCKET", "test-archive")
    monkeypatch.setattr(
        config,
        "CONFIGS",
        MappingProxyType({item.code: item for item in configs}),
    )
    coops_client = MockCoopsApi()
    nwis_client = MockNwisApi()
    clients: dict[str, BaseApiClient] = {"coops": coops_client, "nwis": nwis_client}
    monkeypatch.setattr(update, "create_api_clients", lambda session: clients)
    store = MemoryObjectStore()
    monkeypatch.setattr(archive_store, "GcsObjectStore", lambda bucket: store)
    return coops_client, nwis_client, store


def _summary_record(caplog: pytest.LogCaptureFixture) -> logging.LogRecord:
    """Return the single run summary event emitted by the job."""
    records = [
        record for record in caplog.records if getattr(record, "operation", "") == "run"
    ]
    assert len(records) == 1
    return records[0]


def _archived_rows(store: MemoryObjectStore, key: str, unit: str) -> pd.DataFrame:
    """Read one archived partition back out of the memory store."""
    stored = asyncio.run(store.read(key))
    assert stored is not None
    return read_observations(BytesIO(stored.data), expected_unit=unit)


def test_run_outcome_counts() -> None:
    assert update.run_outcome(0, 0) == "success"
    assert update.run_outcome(3, 3) == "success"
    assert update.run_outcome(1, 3) == "partial"
    assert update.run_outcome(0, 3) == "failed"


def test_run_captures_observations_and_skips_predictions(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    setup_logging_mock: Mock,
) -> None:
    coops_client, nwis_client, store = _install_job_environment(
        monkeypatch, [TEST_CONFIG_OBSERVATION_CURRENTS, TEST_CONFIG_FULL]
    )

    with caplog.at_level(logging.INFO):
        exit_code = update.main([])

    assert exit_code == 0
    setup_logging_mock.assert_called_once()
    assert coops_client.tides_calls == 0
    assert coops_client.currents_calls == 0
    assert nwis_client.currents_calls == 1

    assert len(_archived_rows(store, TEMPERATURE_KEY, TEMPERATURE_UNIT)) == 24
    assert len(_archived_rows(store, CURRENTS_KEY, CURRENTS_UNIT)) == 24
    assert _summary_record(caplog).outcome == "success"


def test_default_run_limits_history_to_current_year(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coops_client, _, _ = _install_job_environment(
        monkeypatch, [MULTI_YEAR_HISTORY_CONFIG]
    )
    clients: dict[str, BaseApiClient] = {"coops": coops_client}

    default_feed = build_feeds(
        MULTI_YEAR_HISTORY_CONFIG, clients, historic_start_year=utc_now().year
    )[feeds.FEED_HISTORIC_TEMPS]
    full_feed = build_feeds(MULTI_YEAR_HISTORY_CONFIG, clients)[
        feeds.FEED_HISTORIC_TEMPS
    ]
    assert isinstance(default_feed, feeds.HistoricalTempsFeed)
    assert isinstance(full_feed, feeds.HistoricalTempsFeed)
    assert default_feed.start_year == utc_now().year
    assert default_feed.end_year == utc_now().year
    assert full_feed.start_year == 2024
    assert full_feed.end_year == utc_now().year

    assert update.main([]) == 0
    assert coops_client.historic_temperature_calls == 1


def test_full_history_flag_widens_the_served_range_not_the_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The flag chooses which years are served, not how many are fetched."""
    coops_client, _, store = _install_job_environment(
        monkeypatch, [MULTI_YEAR_HISTORY_CONFIG]
    )

    assert update.main(["--full-history"]) == 0

    # The top-up fetches this year whatever range the feed serves; the earlier
    # configured years come from the archive, and are gaps until a backfill
    # puts them there.
    assert coops_client.historic_temperature_calls == 1
    assert [key for key in store._objects if key.startswith("archive/")] == [
        f"archive/temperature/coops/6666666/{utc_now().year}.parquet"
    ]


def test_past_history_range_is_skipped_without_failing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    coops_client, _, _ = _install_job_environment(monkeypatch, [RETIRED_HISTORY_CONFIG])

    with caplog.at_level(logging.INFO):
        exit_code = update.main([])

    assert exit_code == 0
    assert coops_client.historic_temperature_calls == 0
    assert coops_client.live_temperature_calls == 1
    assert any(
        "No historical temperature years in range" in record.getMessage()
        for record in caplog.records
    )
    assert _summary_record(caplog).outcome == "success"


def test_one_failing_feed_leaves_run_partial(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    coops_client, nwis_client, _ = _install_job_environment(
        monkeypatch, [TEST_CONFIG_OBSERVATION_CURRENTS]
    )
    coops_client.live_temperature_error = RuntimeError("live temp boom")

    with caplog.at_level(logging.INFO):
        exit_code = update.main([])

    assert exit_code == 0
    assert nwis_client.currents_calls == 1
    assert coops_client.historic_temperature_calls == 1

    summary = _summary_record(caplog)
    assert summary.outcome == "partial"
    assert summary.levelno == logging.WARNING
    feed_errors = [
        record
        for record in caplog.records
        if record.levelno == logging.ERROR
        and getattr(record, "operation", "") == "feed_update"
    ]
    assert len(feed_errors) == 1


def test_all_feeds_failing_is_a_failed_run(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    coops_client, nwis_client, _ = _install_job_environment(
        monkeypatch, [TEST_CONFIG_OBSERVATION_CURRENTS]
    )
    coops_client.live_temperature_error = RuntimeError("live temp boom")
    coops_client.historic_temperature_error = RuntimeError("historic temp boom")
    nwis_client.currents_error = RuntimeError("currents boom")

    with caplog.at_level(logging.INFO):
        exit_code = update.main([])

    assert exit_code == 1
    summary = _summary_record(caplog)
    assert summary.outcome == "failed"
    assert summary.levelno == logging.ERROR
    assert summary.record_count == 0
    assert summary.new_count == 0
    assert summary.revised_count == 0


def test_missing_bucket_fails_before_any_client(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("SHALLWESWIM_ARCHIVE_BUCKET", "")
    client_factory = Mock()
    monkeypatch.setattr(update, "create_api_clients", client_factory)
    session_factory = Mock()
    monkeypatch.setattr(update.aiohttp, "ClientSession", session_factory)

    with caplog.at_level(logging.INFO):
        exit_code = update.main([])

    assert exit_code == 1
    client_factory.assert_not_called()
    session_factory.assert_not_called()
    assert any(
        "SHALLWESWIM_ARCHIVE_BUCKET is required" in record.getMessage()
        for record in caplog.records
    )


def test_summary_event_fields_and_run_id(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _, _, store = _install_job_environment(
        monkeypatch, [TEST_CONFIG_OBSERVATION_CURRENTS]
    )
    monkeypatch.setenv("CLOUD_RUN_EXECUTION", "shallweswim-capture-abcde")

    with caplog.at_level(logging.INFO):
        assert update.main([]) == 0

    summary = _summary_record(caplog)
    assert summary.component == "updater"
    assert summary.outcome == "success"
    assert summary.run_id == "shallweswim-capture-abcde"
    assert isinstance(summary.duration_ms, int)
    assert summary.record_count > 0
    # A first run archives every observation it fetched and revises nothing.
    assert summary.new_count == len(
        _archived_rows(store, TEMPERATURE_KEY, TEMPERATURE_UNIT)
    ) + len(_archived_rows(store, CURRENTS_KEY, CURRENTS_UNIT))
    assert summary.revised_count == 0


def test_identical_second_run_archives_no_new_or_revised_rows(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Re-fetching the same observations overlaps instead of rewriting them."""
    _install_job_environment(monkeypatch, [TEST_CONFIG_OBSERVATION_CURRENTS])

    assert update.main([]) == 0
    caplog.clear()
    with caplog.at_level(logging.INFO):
        assert update.main([]) == 0

    summary = _summary_record(caplog)
    assert summary.outcome == "success"
    assert summary.record_count > 0
    assert summary.new_count == 0
    assert summary.revised_count == 0
    merges = [
        record
        for record in caplog.records
        if getattr(record, "operation", "") == "merge"
    ]
    assert merges
    assert {record.outcome for record in merges} == {"unchanged"}


# =============================================================================
# Snapshot publishing path
# =============================================================================


def _install_publish_environment(
    monkeypatch: pytest.MonkeyPatch,
    configs: list[LocationConfig],
) -> tuple[MockCoopsApi, MockNwisApi, MemoryObjectStore]:
    """Enable publishing on top of the job environment, with no matplotlib.

    Plot generation runs in a thread pool in place of the process pool, and the
    plot functions return fixed bytes so no chart is rendered.
    """
    coops_client, nwis_client, store = _install_job_environment(monkeypatch, configs)
    monkeypatch.setenv("SHALLWESWIM_SNAPSHOT_PUBLISH", "1")
    monkeypatch.setenv("SHALLWESWIM_ARCHIVE_READ_BUCKET", "test-archive")
    monkeypatch.setattr(update, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(
        manager_module, "_generate_live_temp_plot", lambda *args: b"<svg>live</svg>"
    )
    monkeypatch.setattr(
        manager_module,
        "_generate_historic_temp_plots",
        lambda *args: {"2mo": b"<svg>2mo</svg>", "12mo": b"<svg>12mo</svg>"},
    )
    return coops_client, nwis_client, store


def _publish_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """Return every structured snapshot publish event."""
    return [
        record
        for record in caplog.records
        if getattr(record, "component", "") == "snapshot"
        and getattr(record, "operation", "") == "publish"
    ]


def _published_keys(store: MemoryObjectStore, prefix: str) -> list[str]:
    return sorted(key for key in store._objects if key.startswith(prefix))


def test_publish_run_writes_one_generation_with_every_feed_and_plot(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    coops_client, nwis_client, store = _install_publish_environment(
        monkeypatch, [TEST_CONFIG_OBSERVATION_CURRENTS, TEST_CONFIG_FULL]
    )
    monkeypatch.setenv("CLOUD_RUN_EXECUTION", "shallweswim-capture-pub01")

    with caplog.at_level(logging.INFO):
        exit_code = update.main([])

    assert exit_code == 0
    # The full serving cycle fetches the prediction feeds the capture-only
    # path skips. The historical feed serves the full configured range from the
    # archive and fetches only its top-up year, once per location.
    assert coops_client.tides_calls == 2
    assert coops_client.currents_calls == 1
    assert nwis_client.currents_calls == 1
    assert coops_client.historic_temperature_calls == 2
    # Archive capture still runs inside the feed updates.
    assert len(_archived_rows(store, TEMPERATURE_KEY, TEMPERATURE_UNIT)) == 24
    assert len(_archived_rows(store, CURRENTS_KEY, CURRENTS_UNIT)) == 24

    loaded = asyncio.run(load_current(SnapshotStore(store)))
    assert loaded is not None
    assert loaded.pointer.generation_id.endswith("-shallweswim-capture-pub01")
    assert len(_published_keys(store, "published/manifests/")) == 1
    assert set(loaded.manifest.locations) == {"obs", "nyc"}
    for code, location in loaded.manifest.locations.items():
        assert set(location.feeds) == set(feeds.FeedName), code
        assert set(location.plots) == set(feeds.PlotName), code
        assert loaded.plots[code][feeds.PlotName.LIVE_TEMPS] == b"<svg>live</svg>"
        assert loaded.plots[code][feeds.PlotName.HISTORIC_TEMPS_12MO] == (
            b"<svg>12mo</svg>"
        )
        assert len(loaded.frames[code][feeds.FeedName.LIVE_TEMPS]) == 24

    (publish_event,) = _publish_records(caplog)
    assert publish_event.outcome == "success"
    assert publish_event.run_id == "shallweswim-capture-pub01"
    assert publish_event.record_count == len(
        _published_keys(store, "published/objects/")
    )

    summary = _summary_record(caplog)
    assert summary.outcome == "success"
    # Four feeds per location, every one published.
    assert "8 of 8 feeds published; snapshot publish success" in summary.getMessage()
    assert summary.record_count == sum(
        len(frame) for location in loaded.frames.values() for frame in location.values()
    )
    # Two temperature stations and one observational currents partition.
    assert summary.new_count == 3 * 24
    assert summary.revised_count == 0


def _manifest_entry(
    feed: feeds.Feed,
    *,
    next_fetch_after: datetime.datetime | None,
    source_identity: str | None = None,
) -> FeedObject:
    """A published entry for one feed, carrying only its schedule and source."""
    return FeedObject(
        key="published/objects/sha256-0.parquet",
        size_bytes=1,
        source_identity=source_identity or feed.feed_config.citation_key,
        fetch_timestamp=datetime.datetime(2026, 6, 1, tzinfo=datetime.UTC),
        next_fetch_after=next_fetch_after,
        expiration_seconds=600.0,
        record_count=1,
        consecutive_failures=0,
        last_error=None,
        timezone="US/Eastern",
        historical=None,
    )


def _provider_calls(coops: MockCoopsApi, nwis: MockNwisApi) -> tuple[int, ...]:
    """Every provider call count, to assert which feeds a run fetched."""
    return (
        coops.tides_calls,
        coops.currents_calls,
        coops.live_temperature_calls,
        coops.historic_temperature_calls,
        nwis.currents_calls,
    )


def _edit_published_feed(
    store: MemoryObjectStore,
    location: str,
    feed_name: feeds.FeedName,
    **updates: object,
) -> None:
    """Rewrite one feed entry of the published manifest the next run restores.

    The manifest object is replaced in place, under the key the current
    pointer already names, so the next run reads the edited schedule.
    """
    pointer = CurrentPointer.model_validate_json(store._objects[CURRENT_KEY])
    manifest = Manifest.model_validate_json(store._objects[pointer.manifest_key])
    location_manifest = manifest.locations[location]
    feed_objects = dict(location_manifest.feeds)
    feed_objects[feed_name] = feed_objects[feed_name].model_copy(update=updates)
    locations = dict(manifest.locations)
    locations[location] = location_manifest.model_copy(update={"feeds": feed_objects})
    store._objects[pointer.manifest_key] = (
        manifest.model_copy(update={"locations": locations}).model_dump_json().encode()
    )


def test_second_publish_run_holds_every_feed_and_changes_nothing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The manifest is the schedule, so an immediate rerun fetches nothing.

    Under the production cadence the ten-minute live feed is due before the
    next run and fetches every time, which the boundary test below covers; a
    one-minute cadence here leaves every feed genuinely not due.
    """
    coops_client, nwis_client, store = _install_publish_environment(
        monkeypatch, [TEST_CONFIG_OBSERVATION_CURRENTS]
    )
    monkeypatch.setattr(update, "JOB_CADENCE", datetime.timedelta(minutes=1))

    assert update.main([]) == 0
    objects_after_first = _published_keys(store, "published/objects/")
    manifests_after_first = _published_keys(store, "published/manifests/")
    calls_after_first = _provider_calls(coops_client, nwis_client)
    caplog.clear()
    with caplog.at_level(logging.INFO):
        assert update.main([]) == 0

    # No feed was due again, so no provider was contacted and the published
    # generation is byte-identical: the `unchanged` outcome the design wants.
    assert _provider_calls(coops_client, nwis_client) == calls_after_first
    assert _published_keys(store, "published/objects/") == objects_after_first
    assert _published_keys(store, "published/manifests/") == manifests_after_first
    (publish_event,) = _publish_records(caplog)
    assert publish_event.outcome == "unchanged"

    events = {record.feed: record for record in _freshness_records(caplog)}
    assert set(events) == {name.value for name in feeds.FeedName}
    assert {record.outcome for record in events.values()} == {"held"}
    assert all(record.levelno == logging.INFO for record in events.values())
    assert all(record.age_seconds >= 0 for record in events.values())

    summary = _summary_record(caplog)
    assert summary.outcome == "success"
    # A held feed counts as published; its rows belong to the run that fetched
    # them, so this process reports none.
    assert "4 of 4 feeds published; snapshot publish unchanged" in summary.getMessage()
    assert summary.record_count == 0
    assert summary.new_count == 0


def test_a_due_feed_fetches_while_the_rest_are_held(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    coops_client, nwis_client, store = _install_publish_environment(
        monkeypatch, [TEST_CONFIG_OBSERVATION_CURRENTS]
    )

    assert update.main([]) == 0
    base = asyncio.run(load_current(SnapshotStore(store)))
    assert base is not None
    calls_after_first = _provider_calls(coops_client, nwis_client)
    # Live temperature came due; the daily feeds did not.
    _edit_published_feed(
        store,
        "obs",
        feeds.FeedName.LIVE_TEMPS,
        next_fetch_after=utc_now().replace(tzinfo=datetime.UTC)
        - datetime.timedelta(minutes=1),
    )
    caplog.clear()
    with caplog.at_level(logging.INFO):
        assert update.main([]) == 0

    assert coops_client.live_temperature_calls == calls_after_first[2] + 1
    assert _provider_calls(coops_client, nwis_client)[:2] == calls_after_first[:2]
    assert _provider_calls(coops_client, nwis_client)[3] == calls_after_first[3]
    # The observed currents feed refreshes on the live interval, so it comes
    # due before the next run starts and fetches on every run.
    assert nwis_client.currents_calls == calls_after_first[4] + 1

    loaded = asyncio.run(load_current(SnapshotStore(store)))
    assert loaded is not None
    assert loaded.manifest.generation_id != base.manifest.generation_id
    held = loaded.manifest.locations["obs"].feeds[feeds.FeedName.HISTORIC_TEMPS]
    assert held == base.manifest.locations["obs"].feeds[feeds.FeedName.HISTORIC_TEMPS]
    # The held feed's plots come with its entry, unchanged.
    for plot_name in (
        feeds.PlotName.HISTORIC_TEMPS_2MO,
        feeds.PlotName.HISTORIC_TEMPS_12MO,
    ):
        assert (
            loaded.manifest.locations["obs"].plots[plot_name]
            == (base.manifest.locations["obs"].plots[plot_name])
        )
    refetched = loaded.manifest.locations["obs"].feeds[feeds.FeedName.LIVE_TEMPS]
    assert refetched.fetch_timestamp > (
        base.manifest.locations["obs"].feeds[feeds.FeedName.LIVE_TEMPS].fetch_timestamp
    )

    events = {record.feed: record for record in _freshness_records(caplog)}
    assert events[feeds.FeedName.LIVE_TEMPS].outcome == "success"
    assert events[feeds.FeedName.HISTORIC_TEMPS].outcome == "held"
    assert _summary_record(caplog).outcome == "success"


def test_a_changed_source_identity_fetches_instead_of_holding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coops_client, nwis_client, store = _install_publish_environment(
        monkeypatch, [TEST_CONFIG_OBSERVATION_CURRENTS]
    )

    assert update.main([]) == 0
    calls_after_first = _provider_calls(coops_client, nwis_client)
    _edit_published_feed(
        store, "obs", feeds.FeedName.LIVE_TEMPS, source_identity="coops:temp:9999999"
    )

    assert update.main([]) == 0

    # The entry no longer describes the configured source, so the feed is a
    # fresh feed again and fetches; the daily feeds still hold.
    assert coops_client.live_temperature_calls == calls_after_first[2] + 1
    assert _provider_calls(coops_client, nwis_client)[3] == calls_after_first[3]
    # The observed currents feed refreshes on the live interval, so it comes
    # due before the next run starts and fetches on every run.
    assert nwis_client.currents_calls == calls_after_first[4] + 1


def test_a_run_without_a_published_generation_fetches_everything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coops_client, nwis_client, store = _install_publish_environment(
        monkeypatch, [TEST_CONFIG_OBSERVATION_CURRENTS]
    )

    assert update.main([]) == 0
    calls_after_first = _provider_calls(coops_client, nwis_client)
    del store._objects[CURRENT_KEY]

    assert update.main([]) == 0

    assert _provider_calls(coops_client, nwis_client) == tuple(
        count * 2 for count in calls_after_first
    )


def test_full_history_run_ignores_the_published_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coops_client, nwis_client, _ = _install_publish_environment(
        monkeypatch, [TEST_CONFIG_OBSERVATION_CURRENTS]
    )

    assert update.main([]) == 0
    calls_after_first = _provider_calls(coops_client, nwis_client)

    assert update.main(["--full-history"]) == 0

    assert _provider_calls(coops_client, nwis_client) == tuple(
        count * 2 for count in calls_after_first
    )


def test_restore_schedule_only_restores_matching_entries() -> None:
    """A manifest entry restores a feed only while it names its source."""
    manager = fresh_manager()
    tides = manager._feeds[feeds.FeedName.TIDES]
    live = manager._feeds[feeds.FeedName.LIVE_TEMPS]
    currents = manager._feeds[feeds.FeedName.CURRENTS]
    historic = manager._feeds[feeds.FeedName.HISTORIC_TEMPS]
    assert tides is not None and live is not None
    assert currents is not None and historic is not None
    # Well past the next run, so a restored feed is genuinely not due.
    due = utc_now().replace(tzinfo=datetime.UTC) + datetime.timedelta(days=1)

    update.restore_schedule(
        manager,
        LocationManifest(
            feeds={
                # Restored: the entry names the feed's configured source.
                feeds.FeedName.TIDES: _manifest_entry(tides, next_fetch_after=due),
                # A never-refreshing feed keeps its None.
                feeds.FeedName.LIVE_TEMPS: _manifest_entry(live, next_fetch_after=None),
                # Published from another source, so it is not this feed's state.
                feeds.FeedName.CURRENTS: _manifest_entry(
                    currents, next_fetch_after=due, source_identity="coops:other:1"
                ),
            },
            plots={},
        ),
    )

    assert tides._next_fetch_after == due.replace(tzinfo=None)
    assert live._next_fetch_after is None
    assert currents._next_fetch_after is None
    # A feed the manifest does not describe keeps its fresh-feed state.
    assert historic._next_fetch_after is None
    assert all(feed.is_expired for feed in (live, currents, historic))


def test_restore_schedule_fetches_a_feed_due_before_the_next_run() -> None:
    """A ten-minute feed under a ten-minute cadence must fetch every run.

    Restored literally, its next fetch time lands seconds after the next run
    begins, so it would be held on every other run and refresh every twenty
    minutes; production showed exactly that on 2026-09-14.
    """
    manager = fresh_manager()
    live = manager._feeds[feeds.FeedName.LIVE_TEMPS]
    tides = manager._feeds[feeds.FeedName.TIDES]
    assert live is not None and tides is not None
    now = utc_now().replace(tzinfo=datetime.UTC)
    soon = now + datetime.timedelta(seconds=30)
    later = now + datetime.timedelta(minutes=11)

    update.restore_schedule(
        manager,
        LocationManifest(
            feeds={
                feeds.FeedName.LIVE_TEMPS: _manifest_entry(live, next_fetch_after=soon),
                feeds.FeedName.TIDES: _manifest_entry(tides, next_fetch_after=later),
            },
            plots={},
        ),
        cadence=datetime.timedelta(minutes=10),
    )

    # Due before the next run: left in its fresh, always-due state.
    assert live._next_fetch_after is None
    assert live.is_expired
    # Not due until after the next run: restored and held.
    assert tides._next_fetch_after == later.replace(tzinfo=None)
    assert not tides.is_expired


def test_restore_schedule_without_a_manifest_leaves_every_feed_due() -> None:
    manager = fresh_manager()

    update.restore_schedule(manager, None)

    for feed in manager._feeds.values():
        assert feed is not None
        assert feed._next_fetch_after is None
        assert feed.is_expired


def test_publish_failure_leaves_run_outcome_and_exit_code(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _, _, store = _install_publish_environment(
        monkeypatch, [TEST_CONFIG_OBSERVATION_CURRENTS]
    )

    async def failing_compare_and_swap(key: str, **kwargs: object) -> str:
        if key.startswith("published/"):
            raise RuntimeError("bucket write refused")
        return await original_compare_and_swap(key, **kwargs)

    original_compare_and_swap = store.compare_and_swap
    monkeypatch.setattr(store, "compare_and_swap", failing_compare_and_swap)

    with caplog.at_level(logging.INFO):
        exit_code = update.main([])

    assert exit_code == 0
    assert _published_keys(store, "published/") == []
    assert len(_archived_rows(store, TEMPERATURE_KEY, TEMPERATURE_UNIT)) == 24
    (publish_event,) = _publish_records(caplog)
    assert publish_event.outcome == "failed"
    assert publish_event.levelno == logging.ERROR
    summary = _summary_record(caplog)
    assert summary.outcome == "success"
    assert summary.levelno == logging.INFO
    assert "snapshot publish failed" in summary.getMessage()
    # The publish event is the only ERROR the failure produces.
    assert [r for r in caplog.records if r.levelno == logging.ERROR] == [publish_event]


def _gc_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """Return every structured generation collection event."""
    return [
        record
        for record in caplog.records
        if getattr(record, "component", "") == "snapshot"
        and getattr(record, "operation", "") == "gc"
    ]


def test_publish_run_sweeps_generations_after_publishing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Every publishing run ends with one collection over what it published."""
    _, _, store = _install_publish_environment(
        monkeypatch, [TEST_CONFIG_OBSERVATION_CURRENTS]
    )

    with caplog.at_level(logging.INFO):
        assert update.main([]) == 0

    (gc_event,) = _gc_records(caplog)
    assert gc_event.levelno == logging.INFO
    assert gc_event.outcome == "success"
    # This run's own generation is current and its objects are seconds old, so
    # the sweep examines everything and deletes nothing.
    assert gc_event.record_count == 0
    assert len(_published_keys(store, "published/manifests/")) == 1
    assert _published_keys(store, "published/objects/")


def test_sweep_failure_leaves_the_run_outcome_and_the_generation_unchanged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A broken listing is the sweep's failure alone, like a failed publish."""
    _, _, store = _install_publish_environment(
        monkeypatch, [TEST_CONFIG_OBSERVATION_CURRENTS]
    )

    async def failing_list(prefix: str) -> list[object]:
        raise RuntimeError("bucket listing refused")

    monkeypatch.setattr(store, "list", failing_list)

    with caplog.at_level(logging.INFO):
        exit_code = update.main([])

    assert exit_code == 0
    summary = _summary_record(caplog)
    assert summary.outcome == "success"
    assert summary.levelno == logging.INFO
    (publish_event,) = _publish_records(caplog)
    assert publish_event.outcome == "success"
    # The generation the run published is untouched and still loadable.
    assert asyncio.run(load_current(SnapshotStore(store))) is not None
    (gc_event,) = _gc_records(caplog)
    assert gc_event.levelno == logging.ERROR
    assert gc_event.outcome == "failed"
    assert [r for r in caplog.records if r.levelno == logging.ERROR] == [gc_event]


def test_publish_run_with_one_failing_feed_is_partial_and_still_publishes(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    coops_client, nwis_client, store = _install_publish_environment(
        monkeypatch, [TEST_CONFIG_OBSERVATION_CURRENTS]
    )
    coops_client.live_temperature_error = RuntimeError("live temp boom")

    with caplog.at_level(logging.INFO):
        exit_code = update.main([])

    assert exit_code == 0
    # The feed after the failing one in the cycle still ran.
    assert nwis_client.currents_calls == 1
    assert coops_client.historic_temperature_calls > 0
    assert coops_client.live_temperature_calls == 1
    summary = _summary_record(caplog)
    assert summary.outcome == "partial"
    assert "3 of 4 feeds published; snapshot publish success" in summary.getMessage()
    loaded = asyncio.run(load_current(SnapshotStore(store)))
    assert loaded is not None
    assert set(loaded.manifest.locations["obs"].feeds) == {
        feeds.FeedName.TIDES,
        feeds.FeedName.CURRENTS,
        feeds.FeedName.HISTORIC_TEMPS,
    }
    assert set(loaded.manifest.locations["obs"].plots) == {
        feeds.PlotName.HISTORIC_TEMPS_2MO,
        feeds.PlotName.HISTORIC_TEMPS_12MO,
    }


def _freshness_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """Return every structured per-feed snapshot freshness event."""
    return [
        record
        for record in caplog.records
        if getattr(record, "component", "") == "snapshot"
        and getattr(record, "operation", "") == "freshness"
    ]


def test_publish_run_includes_a_location_whose_feeds_all_failed(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Every enabled location reaches the snapshot, even with nothing to serve."""
    coops_client, nwis_client, store = _install_publish_environment(
        monkeypatch, [TEST_CONFIG_OBSERVATION_CURRENTS, TEST_CONFIG_FULL]
    )
    # Every CO-OPS feed fails, so "nyc" has no data at all and "obs" keeps only
    # its NWIS observational currents.
    coops_client.live_temperature_error = RuntimeError("live temp boom")
    coops_client.historic_temperature_error = RuntimeError("historic temp boom")
    coops_client.tides_error = RuntimeError("tides boom")
    coops_client.currents_error = RuntimeError("currents boom")

    with caplog.at_level(logging.INFO):
        exit_code = update.main([])

    assert exit_code == 0
    assert nwis_client.currents_calls == 1
    loaded = asyncio.run(load_current(SnapshotStore(store)))
    assert loaded is not None
    assert set(loaded.manifest.locations) == {"obs", "nyc"}
    assert loaded.manifest.locations["nyc"].feeds == {}
    assert loaded.manifest.locations["nyc"].plots == {}
    assert set(loaded.manifest.locations["obs"].feeds) == {feeds.FeedName.CURRENTS}

    events = {
        (record.location, record.feed): record for record in _freshness_records(caplog)
    }
    assert len(events) == 2 * len(feeds.FeedName)
    served = events[("obs", feeds.FeedName.CURRENTS)]
    assert served.outcome == "success"
    assert served.levelno == logging.INFO
    assert served.age_seconds >= 0
    assert {
        record.outcome for (code, _feed), record in events.items() if code == "nyc"
    } == {"absent"}
    assert all(
        record.levelno == logging.WARNING
        for (code, _feed), record in events.items()
        if code == "nyc"
    )


def test_publish_run_with_no_data_publishes_nothing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    coops_client, nwis_client, store = _install_publish_environment(
        monkeypatch, [TEST_CONFIG_OBSERVATION_CURRENTS]
    )
    coops_client.live_temperature_error = RuntimeError("live temp boom")
    coops_client.historic_temperature_error = RuntimeError("historic temp boom")
    coops_client.tides_error = RuntimeError("tides boom")
    nwis_client.currents_error = RuntimeError("currents boom")

    with caplog.at_level(logging.INFO):
        exit_code = update.main([])

    assert exit_code == 1
    assert _published_keys(store, "published/") == []
    assert _publish_records(caplog) == []
    summary = _summary_record(caplog)
    assert summary.outcome == "failed"
    assert "snapshot publish skipped" in summary.getMessage()


def test_publish_mode_requires_read_bucket_before_any_client(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("SHALLWESWIM_ARCHIVE_BUCKET", "test-archive")
    monkeypatch.setenv("SHALLWESWIM_SNAPSHOT_PUBLISH", "1")
    monkeypatch.delenv("SHALLWESWIM_ARCHIVE_READ_BUCKET", raising=False)
    client_factory = Mock()
    monkeypatch.setattr(update, "create_api_clients", client_factory)
    session_factory = Mock()
    monkeypatch.setattr(update.aiohttp, "ClientSession", session_factory)

    with caplog.at_level(logging.INFO):
        exit_code = update.main([])

    assert exit_code == 1
    client_factory.assert_not_called()
    session_factory.assert_not_called()
    assert any(
        "SHALLWESWIM_ARCHIVE_READ_BUCKET is required" in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.parametrize("publish_flag", [None, "0", "true"])
def test_publish_disabled_keeps_the_capture_only_path(
    monkeypatch: pytest.MonkeyPatch, publish_flag: str | None
) -> None:
    coops_client, _, store = _install_job_environment(monkeypatch, [TEST_CONFIG_FULL])
    if publish_flag is None:
        monkeypatch.delenv("SHALLWESWIM_SNAPSHOT_PUBLISH", raising=False)
    else:
        monkeypatch.setenv("SHALLWESWIM_SNAPSHOT_PUBLISH", publish_flag)
    monkeypatch.delenv("SHALLWESWIM_ARCHIVE_READ_BUCKET", raising=False)
    pool_factory = Mock()
    monkeypatch.setattr(update, "ProcessPoolExecutor", pool_factory)

    assert update.main([]) == 0

    assert coops_client.tides_calls == 0
    assert coops_client.currents_calls == 0
    pool_factory.assert_not_called()
    assert _published_keys(store, "published/") == []
    # The historical feed serves from the archive here too, so the run names
    # the read locator itself rather than requiring a second variable.
    assert os.environ["SHALLWESWIM_ARCHIVE_READ_BUCKET"] == "test-archive"
    assert coops_client.historic_temperature_calls == 1


# =============================================================================
# Deep history backfill
# =============================================================================

# A fixed clock keeps the walk's years, windows, and partition keys the same
# whenever the suite runs.
BACKFILL_NOW = datetime.datetime(2026, 9, 15, 12, 0)
BACKFILL_YEAR = BACKFILL_NOW.year

COOPS_BACKFILL_STATION = 7777777
NWIS_BACKFILL_SITE = "87654321"

# Rows one complete year archives: a day of hourly readings, and for CO-OPS two
# six-minute readings in each of the twelve months.
HOURLY_ROWS_PER_YEAR = 24
SIX_MINUTE_ROWS_PER_MONTH = 2

COOPS_BACKFILL_CONFIG = LocationConfig(
    code="bkc",
    name="Backfill CO-OPS Location",
    swim_location="Deep Record Beach",
    swim_location_link="http://example.com/deep",
    description="Test location whose CO-OPS station is walked back by year",
    latitude=26.0,
    longitude=-80.0,
    timezone=pytz.timezone("US/Eastern"),
    default_temperature_unit="F",
    historic_temp_source=CoopsTempFeedConfig(
        station=COOPS_BACKFILL_STATION, name="Deep Record Temp"
    ),
    enabled=True,
)

NWIS_BACKFILL_CONFIG = LocationConfig(
    code="bkn",
    name="Backfill NWIS Location",
    swim_location="River Record",
    swim_location_link="http://example.com/river-record",
    description="Test location whose NWIS site is walked back by year",
    latitude=38.0,
    longitude=-85.0,
    timezone=pytz.timezone("US/Eastern"),
    default_temperature_unit="F",
    historic_temp_source=NwisTempFeedConfig(
        site_no=NWIS_BACKFILL_SITE, name="River Record Temp"
    ),
    enabled=True,
)

NO_HISTORY_CONFIG = LocationConfig(
    code="nhi",
    name="No History Location",
    swim_location="Live Only Beach",
    swim_location_link="http://example.com/live-only",
    description="Test location whose source serves live readings only",
    latitude=30.0,
    longitude=-81.0,
    timezone=pytz.timezone("US/Eastern"),
    default_temperature_unit="F",
    live_temp_source=CoopsTempFeedConfig(
        station=COOPS_BACKFILL_STATION, name="Live Only Temp"
    ),
    historic_temp_source=CoopsTempFeedConfig(
        station=COOPS_BACKFILL_STATION, name="Live Only Temp", historic_enabled=False
    ),
    enabled=True,
)


def _hourly_year_frame(year: int) -> pd.DataFrame:
    """Return a day of on-the-hour readings at the start of one year."""
    index = pd.date_range(
        f"{year}-01-01", periods=HOURLY_ROWS_PER_YEAR, freq="h", tz="UTC", name="time"
    )
    return pd.DataFrame({"water_temp": [60.0] * len(index)}, index=index)


def _six_minute_month_frame(year: int, month: int) -> pd.DataFrame:
    """Return six-minute readings at the start of one month, off the hour.

    Off the hour so they never coincide with the hourly product's readings,
    which would archive as overlaps rather than as rows of their own.
    """
    index = pd.date_range(
        f"{year}-{month:02d}-01 00:30",
        periods=SIX_MINUTE_ROWS_PER_MONTH,
        freq="6min",
        tz="UTC",
        name="time",
    )
    return pd.DataFrame({"water_temp": [61.0, 61.5]}, index=index)


class BackfillCoopsApi(CoopsApi):
    """CO-OPS client answering each backfill window from a fixed record."""

    def __init__(
        self,
        *,
        data_years: set[int],
        six_minute_gaps: set[tuple[int, int]] | None = None,
        errors: dict[int, Exception] | None = None,
        order: list[str] | None = None,
    ) -> None:
        """Record which years hold data, which months lack six-minute readings."""
        super().__init__(session=cast(aiohttp.ClientSession, None))
        self.data_years = data_years
        self.six_minute_gaps = six_minute_gaps or set()
        self.errors = errors or {}
        # A list both fakes append to, so a test can see the order requests to
        # different providers were made in.
        self.order = order
        self.requests: list[tuple[str | None, datetime.datetime]] = []

    async def temperature(
        self,
        station: int,
        product: str,
        begin_date: object,
        end_date: object,
        timezone: str,
        interval: str | None = None,
        location_code: str = "unknown",
    ) -> pd.DataFrame:
        """Return the window's readings, or report the station has none."""
        assert isinstance(begin_date, datetime.datetime)
        self.requests.append((interval, begin_date))
        if self.order is not None:
            self.order.append("coops")
        year, month = begin_date.year, begin_date.month
        error = self.errors.get(year)
        if error is not None:
            raise error
        if year not in self.data_years:
            raise StationUnavailableError(f"No data was found for {year}")
        if interval == "6-min":
            if (year, month) in self.six_minute_gaps:
                raise StationUnavailableError(f"No data was found for {year}-{month}")
            return _six_minute_month_frame(year, month)
        return _hourly_year_frame(year)


class BackfillNwisApi(NwisApi):
    """NWIS client answering one temperature request per backfilled year."""

    def __init__(
        self,
        *,
        data_years: set[int],
        errors: dict[int, Exception] | None = None,
        blocks: dict[int, int] | None = None,
        order: list[str] | None = None,
    ) -> None:
        """Record which years hold data and which raise an unexpected error."""
        super().__init__(session=cast(aiohttp.ClientSession, None))
        self.data_years = data_years
        self.errors = errors or {}
        # How many times a year is answered with a provider block before it is
        # answered normally, counted down as the walk retries.
        self.blocks = dict(blocks or {})
        self.order = order
        self.requests: list[datetime.datetime] = []

    async def temperature(
        self,
        site_no: str,
        begin_date: object,
        end_date: object,
        timezone: str,
        location_code: str = "unknown",
        parameter_cd: str = "00010",
    ) -> pd.DataFrame:
        """Return the year's readings, or report the site has none."""
        assert isinstance(begin_date, datetime.datetime)
        self.requests.append(begin_date)
        if self.order is not None:
            self.order.append("nwis")
        blocks_left = self.blocks.get(begin_date.year, 0)
        if blocks_left:
            self.blocks[begin_date.year] = blocks_left - 1
            raise NwisConnectionError(
                f"NWIS request for site {site_no} returned HTTP 403",
                status=backfill.BACKFILL_BLOCK_STATUS,
            )
        error = self.errors.get(begin_date.year)
        if error is not None:
            raise error
        if begin_date.year not in self.data_years:
            raise StationUnavailableError(f"No data for {begin_date.year}")
        return _hourly_year_frame(begin_date.year)


def _record_backfill_sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Replace the walk's sleep with a recorder, so its pauses cost no time.

    The recorder still yields to the event loop, so a walk that ran locations
    concurrently would interleave their requests rather than hide it.

    Returns:
        The seconds the walk sleeps for, in order.
    """
    sleeps: list[float] = []

    async def record(seconds: float) -> None:
        sleeps.append(seconds)
        await asyncio.sleep(0)

    monkeypatch.setattr(backfill, "_sleep", record)
    return sleeps


def _install_backfill_environment(
    monkeypatch: pytest.MonkeyPatch,
    configs: list[LocationConfig],
    *,
    coops_client: BackfillCoopsApi | None = None,
    nwis_client: BackfillNwisApi | None = None,
) -> tuple[BackfillCoopsApi, BackfillNwisApi, MemoryObjectStore]:
    """Point a backfill run at fake locations, fixed clients, and a fixed clock."""
    monkeypatch.setenv("SHALLWESWIM_ARCHIVE_BUCKET", "test-archive")
    monkeypatch.delenv("SHALLWESWIM_SNAPSHOT_PUBLISH", raising=False)
    monkeypatch.delenv("SHALLWESWIM_ARCHIVE_READ_BUCKET", raising=False)
    monkeypatch.setattr(
        config,
        "CONFIGS",
        MappingProxyType({item.code: item for item in configs}),
    )
    coops_client = coops_client or BackfillCoopsApi(data_years=set())
    nwis_client = nwis_client or BackfillNwisApi(data_years=set())
    clients: dict[str, BaseApiClient] = {"coops": coops_client, "nwis": nwis_client}
    monkeypatch.setattr(update, "create_api_clients", lambda session: clients)
    store = MemoryObjectStore()
    monkeypatch.setattr(archive_store, "GcsObjectStore", lambda bucket: store)
    monkeypatch.setattr(backfill, "utc_now", lambda: BACKFILL_NOW)
    # Tests that assert on the pauses install their own recorder for it.
    _record_backfill_sleeps(monkeypatch)
    return coops_client, nwis_client, store


def _backfill_summary(caplog: pytest.LogCaptureFixture) -> logging.LogRecord:
    """Return the single backfill run summary event."""
    records = [
        record
        for record in caplog.records
        if getattr(record, "operation", "") == "backfill"
    ]
    assert len(records) == 1
    return records[0]


def _coops_key(year: int) -> str:
    return f"archive/temperature/coops/{COOPS_BACKFILL_STATION}/{year}.parquet"


def _nwis_key(year: int) -> str:
    return f"archive/temperature/nwis/{NWIS_BACKFILL_SITE}%3A00010/{year}.parquet"


def _archived_years(store: MemoryObjectStore, prefix: str) -> list[int]:
    """Return the years this source holds partitions for, oldest first."""
    return sorted(
        int(key.rsplit("/", 1)[1].removesuffix(".parquet"))
        for key in store._objects
        if key.startswith(prefix)
    )


def test_backfill_argument_defaults_to_the_floor_year() -> None:
    assert _parse_args_backfill([]) is None
    assert _parse_args_backfill(["--backfill-from"]) == backfill.BACKFILL_FLOOR_YEAR
    assert backfill.BACKFILL_FLOOR_YEAR == 1900
    assert _parse_args_backfill(["--backfill-from", "2009"]) == 2009


def _parse_args_backfill(argv: list[str]) -> int | None:
    """Return the floor year the given arguments select."""
    parsed: int | None = update._parse_args(argv).backfill_from
    return parsed


def test_backfill_rejects_an_unknown_location_before_any_request(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _install_backfill_environment(monkeypatch, [NWIS_BACKFILL_CONFIG])
    session_factory = Mock()
    monkeypatch.setattr(update.aiohttp, "ClientSession", session_factory)

    with caplog.at_level(logging.INFO):
        exit_code = update.main(["--backfill-from", "2000", "--location", "zzz", "bkn"])

    assert exit_code == 1
    session_factory.assert_not_called()
    assert any(
        "Unknown location code(s): zzz" in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.parametrize(
    "argv,publish,expected",
    [
        (["--backfill-from", "2000", "--full-history"], False, "--full-history"),
        (["--backfill-from"], True, "SHALLWESWIM_SNAPSHOT_PUBLISH=1"),
        (["--location", "bkn"], False, "--location selects"),
    ],
)
def test_backfill_usage_errors_stop_before_any_request(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    argv: list[str],
    publish: bool,
    expected: str,
) -> None:
    _install_backfill_environment(monkeypatch, [NWIS_BACKFILL_CONFIG])
    if publish:
        monkeypatch.setenv("SHALLWESWIM_SNAPSHOT_PUBLISH", "1")
    session_factory = Mock()
    monkeypatch.setattr(update.aiohttp, "ClientSession", session_factory)

    with caplog.at_level(logging.INFO):
        exit_code = update.main(argv)

    assert exit_code == 1
    session_factory.assert_not_called()
    assert any(expected in record.getMessage() for record in caplog.records)


def test_backfill_requires_the_archive_bucket(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _install_backfill_environment(monkeypatch, [NWIS_BACKFILL_CONFIG])
    monkeypatch.setenv("SHALLWESWIM_ARCHIVE_BUCKET", "")
    session_factory = Mock()
    monkeypatch.setattr(update.aiohttp, "ClientSession", session_factory)

    with caplog.at_level(logging.INFO):
        exit_code = update.main(["--backfill-from"])

    assert exit_code == 1
    session_factory.assert_not_called()


def test_backfill_walks_newest_first_and_archives_one_partition_per_year(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    nwis_client = BackfillNwisApi(data_years={2026, 2025, 2024, 2023})
    _, _, store = _install_backfill_environment(
        monkeypatch, [NWIS_BACKFILL_CONFIG], nwis_client=nwis_client
    )

    with caplog.at_level(logging.INFO):
        exit_code = update.main(["--backfill-from", "2020"])

    assert exit_code == 0
    # One request per year, walked from this year down to the floor.
    assert [request.year for request in nwis_client.requests] == list(
        range(BACKFILL_YEAR, 2019, -1)
    )
    assert _archived_years(store, "archive/temperature/nwis/") == [
        2023,
        2024,
        2025,
        2026,
    ]
    for year in (2023, 2024, 2025, 2026):
        rows = _archived_rows(store, _nwis_key(year), TEMPERATURE_UNIT)
        assert len(rows) == HOURLY_ROWS_PER_YEAR

    summary = _backfill_summary(caplog)
    assert summary.outcome == "success"
    assert "archived 2023, 2024, 2025, 2026" in summary.getMessage()
    assert "empty 2020, 2021, 2022" in summary.getMessage()
    assert "earliest 2023" in summary.getMessage()
    assert "stopped at floor" in summary.getMessage()


def test_backfill_stops_after_five_empty_years_and_resets_on_data(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A year with data restarts the count; five empty ones in a row end it."""
    nwis_client = BackfillNwisApi(data_years={2026, 2023})
    _, _, store = _install_backfill_environment(
        monkeypatch, [NWIS_BACKFILL_CONFIG], nwis_client=nwis_client
    )

    with caplog.at_level(logging.INFO):
        exit_code = update.main(["--backfill-from"])

    assert exit_code == 0
    # 2025 and 2024 are empty, 2023 resets the count, and the walk ends after
    # the five empty years 2022 through 2018 rather than at the 1900 floor.
    assert [request.year for request in nwis_client.requests] == list(
        range(BACKFILL_YEAR, 2017, -1)
    )
    assert _archived_years(store, "archive/temperature/nwis/") == [2023, 2026]
    summary = _backfill_summary(caplog)
    assert summary.outcome == "success"
    assert "earliest 2023" in summary.getMessage()
    assert "stopped at empty years" in summary.getMessage()


def test_backfill_of_an_empty_source_is_a_successful_run(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    nwis_client = BackfillNwisApi(data_years=set())
    _, _, store = _install_backfill_environment(
        monkeypatch, [NWIS_BACKFILL_CONFIG], nwis_client=nwis_client
    )

    with caplog.at_level(logging.INFO):
        exit_code = update.main(["--backfill-from"])

    assert exit_code == 0
    assert len(nwis_client.requests) == backfill.BACKFILL_EMPTY_YEARS_STOP
    assert store._objects == {}
    summary = _backfill_summary(caplog)
    assert summary.outcome == "success"
    assert summary.record_count == 0
    assert "earliest none" in summary.getMessage()


def test_backfill_fetches_both_coops_products_for_every_year(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Thirteen requests a complete year, and an empty month is skipped."""
    coops_client = BackfillCoopsApi(
        data_years={2026, 2025}, six_minute_gaps={(2025, 3)}
    )
    _, _, store = _install_backfill_environment(
        monkeypatch, [COOPS_BACKFILL_CONFIG], coops_client=coops_client
    )

    with caplog.at_level(logging.INFO):
        exit_code = update.main(["--backfill-from", "2025"])

    assert exit_code == 0
    requests_2025 = [
        request for request in coops_client.requests if request[1].year == 2025
    ]
    assert [interval for interval, _ in requests_2025] == ["h"] + ["6-min"] * 12
    assert [begin.month for _, begin in requests_2025] == [1, *range(1, 13)]
    # The current year stops at the current month; nothing asks for the future.
    requests_2026 = [
        request for request in coops_client.requests if request[1].year == 2026
    ]
    assert [begin.month for _, begin in requests_2026] == [
        1,
        *range(1, BACKFILL_NOW.month + 1),
    ]

    # The empty March leaves the rest of 2025 archived.
    assert len(_archived_rows(store, _coops_key(2025), TEMPERATURE_UNIT)) == (
        HOURLY_ROWS_PER_YEAR + 11 * SIX_MINUTE_ROWS_PER_MONTH
    )
    assert len(_archived_rows(store, _coops_key(2026), TEMPERATURE_UNIT)) == (
        HOURLY_ROWS_PER_YEAR + BACKFILL_NOW.month * SIX_MINUTE_ROWS_PER_MONTH
    )
    assert any(
        "1 of 13 requests empty" in record.getMessage() for record in caplog.records
    )
    assert _backfill_summary(caplog).outcome == "success"


def test_backfill_publishes_nothing_and_never_hydrates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, store = _install_backfill_environment(
        monkeypatch,
        [COOPS_BACKFILL_CONFIG],
        coops_client=BackfillCoopsApi(data_years={2026}),
    )
    # A read bucket must not be consulted: the walk fetches every year from the
    # provider, so nothing in it may open the hydration store.
    monkeypatch.setenv("SHALLWESWIM_ARCHIVE_READ_BUCKET", "test-archive")
    hydration_store = Mock(side_effect=AssertionError("backfill must not hydrate"))
    monkeypatch.setattr(feeds, "object_store", hydration_store)
    pool_factory = Mock()
    monkeypatch.setattr(update, "ProcessPoolExecutor", pool_factory)

    assert update.main(["--backfill-from", "2025"]) == 0

    hydration_store.assert_not_called()
    pool_factory.assert_not_called()
    assert _published_keys(store, "published/") == []
    assert _archived_years(store, "archive/temperature/coops/") == [2026]


def test_backfill_location_filter_walks_only_the_named_locations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coops_client = BackfillCoopsApi(data_years={2026})
    nwis_client = BackfillNwisApi(data_years={2026})
    _, _, store = _install_backfill_environment(
        monkeypatch,
        [COOPS_BACKFILL_CONFIG, NWIS_BACKFILL_CONFIG, NO_HISTORY_CONFIG],
        coops_client=coops_client,
        nwis_client=nwis_client,
    )

    assert update.main(["--backfill-from", "2025", "--location", "bkn"]) == 0

    assert coops_client.requests == []
    assert [request.year for request in nwis_client.requests] == [2026, 2025]
    assert _archived_years(store, "archive/temperature/nwis/") == [2026]


def test_backfill_skips_a_location_without_a_historical_source(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    coops_client = BackfillCoopsApi(data_years={2026})
    _, _, store = _install_backfill_environment(
        monkeypatch, [NO_HISTORY_CONFIG], coops_client=coops_client
    )

    with caplog.at_level(logging.INFO):
        exit_code = update.main(["--backfill-from", "2025"])

    assert exit_code == 0
    assert coops_client.requests == []
    assert store._objects == {}
    assert any(
        "No historical temperature source to backfill" in record.getMessage()
        for record in caplog.records
    )
    summary = _backfill_summary(caplog)
    assert summary.outcome == "success"
    assert "no sources to walk" in summary.getMessage()


def test_backfill_isolates_a_failing_source_and_reports_partial(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    coops_client = BackfillCoopsApi(data_years={2026, 2025})
    nwis_client = BackfillNwisApi(
        data_years={2026, 2025}, errors={2025: RuntimeError("nwis boom")}
    )
    _, _, store = _install_backfill_environment(
        monkeypatch,
        [COOPS_BACKFILL_CONFIG, NWIS_BACKFILL_CONFIG],
        coops_client=coops_client,
        nwis_client=nwis_client,
    )

    with caplog.at_level(logging.INFO):
        exit_code = update.main(["--backfill-from", "2024"])

    # The failing walk stopped at its bad year; the other location finished.
    assert exit_code == 0
    assert [request.year for request in nwis_client.requests] == [2026, 2025]
    assert _archived_years(store, "archive/temperature/nwis/") == [2026]
    assert _archived_years(store, "archive/temperature/coops/") == [2025, 2026]
    errors = [
        record
        for record in caplog.records
        if record.levelno == logging.ERROR
        and "Backfill of nwis:temperature" in record.getMessage()
    ]
    assert len(errors) == 1
    assert "2025" in errors[0].getMessage()
    summary = _backfill_summary(caplog)
    assert summary.outcome == "partial"
    assert summary.levelno == logging.WARNING
    assert "stopped at error" in summary.getMessage()


def test_backfill_summary_event_fields_are_its_own_operation(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The summary never reports itself as a scheduled capture run."""
    _install_backfill_environment(
        monkeypatch,
        [NWIS_BACKFILL_CONFIG],
        nwis_client=BackfillNwisApi(data_years={2026, 2025}),
    )
    monkeypatch.setenv("CLOUD_RUN_EXECUTION", "shallweswim-backfill-01")

    with caplog.at_level(logging.INFO):
        assert update.main(["--backfill-from", "2025"]) == 0

    summary = _backfill_summary(caplog)
    assert summary.component == "updater"
    assert summary.operation == "backfill"
    assert summary.outcome == "success"
    assert summary.run_id == "shallweswim-backfill-01"
    assert isinstance(summary.duration_ms, int)
    assert summary.new_count == 2 * HOURLY_ROWS_PER_YEAR
    assert summary.revised_count == 0
    assert summary.record_count == summary.new_count + summary.revised_count
    assert [
        record for record in caplog.records if getattr(record, "operation", "") == "run"
    ] == []


def test_backfill_waits_out_a_provider_block_and_retries_the_request(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A 403 is a temporary block: the walk waits, asks again, and goes on."""
    nwis_client = BackfillNwisApi(data_years={2026, 2025}, blocks={2025: 1})
    _, _, store = _install_backfill_environment(
        monkeypatch, [NWIS_BACKFILL_CONFIG], nwis_client=nwis_client
    )
    sleeps = _record_backfill_sleeps(monkeypatch)

    with caplog.at_level(logging.INFO):
        exit_code = update.main(["--backfill-from", "2024"])

    assert exit_code == 0
    # The blocked year was asked for twice, and the walk reached the floor.
    assert [request.year for request in nwis_client.requests] == [
        2026,
        2025,
        2025,
        2024,
    ]
    assert _archived_years(store, "archive/temperature/nwis/") == [2025, 2026]
    assert sleeps.count(backfill.BACKFILL_BLOCK_PAUSE.total_seconds()) == 1
    blocked = [
        record
        for record in caplog.records
        if "Provider blocked" in record.getMessage()
        and record.levelno == logging.WARNING
    ]
    assert len(blocked) == 1
    assert "retry 1 of 3" in blocked[0].getMessage()
    summary = _backfill_summary(caplog)
    assert summary.outcome == "success"
    assert "stopped at floor" in summary.getMessage()


def test_backfill_gives_up_on_a_source_that_stays_blocked(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Three waits, and a fourth block ends the source like any other error."""
    nwis_client = BackfillNwisApi(
        data_years={2026, 2025}, blocks={2025: backfill.BACKFILL_BLOCK_RETRIES + 1}
    )
    _, _, store = _install_backfill_environment(
        monkeypatch, [NWIS_BACKFILL_CONFIG], nwis_client=nwis_client
    )
    sleeps = _record_backfill_sleeps(monkeypatch)

    with caplog.at_level(logging.INFO):
        exit_code = update.main(["--backfill-from", "2024"])

    assert exit_code == 0
    assert [request.year for request in nwis_client.requests] == [2026] + [2025] * (
        backfill.BACKFILL_BLOCK_RETRIES + 1
    )
    assert (
        sleeps.count(backfill.BACKFILL_BLOCK_PAUSE.total_seconds())
        == backfill.BACKFILL_BLOCK_RETRIES
    )
    # The years reached before the block stay archived.
    assert _archived_years(store, "archive/temperature/nwis/") == [2026]
    errors = [
        record
        for record in caplog.records
        if record.levelno == logging.ERROR
        and "Backfill of nwis:temperature" in record.getMessage()
    ]
    assert len(errors) == 1
    assert "403" in errors[0].getMessage()
    summary = _backfill_summary(caplog)
    assert summary.outcome == "partial"
    assert "stopped at error" in summary.getMessage()


def test_backfill_ends_a_source_at_once_on_a_status_that_is_not_a_block(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    nwis_client = BackfillNwisApi(
        data_years={2026},
        errors={
            2025: NwisConnectionError(
                "NWIS request for site 87654321 returned HTTP 404", status=404
            )
        },
    )
    _, _, store = _install_backfill_environment(
        monkeypatch, [NWIS_BACKFILL_CONFIG], nwis_client=nwis_client
    )
    sleeps = _record_backfill_sleeps(monkeypatch)

    with caplog.at_level(logging.INFO):
        exit_code = update.main(["--backfill-from", "2024"])

    assert exit_code == 0
    # The failing year was asked for once, and the walk stopped there.
    assert [request.year for request in nwis_client.requests] == [2026, 2025]
    assert backfill.BACKFILL_BLOCK_PAUSE.total_seconds() not in sleeps
    assert _archived_years(store, "archive/temperature/nwis/") == [2026]
    summary = _backfill_summary(caplog)
    assert summary.outcome == "partial"
    assert "stopped at error" in summary.getMessage()


def test_backfill_pauses_between_requests_within_a_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One pause between consecutive requests, and none after the last."""
    coops_client = BackfillCoopsApi(data_years={2026, 2025})
    _install_backfill_environment(
        monkeypatch, [COOPS_BACKFILL_CONFIG], coops_client=coops_client
    )
    sleeps = _record_backfill_sleeps(monkeypatch)

    assert update.main(["--backfill-from", "2025"]) == 0

    # The current year stops at the current month; 2025 is a whole year.
    assert len(coops_client.requests) == (1 + BACKFILL_NOW.month) + 13
    assert sleeps == [backfill.BACKFILL_REQUEST_PAUSE.total_seconds()] * (
        len(coops_client.requests) - 1
    )


def test_backfill_walks_locations_one_after_another(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A location's requests all land before the next location's first."""
    order: list[str] = []
    coops_client = BackfillCoopsApi(data_years={2026}, order=order)
    nwis_client = BackfillNwisApi(data_years={2026}, order=order)
    _install_backfill_environment(
        monkeypatch,
        [COOPS_BACKFILL_CONFIG, NWIS_BACKFILL_CONFIG],
        coops_client=coops_client,
        nwis_client=nwis_client,
    )

    assert update.main(["--backfill-from", "2025"]) == 0

    assert coops_client.requests and nwis_client.requests
    assert order == ["coops"] * len(coops_client.requests) + ["nwis"] * len(
        nwis_client.requests
    )


def test_observed_currents_refresh_on_the_live_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An observed current is a reading, not a prediction, so it expires like one."""
    coops_client, nwis_client, _ = _install_job_environment(
        monkeypatch, [TEST_CONFIG_OBSERVATION_CURRENTS, TEST_CONFIG_FULL]
    )
    clients: dict[str, BaseApiClient] = {"coops": coops_client, "nwis": nwis_client}
    observed = build_feeds(TEST_CONFIG_OBSERVATION_CURRENTS, clients)[
        feeds.FEED_CURRENTS
    ]
    predicted = build_feeds(TEST_CONFIG_FULL, clients)[feeds.FEED_CURRENTS]

    assert observed is not None and predicted is not None
    assert (
        observed.expiration_interval
        == manager_module.EXPIRATION_PERIODS[feeds.FEED_LIVE_TEMPS]
    )
    assert (
        predicted.expiration_interval
        == manager_module.EXPIRATION_PERIODS[feeds.FEED_CURRENTS]
    )
