"""One-shot capture job behavior over mocked clients and a memory archive.

These tests exercise ``shallweswim.update`` through the real capture hook
(``shallweswim.archive.capture``) so archived keys and rows are asserted from an
in-memory object store rather than mocked out.
"""

import asyncio
import datetime
import logging
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
from shallweswim.clients.base import BaseApiClient
from shallweswim.clients.coops import CoopsApi
from shallweswim.clients.nwis import NwisApi
from shallweswim.config import CoopsTempFeedConfig, LocationConfig
from shallweswim.core import feeds
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


def test_full_history_flag_fetches_configured_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coops_client, _, _ = _install_job_environment(
        monkeypatch, [MULTI_YEAR_HISTORY_CONFIG]
    )

    assert update.main(["--full-history"]) == 0
    assert coops_client.historic_temperature_calls == utc_now().year - 2024 + 1


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
    # path skips, and the full historical range without --full-history.
    assert coops_client.tides_calls == 2
    assert coops_client.currents_calls == 1
    assert nwis_client.currents_calls == 1
    assert coops_client.historic_temperature_calls == 2 * (utc_now().year - 2011 + 1)
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
    """The manifest is the schedule, so an immediate rerun fetches nothing."""
    coops_client, nwis_client, store = _install_publish_environment(
        monkeypatch, [TEST_CONFIG_OBSERVATION_CURRENTS]
    )

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
    # Live temperature came due; nothing else did.
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
    assert _provider_calls(coops_client, nwis_client)[3:] == calls_after_first[3:]

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
    # fresh feed again and fetches; every other feed still holds.
    assert coops_client.live_temperature_calls == calls_after_first[2] + 1
    assert _provider_calls(coops_client, nwis_client)[3:] == calls_after_first[3:]


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
    due = datetime.datetime(2026, 6, 1, 12, 0, tzinfo=datetime.UTC)

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
