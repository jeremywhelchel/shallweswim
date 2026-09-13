"""Serving isolation and observation capture through real feed updates."""

import datetime
import logging
from collections.abc import Iterator
from io import BytesIO
from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest

from shallweswim import config
from shallweswim.archive import capture
from shallweswim.archive import store as archive_store
from shallweswim.archive.observations import (
    CURRENTS_MEASUREMENT,
    CURRENTS_UNIT,
    CURRENTS_VALUE_COLUMN,
    TEMPERATURE_MEASUREMENT,
    TEMPERATURE_UNIT,
    TEMPERATURE_VALUE_COLUMN,
    read_observations,
)
from shallweswim.archive.store import MemoryObjectStore
from shallweswim.clients.base import StationUnavailableError
from shallweswim.core import feeds
from shallweswim.types import TIDE_TYPE_CATEGORIES


@pytest.fixture(autouse=True)
def clear_store_cache() -> Iterator[None]:
    """Keep mocked stores from leaking between capture tests."""
    archive_store.gcs_store.cache_clear()
    yield
    archive_store.gcs_store.cache_clear()


@pytest.mark.asyncio
async def test_capture_reuses_store_per_bucket(monkeypatch) -> None:
    factory = Mock(side_effect=lambda bucket: MemoryObjectStore())
    monkeypatch.setattr(archive_store, "GcsObjectStore", factory)
    feed = _feed()
    for bucket in ("first", "first", "second", "first"):
        await capture.capture_observations(
            bucket,
            frame=_frame("2026-01-01"),
            source_identity=feed.feed_config.citation_key,
            measurement=TEMPERATURE_MEASUREMENT,
            value_column=TEMPERATURE_VALUE_COLUMN,
            unit=TEMPERATURE_UNIT,
            retrieved_at=datetime.datetime(2026, 1, 2),
        )
    assert [call.args for call in factory.call_args_list] == [("first",), ("second",)]


def _feed() -> feeds.CoopsTempFeed:
    location = next(item for item in config.get_all_configs() if item.code == "nyc")
    return feeds.CoopsTempFeed(
        location_config=location,
        feed_config=config.CoopsTempFeedConfig(station=8518750),
        interval="h",
        expiration_interval=datetime.timedelta(minutes=10),
    )


def _frame(*times: str) -> pd.DataFrame:
    """Build a client-style temperature frame indexed by UTC instants."""
    return pd.DataFrame(
        {"water_temp": [60.0] * len(times)},
        index=pd.DatetimeIndex(times, tz="UTC", name="time"),
    )


def _served(frame: pd.DataFrame, location: config.LocationConfig) -> pd.DataFrame:
    """The naive location-local frame a feed publishes from a UTC client frame."""
    return feeds.to_serving_index(frame, location.timezone)


@pytest.mark.asyncio
async def test_update_archives_by_utc_year_and_preserves_serving(monkeypatch) -> None:
    monkeypatch.setenv("SHALLWESWIM_ARCHIVE_BUCKET", "test-archive")
    store = MemoryObjectStore()
    monkeypatch.setattr(archive_store, "GcsObjectStore", lambda bucket: store)
    # One instant each side of the UTC year boundary, on the same local day.
    frame = _frame("2025-12-31 23:00", "2026-01-01 00:00")
    monkeypatch.setattr(feeds.CoopsTempFeed, "_fetch", AsyncMock(return_value=frame))
    feed = _feed()
    await feed.update({})
    pd.testing.assert_frame_equal(feed.values, _served(frame, feed.location_config))
    assert feed.values.index.strftime("%Y-%m-%d %H:%M").tolist() == [
        "2025-12-31 18:00",
        "2025-12-31 19:00",
    ]
    assert feed._fetch_timestamp is not None
    assert feed._next_fetch_after == feed._fetch_timestamp + feed.expiration_interval
    for year in (2025, 2026):
        stored = await store.read(f"archive/temperature/coops/8518750/{year}.parquet")
        assert stored is not None
        rows = read_observations(BytesIO(stored.data), expected_unit="F")
        assert len(rows) == 1
        assert rows["retrieved_at"].iloc[0] == pd.Timestamp(
            feed._fetch_timestamp, tz="UTC"
        )


@pytest.mark.asyncio
async def test_identical_second_update_archives_nothing_new(
    monkeypatch, caplog
) -> None:
    """A repeated fetch of the same readings overlaps and writes nothing."""
    monkeypatch.setenv("SHALLWESWIM_ARCHIVE_BUCKET", "test-archive")
    store = MemoryObjectStore()
    monkeypatch.setattr(archive_store, "GcsObjectStore", lambda bucket: store)
    frame = _frame("2026-01-01 12:00", "2026-01-01 13:00")
    monkeypatch.setattr(feeds.CoopsTempFeed, "_fetch", AsyncMock(return_value=frame))

    first = _feed()
    await first.update({})
    stored = await store.read("archive/temperature/coops/8518750/2026.parquet")
    assert stored is not None

    # A fresh feed is the next run's fetch of identical upstream readings.
    second = _feed()
    with caplog.at_level(logging.INFO):
        await second.update({})

    assert first.last_capture == capture.CaptureResult(2, 0)
    assert second.last_capture == capture.CaptureResult(0, 0)
    merges = [r for r in caplog.records if getattr(r, "operation", None) == "merge"]
    assert [record.outcome for record in merges] == ["unchanged"]
    assert merges[0].overlap_count == 2
    assert (
        await store.read("archive/temperature/coops/8518750/2026.parquet")
    ) == stored


@pytest.mark.asyncio
async def test_last_capture_reports_counts_only_when_capture_ran(monkeypatch) -> None:
    """The feed keeps its merge counts so the job can sum them per run."""
    store = MemoryObjectStore()
    monkeypatch.setattr(archive_store, "GcsObjectStore", lambda bucket: store)
    monkeypatch.setattr(
        feeds.CoopsTempFeed,
        "_fetch",
        AsyncMock(return_value=_frame("2026-01-01 12:00", "2026-01-01 13:00")),
    )

    monkeypatch.delenv("SHALLWESWIM_ARCHIVE_BUCKET", raising=False)
    disabled = _feed()
    await disabled.update({})
    assert disabled.last_capture is None

    monkeypatch.setenv("SHALLWESWIM_ARCHIVE_BUCKET", "test-archive")
    captured = _feed()
    await captured.update({})
    assert captured.last_capture == capture.CaptureResult(2, 0)

    # Update resets the counts, so a second update of the same feed reports
    # only its own captures - here a re-fetch of readings already archived.
    captured._next_fetch_after = feeds.utc_now() - datetime.timedelta(seconds=1)
    await captured.update({})
    assert captured.last_capture == capture.CaptureResult(0, 0)

    monkeypatch.setattr(store, "read", AsyncMock(side_effect=OSError("offline")))
    failed = _feed()
    await failed.update({})
    assert failed.last_capture == capture.CaptureResult(0, 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["naive_index", "credentials", "storage"])
async def test_archive_failure_keeps_success_state(
    monkeypatch, caplog, failure: str
) -> None:
    monkeypatch.setenv("SHALLWESWIM_ARCHIVE_BUCKET", "test-archive")
    if failure == "naive_index":
        # A client that forgot to return UTC fails normalization, not serving.
        frame = pd.DataFrame(
            {"water_temp": [60.0]},
            index=pd.DatetimeIndex(["2026-01-01 12:00"], name="time"),
        )
    else:
        frame = _frame("2026-01-01 12:00")
    monkeypatch.setattr(feeds.CoopsTempFeed, "_fetch", AsyncMock(return_value=frame))
    store = MemoryObjectStore()
    if failure == "credentials":

        def make_store(bucket: str) -> MemoryObjectStore:
            raise RuntimeError("credentials unavailable")

        monkeypatch.setattr(archive_store, "GcsObjectStore", make_store)
    else:
        monkeypatch.setattr(archive_store, "GcsObjectStore", lambda bucket: store)
    if failure == "storage":
        monkeypatch.setattr(store, "read", AsyncMock(side_effect=OSError("offline")))
    feed = _feed()
    with caplog.at_level(logging.INFO):
        await feed.update({})
    assert feed._last_error is None
    assert feed._consecutive_failures == 0
    assert feed._ready_event.is_set()
    assert feed._fetch_timestamp is not None
    assert feed._next_fetch_after == feed._fetch_timestamp + feed.expiration_interval
    # A naive frame passes through the serving step; a UTC frame is converted.
    expected = (
        frame if failure == "naive_index" else _served(frame, feed.location_config)
    )
    pd.testing.assert_frame_equal(feed.values, expected)
    failures = [r for r in caplog.records if getattr(r, "operation", None) == "merge"]
    assert len(failures) == 1
    assert failures[0].outcome == "failed"


@pytest.mark.asyncio
async def test_configured_outlier_is_served_out_but_archived(monkeypatch) -> None:
    """The archive keeps provider readings a configured outlier hides from serving."""
    monkeypatch.setenv("SHALLWESWIM_ARCHIVE_BUCKET", "test-archive")
    store = MemoryObjectStore()
    monkeypatch.setattr(archive_store, "GcsObjectStore", lambda bucket: store)
    frame = _frame("2026-01-01 12:00", "2026-01-01 13:00")
    monkeypatch.setattr(feeds.CoopsTempFeed, "_fetch", AsyncMock(return_value=frame))

    location = next(item for item in config.get_all_configs() if item.code == "nyc")
    feed = feeds.CoopsTempFeed(
        location_config=location,
        feed_config=config.CoopsTempFeedConfig(
            station=8518750, outliers=["2026-01-01 08:00:00"]
        ),
        interval="h",
        expiration_interval=datetime.timedelta(minutes=10),
    )
    await feed.update({})

    # 12:00 and 13:00 UTC serve as local 07:00 and 08:00; the outlier hides 08:00.
    assert feed.values.index.strftime("%H:%M").tolist() == ["07:00"]

    stored = await store.read("archive/temperature/coops/8518750/2026.parquet")
    assert stored is not None
    rows = read_observations(BytesIO(stored.data), expected_unit="F")
    local_times = (
        rows["observed_at"].dt.tz_convert(location.timezone).dt.strftime("%H:%M")
    )
    assert local_times.tolist() == ["07:00", "08:00"]


@pytest.mark.asyncio
async def test_conflicting_repeated_instant_is_dropped_with_one_warning(
    monkeypatch, caplog
) -> None:
    """A raw year frame repeating an instant with a different value warns once."""
    monkeypatch.setenv("SHALLWESWIM_ARCHIVE_BUCKET", "test-archive")
    store = MemoryObjectStore()
    monkeypatch.setattr(archive_store, "GcsObjectStore", lambda bucket: store)
    live = _feed()
    history = feeds.HistoricalTempsFeed(
        location_config=live.location_config,
        feed_config=live.feed_config,
        start_year=2026,
        end_year=2026,
        expiration_interval=datetime.timedelta(days=1),
    )
    raw = pd.DataFrame(
        {"water_temp": [60.0, 61.0, 62.0]},
        index=pd.DatetimeIndex(
            ["2026-01-01 12:00", "2026-01-01 12:00", "2026-01-01 13:00"],
            tz="UTC",
            name="time",
        ),
    )
    monkeypatch.setattr(feeds.CoopsTempFeed, "_fetch", AsyncMock(return_value=raw))
    with caplog.at_level(logging.INFO):
        await history.update({})

    conflicts = [
        record
        for record in caplog.records
        if getattr(record, "outcome", None) == "conflict_dropped"
    ]
    assert len(conflicts) == 1
    assert conflicts[0].levelno == logging.WARNING
    assert conflicts[0].component == "archive"
    assert conflicts[0].operation == "normalize"
    assert conflicts[0].source_identity == history.feed_config.citation_key
    assert conflicts[0].record_count == 1

    stored = await store.read("archive/temperature/coops/8518750/2026.parquet")
    assert stored is not None
    rows = read_observations(BytesIO(stored.data), expected_unit="F")
    assert rows["value"].tolist() == [60.0, 62.0]


@pytest.mark.asyncio
async def test_disabled_capture_does_not_construct_store(monkeypatch) -> None:
    monkeypatch.delenv("SHALLWESWIM_ARCHIVE_BUCKET", raising=False)
    archive = AsyncMock()
    monkeypatch.setattr(feeds, "capture_observations", archive)
    monkeypatch.setattr(
        feeds.CoopsTempFeed, "_fetch", AsyncMock(return_value=_frame("2026-01-01"))
    )
    await _feed().update({})
    archive.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_fetch_does_not_archive(monkeypatch) -> None:
    monkeypatch.setenv("SHALLWESWIM_ARCHIVE_BUCKET", "test-archive")
    archive = AsyncMock()
    monkeypatch.setattr(feeds, "capture_observations", archive)
    monkeypatch.setattr(
        feeds.CoopsTempFeed,
        "_fetch",
        AsyncMock(side_effect=StationUnavailableError("offline")),
    )
    await _feed().update({})
    archive.assert_not_awaited()


@pytest.mark.asyncio
async def test_historical_capture_only_fresh_years_even_on_partial_failure(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SHALLWESWIM_ARCHIVE_BUCKET", "test-archive")
    live = _feed()
    history = feeds.HistoricalTempsFeed(
        location_config=live.location_config,
        feed_config=live.feed_config,
        start_year=2023,
        end_year=2025,
        expiration_interval=datetime.timedelta(days=1),
    )
    history._year_cache[2023] = _served(_frame("2023-06-01"), live.location_config)
    old_retrieval = datetime.datetime(2023, 6, 2)
    history._year_cache_fetch_timestamp[2023] = old_retrieval
    fresh = _frame("2024-06-01")
    fetch = AsyncMock(side_effect=[fresh, StationUnavailableError("offline")])
    monkeypatch.setattr(feeds.CoopsTempFeed, "_fetch", fetch)
    archive = AsyncMock()
    monkeypatch.setattr(feeds, "capture_observations", archive)
    await history.update({})
    assert fetch.await_count == 2
    archive.assert_awaited_once()
    captured = archive.call_args.kwargs
    assert captured["frame"].index.year.tolist() == [2024]
    assert captured["retrieved_at"] == history._year_cache_fetch_timestamp[2024]
    assert history._year_cache_fetch_timestamp[2023] == old_retrieval
    assert history._last_error is not None
    assert history._consecutive_failures == 1


def _ten_minute_frame(*utc_hours: str) -> pd.DataFrame:
    """Native ten-minute cadence for each listed UTC hour, in order."""
    return _frame(
        *(f"{hour}:{minute:02d}" for hour in utc_hours for minute in range(0, 60, 10))
    )


@pytest.mark.asyncio
async def test_historical_capture_archives_native_cadence_and_both_folds(
    monkeypatch, caplog
) -> None:
    monkeypatch.setenv("SHALLWESWIM_ARCHIVE_BUCKET", "test-archive")
    store = MemoryObjectStore()
    monkeypatch.setattr(archive_store, "GcsObjectStore", lambda bucket: store)
    live = _feed()
    history = feeds.HistoricalTempsFeed(
        location_config=live.location_config,
        feed_config=live.feed_config,
        start_year=2024,
        end_year=2025,
        expiration_interval=datetime.timedelta(days=1),
    )
    # US/Eastern falls back at 06:00 UTC on both days, so the UTC hour before
    # and the UTC hour after each boundary share a local wall time.
    # 2024 is an hourly year: 05:00 and 06:00 UTC are both local 01:00.
    hourly = _frame(
        "2024-11-03 04:00", "2024-11-03 05:00", "2024-11-03 06:00", "2024-11-03 07:00"
    )
    # 2025 is a native ten-minute year carrying the same two folds.
    both_folds = _ten_minute_frame(
        "2025-11-02 04",
        "2025-11-02 05",
        "2025-11-02 06",
        "2025-11-02 07",
    )
    monkeypatch.setattr(
        feeds.CoopsTempFeed,
        "_fetch",
        AsyncMock(side_effect=[hourly, both_folds]),
    )
    with caplog.at_level(logging.INFO):
        await history.update({})

    stored = await store.read("archive/temperature/coops/8518750/2025.parquet")
    assert stored is not None
    rows = read_observations(BytesIO(stored.data), expected_unit="F")
    assert len(rows) == len(both_folds)
    assert rows["observed_at"].is_unique
    local = rows["observed_at"].dt.tz_convert(live.location_config.timezone)
    assert local.dt.strftime("%H:%M").tolist().count("01:00") == 2

    # Serving still resamples to hourly, collapsing the repeated local hour.
    served = history.values.loc["2025-11-02 00:00":"2025-11-02 02:00"]
    assert served.index.strftime("%H:%M").tolist() == ["00:00", "01:00", "02:00"]

    stored_2024 = await store.read("archive/temperature/coops/8518750/2024.parquet")
    assert stored_2024 is not None
    rows_2024 = read_observations(BytesIO(stored_2024.data), expected_unit="F")
    # Every provider row reaches the archive, including both fall-back folds.
    assert len(rows_2024) == len(hourly)
    local_2024 = rows_2024["observed_at"].dt.tz_convert(live.location_config.timezone)
    assert local_2024.dt.strftime("%H:%M").tolist() == [
        "00:00",
        "01:00",
        "01:00",
        "02:00",
    ]
    assert not [
        record
        for record in caplog.records
        if getattr(record, "operation", None) == "normalize"
    ]
    # One update captures once per fetched year, and the run summary needs the
    # sum of every year, not only the last one captured.
    assert history.last_capture == capture.CaptureResult(len(rows_2024) + len(rows), 0)


@pytest.mark.asyncio
async def test_historical_capture_sums_years_around_a_failed_one(monkeypatch) -> None:
    """A year whose capture fails contributes zeros, not a lost total."""
    monkeypatch.setenv("SHALLWESWIM_ARCHIVE_BUCKET", "test-archive")
    store = MemoryObjectStore()
    monkeypatch.setattr(archive_store, "GcsObjectStore", lambda bucket: store)
    live = _feed()
    history = feeds.HistoricalTempsFeed(
        location_config=live.location_config,
        feed_config=live.feed_config,
        start_year=2024,
        end_year=2025,
        expiration_interval=datetime.timedelta(days=1),
    )
    first_year = _frame("2024-06-01 12:00", "2024-06-01 13:00")
    second_year = _frame("2025-06-01 12:00", "2025-06-01 13:00", "2025-06-01 14:00")
    monkeypatch.setattr(
        feeds.CoopsTempFeed,
        "_fetch",
        AsyncMock(side_effect=[first_year, second_year]),
    )

    real_capture = capture.capture_observations
    calls = 0

    async def flaky(*args, **kwargs) -> capture.CaptureResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("offline")
        return await real_capture(*args, **kwargs)

    monkeypatch.setattr(feeds, "capture_observations", flaky)
    await history.update({})

    assert calls == 2
    # 2024 is captured first and fails, so only 2025's rows reach the archive.
    assert await store.read("archive/temperature/coops/8518750/2024.parquet") is None
    stored = await store.read("archive/temperature/coops/8518750/2025.parquet")
    assert stored is not None
    assert len(read_observations(BytesIO(stored.data), expected_unit="F")) == len(
        second_year
    )
    assert history.last_capture == capture.CaptureResult(len(second_year), 0)


def test_source_paths_preserve_station_parameter_identity() -> None:
    partitions = capture._partitions(
        _frame("2026-01-01"),
        "nwis:temperature:08155500:00010",
        TEMPERATURE_MEASUREMENT,
        TEMPERATURE_VALUE_COLUMN,
        TEMPERATURE_UNIT,
        datetime.datetime(2026, 1, 2),
    )
    assert partitions[0][0] == "archive/temperature/nwis/08155500%3A00010/2026.parquet"


@pytest.mark.asyncio
async def test_prediction_feed_does_not_capture(monkeypatch) -> None:
    monkeypatch.setenv("SHALLWESWIM_ARCHIVE_BUCKET", "test-archive")
    archive = AsyncMock()
    monkeypatch.setattr(feeds, "capture_observations", archive)
    frame = pd.DataFrame(
        {
            "prediction": [2.0],
            "type": pd.Categorical(["high"], categories=TIDE_TYPE_CATEGORIES),
        },
        index=pd.DatetimeIndex(["2026-01-01"], name="time"),
    )
    monkeypatch.setattr(feeds.CoopsTidesFeed, "_fetch", AsyncMock(return_value=frame))
    tide = feeds.CoopsTidesFeed(
        location_config=_feed().location_config,
        feed_config=config.CoopsTideFeedConfig(station=8517741),
        expiration_interval=datetime.timedelta(hours=1),
    )
    await tide.update({})
    archive.assert_not_awaited()
    assert tide._fetch_timestamp is not None


def _currents_feed() -> feeds.NwisCurrentFeed:
    location = next(item for item in config.get_all_configs() if item.code == "sdf")
    assert isinstance(location.currents_source, config.NwisCurrentFeedConfig)
    return feeds.NwisCurrentFeed(
        location_config=location,
        feed_config=location.currents_source,
        expiration_interval=datetime.timedelta(minutes=10),
    )


def _currents_frame(*times: str) -> pd.DataFrame:
    """Build a client-style currents frame indexed by UTC instants."""
    return pd.DataFrame(
        {"velocity": [1.5] * len(times)},
        index=pd.DatetimeIndex(times, tz="UTC", name="time"),
    )


def test_currents_bindings_match_archive_contract() -> None:
    assert CURRENTS_MEASUREMENT == "currents"
    assert CURRENTS_VALUE_COLUMN == "velocity"
    assert CURRENTS_UNIT == "kt"


def test_currents_source_paths_use_currents_prefix() -> None:
    feed = _currents_feed()
    partitions = capture._partitions(
        _currents_frame("2026-01-01"),
        feed.feed_config.citation_key,
        CURRENTS_MEASUREMENT,
        CURRENTS_VALUE_COLUMN,
        CURRENTS_UNIT,
        datetime.datetime(2026, 1, 2),
    )
    assert partitions[0][0] == "archive/currents/nwis/03292494%3A72255/2026.parquet"
    assert partitions[0][1]["unit"].tolist() == ["kt"]
    assert partitions[0][1]["value"].tolist() == [1.5]


def test_currents_capture_rejects_mismatched_measurement() -> None:
    feed = _currents_feed()
    with pytest.raises(ValueError, match="Expected a temperature source identity"):
        capture._partitions(
            _currents_frame("2026-01-01"),
            feed.feed_config.citation_key,
            TEMPERATURE_MEASUREMENT,
            TEMPERATURE_VALUE_COLUMN,
            TEMPERATURE_UNIT,
            datetime.datetime(2026, 1, 2),
        )


@pytest.mark.asyncio
async def test_observational_currents_update_archives_by_utc_year(monkeypatch) -> None:
    monkeypatch.setenv("SHALLWESWIM_ARCHIVE_BUCKET", "test-archive")
    store = MemoryObjectStore()
    monkeypatch.setattr(archive_store, "GcsObjectStore", lambda bucket: store)
    frame = _currents_frame("2025-12-31 23:00", "2026-01-01 00:00")
    monkeypatch.setattr(feeds.NwisCurrentFeed, "_fetch", AsyncMock(return_value=frame))
    feed = _currents_feed()
    await feed.update({})
    pd.testing.assert_frame_equal(feed.values, _served(frame, feed.location_config))
    assert feed._fetch_timestamp is not None
    assert feed._next_fetch_after == feed._fetch_timestamp + feed.expiration_interval
    for year in (2025, 2026):
        stored = await store.read(
            f"archive/currents/nwis/03292494%3A72255/{year}.parquet"
        )
        assert stored is not None
        rows = read_observations(BytesIO(stored.data), expected_unit=CURRENTS_UNIT)
        assert len(rows) == 1
        assert rows["value"].iloc[0] == 1.5
        assert rows["retrieved_at"].iloc[0] == pd.Timestamp(
            feed._fetch_timestamp, tz="UTC"
        )


@pytest.mark.asyncio
async def test_currents_archive_failure_keeps_success_state(
    monkeypatch, caplog
) -> None:
    monkeypatch.setenv("SHALLWESWIM_ARCHIVE_BUCKET", "test-archive")
    store = MemoryObjectStore()
    monkeypatch.setattr(archive_store, "GcsObjectStore", lambda bucket: store)
    monkeypatch.setattr(store, "read", AsyncMock(side_effect=OSError("offline")))
    frame = _currents_frame("2026-01-01 12:00")
    monkeypatch.setattr(feeds.NwisCurrentFeed, "_fetch", AsyncMock(return_value=frame))
    feed = _currents_feed()
    with caplog.at_level(logging.INFO):
        await feed.update({})
    assert feed._last_error is None
    assert feed._consecutive_failures == 0
    assert feed._next_fetch_after == feed._fetch_timestamp + feed.expiration_interval
    pd.testing.assert_frame_equal(feed.values, _served(frame, feed.location_config))
    failures = [r for r in caplog.records if getattr(r, "operation", None) == "merge"]
    assert len(failures) == 1
    assert failures[0].outcome == "failed"
    assert failures[0].source_identity == "nwis:currents:03292494:72255"


@pytest.mark.asyncio
async def test_prediction_currents_feed_does_not_capture(monkeypatch) -> None:
    monkeypatch.setenv("SHALLWESWIM_ARCHIVE_BUCKET", "test-archive")
    archive = AsyncMock()
    monkeypatch.setattr(feeds, "capture_observations", archive)
    location = _feed().location_config
    assert isinstance(location.currents_source, config.CoopsCurrentsFeedConfig)
    monkeypatch.setattr(
        feeds.CoopsCurrentsFeed,
        "_fetch",
        AsyncMock(return_value=_currents_frame("2026-01-01 12:00")),
    )
    currents = feeds.CoopsCurrentsFeed(
        location_config=location,
        feed_config=location.currents_source,
        expiration_interval=datetime.timedelta(hours=1),
    )
    await currents.update({})
    archive.assert_not_awaited()
    assert currents._fetch_timestamp is not None
