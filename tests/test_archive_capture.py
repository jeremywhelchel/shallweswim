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
    capture._store_for.cache_clear()
    yield
    capture._store_for.cache_clear()


@pytest.mark.asyncio
async def test_capture_reuses_store_per_bucket(monkeypatch) -> None:
    factory = Mock(side_effect=lambda bucket: MemoryObjectStore())
    monkeypatch.setattr(capture, "GcsObjectStore", factory)
    feed = _feed()
    for bucket in ("first", "first", "second", "first"):
        await capture.capture_observations(
            bucket,
            frame=_frame("2026-01-01"),
            source_identity=feed.feed_config.citation_key,
            measurement=TEMPERATURE_MEASUREMENT,
            value_column=TEMPERATURE_VALUE_COLUMN,
            unit=TEMPERATURE_UNIT,
            timezone=feed.location_config.timezone,
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
    return pd.DataFrame(
        {"water_temp": [60.0] * len(times)},
        index=pd.DatetimeIndex(times, name="time"),
    )


@pytest.mark.asyncio
async def test_update_archives_by_utc_year_and_preserves_serving(monkeypatch) -> None:
    monkeypatch.setenv("SHALLWESWIM_ARCHIVE_BUCKET", "test-archive")
    store = MemoryObjectStore()
    monkeypatch.setattr(capture, "GcsObjectStore", lambda bucket: store)
    frame = _frame("2025-12-31 18:00", "2025-12-31 19:00")
    monkeypatch.setattr(feeds.CoopsTempFeed, "_fetch", AsyncMock(return_value=frame))
    feed = _feed()
    await feed.update({})
    pd.testing.assert_frame_equal(feed.values, frame)
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
@pytest.mark.parametrize("failure", ["timezone", "credentials", "storage"])
async def test_archive_failure_keeps_success_state(
    monkeypatch, caplog, failure: str
) -> None:
    monkeypatch.setenv("SHALLWESWIM_ARCHIVE_BUCKET", "test-archive")
    frame = _frame("2026-03-08 02:30" if failure == "timezone" else "2026-01-01 12:00")
    monkeypatch.setattr(feeds.CoopsTempFeed, "_fetch", AsyncMock(return_value=frame))
    store = MemoryObjectStore()
    if failure == "credentials":

        def make_store(bucket: str) -> MemoryObjectStore:
            raise RuntimeError("credentials unavailable")

        monkeypatch.setattr(capture, "GcsObjectStore", make_store)
    else:
        monkeypatch.setattr(capture, "GcsObjectStore", lambda bucket: store)
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
    pd.testing.assert_frame_equal(feed.values, frame)
    failures = [r for r in caplog.records if getattr(r, "operation", None) == "merge"]
    assert len(failures) == 1
    assert failures[0].outcome == "failed"


@pytest.mark.asyncio
async def test_unresolvable_fall_back_row_is_dropped_with_one_warning(
    monkeypatch, caplog
) -> None:
    monkeypatch.setenv("SHALLWESWIM_ARCHIVE_BUCKET", "test-archive")
    store = MemoryObjectStore()
    monkeypatch.setattr(capture, "GcsObjectStore", lambda bucket: store)
    frame = _frame("2026-11-01 00:30", "2026-11-01 01:30", "2026-11-01 02:00")
    monkeypatch.setattr(feeds.CoopsTempFeed, "_fetch", AsyncMock(return_value=frame))
    feed = _feed()
    with caplog.at_level(logging.INFO):
        await feed.update({})

    pd.testing.assert_frame_equal(feed.values, frame)
    dropped = [
        record
        for record in caplog.records
        if getattr(record, "operation", None) == "normalize"
    ]
    assert len(dropped) == 1
    assert dropped[0].levelno == logging.WARNING
    assert dropped[0].component == "archive"
    assert dropped[0].outcome == "ambiguous_dropped"
    assert dropped[0].source_identity == feed.feed_config.citation_key
    assert dropped[0].record_count == 1

    stored = await store.read("archive/temperature/coops/8518750/2026.parquet")
    assert stored is not None
    rows = read_observations(BytesIO(stored.data), expected_unit="F")
    assert rows["observed_at"].dt.strftime("%H:%M").tolist() == ["04:30", "07:00"]


@pytest.mark.asyncio
async def test_conflicting_repeated_instant_is_dropped_with_one_warning(
    monkeypatch, caplog
) -> None:
    """A raw year frame repeating an instant with a different value warns once."""
    monkeypatch.setenv("SHALLWESWIM_ARCHIVE_BUCKET", "test-archive")
    store = MemoryObjectStore()
    monkeypatch.setattr(capture, "GcsObjectStore", lambda bucket: store)
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
            ["2026-01-01 12:00", "2026-01-01 12:00", "2026-01-01 13:00"], name="time"
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
    history._year_cache[2023] = _frame("2023-06-01")
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


def _ten_minute_frame(*local_hours: str) -> pd.DataFrame:
    """Native ten-minute cadence for each listed local hour, in order."""
    return _frame(
        *(f"{hour}:{minute:02d}" for hour in local_hours for minute in range(0, 60, 10))
    )


@pytest.mark.asyncio
async def test_historical_capture_archives_native_cadence_and_both_folds(
    monkeypatch, caplog
) -> None:
    monkeypatch.setenv("SHALLWESWIM_ARCHIVE_BUCKET", "test-archive")
    store = MemoryObjectStore()
    monkeypatch.setattr(capture, "GcsObjectStore", lambda bucket: store)
    live = _feed()
    history = feeds.HistoricalTempsFeed(
        location_config=live.location_config,
        feed_config=live.feed_config,
        start_year=2024,
        end_year=2025,
        expiration_interval=datetime.timedelta(days=1),
    )
    # 2024 is a CO-OPS-style hourly year whose fall-back 01:00 appears once.
    single_fold = _frame(
        "2024-11-03 00:00", "2024-11-03 01:00", "2024-11-03 02:00", "2024-11-03 03:00"
    )
    # 2025 is a native ten-minute year carrying both fall-back folds.
    both_folds = _ten_minute_frame(
        "2025-11-02 00",
        "2025-11-02 01",
        "2025-11-02 01",
        "2025-11-02 02",
        "2025-11-02 03",
    )
    monkeypatch.setattr(
        feeds.CoopsTempFeed,
        "_fetch",
        AsyncMock(side_effect=[single_fold, both_folds]),
    )
    with caplog.at_level(logging.INFO):
        await history.update({})

    stored = await store.read("archive/temperature/coops/8518750/2025.parquet")
    assert stored is not None
    rows = read_observations(BytesIO(stored.data), expected_unit="F")
    assert len(rows) == len(both_folds)
    assert rows["observed_at"].is_unique
    assert rows["observed_at"].dt.strftime("%H:%M").tolist().count("05:00") == 1
    assert rows["observed_at"].dt.strftime("%H:%M").tolist().count("06:00") == 1

    # Serving still resamples to hourly, collapsing the repeated local hour.
    served = history.values.loc["2025-11-02 00:00":"2025-11-02 03:00"]
    assert served.index.strftime("%H:%M").tolist() == [
        "00:00",
        "01:00",
        "02:00",
        "03:00",
    ]

    stored_2024 = await store.read("archive/temperature/coops/8518750/2024.parquet")
    assert stored_2024 is not None
    rows_2024 = read_observations(BytesIO(stored_2024.data), expected_unit="F")
    assert len(rows_2024) == len(single_fold) - 1
    dropped = [
        record
        for record in caplog.records
        if getattr(record, "operation", None) == "normalize"
    ]
    assert len(dropped) == 1
    assert dropped[0].levelno == logging.WARNING
    assert dropped[0].outcome == "ambiguous_dropped"
    assert dropped[0].source_identity == history.feed_config.citation_key
    assert dropped[0].record_count == 1


def test_source_paths_preserve_station_parameter_identity() -> None:
    live = _feed()
    partitions = capture._partitions(
        _frame("2026-01-01"),
        "nwis:temperature:08155500:00010",
        TEMPERATURE_MEASUREMENT,
        TEMPERATURE_VALUE_COLUMN,
        TEMPERATURE_UNIT,
        live.location_config.timezone,
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
    return pd.DataFrame(
        {"velocity": [1.5] * len(times)},
        index=pd.DatetimeIndex(times, name="time"),
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
        feed.location_config.timezone,
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
            feed.location_config.timezone,
            datetime.datetime(2026, 1, 2),
        )


@pytest.mark.asyncio
async def test_observational_currents_update_archives_by_utc_year(monkeypatch) -> None:
    monkeypatch.setenv("SHALLWESWIM_ARCHIVE_BUCKET", "test-archive")
    store = MemoryObjectStore()
    monkeypatch.setattr(capture, "GcsObjectStore", lambda bucket: store)
    frame = _currents_frame("2025-12-31 18:00", "2025-12-31 19:00")
    monkeypatch.setattr(feeds.NwisCurrentFeed, "_fetch", AsyncMock(return_value=frame))
    feed = _currents_feed()
    await feed.update({})
    pd.testing.assert_frame_equal(feed.values, frame)
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
    monkeypatch.setattr(capture, "GcsObjectStore", lambda bucket: store)
    monkeypatch.setattr(store, "read", AsyncMock(side_effect=OSError("offline")))
    frame = _currents_frame("2026-01-01 12:00")
    monkeypatch.setattr(feeds.NwisCurrentFeed, "_fetch", AsyncMock(return_value=frame))
    feed = _currents_feed()
    with caplog.at_level(logging.INFO):
        await feed.update({})
    assert feed._last_error is None
    assert feed._consecutive_failures == 0
    assert feed._next_fetch_after == feed._fetch_timestamp + feed.expiration_interval
    pd.testing.assert_frame_equal(feed.values, frame)
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
