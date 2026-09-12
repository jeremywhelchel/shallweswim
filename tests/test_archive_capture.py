"""Serving isolation and temperature capture through real feed updates."""

import datetime
import logging
from collections.abc import Iterator
from io import BytesIO
from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest

from shallweswim import config
from shallweswim.archive import capture
from shallweswim.archive.observations import read_observations
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
        await capture.capture_temperature(
            bucket,
            frame=_frame("2026-01-01"),
            source_identity=feed.feed_config.citation_key,
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
    frame = _frame("2026-11-01 01:00" if failure == "timezone" else "2026-01-01 12:00")
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
async def test_disabled_capture_does_not_construct_store(monkeypatch) -> None:
    monkeypatch.delenv("SHALLWESWIM_ARCHIVE_BUCKET", raising=False)
    archive = AsyncMock()
    monkeypatch.setattr(feeds, "capture_temperature", archive)
    monkeypatch.setattr(
        feeds.CoopsTempFeed, "_fetch", AsyncMock(return_value=_frame("2026-01-01"))
    )
    await _feed().update({})
    archive.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_fetch_does_not_archive(monkeypatch) -> None:
    monkeypatch.setenv("SHALLWESWIM_ARCHIVE_BUCKET", "test-archive")
    archive = AsyncMock()
    monkeypatch.setattr(feeds, "capture_temperature", archive)
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
    monkeypatch.setattr(feeds, "capture_temperature", archive)
    await history.update({})
    assert fetch.await_count == 2
    archive.assert_awaited_once()
    captured = archive.call_args.kwargs
    assert captured["frame"].index.year.tolist() == [2024]
    assert captured["retrieved_at"] == history._year_cache_fetch_timestamp[2024]
    assert history._year_cache_fetch_timestamp[2023] == old_retrieval
    assert history._last_error is not None
    assert history._consecutive_failures == 1


def test_source_paths_preserve_station_parameter_identity() -> None:
    live = _feed()
    partitions = capture._partitions(
        _frame("2026-01-01"),
        "nwis:temperature:08155500:00010",
        live.location_config.timezone,
        datetime.datetime(2026, 1, 2),
    )
    assert partitions[0][0] == "archive/temperature/nwis/08155500%3A00010/2026.parquet"


@pytest.mark.asyncio
async def test_prediction_feed_does_not_capture(monkeypatch) -> None:
    monkeypatch.setenv("SHALLWESWIM_ARCHIVE_BUCKET", "test-archive")
    archive = AsyncMock()
    monkeypatch.setattr(feeds, "capture_temperature", archive)
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
