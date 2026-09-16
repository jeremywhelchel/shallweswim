"""Serving index derivation from client frames, including UTC input."""

import datetime
from typing import Any
from unittest.mock import AsyncMock

import pandas as pd
import pytest
import pytz

from shallweswim import config
from shallweswim.archive.capture import CaptureResult, capture_observations
from shallweswim.archive.store import MEMORY_LOCATOR, memory_store
from shallweswim.core import feeds

EASTERN = pytz.timezone("US/Eastern")


def _location() -> config.LocationConfig:
    return next(item for item in config.get_all_configs() if item.code == "nyc")


def _temp_feed() -> feeds.CoopsTempFeed:
    return feeds.CoopsTempFeed(
        location_config=_location(),
        feed_config=config.CoopsTempFeedConfig(station=8518750),
        interval="h",
        expiration_interval=datetime.timedelta(minutes=10),
    )


def _utc_frame(start: str, end: str, freq: str, fold_boundary: str) -> pd.DataFrame:
    """Build a UTC frame whose values differ before and after an instant."""
    index = pd.date_range(start, end, freq=freq, tz="UTC", name="time")
    boundary = pd.Timestamp(fold_boundary, tz="UTC")
    return pd.DataFrame(
        {"water_temp": [60.0 if t < boundary else 70.0 for t in index]},
        index=index,
    )


def test_naive_frame_is_unchanged() -> None:
    """The transitional path leaves a naive client frame exactly as it is."""
    frame = pd.DataFrame(
        {"water_temp": [60.0, 61.0]},
        index=pd.DatetimeIndex(["2025-06-01 10:00", "2025-06-01 11:00"], name="time"),
    )

    result = feeds.to_serving_index(frame, EASTERN)

    assert result is frame
    pd.testing.assert_frame_equal(result, frame)


def test_fall_back_keeps_first_fold_and_drops_repeats() -> None:
    """Repeated fall-back wall times collapse to the daylight-time reading."""
    # 05:00-06:00 UTC is the first (daylight) 01:xx fold, 06:00-07:00 the second.
    frame = _utc_frame(
        "2025-11-02 04:00", "2025-11-02 08:00", "10min", "2025-11-02 06:00"
    )

    result = feeds.to_serving_index(frame, EASTERN)

    assert result.index.tz is None
    assert result.index.name == "time"
    assert result.index.is_unique
    assert result.index.is_monotonic_increasing
    # Six 10-minute wall times (01:00 through 01:50) repeat in the second fold.
    assert len(result) == len(frame) - 6
    repeated = result.loc["2025-11-02 01:00":"2025-11-02 01:50"]  # type: ignore[misc]
    assert len(repeated) == 6
    assert (repeated["water_temp"] == 60.0).all()


def test_spring_forward_has_no_missing_hour_rows_and_no_drops() -> None:
    """Spring-forward skips a wall hour without losing any observation."""
    frame = _utc_frame(
        "2025-03-09 06:00", "2025-03-09 09:00", "10min", "2025-03-09 07:00"
    )

    result = feeds.to_serving_index(frame, EASTERN)

    assert len(result) == len(frame)
    assert result.index.tz is None
    assert result.index.is_unique
    assert result.index.is_monotonic_increasing
    assert not any(stamp.hour == 2 for stamp in result.index)


def test_non_datetime_index_raises() -> None:
    """A frame without a DatetimeIndex is a bug in the client, not bad data."""
    frame = pd.DataFrame({"water_temp": [60.0]}, index=pd.Index(["not-a-time"]))

    with pytest.raises(ValueError, match="DatetimeIndex"):
        feeds.to_serving_index(frame, EASTERN)


@pytest.mark.asyncio
async def test_update_serves_naive_and_captures_utc(monkeypatch) -> None:
    """Serving derives naive local times while capture keeps the UTC frame."""
    monkeypatch.setenv("SHALLWESWIM_ARCHIVE_BUCKET", "test-archive")
    captured: list[pd.DataFrame] = []

    async def record(*args: Any, **kwargs: Any) -> CaptureResult:
        captured.append(kwargs["frame"])
        return CaptureResult(0, 0)

    monkeypatch.setattr(feeds, "capture_observations", record)
    frame = _utc_frame(
        "2025-11-02 04:00", "2025-11-02 08:00", "10min", "2025-11-02 06:00"
    )
    monkeypatch.setattr(feeds.CoopsTempFeed, "_fetch", AsyncMock(return_value=frame))

    feed = _temp_feed()
    await feed.update({})

    assert feed.values.index.tz is None
    assert feed.values.index.is_unique
    assert len(feed.values) == len(frame) - 6
    assert len(captured) == 1
    assert captured[0].index.tz is not None
    pd.testing.assert_frame_equal(captured[0], frame)


@pytest.mark.asyncio
async def test_historical_top_up_captures_utc_and_serves_naive(monkeypatch) -> None:
    """The top-up captures the UTC frame; the archive serves it back as local."""
    monkeypatch.setenv("SHALLWESWIM_ARCHIVE_BUCKET", MEMORY_LOCATOR)
    monkeypatch.setenv("SHALLWESWIM_ARCHIVE_READ_BUCKET", MEMORY_LOCATOR)
    memory_store.cache_clear()
    captured: list[pd.DataFrame] = []
    real_capture = capture_observations

    async def record(*args: Any, **kwargs: Any) -> CaptureResult:
        captured.append(kwargs["frame"])
        return await real_capture(*args, **kwargs)

    monkeypatch.setattr(feeds, "capture_observations", record)

    year = feeds.utc_now().year
    historical = feeds.HistoricalTempsFeed(
        location_config=_location(),
        feed_config=config.CoopsTempFeedConfig(station=8518750),
        start_year=year,
        end_year=year,
        expiration_interval=datetime.timedelta(hours=3),
    )
    frame = _utc_frame(
        f"{year}-06-02 04:00", f"{year}-06-02 08:00", "10min", f"{year}-06-02 06:00"
    )
    monkeypatch.setattr(feeds.CoopsTempFeed, "_fetch", AsyncMock(return_value=frame))

    try:
        result = await historical._fetch(clients={})
    finally:
        memory_store.cache_clear()

    assert result.index.tz is None
    assert result.index.is_unique
    assert result.index.is_monotonic_increasing
    assert result.index.name == "time"
    assert (result.index.minute == 0).all()

    # One capture, of the unconverted client frame; the served rows above came
    # back out of the archive rather than from this frame.
    assert len(captured) == 1
    assert captured[0].index.tz is not None
    pd.testing.assert_frame_equal(captured[0], frame)
