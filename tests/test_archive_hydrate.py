"""Local development hydration of historical temperature years from the archive."""

import datetime
import logging
from collections.abc import Iterator
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from shallweswim import config
from shallweswim.archive import capture, hydrate
from shallweswim.archive import store as archive_store
from shallweswim.archive.observations import (
    CURRENTS_MEASUREMENT,
    TEMPERATURE_MEASUREMENT,
    TEMPERATURE_UNIT,
    TEMPERATURE_VALUE_COLUMN,
    partition_key,
)
from shallweswim.archive.store import MemoryObjectStore
from shallweswim.core import feeds

READ_BUCKET = "test-read"
WRITE_BUCKET = "test-archive"
NYC_TEMPERATURE_KEY = "archive/temperature/coops/8518750/{year}.parquet"


@pytest.fixture(autouse=True)
def clear_store_cache() -> Iterator[None]:
    """Keep mocked stores from leaking between hydration tests."""
    archive_store.gcs_store.cache_clear()
    yield
    archive_store.gcs_store.cache_clear()


def _location() -> config.LocationConfig:
    return next(item for item in config.get_all_configs() if item.code == "nyc")


def _temp_config() -> config.CoopsTempFeedConfig:
    return config.CoopsTempFeedConfig(station=8518750)


def _history(start_year: int, end_year: int) -> feeds.HistoricalTempsFeed:
    return feeds.HistoricalTempsFeed(
        location_config=_location(),
        feed_config=_temp_config(),
        start_year=start_year,
        end_year=end_year,
        expiration_interval=datetime.timedelta(hours=3),
    )


def _frame(*times: str, value: float = 60.0) -> pd.DataFrame:
    """Build a client-style temperature frame indexed by UTC instants."""
    return pd.DataFrame(
        {TEMPERATURE_VALUE_COLUMN: [value] * len(times)},
        index=pd.DatetimeIndex(times, tz="UTC", name="time"),
    )


def _year_frame(year: int) -> pd.DataFrame:
    """Two readings in a UTC year, enough to distinguish it from its neighbors."""
    return _frame(f"{year}-06-01 12:00", f"{year}-06-01 13:00", value=float(year % 100))


def _ten_minute_fold_frame() -> pd.DataFrame:
    """Ten-minute UTC readings spanning both folds of US/Eastern's 2025 fall back."""
    return _frame(
        *(
            f"2025-11-02 {hour:02d}:{minute:02d}"
            for hour in range(4, 8)
            for minute in range(0, 60, 10)
        )
    )


async def _archive(
    store: MemoryObjectStore, frame: pd.DataFrame, bucket: str = WRITE_BUCKET
) -> None:
    """Write a client frame to the archive through the real capture hook."""
    await capture.capture_observations(
        bucket,
        frame=frame,
        source_identity=_temp_config().citation_key,
        measurement=TEMPERATURE_MEASUREMENT,
        value_column=TEMPERATURE_VALUE_COLUMN,
        unit=TEMPERATURE_UNIT,
        retrieved_at=datetime.datetime(2026, 1, 2, tzinfo=datetime.UTC),
    )
    key = NYC_TEMPERATURE_KEY.format(year=frame.index[0].year)
    assert await store.read(key) is not None


async def _hydrate(store: MemoryObjectStore, year: int) -> pd.DataFrame | None:
    return await hydrate.hydrate_year(
        store,
        source_identity=_temp_config().citation_key,
        measurement=TEMPERATURE_MEASUREMENT,
        value_column=TEMPERATURE_VALUE_COLUMN,
        unit=TEMPERATURE_UNIT,
        year=year,
        timezone=_location().timezone,
    )


def _in_nanoseconds(frame: pd.DataFrame) -> pd.DataFrame:
    """The same frame with its index at the archive's canonical resolution.

    The provider's parser produces microsecond instants and the archive stores
    nanosecond ones, so equality is compared on instants, not on index dtype.
    """
    return frame.set_axis(frame.index.as_unit("ns"), axis=0)


def _served(frame: pd.DataFrame, feed: feeds.HistoricalTempsFeed) -> pd.DataFrame:
    """The published frame a feed derives from one year's client frame."""
    return feed._combine_feeds(
        [feeds.to_serving_index(frame, feed.location_config.timezone)]
    )


def _hydrate_events(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if getattr(record, "operation", None) == "hydrate"
    ]


def test_partition_key_matches_the_keys_capture_writes() -> None:
    """Hydration reads exactly the keys capture wrote, colons included."""
    nwis_identity = "nwis:temperature:08155500:00010"
    written = capture._partitions(
        _frame("2026-01-01"),
        nwis_identity,
        TEMPERATURE_MEASUREMENT,
        TEMPERATURE_VALUE_COLUMN,
        TEMPERATURE_UNIT,
        datetime.datetime(2026, 1, 2),
    )
    assert written[0][0] == partition_key(nwis_identity, TEMPERATURE_MEASUREMENT, 2026)

    coops_identity = _temp_config().citation_key
    coops_written = capture._partitions(
        _frame("2026-01-01"),
        coops_identity,
        TEMPERATURE_MEASUREMENT,
        TEMPERATURE_VALUE_COLUMN,
        TEMPERATURE_UNIT,
        datetime.datetime(2026, 1, 2),
    )
    assert coops_written[0][0] == partition_key(
        coops_identity, TEMPERATURE_MEASUREMENT, 2026
    )
    assert coops_written[0][0] == NYC_TEMPERATURE_KEY.format(year=2026)


def test_partition_key_rejects_a_mismatched_measurement() -> None:
    with pytest.raises(ValueError, match="Expected a currents source identity"):
        partition_key(_temp_config().citation_key, CURRENTS_MEASUREMENT, 2026)


@pytest.mark.asyncio
async def test_hydrate_returns_none_for_a_missing_partition(monkeypatch) -> None:
    store = MemoryObjectStore()
    monkeypatch.setattr(archive_store, "GcsObjectStore", lambda bucket: store)
    assert await _hydrate(store, 2024) is None

    # The following partition alone is not this local year's data.
    await _archive(store, _year_frame(2025))
    assert await _hydrate(store, 2024) is None


@pytest.mark.asyncio
async def test_hydrate_returns_a_utc_frame_with_the_value_column(monkeypatch) -> None:
    store = MemoryObjectStore()
    monkeypatch.setattr(archive_store, "GcsObjectStore", lambda bucket: store)
    provider = _year_frame(2024)
    await _archive(store, provider)

    hydrated = await _hydrate(store, 2024)

    assert hydrated is not None
    assert list(hydrated.columns) == [TEMPERATURE_VALUE_COLUMN]
    assert hydrated.index.name == "time"
    assert str(hydrated.index.tz) == "UTC"
    assert hydrated.index.is_monotonic_increasing
    assert str(hydrated.index.dtype) == "datetime64[ns, UTC]"
    pd.testing.assert_frame_equal(hydrated, _in_nanoseconds(provider))


@pytest.mark.asyncio
async def test_hydrated_year_serves_identically_to_the_provider_year(
    monkeypatch,
) -> None:
    """A fixture year spanning both fall-back folds round-trips through capture."""
    store = MemoryObjectStore()
    monkeypatch.setattr(archive_store, "GcsObjectStore", lambda bucket: store)
    provider = _ten_minute_fold_frame()
    await _archive(store, provider)

    hydrated = await _hydrate(store, 2025)

    assert hydrated is not None
    assert len(hydrated) == len(provider)
    pd.testing.assert_frame_equal(hydrated, _in_nanoseconds(provider))
    feed = _history(2025, 2025)
    pd.testing.assert_frame_equal(
        _in_nanoseconds(_served(hydrated, feed)),
        _in_nanoseconds(_served(provider, feed)),
    )


def _straddling_frame(year: int) -> pd.DataFrame:
    """Readings just inside and just outside both edges of a US/Eastern year."""
    return pd.DataFrame(
        {TEMPERATURE_VALUE_COLUMN: [1.0, 2.0, 3.0, 4.0]},
        index=pd.DatetimeIndex(
            [
                # Local 18:00 and 22:00 on December 31 of the previous year.
                f"{year - 1}-12-31 23:00",
                f"{year}-01-01 03:00",
                # Local 18:00 and 22:00 on December 31 of this year.
                f"{year}-12-31 23:00",
                f"{year + 1}-01-01 03:00",
            ],
            tz="UTC",
            name="time",
        ),
    )


@pytest.mark.asyncio
async def test_hydrate_returns_exactly_the_station_local_year(monkeypatch) -> None:
    """A local year's edge hours live in the neighboring UTC partitions."""
    store = MemoryObjectStore()
    monkeypatch.setattr(archive_store, "GcsObjectStore", lambda bucket: store)
    provider = _straddling_frame(2025)
    await _archive(store, provider)

    hydrated = await _hydrate(store, 2025)

    assert hydrated is not None
    # The rows local to 2024 stay out, and 2026's partition supplies the local
    # year's final hours that a provider fetch of 2025 would have returned.
    pd.testing.assert_frame_equal(hydrated, _in_nanoseconds(provider.iloc[2:]))
    local = hydrated.index.tz_convert(_location().timezone)
    assert local.strftime("%Y-%m-%d %H:%M").tolist() == [
        "2025-12-31 18:00",
        "2025-12-31 22:00",
    ]


@pytest.mark.asyncio
async def test_feed_serves_the_final_local_hours_of_a_hydrated_year(
    monkeypatch,
) -> None:
    """The hydrated year's last local hours must not serve as a gap."""
    store = MemoryObjectStore()
    monkeypatch.setattr(archive_store, "GcsObjectStore", lambda bucket: store)
    monkeypatch.setenv("SHALLWESWIM_ARCHIVE_READ_BUCKET", READ_BUCKET)
    monkeypatch.delenv("SHALLWESWIM_ARCHIVE_BUCKET", raising=False)
    current_year = feeds.utc_now().year
    hydrated_year = current_year - 1
    await _archive(store, _straddling_frame(hydrated_year))

    monkeypatch.setattr(
        feeds.CoopsTempFeed, "_fetch", AsyncMock(return_value=_year_frame(current_year))
    )
    feed = _history(hydrated_year, current_year)
    await feed.update({})

    served = feed.values[TEMPERATURE_VALUE_COLUMN]
    assert served.loc[f"{hydrated_year}-12-31 18:00"] == 3.0
    assert served.loc[f"{hydrated_year}-12-31 22:00"] == 4.0
    # The previous local year's rows belong to that year, not this one.
    assert f"{hydrated_year - 1}-12-31 22:00" not in served.index


@pytest.mark.asyncio
async def test_feed_hydrates_past_years_and_fetches_only_the_current_year(
    monkeypatch, caplog
) -> None:
    store = MemoryObjectStore()
    monkeypatch.setattr(archive_store, "GcsObjectStore", lambda bucket: store)
    monkeypatch.setenv("SHALLWESWIM_ARCHIVE_READ_BUCKET", READ_BUCKET)
    monkeypatch.delenv("SHALLWESWIM_ARCHIVE_BUCKET", raising=False)
    current_year = feeds.utc_now().year
    past_years = (current_year - 2, current_year - 1)
    archived_rows = 0
    for year in past_years:
        frame = _year_frame(year)
        archived_rows += len(frame)
        await _archive(store, frame)

    current = _year_frame(current_year)
    fetch = AsyncMock(return_value=current)
    monkeypatch.setattr(feeds.CoopsTempFeed, "_fetch", fetch)
    feed = _history(past_years[0], current_year)
    with caplog.at_level(logging.INFO):
        await feed.update({})

    assert fetch.await_count == 1
    assert feed.last_fetched_years == (current_year,)
    assert feed.last_available_years == (*past_years, current_year)
    assert set(feed.values[TEMPERATURE_VALUE_COLUMN].dropna().unique()) == {
        float(year % 100) for year in (*past_years, current_year)
    }

    status = feed.status.historical_temp_status
    assert status is not None
    assert status.cached_years == [*past_years, current_year]
    assert status.available_years == [*past_years, current_year]
    assert status.missing_years == []
    assert status.fetched_years == [current_year]
    assert status.failed_years == {}

    events = _hydrate_events(caplog)
    assert len(events) == 1
    assert events[0].levelno == logging.INFO
    assert events[0].component == "archive"
    assert events[0].outcome == "success"
    assert events[0].record_count == archived_rows
    assert events[0].location == feed.location_config.code
    assert events[0].feed == feeds.FeedName.HISTORIC_TEMPS.value


@pytest.mark.asyncio
async def test_year_missing_from_the_archive_falls_back_to_the_provider(
    monkeypatch, caplog
) -> None:
    store = MemoryObjectStore()
    monkeypatch.setattr(archive_store, "GcsObjectStore", lambda bucket: store)
    monkeypatch.setenv("SHALLWESWIM_ARCHIVE_READ_BUCKET", READ_BUCKET)
    monkeypatch.delenv("SHALLWESWIM_ARCHIVE_BUCKET", raising=False)
    current_year = feeds.utc_now().year
    archived_year, missing_year = current_year - 2, current_year - 1
    await _archive(store, _year_frame(archived_year))

    fetch = AsyncMock(
        side_effect=[_year_frame(missing_year), _year_frame(current_year)]
    )
    monkeypatch.setattr(feeds.CoopsTempFeed, "_fetch", fetch)
    feed = _history(archived_year, current_year)
    with caplog.at_level(logging.INFO):
        await feed.update({})

    assert fetch.await_count == 2
    assert feed.last_fetched_years == (missing_year, current_year)
    assert feed.last_available_years == (archived_year, missing_year, current_year)
    events = _hydrate_events(caplog)
    assert len(events) == 1
    assert events[0].outcome == "success"
    assert events[0].record_count == len(_year_frame(archived_year))


@pytest.mark.asyncio
async def test_failed_archive_read_warns_and_falls_back_to_the_provider(
    monkeypatch, caplog
) -> None:
    store = MemoryObjectStore()
    monkeypatch.setattr(archive_store, "GcsObjectStore", lambda bucket: store)
    monkeypatch.setenv("SHALLWESWIM_ARCHIVE_READ_BUCKET", READ_BUCKET)
    monkeypatch.delenv("SHALLWESWIM_ARCHIVE_BUCKET", raising=False)
    current_year = feeds.utc_now().year
    broken_year, archived_year = current_year - 2, current_year - 1
    await _archive(store, _year_frame(broken_year))
    await _archive(store, _year_frame(archived_year))

    real_read = store.read

    async def flaky_read(key: str) -> archive_store.StoredObject | None:
        if key == NYC_TEMPERATURE_KEY.format(year=broken_year):
            raise OSError("offline")
        return await real_read(key)

    monkeypatch.setattr(store, "read", flaky_read)
    fetch = AsyncMock(side_effect=[_year_frame(broken_year), _year_frame(current_year)])
    monkeypatch.setattr(feeds.CoopsTempFeed, "_fetch", fetch)
    feed = _history(broken_year, current_year)
    with caplog.at_level(logging.INFO):
        await feed.update({})

    assert fetch.await_count == 2
    assert feed.last_fetched_years == (broken_year, current_year)
    assert feed.last_available_years == (broken_year, archived_year, current_year)
    warnings = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING and "hydration failed" in record.message
    ]
    assert len(warnings) == 1
    assert str(broken_year) in warnings[0].message
    events = _hydrate_events(caplog)
    assert len(events) == 1
    assert events[0].outcome == "failed"
    assert events[0].record_count == len(_year_frame(archived_year))


@pytest.mark.asyncio
async def test_hydrated_years_are_not_captured(monkeypatch) -> None:
    """Capture applies only to years the provider supplied this update."""
    read_store = MemoryObjectStore()
    write_store = MemoryObjectStore()
    stores = {READ_BUCKET: read_store, WRITE_BUCKET: write_store}
    monkeypatch.setattr(archive_store, "GcsObjectStore", lambda bucket: stores[bucket])
    monkeypatch.setenv("SHALLWESWIM_ARCHIVE_READ_BUCKET", READ_BUCKET)
    current_year = feeds.utc_now().year
    archived_year = current_year - 1
    await _archive(read_store, _year_frame(archived_year), bucket=READ_BUCKET)

    monkeypatch.setenv("SHALLWESWIM_ARCHIVE_BUCKET", WRITE_BUCKET)
    monkeypatch.setattr(
        feeds.CoopsTempFeed, "_fetch", AsyncMock(return_value=_year_frame(current_year))
    )
    feed = _history(archived_year, current_year)
    await feed.update({})

    # Only the provider-fetched current year reached the write bucket.
    assert (
        await write_store.read(NYC_TEMPERATURE_KEY.format(year=current_year))
    ) is not None
    assert (
        await write_store.read(NYC_TEMPERATURE_KEY.format(year=archived_year))
    ) is None
    assert feed.last_capture == capture.CaptureResult(len(_year_frame(current_year)), 0)


@pytest.mark.asyncio
async def test_unset_read_bucket_fetches_every_year(monkeypatch, caplog) -> None:
    store = MemoryObjectStore()
    monkeypatch.setattr(archive_store, "GcsObjectStore", lambda bucket: store)
    monkeypatch.delenv("SHALLWESWIM_ARCHIVE_READ_BUCKET", raising=False)
    monkeypatch.delenv("SHALLWESWIM_ARCHIVE_BUCKET", raising=False)
    current_year = feeds.utc_now().year
    archived_year = current_year - 1
    await _archive(store, _year_frame(archived_year))

    fetch = AsyncMock(
        side_effect=[_year_frame(archived_year), _year_frame(current_year)]
    )
    monkeypatch.setattr(feeds.CoopsTempFeed, "_fetch", fetch)
    feed = _history(archived_year, current_year)
    with caplog.at_level(logging.INFO):
        await feed.update({})

    assert fetch.await_count == 2
    assert feed.last_fetched_years == (archived_year, current_year)
    assert _hydrate_events(caplog) == []
