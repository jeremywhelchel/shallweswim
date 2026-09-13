"""Observation archive merge semantics and event tests."""

import datetime
import logging
from io import BytesIO

import pandas as pd
import pytest
import pytz

from shallweswim.archive import merge as merge_module
from shallweswim.archive.merge import ArchiveIntegrityError, merge_observations
from shallweswim.archive.observations import (
    TEMPERATURE_UNIT,
    TEMPERATURE_VALUE_COLUMN,
    normalize_observations,
    read_observations,
)
from shallweswim.archive.store import (
    MemoryObjectStore,
    StoredObject,
    VersionConflictError,
)

SOURCE = "coops:temperature:8518750"
KEY = "archive/temperature/coops/8518750/2026.parquet"
EASTERN = pytz.timezone("US/Eastern")


def _rows(values: dict[str, float], retrieved_at: datetime.datetime) -> pd.DataFrame:
    frame = pd.DataFrame(
        {TEMPERATURE_VALUE_COLUMN: list(values.values())},
        index=pd.DatetimeIndex(values, name="time"),
    )
    return normalize_observations(
        frame,
        value_column=TEMPERATURE_VALUE_COLUMN,
        unit=TEMPERATURE_UNIT,
        timezone=EASTERN,
        retrieved_at=retrieved_at,
    ).frame


async def _read_frame(store: MemoryObjectStore) -> pd.DataFrame:
    stored = await store.read(KEY)
    assert stored is not None
    return read_observations(BytesIO(stored.data), expected_unit=TEMPERATURE_UNIT)


@pytest.mark.asyncio
async def test_merge_inserts_and_revises_by_newest_retrieval() -> None:
    store = MemoryObjectStore()
    await merge_observations(
        store,
        key=KEY,
        source_identity=SOURCE,
        incoming=_rows(
            {"2026-01-01 12:00": 50.0}, datetime.datetime(2026, 1, 1, 18, 0)
        ),
        expected_unit=TEMPERATURE_UNIT,
    )
    await merge_observations(
        store,
        key=KEY,
        source_identity=SOURCE,
        incoming=_rows(
            {"2026-01-01 12:00": 51.0}, datetime.datetime(2026, 1, 1, 19, 0)
        ),
        expected_unit=TEMPERATURE_UNIT,
    )

    frame = await _read_frame(store)
    assert frame["value"].tolist() == [51.0]


@pytest.mark.asyncio
async def test_fully_identical_duplicate_collapses_without_write() -> None:
    store = MemoryObjectStore()
    rows = _rows({"2026-01-01 12:00": 50.0}, datetime.datetime(2026, 1, 1, 18, 0))
    await merge_observations(
        store,
        key=KEY,
        source_identity=SOURCE,
        incoming=rows,
        expected_unit=TEMPERATURE_UNIT,
    )
    result = await merge_observations(
        store,
        key=KEY,
        source_identity=SOURCE,
        incoming=rows,
        expected_unit=TEMPERATURE_UNIT,
    )

    assert result.outcome == "unchanged"
    assert result.record_count == 1


@pytest.mark.asyncio
async def test_equal_retrieval_conflict_fails_then_newer_fetch_recovers(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = MemoryObjectStore()
    retrieved_at = datetime.datetime(2026, 1, 1, 18, 0)
    await merge_observations(
        store,
        key=KEY,
        source_identity=SOURCE,
        incoming=_rows({"2026-01-01 12:00": 50.0}, retrieved_at),
        expected_unit=TEMPERATURE_UNIT,
    )

    with (
        caplog.at_level(logging.WARNING),
        pytest.raises(
            ArchiveIntegrityError,
            match=r"source=coops:temperature:8518750 observed_at=2026-01-01T17:00:00\+00:00",
        ),
    ):
        await merge_observations(
            store,
            key=KEY,
            source_identity=SOURCE,
            incoming=_rows({"2026-01-01 12:00": 52.0}, retrieved_at),
            expected_unit=TEMPERATURE_UNIT,
        )

    failure = caplog.records[-1]
    assert failure.component == "archive"
    assert failure.operation == "merge"
    assert failure.outcome == "failed"
    assert failure.levelno == logging.WARNING
    assert failure.source_identity == SOURCE
    assert failure.observed_at == "2026-01-01T17:00:00+00:00"

    await merge_observations(
        store,
        key=KEY,
        source_identity=SOURCE,
        incoming=_rows(
            {"2026-01-01 12:00": 52.0}, datetime.datetime(2026, 1, 1, 19, 0)
        ),
        expected_unit=TEMPERATURE_UNIT,
    )
    assert (await _read_frame(store))["value"].tolist() == [52.0]


class _ConflictOnceStore(MemoryObjectStore):
    def __init__(self) -> None:
        super().__init__()
        self.conflicted = False

    async def compare_and_swap(
        self,
        key: str,
        *,
        expected_version: str | None,
        data: bytes,
    ) -> str:
        if not self.conflicted:
            self.conflicted = True
            raise VersionConflictError(key)
        return await super().compare_and_swap(
            key, expected_version=expected_version, data=data
        )


@pytest.mark.asyncio
async def test_merge_retries_version_conflict() -> None:
    store = _ConflictOnceStore()

    result = await merge_observations(
        store,
        key=KEY,
        source_identity=SOURCE,
        incoming=_rows(
            {"2026-01-01 12:00": 50.0}, datetime.datetime(2026, 1, 1, 18, 0)
        ),
        expected_unit=TEMPERATURE_UNIT,
        retry_base_seconds=0,
    )

    assert result.outcome == "success"
    assert result.attempt_count == 2


@pytest.mark.asyncio
async def test_merge_offloads_dataframe_and_parquet_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    async def recording_to_thread(function, *args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(function.__name__)
        return function(*args, **kwargs)

    monkeypatch.setattr(merge_module.asyncio, "to_thread", recording_to_thread)

    await merge_observations(
        MemoryObjectStore(),
        key=KEY,
        source_identity=SOURCE,
        incoming=_rows(
            {"2026-01-01 12:00": 50.0}, datetime.datetime(2026, 1, 1, 18, 0)
        ),
        expected_unit=TEMPERATURE_UNIT,
    )

    assert calls == ["normalize_archive_frame", "_prepare_merge"]


class _AlwaysConflictStore:
    async def read(self, key: str) -> StoredObject | None:
        return None

    async def compare_and_swap(
        self,
        key: str,
        *,
        expected_version: str | None,
        data: bytes,
    ) -> str:
        raise VersionConflictError(key)


@pytest.mark.asyncio
async def test_merge_emits_failed_event_after_cas_exhaustion(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.ERROR), pytest.raises(VersionConflictError):
        await merge_observations(
            _AlwaysConflictStore(),
            key=KEY,
            source_identity=SOURCE,
            incoming=_rows(
                {"2026-01-01 12:00": 50.0},
                datetime.datetime(2026, 1, 1, 18, 0),
            ),
            expected_unit=TEMPERATURE_UNIT,
            max_attempts=2,
            retry_base_seconds=0,
        )

    failure = caplog.records[-1]
    assert failure.operation == "merge"
    assert failure.outcome == "failed"
    assert failure.attempt_count == 2
