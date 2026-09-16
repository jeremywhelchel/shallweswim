"""Observation archive merge semantics and event tests."""

import datetime
import logging
from io import BytesIO

import pandas as pd
import pytest

from shallweswim.archive import merge as merge_module
from shallweswim.archive.merge import ArchiveIntegrityError, merge_observations
from shallweswim.archive.observations import (
    COOPS_HOURLY_PRODUCT,
    COOPS_SIX_MINUTE_PRODUCT,
    NDBC_FILES_PRODUCT,
    NDBC_REALTIME_PRODUCT,
    PRODUCT_COLUMN,
    TEMPERATURE_UNIT,
    TEMPERATURE_VALUE_COLUMN,
    normalize_archive_frame,
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


def _rows(
    values: dict[str, float],
    retrieved_at: datetime.datetime,
    product: str | None = COOPS_HOURLY_PRODUCT,
) -> pd.DataFrame:
    """Normalize a client-style UTC-indexed frame into archive rows."""
    frame = pd.DataFrame(
        {TEMPERATURE_VALUE_COLUMN: list(values.values())},
        index=pd.DatetimeIndex(values, tz="UTC", name="time"),
    )
    return normalize_observations(
        frame,
        value_column=TEMPERATURE_VALUE_COLUMN,
        unit=TEMPERATURE_UNIT,
        retrieved_at=retrieved_at,
        product=product,
    ).frame


async def _read_frame(store: MemoryObjectStore) -> pd.DataFrame:
    stored = await store.read(KEY)
    assert stored is not None
    return read_observations(BytesIO(stored.data), expected_unit=TEMPERATURE_UNIT)


async def _merge(
    store: MemoryObjectStore,
    values: dict[str, float],
    retrieved_hour: int,
    product: str | None = COOPS_HOURLY_PRODUCT,
) -> merge_module.MergeResult:
    return await merge_observations(
        store,
        key=KEY,
        source_identity=SOURCE,
        incoming=_rows(
            values, datetime.datetime(2026, 1, 1, retrieved_hour, 0), product
        ),
        expected_unit=TEMPERATURE_UNIT,
    )


def _merge_event(caplog: pytest.LogCaptureFixture) -> logging.LogRecord:
    records = [
        record
        for record in caplog.records
        if getattr(record, "operation", None) == "merge"
    ]
    assert len(records) == 1
    return records[0]


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
async def test_overlapping_merge_leaves_the_object_bytes_untouched() -> None:
    """A later fetch of the same value keeps the original stored row verbatim."""
    store = MemoryObjectStore()
    await _merge(store, {"2026-01-01 12:00": 50.0}, 18)
    before = await store.read(KEY)
    assert before is not None

    result = await _merge(store, {"2026-01-01 12:00": 50.0}, 19)

    assert result.outcome == "unchanged"
    assert result.record_count == 1
    after = await store.read(KEY)
    assert after is not None
    assert after.version == before.version
    assert after.data == before.data
    frame = await _read_frame(store)
    assert frame["retrieved_at"].tolist() == [
        pd.Timestamp("2026-01-01 18:00", tz="UTC")
    ]


# Stored baseline for the count matrix: two hourly readings from one fetch.
STORED_VALUES = {"2026-01-01 12:00": 50.0, "2026-01-01 13:00": 60.0}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("values", "retrieved_hour", "outcome", "counts", "record_count"),
    [
        # incoming, new, overlap, revised
        ({"2026-01-01 14:00": 70.0}, 19, "success", (1, 1, 0, 0), 3),
        ({"2026-01-01 12:00": 50.0}, 19, "unchanged", (1, 0, 1, 0), 2),
        ({"2026-01-01 12:00": 51.0}, 19, "success", (1, 0, 0, 1), 2),
        # A differing value the stored row already supersedes changes nothing,
        # so it is overlapping: overlap counts incoming rows that left the
        # partition alone, not only rows identical to a stored row.
        ({"2026-01-01 12:00": 51.0}, 17, "unchanged", (1, 0, 1, 0), 2),
        (
            {
                "2026-01-01 12:00": 50.0,
                "2026-01-01 13:00": 61.0,
                "2026-01-01 14:00": 70.0,
            },
            19,
            "success",
            (3, 1, 1, 1),
            3,
        ),
    ],
    ids=["new", "overlap", "revised", "superseded", "mixed"],
)
async def test_merge_counts_classify_every_incoming_row(
    caplog: pytest.LogCaptureFixture,
    values: dict[str, float],
    retrieved_hour: int,
    outcome: str,
    counts: tuple[int, int, int, int],
    record_count: int,
) -> None:
    store = MemoryObjectStore()
    await _merge(store, STORED_VALUES, 18)

    with caplog.at_level(logging.INFO):
        result = await _merge(store, values, retrieved_hour)

    incoming_count, new_count, overlap_count, revised_count = counts
    assert result.outcome == outcome
    assert result.record_count == record_count
    assert result.incoming_count == incoming_count
    assert result.new_count == new_count
    assert result.overlap_count == overlap_count
    assert result.revised_count == revised_count
    assert result.incoming_count == (
        result.new_count + result.overlap_count + result.revised_count
    )

    event = _merge_event(caplog)
    assert event.component == "archive"
    assert event.outcome == outcome
    assert event.record_count == record_count
    assert event.incoming_count == incoming_count
    assert event.new_count == new_count
    assert event.overlap_count == overlap_count
    assert event.revised_count == revised_count


@pytest.mark.asyncio
async def test_merge_without_observations_reports_zero_counts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = MemoryObjectStore()

    with caplog.at_level(logging.INFO):
        result = await _merge(store, {}, 18)

    assert result.outcome == "unchanged"
    assert result.record_count == 0
    event = _merge_event(caplog)
    assert event.incoming_count == 0
    assert event.new_count == 0
    assert event.overlap_count == 0
    assert event.revised_count == 0
    assert await store.read(KEY) is None


@pytest.mark.asyncio
async def test_duplicate_incoming_key_fails_without_writing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Normalization collapses repeated instants, so a duplicate is a defect."""
    store = MemoryObjectStore()
    rows = _rows({"2026-01-01 12:00": 50.0}, datetime.datetime(2026, 1, 1, 18, 0))
    duplicated = pd.concat([rows, rows], ignore_index=True)

    with (
        caplog.at_level(logging.ERROR),
        pytest.raises(ValueError, match="unique observed_at"),
    ):
        await merge_observations(
            store,
            key=KEY,
            source_identity=SOURCE,
            incoming=duplicated,
            expected_unit=TEMPERATURE_UNIT,
        )

    assert await store.read(KEY) is None
    failure = _merge_event(caplog)
    assert failure.outcome == "failed"
    assert failure.levelno == logging.ERROR
    assert failure.incoming_count == 2
    assert failure.record_count == 0
    assert failure.attempt_count == 0


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
            match=r"source=coops:temperature:8518750 observed_at=2026-01-01T12:00:00\+00:00",
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
    assert failure.observed_at == "2026-01-01T12:00:00+00:00"
    # A conflict aborts classification, so only the validated incoming count is
    # known and the unclassified counts stay zero.
    assert failure.incoming_count == 1
    assert failure.new_count == 0
    assert failure.overlap_count == 0
    assert failure.revised_count == 0

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
    # Classification succeeded on every attempt; only the write failed.
    assert failure.record_count == 1
    assert failure.incoming_count == 1
    assert failure.new_count == 1
    assert failure.overlap_count == 0
    assert failure.revised_count == 0


# =============================================================================
# Product ranks
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stored_product", "stored_hour", "incoming_product", "incoming_hour", "value"),
    [
        # A higher-ranked product replaces a lower-ranked one whichever was
        # fetched first, so the files product wins both ways round.
        (NDBC_REALTIME_PRODUCT, 18, NDBC_FILES_PRODUCT, 19, 51.0),
        (NDBC_REALTIME_PRODUCT, 19, NDBC_FILES_PRODUCT, 18, 51.0),
        # An unknown product, including a row archived before the column
        # existed, ranks below every named product.
        (None, 19, COOPS_HOURLY_PRODUCT, 18, 51.0),
    ],
    ids=["higher-rank-later", "higher-rank-earlier", "unknown-loses"],
)
async def test_higher_ranked_product_replaces_lower_whichever_was_fetched_first(
    stored_product: str | None,
    stored_hour: int,
    incoming_product: str,
    incoming_hour: int,
    value: float,
) -> None:
    store = MemoryObjectStore()
    await _merge(store, {"2026-01-01 12:00": 50.0}, stored_hour, stored_product)

    result = await _merge(
        store, {"2026-01-01 12:00": value}, incoming_hour, incoming_product
    )

    assert result.revised_count == 1
    frame = await _read_frame(store)
    assert frame["value"].tolist() == [value]
    assert frame[PRODUCT_COLUMN].tolist() == [incoming_product]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("incoming_product", "incoming_hour"),
    [
        # Ranked below the stored row, so the stored row already supersedes it.
        (NDBC_REALTIME_PRODUCT, 19),
        # Same rank, retrieved earlier.
        (NDBC_FILES_PRODUCT, 17),
    ],
    ids=["lower-rank", "same-rank-earlier"],
)
async def test_a_row_the_partition_supersedes_overlaps(
    incoming_product: str, incoming_hour: int
) -> None:
    store = MemoryObjectStore()
    await _merge(store, {"2026-01-01 12:00": 50.0}, 18, NDBC_FILES_PRODUCT)
    before = await store.read(KEY)
    assert before is not None

    result = await _merge(
        store, {"2026-01-01 12:00": 51.0}, incoming_hour, incoming_product
    )

    assert result.outcome == "unchanged"
    assert result.overlap_count == 1
    assert result.revised_count == 0
    after = await store.read(KEY)
    assert after is not None
    assert after.data == before.data


@pytest.mark.asyncio
async def test_equal_rank_falls_back_to_retrieval_time() -> None:
    """CO-OPS's two products rank equally, so their retrieval order decides."""
    store = MemoryObjectStore()
    await _merge(store, {"2026-01-01 12:00": 50.0}, 18, COOPS_SIX_MINUTE_PRODUCT)

    result = await _merge(store, {"2026-01-01 12:00": 51.0}, 19, COOPS_HOURLY_PRODUCT)

    assert result.revised_count == 1
    frame = await _read_frame(store)
    assert frame["value"].tolist() == [51.0]
    assert frame[PRODUCT_COLUMN].tolist() == [COOPS_HOURLY_PRODUCT]


@pytest.mark.asyncio
async def test_equal_values_overlap_whatever_the_products() -> None:
    """Agreement is agreement: an equal value never revises, or gains a product."""
    store = MemoryObjectStore()
    await _merge(store, {"2026-01-01 12:00": 50.0}, 19, None)

    result = await _merge(store, {"2026-01-01 12:00": 50.0}, 18, NDBC_FILES_PRODUCT)

    assert result.outcome == "unchanged"
    assert result.overlap_count == 1
    frame = await _read_frame(store)
    assert frame[PRODUCT_COLUMN].isna().all()
    assert frame["retrieved_at"].tolist() == [
        pd.Timestamp("2026-01-01 19:00", tz="UTC")
    ]


@pytest.mark.asyncio
async def test_integrity_error_fires_only_for_one_rank_at_one_retrieval_time() -> None:
    """Differing ranks decide; only an equally authoritative pair conflicts."""
    store = MemoryObjectStore()
    await _merge(store, {"2026-01-01 12:00": 50.0}, 18, NDBC_REALTIME_PRODUCT)

    # Same instant, same retrieval time, higher rank: ranked, not a conflict.
    result = await _merge(store, {"2026-01-01 12:00": 51.0}, 18, NDBC_FILES_PRODUCT)
    assert result.revised_count == 1

    with pytest.raises(ArchiveIntegrityError):
        await _merge(store, {"2026-01-01 12:00": 52.0}, 18, NDBC_FILES_PRODUCT)


@pytest.mark.asyncio
async def test_an_object_written_without_the_column_merges_as_null_products() -> None:
    """Older objects read back with no product and rank below every product."""
    store = MemoryObjectStore()
    legacy = _rows(
        {"2026-01-01 12:00": 50.0, "2026-01-01 13:00": 60.0},
        datetime.datetime(2026, 1, 1, 19, 0),
    ).drop(columns=[PRODUCT_COLUMN])
    parquet = BytesIO()
    legacy.to_parquet(parquet, index=False)
    await store.compare_and_swap(KEY, expected_version=None, data=parquet.getvalue())

    stored_frame = await _read_frame(store)
    assert stored_frame[PRODUCT_COLUMN].isna().all()

    result = await _merge(
        store,
        {"2026-01-01 12:00": 51.0, "2026-01-01 13:00": 60.0},
        18,
        COOPS_HOURLY_PRODUCT,
    )

    # The differing row is superseded by the known product despite its earlier
    # retrieval; the agreeing row overlaps and keeps its null product.
    assert (result.new_count, result.overlap_count, result.revised_count) == (0, 1, 1)
    merged = await _read_frame(store)
    assert merged["value"].tolist() == [51.0, 60.0]
    assert merged[PRODUCT_COLUMN].tolist() == [COOPS_HOURLY_PRODUCT, pd.NA]


def test_every_configured_product_is_ranked() -> None:
    """A product no rank names would silently rank below every other one."""
    ranked = set(merge_module.PRODUCT_RANKS)
    assert ranked == {
        "coops:h",
        "coops:6-min",
        "ndbc:files",
        "ndbc:realtime",
        "nwis",
        "cspf",
        "irish-lights",
    }
    assert min(merge_module.PRODUCT_RANKS.values()) > merge_module.UNRANKED_PRODUCT
    # An empty frame still carries the column the ranks read.
    assert PRODUCT_COLUMN in normalize_archive_frame(
        _rows({}, datetime.datetime(2026, 1, 1, 18, 0)), expected_unit=TEMPERATURE_UNIT
    )
