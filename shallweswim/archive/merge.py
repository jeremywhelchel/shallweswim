"""Conditional observation-archive merge writer."""

import asyncio
import logging
import time
from dataclasses import dataclass
from io import BytesIO

import pandas as pd

from shallweswim.archive.observations import (
    COOPS_HOURLY_PRODUCT,
    COOPS_SIX_MINUTE_PRODUCT,
    CSPF_PRODUCT,
    IRISH_LIGHTS_PRODUCT,
    NDBC_FILES_PRODUCT,
    NDBC_REALTIME_PRODUCT,
    NWIS_PRODUCT,
    PRODUCT_COLUMN,
    normalize_archive_frame,
    read_observations,
)
from shallweswim.archive.store import ObjectStore, VersionConflictError

# How authoritative each provider product is about an instant it reports. A
# merge only ever compares rows of one source identity, so ranks are compared
# only within one provider and equal numbers across providers never meet.
#
# NDBC's monthly and yearly files are quality controlled after the fact, so they
# supersede the realtime file the same station's rows may have come from first.
# CO-OPS's hourly product is the on-the-hour sample of its six-minute product,
# so neither supersedes the other. Every other provider publishes one product,
# which is trivially its own rank.
PRODUCT_RANKS: dict[str, int] = {
    NDBC_FILES_PRODUCT: 2,
    NDBC_REALTIME_PRODUCT: 1,
    COOPS_HOURLY_PRODUCT: 1,
    COOPS_SIX_MINUTE_PRODUCT: 1,
    NWIS_PRODUCT: 1,
    CSPF_PRODUCT: 1,
    IRISH_LIGHTS_PRODUCT: 1,
}

# A row whose product the table does not name, including one archived before the
# column existed, ranks below every named product, so one capture with a known
# product supersedes it.
UNRANKED_PRODUCT = 0


def _product_ranks(products: pd.Series) -> pd.Series:
    """Return each row's product rank, unranked where the product is unknown."""
    return products.map(PRODUCT_RANKS).fillna(UNRANKED_PRODUCT).astype("int64")


class ArchiveIntegrityError(Exception):
    """Equally authoritative rows make incompatible claims."""

    def __init__(self, source_identity: str, observed_at: pd.Timestamp) -> None:
        self.source_identity = source_identity
        self.observed_at = observed_at
        super().__init__(
            "Conflicting archive claims for "
            f"source={source_identity} observed_at={observed_at.isoformat()}"
        )


@dataclass(frozen=True)
class MergeResult:
    """Outcome and row counts of one partition merge.

    `record_count` is the partition size after the merge. The other three counts
    classify the incoming rows and always sum to `incoming_count`: `new_count`
    for keys absent from the partition, `revised_count` for keys whose stored
    value this fetch replaced, and `overlap_count` for every incoming row that
    left the partition unchanged.
    """

    outcome: str
    record_count: int
    attempt_count: int
    incoming_count: int
    new_count: int
    overlap_count: int
    revised_count: int


@dataclass(frozen=True)
class _PreparedMerge:
    record_count: int
    new_count: int
    overlap_count: int
    revised_count: int
    parquet_data: bytes | None


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    output = BytesIO()
    frame.to_parquet(
        output,
        index=False,
        engine="pyarrow",
        use_dictionary=["unit", PRODUCT_COLUMN],
    )
    return output.getvalue()


def _prepare_merge(
    stored_data: bytes | None,
    incoming: pd.DataFrame,
    *,
    source_identity: str,
    expected_unit: str,
) -> _PreparedMerge:
    """Classify incoming rows against the partition and write only if it changes.

    The deduplication key is `observed_at`; source identity is fixed per
    partition. A key absent from the partition is new. A key whose stored value
    is identical keeps the stored row, including its original `retrieved_at` and
    product. A key whose stored value differs is decided first by product rank,
    so a higher-ranked product replaces a lower-ranked one whichever was fetched
    first; within one rank the newest `retrieved_at` wins, which is a provider
    revising its own reading; an equally recent differing claim of the same rank
    is an integrity conflict.
    """
    if stored_data is None:
        current = incoming.iloc[0:0].copy()
    else:
        current = read_observations(BytesIO(stored_data), expected_unit=expected_unit)

    keys = incoming["observed_at"]
    # Merges only ever write unique keys, so a duplicated stored key is a defect
    # and reindex fails fast rather than silently multiplying rows.
    stored = current.set_index("observed_at")
    stored_value = stored["value"].reindex(keys).to_numpy()
    stored_retrieved = stored["retrieved_at"].reindex(keys).to_numpy()

    incoming_rank = _product_ranks(incoming[PRODUCT_COLUMN]).to_numpy()
    stored_rank = _product_ranks(
        stored[PRODUCT_COLUMN].reindex(keys).astype("string")
    ).to_numpy()

    matched = ~pd.isna(stored_value)
    differing = matched & (incoming["value"].to_numpy() != stored_value)
    equally_ranked = differing & (incoming_rank == stored_rank)
    conflicting = equally_ranked & (
        incoming["retrieved_at"].to_numpy() == stored_retrieved
    )
    if conflicting.any():
        raise ArchiveIntegrityError(source_identity, keys.iloc[conflicting.argmax()])

    new = ~matched
    # A higher-ranked product supersedes whatever is stored whenever it was
    # fetched; within one rank, retrieval order decides.
    revised = (differing & (incoming_rank > stored_rank)) | (
        equally_ranked & (incoming["retrieved_at"].to_numpy() > stored_retrieved)
    )
    new_count = int(new.sum())
    revised_count = int(revised.sum())
    # Overlapping means "did not change the partition": rows identical to the
    # stored row, and differing rows the stored row already supersedes because
    # its product ranks higher, or ranks equal and it was retrieved later.
    overlap_count = len(incoming) - new_count - revised_count

    replacements = new | revised
    if not replacements.any():
        return _PreparedMerge(len(current), 0, overlap_count, 0, None)

    kept = current.loc[~current["observed_at"].isin(keys.loc[revised])]
    merged = (
        pd.concat([kept, incoming.loc[replacements]], ignore_index=True)
        .sort_values("observed_at", kind="stable")
        .reset_index(drop=True)
    )
    return _PreparedMerge(
        len(merged),
        new_count,
        overlap_count,
        revised_count,
        _parquet_bytes(merged),
    )


def _event_fields(
    result: MergeResult,
    *,
    source_identity: str,
    started_at: float,
    observed_at: str | None = None,
) -> dict[str, object]:
    fields: dict[str, object] = {
        "component": "archive",
        "operation": "merge",
        "source_identity": source_identity,
        "outcome": result.outcome,
        "duration_ms": max(0, round((time.monotonic() - started_at) * 1000)),
        "record_count": result.record_count,
        "attempt_count": result.attempt_count,
        "incoming_count": result.incoming_count,
        "new_count": result.new_count,
        "overlap_count": result.overlap_count,
        "revised_count": result.revised_count,
    }
    if observed_at is not None:
        fields["observed_at"] = observed_at
    return fields


async def merge_observations(
    store: ObjectStore,
    *,
    key: str,
    source_identity: str,
    incoming: pd.DataFrame,
    expected_unit: str,
    max_attempts: int = 5,
    retry_base_seconds: float = 0.01,
) -> MergeResult:
    """Merge one source/year partition using bounded CAS retries."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least one")
    started_at = time.monotonic()
    attempt_count = 0
    record_count = 0
    incoming_count = 0
    new_count = 0
    overlap_count = 0
    revised_count = 0
    try:
        validated_incoming = await asyncio.to_thread(
            normalize_archive_frame, incoming, expected_unit=expected_unit
        )
        incoming_count = len(validated_incoming)
        if validated_incoming["observed_at"].duplicated().any():
            # Normalization collapses repeated instants, so a duplicate key here
            # is a caller defect. Rejecting it keeps stored partitions unique.
            raise ValueError("incoming observations must have unique observed_at")
        if validated_incoming.empty:
            result = MergeResult("unchanged", 0, 0, 0, 0, 0, 0)
            logging.info(
                "Archive merge had no observations for %s",
                source_identity,
                extra=_event_fields(
                    result, source_identity=source_identity, started_at=started_at
                ),
            )
            return result

        for attempt_count in range(1, max_attempts + 1):
            stored = await store.read(key)
            expected_version = None if stored is None else stored.version

            prepared = await asyncio.to_thread(
                _prepare_merge,
                None if stored is None else stored.data,
                validated_incoming,
                source_identity=source_identity,
                expected_unit=expected_unit,
            )
            record_count = prepared.record_count
            new_count = prepared.new_count
            overlap_count = prepared.overlap_count
            revised_count = prepared.revised_count
            if prepared.parquet_data is None:
                # Nothing new and nothing revised, so the stored bytes already
                # express this fetch and rewriting them would only churn.
                result = MergeResult(
                    "unchanged",
                    record_count,
                    attempt_count,
                    incoming_count,
                    new_count,
                    overlap_count,
                    revised_count,
                )
                logging.info(
                    "Archive merge was unchanged for %s",
                    source_identity,
                    extra=_event_fields(
                        result, source_identity=source_identity, started_at=started_at
                    ),
                )
                return result

            try:
                await store.compare_and_swap(
                    key,
                    expected_version=expected_version,
                    data=prepared.parquet_data,
                )
            except VersionConflictError:
                if attempt_count == max_attempts:
                    raise
                await asyncio.sleep(retry_base_seconds * 2 ** (attempt_count - 1))
                continue

            result = MergeResult(
                "success",
                record_count,
                attempt_count,
                incoming_count,
                new_count,
                overlap_count,
                revised_count,
            )
            logging.info(
                "Archive merge succeeded for %s",
                source_identity,
                extra=_event_fields(
                    result, source_identity=source_identity, started_at=started_at
                ),
            )
            return result

        raise RuntimeError("Archive merge retry loop ended unexpectedly")
    except Exception as error:
        observed_at: str | None = None
        if isinstance(error, ArchiveIntegrityError):
            observed_at = error.observed_at.isoformat()
        log = (
            logging.warning
            if isinstance(error, ArchiveIntegrityError)
            else logging.error
        )
        log(
            "Archive merge failed for %s: %s",
            source_identity,
            error,
            extra=_event_fields(
                MergeResult(
                    "failed",
                    record_count,
                    attempt_count,
                    incoming_count,
                    new_count,
                    overlap_count,
                    revised_count,
                ),
                source_identity=source_identity,
                started_at=started_at,
                observed_at=observed_at,
            ),
        )
        raise
