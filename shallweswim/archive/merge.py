"""Conditional observation-archive merge writer."""

import asyncio
import logging
import time
from dataclasses import dataclass
from io import BytesIO

import pandas as pd

from shallweswim.archive.observations import normalize_archive_frame, read_observations
from shallweswim.archive.store import ObjectStore, VersionConflictError


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
        use_dictionary=["unit"],
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
    is identical keeps the stored row, including its original `retrieved_at`. A
    key whose stored value differs is decided by the newest `retrieved_at`, and
    an equally recent differing claim is an integrity conflict.
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

    matched = ~pd.isna(stored_value)
    differing = matched & (incoming["value"].to_numpy() != stored_value)
    conflicting = differing & (incoming["retrieved_at"].to_numpy() == stored_retrieved)
    if conflicting.any():
        raise ArchiveIntegrityError(source_identity, keys.iloc[conflicting.argmax()])

    new = ~matched
    revised = differing & (incoming["retrieved_at"].to_numpy() > stored_retrieved)
    new_count = int(new.sum())
    revised_count = int(revised.sum())
    # Overlapping means "did not change the partition": rows identical to the
    # stored row, and differing rows the stored row already supersedes because
    # it was retrieved more recently.
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
