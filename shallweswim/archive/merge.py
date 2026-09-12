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
    """Outcome of one partition merge."""

    outcome: str
    record_count: int
    attempt_count: int


@dataclass(frozen=True)
class _PreparedMerge:
    record_count: int
    unchanged: bool
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


def _merge_frames(
    current: pd.DataFrame,
    incoming: pd.DataFrame,
    *,
    source_identity: str,
) -> pd.DataFrame:
    combined = pd.concat([current, incoming], ignore_index=True)
    distinct_claims = combined.drop_duplicates()
    conflicting_claims = distinct_claims.duplicated(
        subset=["observed_at", "retrieved_at"], keep=False
    )
    if conflicting_claims.any():
        conflict = distinct_claims.loc[conflicting_claims].iloc[0]
        raise ArchiveIntegrityError(source_identity, conflict["observed_at"])

    return (
        combined.sort_values(["observed_at", "retrieved_at"], kind="stable")
        .drop_duplicates(subset=["observed_at"], keep="last")
        .sort_values("observed_at", kind="stable")
        .reset_index(drop=True)
    )


def _prepare_merge(
    stored_data: bytes | None,
    incoming: pd.DataFrame,
    *,
    source_identity: str,
    expected_unit: str,
) -> _PreparedMerge:
    if stored_data is None:
        current = incoming.iloc[0:0].copy()
    else:
        current = read_observations(BytesIO(stored_data), expected_unit=expected_unit)
    merged = _merge_frames(current, incoming, source_identity=source_identity)
    if merged.equals(current):
        return _PreparedMerge(len(merged), True, None)
    return _PreparedMerge(len(merged), False, _parquet_bytes(merged))


def _event_fields(
    *,
    source_identity: str,
    outcome: str,
    started_at: float,
    record_count: int,
    attempt_count: int,
    observed_at: str | None = None,
) -> dict[str, object]:
    fields: dict[str, object] = {
        "component": "archive",
        "operation": "merge",
        "source_identity": source_identity,
        "outcome": outcome,
        "duration_ms": max(0, round((time.monotonic() - started_at) * 1000)),
        "record_count": record_count,
        "attempt_count": attempt_count,
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
    try:
        validated_incoming = await asyncio.to_thread(
            normalize_archive_frame, incoming, expected_unit=expected_unit
        )
        if validated_incoming.empty:
            result = MergeResult("unchanged", 0, 0)
            logging.info(
                "Archive merge had no observations for %s",
                source_identity,
                extra=_event_fields(
                    source_identity=source_identity,
                    outcome=result.outcome,
                    started_at=started_at,
                    record_count=result.record_count,
                    attempt_count=result.attempt_count,
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
            if prepared.unchanged:
                result = MergeResult("unchanged", record_count, attempt_count)
                logging.info(
                    "Archive merge was unchanged for %s",
                    source_identity,
                    extra=_event_fields(
                        source_identity=source_identity,
                        outcome=result.outcome,
                        started_at=started_at,
                        record_count=result.record_count,
                        attempt_count=result.attempt_count,
                    ),
                )
                return result

            if prepared.parquet_data is None:
                raise RuntimeError("Changed archive merge produced no Parquet data")
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

            result = MergeResult("success", record_count, attempt_count)
            logging.info(
                "Archive merge succeeded for %s",
                source_identity,
                extra=_event_fields(
                    source_identity=source_identity,
                    outcome=result.outcome,
                    started_at=started_at,
                    record_count=result.record_count,
                    attempt_count=result.attempt_count,
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
                source_identity=source_identity,
                outcome="failed",
                started_at=started_at,
                record_count=record_count,
                attempt_count=attempt_count,
                observed_at=observed_at,
            ),
        )
        raise
