"""Scalar-observation capture for the transitional in-process feed updater."""

import asyncio
import dataclasses
import datetime
import logging

import pandas as pd

from shallweswim.archive.merge import merge_observations
from shallweswim.archive.observations import (
    normalize_observations,
    partition_key,
    validate_source_identity,
)
from shallweswim.archive.store import object_store


@dataclasses.dataclass(frozen=True)
class CaptureResult:
    """Archive rows one capture added and revised, summed over its partitions."""

    new_count: int
    revised_count: int


def _partitions(
    frame: pd.DataFrame,
    source_identity: str,
    measurement: str,
    value_column: str,
    unit: str,
    retrieved_at: datetime.datetime,
) -> list[tuple[str, pd.DataFrame]]:
    """Normalize the UTC client frame and partition by UTC year.

    Source identity is preserved in the partition key prefix.
    """
    # Reject a mismatched identity before normalization, so the failure names
    # the identity rather than the value column that identity would not carry.
    validate_source_identity(source_identity, measurement)
    normalized = normalize_observations(
        frame,
        value_column=value_column,
        unit=unit,
        retrieved_at=retrieved_at,
    )
    if normalized.conflicting_dropped:
        logging.warning(
            "Archive normalization dropped %d conflicting repeated row(s) for %s",
            normalized.conflicting_dropped,
            source_identity,
            extra={
                "component": "archive",
                "operation": "normalize",
                "source_identity": source_identity,
                "outcome": "conflict_dropped",
                "record_count": normalized.conflicting_dropped,
            },
        )
    rows = normalized.frame
    return [
        (partition_key(source_identity, measurement, int(year)), partition)
        for year, partition in rows.groupby(rows["observed_at"].dt.year)
    ]


async def capture_observations(
    locator: str,
    *,
    frame: pd.DataFrame,
    source_identity: str,
    measurement: str,
    value_column: str,
    unit: str,
    retrieved_at: datetime.datetime,
) -> CaptureResult:
    """Merge each UTC year; the feed caller isolates preparation failures.

    The frame arrives as the client returned it, indexed by timezone-aware UTC
    instants. Returns the rows this fetch added and revised, which the calling
    feed keeps for the capture job's run summary.

    Args:
        locator: The archive store locator, as `archive.store.object_store`
            resolves it: a bucket name, a filesystem path, or `memory`.
    """
    partitions = await asyncio.to_thread(
        _partitions,
        frame,
        source_identity,
        measurement,
        value_column,
        unit,
        retrieved_at,
    )
    if not partitions:
        return CaptureResult(0, 0)
    store = await asyncio.to_thread(object_store, locator)
    new_count = 0
    revised_count = 0
    for key, incoming in partitions:
        try:
            merged = await merge_observations(
                store,
                key=key,
                source_identity=source_identity,
                incoming=incoming,
                expected_unit=unit,
            )
        except Exception:
            # The merge writer already emitted the failed event. Continue so a
            # failed year does not prevent capture of other years in this fetch.
            continue
        new_count += merged.new_count
        revised_count += merged.revised_count
    return CaptureResult(new_count, revised_count)
